import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from isaacgym import gymapi
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gym.spaces import Dict, Box
from collections import deque
from datetime import datetime

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

logger = CustomLogger("navigation_task_gmm_noise")


class NavigationTaskGmmNoise(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device

        self.reward_params = {}
        for key in self.task_config.reward_parameters.keys():
            self.reward_params[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        # Success condition configuration
        self.success_config = self.task_config.success_config

        logger.info("Building environment for navigation task (GMM noise).")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)

        if self.task_config.vae_config.use_vae:
            self.vae_model = VAEImageEncoder(config=self.task_config.vae_config, device=self.device)
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = lambda x: x

        self.obs_dict = self.sim_env.get_obs()
        if "task_external_force_tensor" not in self.obs_dict:
            self.obs_dict["task_external_force_tensor"] = torch.zeros(
                (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
            )
        self.task_external_force_tensor = self.obs_dict["task_external_force_tensor"]
        if "task_external_torque_tensor" not in self.obs_dict:
            self.obs_dict["task_external_torque_tensor"] = torch.zeros(
                (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
            )
        self.task_external_torque_tensor = self.obs_dict["task_external_torque_tensor"]
        # Physics stepping info (used to keep one-control-step recoil impulse consistent).
        self.physics_steps_per_env_step_mean = float(
            getattr(self.sim_env.cfg.env, "num_physics_steps_per_env_step_mean", 1.0)
        )
        self.physics_steps_per_env_step_std = float(
            getattr(self.sim_env.cfg.env, "num_physics_steps_per_env_step_std", 0.0)
        )
        self._recoil_dt_warning_emitted = False
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        self.base_observation_space_dim = int(
            getattr(self.task_config, "base_observation_dim", 13)
        )
        if self.base_observation_space_dim != 13:
            logger.warning(
                "Overriding base_observation_dim to 13 for asymmetric frame-stacked observations."
            )
            self.base_observation_space_dim = 13

        self.obs_frame_stack = int(getattr(self.task_config, "frame_stack", 6))
        if self.obs_frame_stack < 1:
            logger.warning("frame_stack must be >= 1. Falling back to 1.")
            self.obs_frame_stack = 1

        self.privileged_observation_dim = int(
            getattr(self.task_config, "privileged_observation_space_dim", 13)
        )
        if self.privileged_observation_dim != 13:
            logger.warning(
                "Overriding privileged_observation_space_dim to 13 "
                "for the asymmetric critic privileged inputs."
            )
            self.privileged_observation_dim = 13

        self.actor_observation_space_dim = (
            self.base_observation_space_dim * self.obs_frame_stack
        )
        self.critic_observation_space_dim = (
            self.actor_observation_space_dim + self.privileged_observation_dim
        )
        if int(
            getattr(self.task_config, "observation_space_dim", self.actor_observation_space_dim)
        ) != self.actor_observation_space_dim:
            logger.warning(
                f"Overriding observation_space_dim to {self.actor_observation_space_dim} "
                "for stacked actor observations."
            )
        if int(
            getattr(
                self.task_config,
                "critic_observation_space_dim",
                self.critic_observation_space_dim,
            )
        ) != self.critic_observation_space_dim:
            logger.warning(
                f"Overriding critic_observation_space_dim to {self.critic_observation_space_dim} "
                "for asymmetric critic observations."
            )
        self.use_central_value = bool(getattr(self.task_config, "use_central_value", True))
        self.task_config.base_observation_dim = self.base_observation_space_dim
        self.task_config.frame_stack = self.obs_frame_stack
        self.task_config.observation_space_dim = self.actor_observation_space_dim
        self.task_config.critic_observation_space_dim = self.critic_observation_space_dim
        self.task_config.use_central_value = self.use_central_value

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
                "states": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.critic_observation_space_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        self.action_transformation_function = self.task_config.action_transformation_function

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "states": torch.zeros(
                (self.sim_env.num_envs, self.task_config.critic_observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
        }
        self.base_task_observations = torch.zeros(
            (self.sim_env.num_envs, self.base_observation_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.obs_frame_buffer = torch.zeros(
            (
                self.sim_env.num_envs,
                self.obs_frame_stack,
                self.base_observation_space_dim,
            ),
            device=self.device,
            requires_grad=False,
        )
        self.privileged_observations = torch.zeros(
            (self.sim_env.num_envs, self.privileged_observation_dim),
            device=self.device,
            requires_grad=False,
        )

        self.num_task_steps = 0
        self.infos = {}

        self.env_bounds_min = torch.tensor(
            self.task_config.env_bounds_min, device=self.device, requires_grad=False
        )
        self.env_bounds_max = torch.tensor(
            self.task_config.env_bounds_max, device=self.device, requires_grad=False
        )
        self.d_max = torch.norm(self.env_bounds_max - self.env_bounds_min)

        self._apply_fixed_env_bounds()
        self._set_fixed_obstacle_count()
        self.fixed_env_enabled = False
        self.fixed_r_obs = 1.0
        self.best_total_score = float("-inf")
        self.best_position = None
        
        # Spawn position and d_max buffers (for distance reward calculation)
        self.spawn_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        # Per-episode spawn-height reference used by altitude penalty.
        self.spawn_height_reference = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.d_max = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        
        # GMM Physical Force buffers
        self.gmm_force_config = self.task_config.gmm_force_config
        # Layer 1 (main wind): sampled per-env at reset and fixed in one episode.
        self.main_wind_direction = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.main_wind_speed = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.main_wind_vector = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        # Per-env cap for local GMM wind magnitude (Layer 2), sampled at reset.
        self.local_wind_max_speed = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        # Layer 2 (local GMM disturbance): direction can evolve smoothly.
        self.gmm_force_direction = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.gmm_target_direction = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.gmm_force_update_counter = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        # Simplified Dryden turbulence buffers (three independent first-order filters).
        self.dryden_config = getattr(self.task_config, "dryden_config", None)
        self.dryden_enabled = bool(
            self.dryden_config is not None
            and bool(getattr(self.dryden_config, "enable_dryden", False))
        )
        self.dryden_wind_state = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.dryden_sigma = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.dryden_tau = torch.ones(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        self.drop_model_config = getattr(self.task_config, "drop_model_config", None)
        self.drop_impact_config = getattr(self.task_config, "drop_impact_config", None)
        self.drop_reward_config = getattr(self.task_config, "drop_reward_config", None)
        self.fixed_payload_mount_index = -1
        self.fixed_payload_offset_body = torch.zeros(
            3, device=self.device, dtype=torch.float32, requires_grad=False
        )
        if self.drop_impact_config is not None:
            mount_points = getattr(self.drop_impact_config, "payload_mount_points_body", None)
            if isinstance(mount_points, (list, tuple)) and len(mount_points) > 0:
                n_mount = len(mount_points)
                cfg_idx = int(getattr(self.drop_impact_config, "fixed_payload_mount_index", -1))
                random_fixed = bool(
                    getattr(self.drop_impact_config, "random_fixed_payload_mount", True)
                )
                if 0 <= cfg_idx < n_mount:
                    selected_idx = cfg_idx
                elif random_fixed:
                    selected_idx = int(torch.randint(0, n_mount, (1,), device=self.device).item())
                else:
                    selected_idx = 0
                self.fixed_payload_mount_index = selected_idx
                self.fixed_payload_offset_body = torch.tensor(
                    mount_points[selected_idx],
                    device=self.device,
                    dtype=torch.float32,
                )
                logger.warning(
                    "Fixed DROP mount selected: idx=%d, offset_body=%s",
                    self.fixed_payload_mount_index,
                    [float(x) for x in self.fixed_payload_offset_body.tolist()],
                )
            else:
                self.fixed_payload_offset_body = torch.tensor(
                    getattr(self.drop_impact_config, "payload_offset_body", [0.0, 0.0, 0.0]),
                    device=self.device,
                    dtype=torch.float32,
                )
        self.child_landing_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.child_landing_xy_distance = torch.full(
            (self.sim_env.num_envs,), float("inf"), device=self.device, requires_grad=False
        )
        self.child_drop_triggered = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.child_has_dropped = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.step_drop_event_mask = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.step_impulse_metric = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_attitude_theta = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_roll = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_pitch = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_yaw = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_delta_theta = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_delta_v = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_delta_omega = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_heading_error = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_landing_error_xy = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_release_to_target_xy = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.confidence_history_len = int(
            max(1, getattr(self.drop_model_config, "confidence_history_len", 5))
        )
        confidence_history_init = float(
            max(getattr(self.drop_model_config, "confidence_error_scale", 3.0), 1e-6)
        )
        self.pred_drop_error_history = torch.full(
            (self.sim_env.num_envs, self.confidence_history_len),
            confidence_history_init,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.release_confidence = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.release_confidence_error_component = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.release_confidence_risk_component = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.release_confidence_stability_component = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.confidence_gate_blocked_mask = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.record_drop_trajectory_for_eval = bool(
            getattr(
                self.task_config,
                "record_drop_trajectory_for_eval",
                os.environ.get("AERIAL_GYM_EVAL_MODE", "0") == "1",
            )
        )
        self.eval_drop_trace_num_samples = int(
            max(8, getattr(self.drop_model_config, "eval_trace_num_samples", 64))
        )
        self.step_drop_trace_norm_t = torch.zeros(
            (self.sim_env.num_envs, self.eval_drop_trace_num_samples),
            device=self.device,
            requires_grad=False,
        )
        self.step_drop_trace_error_xy = torch.zeros(
            (self.sim_env.num_envs, self.eval_drop_trace_num_samples),
            device=self.device,
            requires_grad=False,
        )
        self.step_drop_trace_pos_x = torch.zeros(
            (self.sim_env.num_envs, self.eval_drop_trace_num_samples),
            device=self.device,
            requires_grad=False,
        )
        self.step_drop_trace_pos_y = torch.zeros(
            (self.sim_env.num_envs, self.eval_drop_trace_num_samples),
            device=self.device,
            requires_grad=False,
        )
        self.step_drop_trace_pos_z = torch.zeros(
            (self.sim_env.num_envs, self.eval_drop_trace_num_samples),
            device=self.device,
            requires_grad=False,
        )
        self.step_drop_trace_fall_steps = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.step_drop_trace_fall_time = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        # Runtime rigid-body update cache for DROP (mass/COM/inertia update at release time).
        self._drop_rb_cache_ready = False
        self._drop_base_body_index = 0
        self._drop_body_names = []
        self._drop_mount_to_body_index = {}
        self._drop_body_props_default = []
        self._drop_robot_mass_default = None
        self._drop_robot_inertia_default = None
        self._robot_com_body_default = torch.zeros(
            3, device=self.device, dtype=torch.float32, requires_grad=False
        )
        self._robot_com_body = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, dtype=torch.float32, requires_grad=False
        )
        self._mother_rigidbody_updated = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self._payload_mass_reference = 0.0
        self._payload_com_body_reference = torch.zeros(
            3, device=self.device, dtype=torch.float32, requires_grad=False
        )
        
        # Success counter for continuous success check (NEW)
        self.success_counter = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        # Episode outcome tracking for success rate calculation (last 100 episodes)
        self.recent_episodes = deque(maxlen=100)
        
        # Distance tracking for reward shaping
        self.previous_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device
        )
        self.previous_distance = torch.zeros(self.sim_env.num_envs, device=self.device)
        self.initial_distance_xy = torch.zeros(self.sim_env.num_envs, device=self.device)
        self.previous_distance_xy = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        # Velocity tracking for acceleration penalty
        self.previous_velocity = torch.zeros(self.sim_env.num_envs, 3, device=self.device)
        self.previous_actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim), device=self.device
        )
        
        # Hover time counter for cumulative hover reward (NEW)
        self.hover_time_counter = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        self._load_fixed_env_preset()
        
        self.best_point_path = self._resolve_best_point_path()
        self._write_reward_weight_snapshot()
        log_flag = os.environ.get("AERIAL_GYM_LOG_STEP_SCORES", "0").strip().lower()
        self.log_step_scores = log_flag in ("1", "true", "yes", "y", "t")
        self.log_step_env_id = 0
        self._episode_step_log = []
        self._episode_step_idx = 0
        self._episode_log_written = False

        # Early crash handling configuration
        self.early_crash_config = self.task_config.early_crash_config
        self._early_crash_retries = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )

        # Eval mode: exit after first episode ends (only for run_parallel_eval.py --run-runner)
        eval_mode_flag = os.environ.get("AERIAL_GYM_EVAL_MODE", "0").strip().lower()
        self._eval_mode = eval_mode_flag in ("1", "true", "yes", "y", "t")
        self._eval_init_phase_complete = False  # Track if we've passed initialization phase

        # TensorBoard writer for task diagnostics (lazily initialized on first use)
        self._writer_initialized = False
        self.writer = None



        # Ensure assets are placed within the fixed bounds.
        self.sim_env.reset()
        self._initialize_drop_rigidbody_cache()
        
        # Initialize Arrival Metric Buffer
        self.has_arrived = torch.zeros(self.sim_env.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.recent_arrivals = deque(maxlen=100)
        
        # Initialize success_buf before reset_idx
        self.success_buf = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.long)
        
        # Initialize extras dictionary for TensorBoard logging
        self.extras = {}
        
        # Buffer for tracking Final J values of terminated episodes (NEW)
        self.final_J_buffer = deque(maxlen=1000)
        self.final_arrival_buffer = deque(maxlen=100)  # [NEW] Track arrival at episode end only
        # Sliding window for recent arrival rate (last N episodes)
        self.recent_arrival_buffer = deque(maxlen=100)
        
        # Initialize episode statistics buffers
        self.episode_sums = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.float)
        self.episode_lengths = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.int)
        self.episode_pot_sums = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.float)
        self.episode_safe_sums = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.float)
        self.episode_smooth_sums = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.float)
        self.episode_hover_sums = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.float)
        self.episode_threshold_sums = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.float)
        self.episode_anchor_sums = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.float)
        self.hover_good_min = torch.full(
            (self.sim_env.num_envs,), float("inf"), device=self.device
        )
        self.hover_good_max = torch.full(
            (self.sim_env.num_envs,), -float("inf"), device=self.device
        )
        
        # Track minimum J value (closest approach/best state) per episode (NEW)
        self.episode_min_J = torch.full((self.sim_env.num_envs,), 1000.0, device=self.device)
        self.min_J_buffer = deque(maxlen=1000)

        # Rolling window stats for end-of-episode logging (per 1000 steps)
        self.window_done_total = 0
        self.window_done_non_crash = 0
        self.window_done_speed_sum = 0.0
        self.window_done_hover_reward_count = 0
        self.window_drop_end_count = 0
        self.window_no_drop_done_count = 0
        self.window_drop_sample_count = 0
        self.window_landing_error_xy_sum = 0.0
        self.window_release_to_target_xy_sum = 0.0
        self.window_release_to_target_xy_min = float("inf")
        self.window_release_to_target_xy_max = 0.0
        self.window_drop_roll_deg_sum = 0.0
        self.window_drop_roll_deg_min = float("inf")
        self.window_drop_roll_deg_max = -float("inf")
        self.window_drop_pitch_deg_sum = 0.0
        self.window_drop_pitch_deg_min = float("inf")
        self.window_drop_pitch_deg_max = -float("inf")
        self.window_drop_attitude_total_deg_sum = 0.0
        self.window_drop_attitude_total_deg_sq_sum = 0.0
        self.window_impulse_metric_sum = 0.0
        self.window_impulse_metric_sq_sum = 0.0
        self.landing_error_xy_ema = None
        self.landing_error_ema_alpha = float(
            getattr(self.drop_reward_config, "landing_error_ema_alpha", 0.9)
        )
        self.attitude_total_deg_ema = None
        self.attitude_total_ema_alpha = float(
            getattr(self.drop_reward_config, "attitude_total_ema_alpha", 0.9)
        )
        self.current_score_d0 = float(getattr(self.drop_reward_config, "score_d0", 4.0))
        self.score_d0_curriculum_stage = 0
        self._train_epoch = 0
        self.spawn_z_curriculum_min_current = float(
            getattr(self.task_config, "spawn_random_z_min", 2.0)
        )
        self.spawn_z_curriculum_alpha_current = 0.0
        self._last_console_stats_epoch = -1
        self._target_marker_draw_enabled = True
        self._target_marker_warned = False
        # Per-environment world origins (preferred) for converting world XY to
        # env-local XY in release-distance diagnostics.
        self._has_env_origins = "env_origins" in self.obs_dict
        # Fallback: infer grid offsets if env_origins are unavailable.
        env_cfg = self.sim_env.IGE_env.cfg.env
        self._envs_per_row = max(1, int(np.sqrt(self.sim_env.num_envs)))
        self._env_grid_spacing_xy = torch.tensor(
            [
                float(env_cfg.upper_bound_max[0]) - float(env_cfg.lower_bound_min[0]),
                float(env_cfg.upper_bound_max[1]) - float(env_cfg.lower_bound_min[1]),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.prev_predicted_drop_error_xy = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        
        
        self.reset_idx(torch.arange(self.sim_env.num_envs, device=self.device))

    def _get_env_world_offset_xy(self, env_ids):
        """Get per-env world origin offset in XY."""
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if self._has_env_origins:
            return self.obs_dict["env_origins"][env_ids, 0:2]
        row = torch.div(env_ids, self._envs_per_row, rounding_mode="floor").to(torch.float32)
        col = torch.remainder(env_ids, self._envs_per_row).to(torch.float32)
        offset_x = col * self._env_grid_spacing_xy[0]
        offset_y = row * self._env_grid_spacing_xy[1]
        return torch.stack([offset_x, offset_y], dim=1)

    def _target_position_is_local_xy(self):
        """
        Heuristic: if almost all targets lie inside configured env bounds, treat
        them as per-env local coordinates.
        """
        target_xy = self.target_position[:, 0:2]
        lower_xy = self.env_bounds_min[0:2].view(1, 2)
        upper_xy = self.env_bounds_max[0:2].view(1, 2)
        inside = ((target_xy >= (lower_xy - 1e-3)) & (target_xy <= (upper_xy + 1e-3))).all(dim=1)
        return bool((inside.float().mean() > 0.99).item())

    def _position_xy_is_local(self, position_xy):
        """Heuristic: detect whether XY coordinates are already env-local."""
        lower_xy = self.env_bounds_min[0:2].view(1, 2)
        upper_xy = self.env_bounds_max[0:2].view(1, 2)
        inside = ((position_xy >= (lower_xy - 1e-3)) & (position_xy <= (upper_xy + 1e-3))).all(dim=1)
        return bool((inside.float().mean() > 0.99).item())

    def _compute_release_to_target_xy_local(self, env_ids, release_pos_world):
        """
        Compute release->target XY distance using the same frame as task states.
        This keeps the metric aligned with reward/state computations.
        """
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        release_xy = release_pos_world[:, 0:2]
        target_xy = self.target_position[env_ids, 0:2]
        return torch.norm(target_xy - release_xy, dim=1)

    def _resolve_best_point_path(self):
        runs_dir = os.environ.get("AERIAL_GYM_RUNS_DIR", "runs")
        experiment_name = os.environ.get("AERIAL_GYM_EXPERIMENT_NAME", "experiment")
        run_dir = None
        if os.path.isdir(runs_dir):
            candidates = [
                os.path.join(runs_dir, name)
                for name in os.listdir(runs_dir)
                if name.startswith(experiment_name)
            ]
            if candidates:
                run_dir = max(candidates, key=os.path.getmtime)
        if run_dir is None:
            run_dir = os.path.join(runs_dir, experiment_name)
        os.makedirs(run_dir, exist_ok=True)
        self.best_point_path = os.path.join(run_dir, "best_point.txt")
        return self.best_point_path

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [NavigationTaskGmmNoise._to_serializable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): NavigationTaskGmmNoise._to_serializable(v) for k, v in value.items()}
        return str(value)

    def _config_object_to_dict(self, cfg_obj):
        if cfg_obj is None:
            return {}
        out = {}
        for name in dir(cfg_obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(cfg_obj, name)
            except Exception:
                continue
            if callable(value):
                continue
            out[name] = self._to_serializable(value)
        return out

    def _get_ppo_config_snapshot(self):
        """Get the rl_games PPO config passed by runner after CLI overrides."""
        ppo_config_path = os.environ.get("AERIAL_GYM_PPO_CONFIG_PATH", "").strip()
        config_json = os.environ.get("AERIAL_GYM_PPO_CONFIG_JSON", "").strip()
        ppo_config_full = {}
        if config_json:
            try:
                ppo_config_full = json.loads(config_json)
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse AERIAL_GYM_PPO_CONFIG_JSON: {exc}")

        params = ppo_config_full.get("params", {}) if isinstance(ppo_config_full, dict) else {}
        train_cfg = params.get("config", {}) if isinstance(params, dict) else {}
        network_cfg = params.get("network", {}) if isinstance(params, dict) else {}
        algo_cfg = params.get("algo", {}) if isinstance(params, dict) else {}
        model_cfg = params.get("model", {}) if isinstance(params, dict) else {}

        config_keys = (
            "name",
            "env_name",
            "learning_rate",
            "lr_schedule",
            "kl_threshold",
            "grad_norm",
            "entropy_coef",
            "truncate_grads",
            "e_clip",
            "clip_value",
            "num_actors",
            "horizon_length",
            "minibatch_size",
            "mini_epochs",
            "critic_coef",
            "normalize_input",
            "seq_length",
            "bounds_loss_coef",
            "max_epochs",
            "normalize_value",
            "use_diagnostics",
            "value_bootstrap",
            "use_smooth_clamp",
            "gamma",
            "tau",
            "normalize_advantage",
            "save_frequency",
            "save_best_after",
            "score_to_win",
        )
        ppo_hyperparameters = {
            "config_path": ppo_config_path if ppo_config_path else None,
            "seed": self._to_serializable(params.get("seed")) if isinstance(params, dict) else None,
            "algo": self._to_serializable(algo_cfg),
            "model": self._to_serializable(model_cfg),
            "network": self._to_serializable(network_cfg),
            "config": {
                key: self._to_serializable(train_cfg[key])
                for key in config_keys
                if isinstance(train_cfg, dict) and key in train_cfg
            },
        }
        if isinstance(train_cfg, dict) and "env_config" in train_cfg:
            ppo_hyperparameters["env_config"] = self._to_serializable(train_cfg["env_config"])
        if isinstance(train_cfg, dict) and "reward_shaper" in train_cfg:
            ppo_hyperparameters["reward_shaper"] = self._to_serializable(
                train_cfg["reward_shaper"]
            )
        if isinstance(train_cfg, dict) and "task_overrides" in train_cfg:
            ppo_hyperparameters["task_overrides"] = self._to_serializable(
                train_cfg["task_overrides"]
            )

        return {
            "ppo_config_path": ppo_config_path if ppo_config_path else None,
            "ppo_hyperparameters": ppo_hyperparameters,
            "ppo_config_full": self._to_serializable(ppo_config_full),
        }

    def _write_reward_weight_snapshot(self, run_dir=None):
        """Write reward-weight/config snapshot to a run directory."""
        try:
            if run_dir is None:
                self._resolve_best_point_path()
                run_dir = os.path.dirname(self.best_point_path)
            os.makedirs(run_dir, exist_ok=True)

            reward_params_dump = {
                key: self._to_serializable(value) for key, value in self.reward_params.items()
            }
            drop_reward_dump = self._config_object_to_dict(self.drop_reward_config)
            weight_like_keys = (
                "weight",
                "penalty",
                "alpha",
                "beta",
                "lambda",
                "score",
                "threshold",
            )
            drop_reward_weight_fields = {
                k: v for k, v in drop_reward_dump.items() if any(tag in k for tag in weight_like_keys)
            }
            resume_flag = os.environ.get("AERIAL_GYM_IS_RESUME", "0").strip().lower()
            is_resumed_training = resume_flag in ("1", "true", "yes", "y", "t")
            resume_checkpoint = os.environ.get("AERIAL_GYM_RESUME_CHECKPOINT", "").strip()
            if not is_resumed_training:
                resume_checkpoint = ""

            snapshot = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "experiment_name": os.environ.get("AERIAL_GYM_EXPERIMENT_NAME", "experiment"),
                "is_resumed_training": bool(is_resumed_training),
                "resume_checkpoint": resume_checkpoint if resume_checkpoint else None,
                "reward_parameters": reward_params_dump,
                "drop_reward_config_weights": drop_reward_weight_fields,
                "drop_reward_config_full": drop_reward_dump,
                "drop_model_config": self._config_object_to_dict(self.drop_model_config),
                "drop_impact_config": self._config_object_to_dict(self.drop_impact_config),
                "success_config": self._config_object_to_dict(self.success_config),
            }
            snapshot.update(self._get_ppo_config_snapshot())

            snapshot_path = os.path.join(run_dir, "reward_weights_snapshot.json")
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False, indent=2)
            logger.info("Saved reward weight snapshot: %s", snapshot_path)
        except Exception as e:
            logger.warning(f"Failed to save reward weight snapshot: {e}")

    def set_train_info(self, frame, algo):
        """Receive training progress metadata from rl_games."""
        try:
            self._train_epoch = int(getattr(algo, "epoch_num", self._train_epoch))
        except Exception:
            pass

    def _update_score_d0_curriculum(self):
        cfg = self.drop_reward_config
        if not bool(getattr(cfg, "score_d0_curriculum_enable", False)):
            self.current_score_d0 = float(getattr(cfg, "score_d0", self.current_score_d0))
            return

        curriculum_values = list(getattr(cfg, "score_d0_curriculum_values", [4.0, 3.0, 2.5, 2.0]))
        if len(curriculum_values) == 0:
            self.current_score_d0 = float(getattr(cfg, "score_d0", self.current_score_d0))
            return

        min_drop_rates = list(getattr(cfg, "score_d0_curriculum_min_drop_rate", []))
        max_error_ema = list(getattr(cfg, "score_d0_curriculum_max_error_ema", []))
        max_stage = len(curriculum_values) - 1
        self.score_d0_curriculum_stage = int(np.clip(self.score_d0_curriculum_stage, 0, max_stage))

        drop_count_window = int(getattr(self, "window_drop_end_count", 0))
        no_drop_count_window = int(getattr(self, "window_no_drop_done_count", 0))
        drop_total = drop_count_window + no_drop_count_window
        if drop_total <= 0 or self.landing_error_xy_ema is None:
            self.current_score_d0 = float(curriculum_values[self.score_d0_curriculum_stage])
            return

        drop_rate = float(drop_count_window) / float(drop_total)
        landing_error_ema = float(self.landing_error_xy_ema)
        next_stage = self.score_d0_curriculum_stage + 1
        if next_stage <= max_stage:
            rate_req = float(min_drop_rates[next_stage - 1]) if (next_stage - 1) < len(min_drop_rates) else 1.0
            error_req = float(max_error_ema[next_stage - 1]) if (next_stage - 1) < len(max_error_ema) else -1.0
            if drop_rate >= rate_req and landing_error_ema <= error_req:
                self.score_d0_curriculum_stage = next_stage

        self.current_score_d0 = float(curriculum_values[self.score_d0_curriculum_stage])

    def _get_spawn_z_sampling_range(self):
        """Get current spawn-Z sampling range, optionally with epoch-based curriculum."""
        z_min_cfg = float(getattr(self.task_config, "spawn_random_z_min", 2.0))
        z_max_cfg = float(getattr(self.task_config, "spawn_random_z_max", 13.0))
        if z_max_cfg < z_min_cfg:
            z_min_cfg, z_max_cfg = z_max_cfg, z_min_cfg

        use_curriculum = bool(getattr(self.task_config, "spawn_z_curriculum_enable", False))
        if not use_curriculum:
            self.spawn_z_curriculum_min_current = z_min_cfg
            self.spawn_z_curriculum_alpha_current = 1.0
            return z_min_cfg, z_max_cfg

        start_min = float(
            getattr(self.task_config, "spawn_z_curriculum_start_min", z_max_cfg)
        )
        end_min = float(getattr(self.task_config, "spawn_z_curriculum_end_min", z_min_cfg))
        warmup_epochs = int(getattr(self.task_config, "spawn_z_curriculum_warmup_epochs", 0))
        full_epochs = int(getattr(self.task_config, "spawn_z_curriculum_full_epochs", 2000))
        if full_epochs < warmup_epochs:
            full_epochs = warmup_epochs

        epoch = max(int(getattr(self, "_train_epoch", 0)), 0)
        if epoch <= warmup_epochs:
            alpha = 0.0
        elif epoch >= full_epochs:
            alpha = 1.0
        else:
            alpha = float(epoch - warmup_epochs) / float(max(full_epochs - warmup_epochs, 1))

        z_min_curriculum = start_min + (end_min - start_min) * alpha
        z_min_curriculum = float(np.clip(z_min_curriculum, z_min_cfg, z_max_cfg))
        z_max_curriculum = z_max_cfg
        if z_max_curriculum < z_min_curriculum:
            z_max_curriculum = z_min_curriculum

        self.spawn_z_curriculum_min_current = z_min_curriculum
        self.spawn_z_curriculum_alpha_current = alpha
        return z_min_curriculum, z_max_curriculum

    def _apply_fixed_env_bounds(self):
        env = self.sim_env.IGE_env
        num_envs = env.env_lower_bound_min.shape[0]
        bounds_min = self.env_bounds_min.view(1, 3).repeat(num_envs, 1)
        bounds_max = self.env_bounds_max.view(1, 3).repeat(num_envs, 1)
        env.env_lower_bound_min = bounds_min.clone()
        env.env_lower_bound_max = bounds_min.clone()
        env.env_upper_bound_min = bounds_max.clone()
        env.env_upper_bound_max = bounds_max.clone()
        env.env_lower_bound.copy_(bounds_min)
        env.env_upper_bound.copy_(bounds_max)

    def _set_fixed_obstacle_count(self):
        self.curriculum_level = int(self.task_config.num_obstacles_in_env)
        self.obs_dict["curriculum_level"] = self.curriculum_level
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = 0.0

    @staticmethod
    def _quat_xyzw_to_rotmat_np(quat_xyzw):
        """Convert [x, y, z, w] quaternion to 3x3 rotation matrix."""
        x = float(quat_xyzw[0])
        y = float(quat_xyzw[1])
        z = float(quat_xyzw[2])
        w = float(quat_xyzw[3])
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        xw = x * w
        yw = y * w
        zw = z * w
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
                [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
                [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _capture_rigid_body_property(prop):
        return {
            "mass": float(prop.mass),
            "com": np.array([prop.com.x, prop.com.y, prop.com.z], dtype=np.float64),
            "inertia": np.array(
                [
                    [prop.inertia.x.x, prop.inertia.x.y, prop.inertia.x.z],
                    [prop.inertia.y.x, prop.inertia.y.y, prop.inertia.y.z],
                    [prop.inertia.z.x, prop.inertia.z.y, prop.inertia.z.z],
                ],
                dtype=np.float64,
            ),
        }

    @staticmethod
    def _apply_cached_rigid_body_property(prop, cached):
        inertia = cached["inertia"]
        prop.mass = float(cached["mass"])
        prop.com.x = float(cached["com"][0])
        prop.com.y = float(cached["com"][1])
        prop.com.z = float(cached["com"][2])
        prop.inertia.x.x = float(inertia[0, 0])
        prop.inertia.x.y = float(inertia[0, 1])
        prop.inertia.x.z = float(inertia[0, 2])
        prop.inertia.y.x = float(inertia[1, 0])
        prop.inertia.y.y = float(inertia[1, 1])
        prop.inertia.y.z = float(inertia[1, 2])
        prop.inertia.z.x = float(inertia[2, 0])
        prop.inertia.z.y = float(inertia[2, 1])
        prop.inertia.z.z = float(inertia[2, 2])

    def _compute_robot_aggregate_dynamics(self, env_id, rb_props=None):
        """
        Compute aggregate (mass, inertia, COM) in base-link frame for one environment.
        This keeps control tensors consistent after payload mass changes.
        """
        ige_env = self.sim_env.IGE_env
        gym = ige_env.gym
        env_handle = ige_env.env_handles[env_id]
        robot_handle = self.sim_env.robot_manager.robot_handles[env_id]
        if rb_props is None:
            rb_props = gym.get_actor_rigid_body_properties(env_handle, robot_handle)
        rb_states = gym.get_actor_rigid_body_states(env_handle, robot_handle, gymapi.STATE_ALL)
        base_idx = int(self._drop_base_body_index)

        root_pos_raw = rb_states[base_idx][0][0]
        root_quat_raw = rb_states[base_idx][0][1]
        root_pos = np.array(
            [float(root_pos_raw[0]), float(root_pos_raw[1]), float(root_pos_raw[2])], dtype=np.float64
        )
        root_quat = np.array(
            [
                float(root_quat_raw[0]),
                float(root_quat_raw[1]),
                float(root_quat_raw[2]),
                float(root_quat_raw[3]),
            ],
            dtype=np.float64,
        )
        rot_root_world = self._quat_xyzw_to_rotmat_np(root_quat)
        rot_world_root = rot_root_world.T

        body_cache = []
        total_mass = 0.0
        weighted_com_root = np.zeros(3, dtype=np.float64)
        for idx, prop in enumerate(rb_props):
            mass = float(prop.mass)
            if mass <= 1e-9:
                continue
            pos_raw = rb_states[idx][0][0]
            quat_raw = rb_states[idx][0][1]
            pos_world = np.array(
                [float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2])], dtype=np.float64
            )
            quat_world = np.array(
                [float(quat_raw[0]), float(quat_raw[1]), float(quat_raw[2]), float(quat_raw[3])],
                dtype=np.float64,
            )
            rot_body_world = self._quat_xyzw_to_rotmat_np(quat_world)
            com_local = np.array([prop.com.x, prop.com.y, prop.com.z], dtype=np.float64)
            com_world = pos_world + rot_body_world @ com_local
            com_root = rot_world_root @ (com_world - root_pos)
            body_cache.append((mass, rot_body_world, com_root, prop))
            total_mass += mass
            weighted_com_root += mass * com_root

        if total_mass <= 1e-9:
            return (
                1.0,
                torch.eye(3, device=self.device, dtype=torch.float32),
                torch.zeros(3, device=self.device, dtype=torch.float32),
            )

        com_root_total = weighted_com_root / total_mass
        inertia_root_total = np.zeros((3, 3), dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)
        for mass, rot_body_world, com_root, prop in body_cache:
            rot_body_root = rot_world_root @ rot_body_world
            inertia_body = np.array(
                [
                    [prop.inertia.x.x, prop.inertia.x.y, prop.inertia.x.z],
                    [prop.inertia.y.x, prop.inertia.y.y, prop.inertia.y.z],
                    [prop.inertia.z.x, prop.inertia.z.y, prop.inertia.z.z],
                ],
                dtype=np.float64,
            )
            inertia_about_body_com_in_root = rot_body_root @ inertia_body @ rot_body_root.T
            r = com_root - com_root_total
            inertia_shift = mass * ((np.dot(r, r) * identity) - np.outer(r, r))
            inertia_root_total += inertia_about_body_com_in_root + inertia_shift

        return (
            float(total_mass),
            torch.tensor(inertia_root_total, device=self.device, dtype=torch.float32),
            torch.tensor(com_root_total, device=self.device, dtype=torch.float32),
        )

    def _initialize_drop_rigidbody_cache(self):
        """Cache default rigid-body properties and body-index mapping for DROP updates."""
        self._drop_rb_cache_ready = False
        if self.drop_impact_config is None:
            return
        try:
            ige_env = self.sim_env.IGE_env
            gym = ige_env.gym
            env_handles = ige_env.env_handles
            robot_handles = self.sim_env.robot_manager.robot_handles
            if len(env_handles) == 0 or len(robot_handles) == 0:
                return

            body_names = list(gym.get_actor_rigid_body_names(env_handles[0], robot_handles[0]))
            self._drop_body_names = body_names
            base_name = str(
                getattr(self.sim_env.robot_manager.cfg.robot_asset, "base_link_name", "base_link")
            )
            self._drop_base_body_index = body_names.index(base_name) if base_name in body_names else 0

            mount_points = getattr(self.drop_impact_config, "payload_mount_points_body", None)
            num_mounts = len(mount_points) if isinstance(mount_points, (list, tuple)) else 0
            self._drop_mount_to_body_index = {}
            for mount_idx in range(num_mounts):
                body_idx = -1
                for candidate in (
                    f"payload_{mount_idx + 1}_base",
                    f"payload_{mount_idx + 1}",
                    f"payload_{mount_idx + 1}_link",
                ):
                    if candidate in body_names:
                        body_idx = body_names.index(candidate)
                        break
                self._drop_mount_to_body_index[mount_idx] = body_idx

            self._drop_body_props_default = []
            for env_handle, robot_handle in zip(env_handles, robot_handles):
                props = gym.get_actor_rigid_body_properties(env_handle, robot_handle)
                self._drop_body_props_default.append(
                    [self._capture_rigid_body_property(prop) for prop in props]
                )

            self._drop_robot_mass_default = self.obs_dict["robot_mass"].clone()
            self._drop_robot_inertia_default = self.obs_dict["robot_inertia"].clone()
            _, _, default_com_root = self._compute_robot_aggregate_dynamics(0)
            self._robot_com_body_default = default_com_root
            self._robot_com_body[:] = default_com_root.view(1, 3).expand(self.sim_env.num_envs, -1)
            payload_body_idx = self._drop_mount_to_body_index.get(
                int(self.fixed_payload_mount_index), -1
            )
            if payload_body_idx >= 0 and len(self._drop_body_props_default) > 0:
                payload_cached = self._drop_body_props_default[0][payload_body_idx]
                self._payload_mass_reference = float(payload_cached["mass"])
                self._payload_com_body_reference = torch.tensor(
                    payload_cached["com"], device=self.device, dtype=torch.float32
                )
            else:
                self._payload_mass_reference = float(getattr(self.drop_impact_config, "child_mass", 1.0))
                self._payload_com_body_reference.zero_()
            self._mother_rigidbody_updated[:] = False
            self._drop_rb_cache_ready = True
        except Exception as err:
            logger.warning(f"DROP rigid-body cache initialization failed: {err}")
            self._drop_rb_cache_ready = False

    def _update_mother_rigidbody_on_drop(self, env_ids):
        """At DROP, remove selected payload mass and sync aggregate mass/inertia tensors."""
        if (not self._drop_rb_cache_ready) or env_ids.numel() == 0:
            return

        payload_body_idx = self._drop_mount_to_body_index.get(int(self.fixed_payload_mount_index), -1)
        if payload_body_idx < 0:
            return

        residual_mass = float(getattr(self.drop_impact_config, "dropped_payload_residual_mass", 1e-4))
        residual_mass = max(residual_mass, 0.0)
        residual_inertia = float(getattr(self.drop_impact_config, "dropped_payload_residual_inertia", 1e-6))
        residual_inertia = max(residual_inertia, 1e-9)

        ige_env = self.sim_env.IGE_env
        gym = ige_env.gym
        env_handles = ige_env.env_handles
        robot_handles = self.sim_env.robot_manager.robot_handles
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        for env_id in env_ids.tolist():
            if bool(self._mother_rigidbody_updated[env_id].item()):
                continue
            try:
                env_handle = env_handles[env_id]
                robot_handle = robot_handles[env_id]
                rb_props = gym.get_actor_rigid_body_properties(env_handle, robot_handle)
                default_props = self._drop_body_props_default[env_id]
                if len(default_props) != len(rb_props):
                    continue
                for idx, prop in enumerate(rb_props):
                    self._apply_cached_rigid_body_property(prop, default_props[idx])

                payload_prop = rb_props[payload_body_idx]
                payload_prop.mass = residual_mass
                payload_prop.inertia.x.x = residual_inertia
                payload_prop.inertia.x.y = 0.0
                payload_prop.inertia.x.z = 0.0
                payload_prop.inertia.y.x = 0.0
                payload_prop.inertia.y.y = residual_inertia
                payload_prop.inertia.y.z = 0.0
                payload_prop.inertia.z.x = 0.0
                payload_prop.inertia.z.y = 0.0
                payload_prop.inertia.z.z = residual_inertia

                gym.set_actor_rigid_body_properties(
                    env_handle, robot_handle, rb_props, recomputeInertia=False
                )

                mass_new, inertia_new, com_root_new = self._compute_robot_aggregate_dynamics(
                    env_id, rb_props=rb_props
                )
                self.obs_dict["robot_mass"][env_id] = mass_new
                self.obs_dict["robot_inertia"][env_id] = inertia_new
                self.sim_env.robot_manager.robot_masses[env_id] = mass_new
                self.sim_env.robot_manager.robot_inertias[env_id] = inertia_new
                self._robot_com_body[env_id] = com_root_new
                self._mother_rigidbody_updated[env_id] = True
            except Exception as err:
                logger.warning(f"DROP rigid-body update failed in env {env_id}: {err}")

    def _restore_mother_rigidbody(self, env_ids):
        """Restore default rigid-body properties/mass tensors on reset."""
        if (not self._drop_rb_cache_ready) or env_ids.numel() == 0:
            return
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        restore_mask = self._mother_rigidbody_updated[env_ids]
        if not bool(restore_mask.any().item()):
            return

        ige_env = self.sim_env.IGE_env
        gym = ige_env.gym
        env_handles = ige_env.env_handles
        robot_handles = self.sim_env.robot_manager.robot_handles
        restore_ids = env_ids[restore_mask]
        for env_id in restore_ids.tolist():
            try:
                env_handle = env_handles[env_id]
                robot_handle = robot_handles[env_id]
                rb_props = gym.get_actor_rigid_body_properties(env_handle, robot_handle)
                default_props = self._drop_body_props_default[env_id]
                if len(default_props) != len(rb_props):
                    continue
                for idx, prop in enumerate(rb_props):
                    self._apply_cached_rigid_body_property(prop, default_props[idx])
                gym.set_actor_rigid_body_properties(
                    env_handle, robot_handle, rb_props, recomputeInertia=False
                )

                default_mass = self._drop_robot_mass_default[env_id]
                default_inertia = self._drop_robot_inertia_default[env_id]
                self.obs_dict["robot_mass"][env_id] = default_mass
                self.obs_dict["robot_inertia"][env_id] = default_inertia
                self.sim_env.robot_manager.robot_masses[env_id] = default_mass
                self.sim_env.robot_manager.robot_inertias[env_id] = default_inertia
                self._robot_com_body[env_id] = self._robot_com_body_default
                self._mother_rigidbody_updated[env_id] = False
            except Exception as err:
                logger.warning(f"DROP rigid-body restore failed in env {env_id}: {err}")

    def _load_fixed_env_preset(self):
        preset_id = getattr(self.task_config, "preset_id", -1)
        try:
            preset_id = int(preset_id)
        except (TypeError, ValueError):
            preset_id = -1

        self.fixed_env_enabled = False
        self.fixed_env_preset_id = preset_id
        self.fixed_env_preset_name = None

        if preset_id < 0:
            return

        presets = getattr(self.task_config, "fixed_env_presets", None)
        if not presets or preset_id >= len(presets):
            raise ValueError(f"Invalid preset_id={preset_id}. Check fixed_env_presets.")

        preset = presets[preset_id]
        required_keys = (
            "target_position",
            "obstacle_positions",
        )
        for key in required_keys:
            if key not in preset:
                raise ValueError(f"Fixed preset {preset_id} missing key: {key}")

        self.fixed_env_preset_name = preset.get("name", f"preset_{preset_id}")

        target_position = torch.tensor(
            preset["target_position"], device=self.device, dtype=torch.float32
        )
        if target_position.shape != (3,):
            raise ValueError("Fixed target_position must be a 3D vector.")

        obstacle_positions = torch.tensor(
            preset["obstacle_positions"], device=self.device, dtype=torch.float32
        )
        if obstacle_positions.ndim != 2 or obstacle_positions.shape[1] != 3:
            raise ValueError("Fixed obstacle_positions must be shaped as (N, 3).")

        expected_obstacles = int(self.task_config.num_obstacles_in_env)
        if obstacle_positions.shape[0] < expected_obstacles:
            raise ValueError(
                f"Fixed obstacle_positions count {obstacle_positions.shape[0]} "
                f"is less than num_obstacles_in_env={expected_obstacles}."
            )
        if obstacle_positions.shape[0] != expected_obstacles:
            obstacle_positions = obstacle_positions[:expected_obstacles]

        self.fixed_target_position = target_position
        self.fixed_obstacle_positions = obstacle_positions
        self.fixed_obstacle_orientation = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
        )

        self.fixed_env_enabled = True
        logger.info(
            "Fixed env preset enabled: %s (id=%s)",
            self.fixed_env_preset_name,
            self.fixed_env_preset_id,
        )

    def _apply_fixed_target(self, env_ids):
        target = self.fixed_target_position.view(1, 3).expand(env_ids.shape[0], -1)
        self.target_position[env_ids] = target

    def _apply_fixed_obstacles(self, env_ids):
        num_assets = self.obs_dict["obstacle_position"].shape[1]
        keep_in_env = int(getattr(self.sim_env, "keep_in_env", 0) or 0)
        num_obstacles = int(self.fixed_obstacle_positions.shape[0])
        start_idx = keep_in_env
        end_idx = start_idx + num_obstacles
        if end_idx > num_assets:
            raise ValueError(
                f"Fixed obstacles exceed available assets: need {end_idx}, have {num_assets}."
            )

        positions = self.fixed_obstacle_positions.view(1, num_obstacles, 3).expand(
            env_ids.shape[0], -1, -1
        )
        orientations = self.fixed_obstacle_orientation.view(1, 1, 4).expand(
            env_ids.shape[0], num_obstacles, -1
        )

        self.obs_dict["obstacle_position"][env_ids, start_idx:end_idx, :] = positions
        self.obs_dict["obstacle_orientation"][env_ids, start_idx:end_idx, :] = orientations
        self.obs_dict["obstacle_linvel"][env_ids, start_idx:end_idx, :].zero_()
        self.obs_dict["obstacle_angvel"][env_ids, start_idx:end_idx, :].zero_()

        if end_idx < num_assets:
            self.obs_dict["obstacle_position"][env_ids, end_idx:, :] = -1000.0
            self.obs_dict["obstacle_linvel"][env_ids, end_idx:, :].zero_()
            self.obs_dict["obstacle_angvel"][env_ids, end_idx:, :].zero_()

        self.sim_env.IGE_env.write_to_sim()
        if self.sim_env.use_warp:
            self.sim_env.warp_env.reset_idx(env_ids)

    def _draw_env0_debug_markers(self):
        if not self._target_marker_draw_enabled:
            return
        try:
            ige_env = getattr(self.sim_env, "IGE_env", None)
            if ige_env is None:
                return
            viewer_ctrl = getattr(ige_env, "viewer", None)
            if viewer_ctrl is None:
                return
            viewer = getattr(viewer_ctrl, "viewer", None)
            if viewer is None:
                return
            env_handles = getattr(ige_env, "env_handles", None)
            if env_handles is None or len(env_handles) == 0:
                return

            gym = ige_env.gym
            env0 = env_handles[0]

            # Convert target to world frame for env0 visualization.
            target0 = self.target_position[0].detach().cpu().numpy().astype(np.float32)
            cx = float(target0[0])
            cy = float(target0[1])
            cz = float(target0[2])
            if self._target_position_is_local_xy():
                env0_id = torch.tensor([0], device=self.device, dtype=torch.long)
                env0_offset_xy = (
                    self._get_env_world_offset_xy(env0_id).detach().cpu().numpy().astype(np.float32)[0]
                )
                cx += float(env0_offset_xy[0])
                cy += float(env0_offset_xy[1])
                if self._has_env_origins:
                    cz += float(self.obs_dict["env_origins"][0, 2].item())

            z_ground = cz + 0.03
            cross_half = 0.6
            pole_h = 2.0
            score_d0 = float(getattr(self.drop_reward_config, "score_d0", 3.6))
            piecewise_r = float(getattr(self.drop_reward_config, "piecewise_r", 2.0))
            piecewise_thresholds = list(
                getattr(self.drop_reward_config, "piecewise_thresholds", [0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 12.2])
            )
            outer_r = piecewise_r * float(piecewise_thresholds[-1]) if len(piecewise_thresholds) > 0 else 24.4

            segments = []
            colors = []

            # Target cross + vertical pole (green).
            segments.append([cx - cross_half, cy, z_ground, cx + cross_half, cy, z_ground])
            colors.append([0.1, 0.9, 0.1])
            segments.append([cx, cy - cross_half, z_ground, cx, cy + cross_half, z_ground])
            colors.append([0.1, 0.9, 0.1])
            segments.append([cx, cy, z_ground, cx, cy, z_ground + pole_h])
            colors.append([0.1, 0.9, 0.1])

            # Circle rings on ground: d0 (cyan) and outer threshold (yellow).
            nseg = 64
            angles = np.linspace(0.0, 2.0 * np.pi, nseg + 1, dtype=np.float32)
            circle_specs = [
                (max(score_d0, 0.05), [0.1, 0.8, 0.8]),
                (max(float(outer_r), 0.05), [0.95, 0.85, 0.2]),
            ]
            for radius, color in circle_specs:
                for i in range(nseg):
                    a0 = angles[i]
                    a1 = angles[i + 1]
                    x0 = cx + radius * float(np.cos(a0))
                    y0 = cy + radius * float(np.sin(a0))
                    x1 = cx + radius * float(np.cos(a1))
                    y1 = cy + radius * float(np.sin(a1))
                    segments.append([x0, y0, z_ground, x1, y1, z_ground])
                    colors.append(color)

            verts = np.asarray(segments, dtype=np.float32)
            cols = np.asarray(colors, dtype=np.float32)
            gym.clear_lines(viewer)
            gym.add_lines(viewer, env0, int(verts.shape[0]), verts, cols)
        except Exception as e:
            # Disable marker drawing if backend call is unavailable.
            if not self._target_marker_warned:
                logger.warning(f"Target marker draw disabled due to viewer API error: {e}")
                self._target_marker_warned = True
            self._target_marker_draw_enabled = False

    def _sample_unit_directions(self, count, horizontal_only=False):
        if count <= 0:
            return torch.zeros((0, 3), device=self.device)
        if horizontal_only:
            angles = torch.rand((count,), device=self.device) * (2.0 * torch.pi)
            dirs = torch.zeros((count, 3), device=self.device)
            dirs[:, 0] = torch.cos(angles)
            dirs[:, 1] = torch.sin(angles)
            return dirs
        dirs = torch.randn((count, 3), device=self.device)
        return torch.nn.functional.normalize(dirs, dim=1)

    def _as_vec3_tensor(self, value, default):
        if value is None:
            value = default
        vec = torch.tensor(value, device=self.device, dtype=torch.float32)
        if vec.numel() == 1:
            vec = vec.repeat(3)
        vec = vec.flatten()
        if vec.shape[0] != 3:
            raise ValueError("Expected a 3D vector-like value.")
        return vec

    def _resample_dryden_parameters(self, env_ids):
        """Sample per-env sigma/tau for simplified Dryden turbulence."""
        if (not self.dryden_enabled) or env_ids.numel() == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        cfg = self.dryden_config
        sigma_min = self._as_vec3_tensor(getattr(cfg, "sigma_min", None), [0.2, 0.2, 0.15])
        sigma_max = self._as_vec3_tensor(getattr(cfg, "sigma_max", None), [0.3, 0.3, 0.25])
        tau_min = self._as_vec3_tensor(getattr(cfg, "tau_min", None), [0.8, 0.8, 0.8])
        tau_max = self._as_vec3_tensor(getattr(cfg, "tau_max", None), [2.0, 2.0, 2.0])

        sigma_lo = torch.minimum(sigma_min, sigma_max)
        sigma_hi = torch.maximum(sigma_min, sigma_max)
        tau_lo = torch.minimum(tau_min, tau_max).clamp_min(1e-3)
        tau_hi = torch.maximum(tau_min, tau_max).clamp_min(1e-3)

        n = env_ids.shape[0]
        rand_sigma = torch.rand((n, 3), device=self.device, dtype=torch.float32)
        rand_tau = torch.rand((n, 3), device=self.device, dtype=torch.float32)

        self.dryden_sigma[env_ids] = sigma_lo.view(1, 3) + rand_sigma * (
            sigma_hi - sigma_lo
        ).view(1, 3)
        self.dryden_tau[env_ids] = tau_lo.view(1, 3) + rand_tau * (tau_hi - tau_lo).view(1, 3)
        self.dryden_wind_state[env_ids] = 0.0

    def _update_dryden_turbulence(self):
        """Update three-axis first-order Dryden turbulence state for all envs."""
        if not self.dryden_enabled:
            return

        dt = float(self.obs_dict["dt"]) if "dt" in self.obs_dict else 0.01
        dt = max(dt, 1e-5)

        tau = torch.clamp(self.dryden_tau, min=1e-3)
        sigma = torch.clamp(self.dryden_sigma, min=0.0)
        a = torch.exp(-dt / tau)
        noise_std = sigma * torch.sqrt(torch.clamp(1.0 - a * a, min=0.0))
        xi = torch.randn_like(self.dryden_wind_state)
        self.dryden_wind_state = a * self.dryden_wind_state + noise_std * xi

        if bool(getattr(self.dryden_config, "horizontal_only", False)):
            self.dryden_wind_state[:, 2] = 0.0

        clip_sigma = float(getattr(self.dryden_config, "clip_sigma", 0.0))
        if clip_sigma > 0.0:
            clip_bound = clip_sigma * sigma
            self.dryden_wind_state = torch.maximum(
                torch.minimum(self.dryden_wind_state, clip_bound), -clip_bound
            )

    def _resample_main_wind(self, env_ids):
        """Sample per-env main wind and keep it constant during an episode."""
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        cfg = self.gmm_force_config
        speed_min = float(getattr(cfg, "main_wind_speed_min", 2.0))
        speed_max = float(getattr(cfg, "main_wind_speed_max", 6.0))
        if speed_max < speed_min:
            speed_min, speed_max = speed_max, speed_min

        speeds = torch.empty((env_ids.shape[0],), device=self.device)
        if abs(speed_max - speed_min) < 1e-9:
            speeds.fill_(speed_min)
        else:
            speeds.uniform_(speed_min, speed_max)

        horizontal_only = bool(getattr(cfg, "main_wind_horizontal_only", True))
        dirs = self._sample_unit_directions(env_ids.shape[0], horizontal_only=horizontal_only)
        self.main_wind_direction[env_ids] = dirs
        self.main_wind_speed[env_ids] = speeds
        self.main_wind_vector[env_ids] = dirs * speeds.unsqueeze(1)

    def _update_gmm_force_direction(self):
        """
        Update local disturbance direction smoothly (Layer 2 gust).
        Main wind (Layer 1) is sampled at reset and stays constant.
        This local gust update is independent from any spatial source model.
        """
        if self.dryden_enabled:
            # Dryden mode does not use direction-based local gust updates.
            return
        if not self.gmm_force_config.enable_physical_force:
            return

        self.gmm_force_update_counter += 1
        update_mask = self.gmm_force_update_counter >= self.gmm_force_config.force_update_steps
        if update_mask.any():
            env_ids = update_mask.nonzero(as_tuple=False).squeeze(-1)
            horizontal_only = bool(getattr(self.gmm_force_config, "local_wind_horizontal_only", True))
            random_dirs = self._sample_unit_directions(len(env_ids), horizontal_only=horizontal_only)
            self.gmm_target_direction[env_ids] = random_dirs
            self.gmm_force_update_counter[env_ids] = 0

        alpha = 0.1
        self.gmm_force_direction = (1.0 - alpha) * self.gmm_force_direction + alpha * self.gmm_target_direction
        self.gmm_force_direction = torch.nn.functional.normalize(self.gmm_force_direction, dim=1)

    def _resample_local_wind_speed(self, env_ids):
        """Sample per-env local-wind cap and keep it constant during an episode."""
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        cfg = self.gmm_force_config
        speed_min = float(
            getattr(
                cfg,
                "local_wind_max_speed_min",
                getattr(cfg, "local_wind_max_speed", 0.5),
            )
        )
        speed_max = float(
            getattr(
                cfg,
                "local_wind_max_speed_max",
                getattr(cfg, "local_wind_max_speed", 0.5),
            )
        )
        if speed_max < speed_min:
            speed_min, speed_max = speed_max, speed_min

        speeds = torch.empty((env_ids.shape[0],), device=self.device)
        if abs(speed_max - speed_min) < 1e-9:
            speeds.fill_(speed_min)
        else:
            speeds.uniform_(speed_min, speed_max)
        self.local_wind_max_speed[env_ids] = speeds
    
    def _apply_gmm_physical_forces(self):
        """Apply two-layer wind drag force on the base link."""
        if not self.gmm_force_config.enable_physical_force:
            self.task_external_force_tensor.zero_()
            return

        position = self.obs_dict["robot_position"]
        v_w = self._compute_gmm_wind_vector(position)
        c_drag = float(self.gmm_force_config.drag_coefficient)
        v_uav = self.obs_dict["robot_linvel"]

        # Relative-wind drag model: F_drag = c_drag * (v_w - v_uav)
        self.task_external_force_tensor[:] = c_drag * (v_w - v_uav)

    def _compute_gmm_wind_vector(self, position, env_ids=None):
        """Compute world-frame wind vector: w_total = w_main + w_local."""
        if env_ids is not None:
            main_wind = self.main_wind_vector[env_ids]
        else:
            main_wind = self.main_wind_vector

        local_wind = torch.zeros_like(main_wind)
        if self.dryden_enabled:
            if env_ids is not None:
                local_wind = self.dryden_wind_state[env_ids]
            else:
                local_wind = self.dryden_wind_state
        else:
            if env_ids is not None:
                directions = self.gmm_force_direction[env_ids]
                local_v_max = self.local_wind_max_speed[env_ids]
            else:
                directions = self.gmm_force_direction
                local_v_max = self.local_wind_max_speed
            local_wind = local_v_max.unsqueeze(1) * directions

        return main_wind + local_wind

    def _compute_eject_velocity_world(self, env_ids):
        """Compute eject velocity vector in world frame for selected envs."""
        if env_ids.numel() == 0:
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)

        cfg = self.drop_impact_config
        eject_speed = float(getattr(cfg, "eject_speed", 0.5)) if cfg is not None else 0.5
        eject_dir_body = torch.tensor(
            getattr(cfg, "eject_direction_body", [0.0, 0.0, -1.0]) if cfg is not None else [0.0, 0.0, -1.0],
            device=self.device,
            dtype=torch.float32,
        )
        if torch.norm(eject_dir_body) < 1e-6:
            eject_dir_body = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        eject_dir_body = eject_dir_body / torch.norm(eject_dir_body)
        v_eject_body = eject_speed * eject_dir_body.view(1, 3).expand(env_ids.shape[0], -1)
        robot_quat = self.obs_dict["robot_orientation"][env_ids]
        return quat_rotate(robot_quat, v_eject_body)

    def _compute_payload_mount_offset_world(self, env_ids):
        """Compute payload mount offset from mother base origin in world frame."""
        if env_ids.numel() == 0:
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if hasattr(self, "fixed_payload_offset_body"):
            offset_body = self.fixed_payload_offset_body.view(1, 3).expand(env_ids.shape[0], -1)
        else:
            cfg = self.drop_impact_config
            offset_body = torch.tensor(
                getattr(cfg, "payload_offset_body", [0.0, 0.0, 0.0]) if cfg is not None else [0.0, 0.0, 0.0],
                device=self.device,
                dtype=torch.float32,
            ).view(1, 3).expand(env_ids.shape[0], -1)
        robot_quat = self.obs_dict["robot_orientation"][env_ids]
        return quat_rotate(robot_quat, offset_body)

    def _compute_child_release_kinematics(self, env_ids, include_eject_velocity=True):
        """
        Compute child release position and velocity at the payload mount point.

        v_release = v_mother + omega_world x r_mount_world + v_eject_world.
        """
        if env_ids.numel() == 0:
            empty = torch.zeros((0, 3), device=self.device, dtype=torch.float32)
            return empty, empty

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        mount_offset_world = self._compute_payload_mount_offset_world(env_ids)
        release_pos = self.obs_dict["robot_position"][env_ids].clone() + mount_offset_world

        omega_world = self.obs_dict["robot_angvel"][env_ids].clone()
        tangential_vel = torch.cross(omega_world, mount_offset_world, dim=1)
        release_vel = self.obs_dict["robot_linvel"][env_ids].clone() + tangential_vel
        if include_eject_velocity:
            release_vel = release_vel + self._compute_eject_velocity_world(env_ids)
        return release_pos, release_vel

    def _apply_drop_impact(self, env_ids):
        """Apply recoil impact to mother UAV when child is released."""
        if env_ids.numel() == 0:
            return

        cfg = self.drop_impact_config
        if cfg is None or not bool(getattr(cfg, "enable_impact", True)):
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        num_drop = env_ids.shape[0]
        if num_drop == 0:
            return
        # Update mother rigid-body properties first so recoil uses post-drop dynamics.
        self._update_mother_rigidbody_on_drop(env_ids)

        # Release velocity in body/world frame.
        m_child = float(getattr(cfg, "child_mass", 1.0))
        eject_speed = float(getattr(cfg, "eject_speed", 0.5))
        eject_dir_body = torch.tensor(
            getattr(cfg, "eject_direction_body", [0.0, 0.0, -1.0]),
            device=self.device,
            dtype=torch.float32,
        )
        if torch.norm(eject_dir_body) < 1e-6:
            eject_dir_body = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        eject_dir_body = eject_dir_body / torch.norm(eject_dir_body)
        v_eject_body = eject_speed * eject_dir_body.view(1, 3).expand(num_drop, -1)
        robot_quat = self.obs_dict["robot_orientation"][env_ids]
        m_mother_cfg = float(getattr(cfg, "mother_mass", 11.04))
        if "robot_mass" in self.obs_dict:
            dyn_mass = self.obs_dict["robot_mass"][env_ids].view(-1, 1).clamp_min(1e-6)
        else:
            dyn_mass = torch.full(
                (num_drop, 1), max(m_mother_cfg, 1e-6), device=self.device, dtype=torch.float32
            )

        # Preferred mode: apply one-step average recoil force/torque (no direct impulse injection).
        if bool(getattr(cfg, "use_force_recoil", True)):
            sim_dt = float(self.obs_dict["dt"]) if "dt" in self.obs_dict else 0.01
            sim_dt = max(sim_dt, 1e-5)
            # The external-force tensor is applied on every physics substep inside one env step.
            # Use control_dt = sim_dt * num_substeps so total impulse matches target Delta_v.
            num_substeps = max(self.physics_steps_per_env_step_mean, 1.0)
            control_dt = max(sim_dt * num_substeps, 1e-5)
            if (
                (self.physics_steps_per_env_step_std > 1e-9)
                and (not self._recoil_dt_warning_emitted)
            ):
                logger.warning(
                    "Recoil force uses mean control_dt (sim_dt * num_physics_steps_per_env_step_mean). "
                    "num_physics_steps_per_env_step_std > 0 may introduce small impulse mismatch."
                )
                self._recoil_dt_warning_emitted = True
            # Match target recoil delta-v based on current (post-drop) dynamics mass:
            #   Delta_v_target = -(m_child / m_dyn) * v_eject
            # and convert it to one-step equivalent force under current simulator mass.
            delta_v_target_body = -(m_child / dyn_mass) * v_eject_body
            recoil_force_body = (dyn_mass / control_dt) * delta_v_target_body
            recoil_force_world = quat_rotate(robot_quat, recoil_force_body)
            self.task_external_force_tensor[env_ids] += recoil_force_world

            # Recoil torque about updated robot COM: tau = (r_mount - com_new) x F.
            if hasattr(self, "fixed_payload_offset_body"):
                payload_offset_body = self.fixed_payload_offset_body.view(1, 3)
            else:
                payload_offset_body = torch.tensor(
                    getattr(cfg, "payload_offset_body", [0.0, 0.0, 0.0]),
                    device=self.device,
                    dtype=torch.float32,
                ).view(1, 3)
            if torch.norm(payload_offset_body).item() > 1e-9:
                if self._drop_rb_cache_ready:
                    current_com_body = self._robot_com_body[env_ids]
                else:
                    current_com_body = torch.zeros((num_drop, 3), device=self.device)
                r_body = payload_offset_body.expand(num_drop, -1) - current_com_body
                recoil_tau_body = torch.cross(r_body, recoil_force_body, dim=1)
                recoil_tau_world = quat_rotate(robot_quat, recoil_tau_body)
                self.task_external_torque_tensor[env_ids] += recoil_tau_world
            return

        # Legacy mode: direct velocity impulse + optional stochastic angular kick.
        delta_v_body = -(m_child / dyn_mass) * v_eject_body
        delta_v_world = quat_rotate(robot_quat, delta_v_body)
        self.obs_dict["robot_linvel"][env_ids] += delta_v_world

        if bool(getattr(cfg, "enable_random_angular_kick", True)):
            sigma_base = float(getattr(cfg, "angular_sigma_base", 0.1))
            sigma_scale = float(getattr(cfg, "angular_sigma_scale", 0.2))
            max_delta_omega = float(getattr(cfg, "max_delta_omega", 0.5))

            omega_body = self.obs_dict["robot_body_angvel"][env_ids]
            omega_norm = torch.norm(omega_body, dim=1, keepdim=True)
            sigma = sigma_base * (1.0 + sigma_scale * omega_norm)
            delta_omega_body = torch.randn((num_drop, 3), device=self.device) * sigma

            if max_delta_omega > 0.0:
                omega_kick_norm = torch.norm(delta_omega_body, dim=1, keepdim=True).clamp_min(1e-6)
                scale = torch.clamp(max_delta_omega / omega_kick_norm, max=1.0)
                delta_omega_body = delta_omega_body * scale

            delta_omega_world = quat_rotate(robot_quat, delta_omega_body)
            self.obs_dict["robot_angvel"][env_ids] += delta_omega_world

        # Push modified root-state tensors to Isaac Gym before the next physics step.
        self.sim_env.IGE_env.write_to_sim()

    def _resample_child_drop_trace(self, trace_pos_seq, trace_len, env_ids, dt):
        num_cases = int(trace_pos_seq.shape[1])
        num_samples = int(self.eval_drop_trace_num_samples)
        norm_t = torch.linspace(0.0, 1.0, num_samples, device=self.device).view(1, -1).repeat(num_cases, 1)
        pos_x = torch.zeros((num_cases, num_samples), device=self.device, dtype=torch.float32)
        pos_y = torch.zeros((num_cases, num_samples), device=self.device, dtype=torch.float32)
        pos_z = torch.zeros((num_cases, num_samples), device=self.device, dtype=torch.float32)
        err_xy = torch.zeros((num_cases, num_samples), device=self.device, dtype=torch.float32)
        target_xy = self.target_position[env_ids, 0:2]

        for idx in range(num_cases):
            seq_len = int(max(int(trace_len[idx].item()), 1))
            seq = trace_pos_seq[:seq_len, idx, :]
            err_seq = torch.norm(seq[:, 0:2] - target_xy[idx].view(1, 2), dim=1, keepdim=True)
            feat = torch.cat([seq, err_seq], dim=1).transpose(0, 1).unsqueeze(0)
            if seq_len == 1:
                feat_rs = feat.repeat(1, 1, num_samples)
            else:
                feat_rs = F.interpolate(feat, size=num_samples, mode="linear", align_corners=True)
            pos_x[idx] = feat_rs[0, 0]
            pos_y[idx] = feat_rs[0, 1]
            pos_z[idx] = feat_rs[0, 2]
            err_xy[idx] = feat_rs[0, 3]

        duration_steps = torch.clamp(trace_len - 1, min=0)
        duration_s = duration_steps.to(dtype=torch.float32) * float(dt)
        return {
            "norm_t": norm_t,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": pos_z,
            "error_xy": err_xy,
            "duration_steps": duration_steps,
            "duration_s": duration_s,
        }

    def _predict_child_landing_xy(self, env_ids, init_pos=None, init_vel=None, return_trace=False):
        """Predict child landing position/distance without mutating DROP buffers."""
        if env_ids.numel() == 0:
            empty_pos = torch.empty((0, 3), device=self.device)
            empty_dist = torch.empty((0,), device=self.device)
            if return_trace:
                empty_trace = {
                    "norm_t": torch.empty((0, self.eval_drop_trace_num_samples), device=self.device),
                    "pos_x": torch.empty((0, self.eval_drop_trace_num_samples), device=self.device),
                    "pos_y": torch.empty((0, self.eval_drop_trace_num_samples), device=self.device),
                    "pos_z": torch.empty((0, self.eval_drop_trace_num_samples), device=self.device),
                    "error_xy": torch.empty((0, self.eval_drop_trace_num_samples), device=self.device),
                    "duration_steps": torch.empty((0,), device=self.device, dtype=torch.long),
                    "duration_s": torch.empty((0,), device=self.device),
                }
                return empty_pos, empty_dist, empty_trace
            return empty_pos, empty_dist

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if init_pos is None:
            child_pos = self.obs_dict["robot_position"][env_ids].clone()
        else:
            child_pos = init_pos.clone()
        if init_vel is None:
            child_vel = self.obs_dict["robot_linvel"][env_ids].clone()
        else:
            child_vel = init_vel.clone()

        dt = float(self.obs_dict["dt"]) if "dt" in self.obs_dict else 0.01
        gravity = float(getattr(self.drop_model_config, "child_gravity", 9.81))
        c_child = float(getattr(self.drop_model_config, "child_drag_coefficient", 1.0))
        m_child = float(getattr(self.drop_model_config, "child_mass", 1.0))
        if c_child < 0.0:
            c_child = 0.0
        if m_child <= 1e-6:
            m_child = float(getattr(self.drop_impact_config, "child_mass", 1.0))
        m_child = max(m_child, 1e-6)
        rel_wind_gain = c_child / m_child
        # Backward compatibility: if new params are absent, fall back to old alias.
        if not hasattr(self.drop_model_config, "child_drag_coefficient"):
            rel_wind_gain = float(getattr(self.drop_model_config, "child_wind_scale", 1.0))
        max_steps = int(getattr(self.drop_model_config, "max_child_sim_steps", 5000))
        g_vec = torch.tensor([0.0, 0.0, -gravity], device=self.device).view(1, 3)

        landing_pos = child_pos.clone()
        active = torch.ones(env_ids.shape[0], device=self.device, dtype=torch.bool)
        trace_pos_frames = None
        trace_len = None
        if return_trace:
            trace_pos_frames = [child_pos.clone()]
            trace_len = torch.ones(env_ids.shape[0], device=self.device, dtype=torch.long)
        # Local Dryden state for child free-fall integration.
        # Use a per-call copy so child rollout sees time-varying wind without mutating global env state.
        dryden_state_local = None
        dryden_a = None
        dryden_noise_std = None
        dryden_clip_bound = None
        dryden_horizontal_only = False
        if self.dryden_enabled:
            dryden_state_local = self.dryden_wind_state[env_ids].clone()
            tau_local = torch.clamp(self.dryden_tau[env_ids], min=1e-3)
            sigma_local = torch.clamp(self.dryden_sigma[env_ids], min=0.0)
            dryden_a = torch.exp(-dt / tau_local)
            dryden_noise_std = sigma_local * torch.sqrt(torch.clamp(1.0 - dryden_a * dryden_a, min=0.0))
            dryden_horizontal_only = bool(getattr(self.dryden_config, "horizontal_only", False))
            clip_sigma = float(getattr(self.dryden_config, "clip_sigma", 0.0))
            if clip_sigma > 0.0:
                dryden_clip_bound = clip_sigma * sigma_local

        for _ in range(max_steps):
            if not active.any():
                break

            local_ids = active.nonzero(as_tuple=False).squeeze(-1)
            env_active = env_ids[local_ids]

            if self.dryden_enabled:
                xi = torch.randn((local_ids.shape[0], 3), device=self.device, dtype=torch.float32)
                dryden_state_local[local_ids] = (
                    dryden_a[local_ids] * dryden_state_local[local_ids]
                    + dryden_noise_std[local_ids] * xi
                )
                if dryden_horizontal_only:
                    dryden_state_local[local_ids, 2] = 0.0
                if dryden_clip_bound is not None:
                    bound = dryden_clip_bound[local_ids]
                    dryden_state_local[local_ids] = torch.maximum(
                        torch.minimum(dryden_state_local[local_ids], bound),
                        -bound,
                    )
                wind_vec = self.main_wind_vector[env_active] + dryden_state_local[local_ids]
            else:
                wind_vec = self._compute_gmm_wind_vector(child_pos[local_ids], env_ids=env_active)
            relative_wind_acc = rel_wind_gain * (wind_vec - child_vel[local_ids])
            acc = g_vec + relative_wind_acc
            child_vel[local_ids] = child_vel[local_ids] + acc * dt
            next_pos = child_pos[local_ids] + child_vel[local_ids] * dt

            landed_now = next_pos[:, 2] <= 0.0
            if landed_now.any():
                landed_local = local_ids[landed_now]
                landing_pos[landed_local, 0:2] = next_pos[landed_now, 0:2]
                landing_pos[landed_local, 2] = 0.0
                child_pos[landed_local] = landing_pos[landed_local]
                active[landed_local] = False

            flying_now = ~landed_now
            if flying_now.any():
                child_pos[local_ids[flying_now]] = next_pos[flying_now]
            if return_trace:
                trace_len[local_ids] += 1
                trace_pos_frames.append(child_pos.clone())

        # Fallback in case max_steps is reached before touching ground.
        if active.any():
            remain_local = active.nonzero(as_tuple=False).squeeze(-1)
            landing_pos[remain_local, 0:2] = child_pos[remain_local, 0:2]
            landing_pos[remain_local, 2] = torch.clamp(child_pos[remain_local, 2], min=0.0)
            child_pos[remain_local] = landing_pos[remain_local]
            if return_trace:
                trace_len[remain_local] += 1
                trace_pos_frames.append(child_pos.clone())

        landing_dist = torch.norm(
            landing_pos[:, 0:2] - self.target_position[env_ids, 0:2], dim=1
        )
        if return_trace:
            trace_pos_seq = torch.stack(trace_pos_frames, dim=0)
            trace_dict = self._resample_child_drop_trace(
                trace_pos_seq=trace_pos_seq,
                trace_len=trace_len,
                env_ids=env_ids,
                dt=dt,
            )
            return landing_pos, landing_dist, trace_dict
        return landing_pos, landing_dist

    def _simulate_child_free_fall(self, env_ids, init_pos=None, init_vel=None):
        """Simulate child payload free fall with a = g + (c_child/m_child)*(v_w - v_child)."""
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        trace_dict = None
        if self.record_drop_trajectory_for_eval:
            landing_pos, landing_dist, trace_dict = self._predict_child_landing_xy(
                env_ids=env_ids,
                init_pos=init_pos,
                init_vel=init_vel,
                return_trace=True,
            )
        else:
            landing_pos, landing_dist = self._predict_child_landing_xy(
                env_ids=env_ids,
                init_pos=init_pos,
                init_vel=init_vel,
            )
        self.child_landing_position[env_ids] = landing_pos
        self.child_landing_xy_distance[env_ids] = landing_dist
        if trace_dict is not None:
            self.step_drop_trace_norm_t[env_ids] = trace_dict["norm_t"]
            self.step_drop_trace_error_xy[env_ids] = trace_dict["error_xy"]
            self.step_drop_trace_pos_x[env_ids] = trace_dict["pos_x"]
            self.step_drop_trace_pos_y[env_ids] = trace_dict["pos_y"]
            self.step_drop_trace_pos_z[env_ids] = trace_dict["pos_z"]
            self.step_drop_trace_fall_steps[env_ids] = trace_dict["duration_steps"]
            self.step_drop_trace_fall_time[env_ids] = trace_dict["duration_s"]




    def _compute_reward_and_scores(self):
        """
        Reward structure:
        - WAIT: direction-progress shaping + predicted-release-error improvement shaping
        - DROP: landing-accuracy score + impulse penalty
        - OUTSIDE: linear penalty beyond outer threshold, keep impulse penalty
        - HEIGHT: keep the existing soft lower-bound altitude penalty
        """
        cfg = self.drop_reward_config
        altitude_tolerance = float(getattr(cfg, "altitude_tolerance", 0.5))
        altitude_low_penalty_w = float(getattr(cfg, "altitude_low_penalty_weight", 1.0))
        direction_w = float(getattr(cfg, "direction_reward_weight", 0.02))
        pred_error_w = float(getattr(cfg, "pred_error_shaping_weight", 0.0))
        pred_error_clip = float(max(getattr(cfg, "pred_error_improvement_clip", 1.0), 1e-6))
        score_reward_w = float(
            getattr(
                cfg,
                "score_reward_weight",
                float(getattr(cfg, "landing_reward_weight", 1.0))
                * float(getattr(cfg, "piecewise_score_scale", 0.1)),
            )
        )
        impulse_lambda = float(getattr(cfg, "impulse_penalty_weight", 0.1))
        score_max = float(getattr(cfg, "score_max", 20.0))
        score_d0 = float(getattr(self, "current_score_d0", getattr(cfg, "score_d0", 4.0)))
        score_p = float(getattr(cfg, "score_p", 1.0))
        piecewise_r = float(getattr(cfg, "piecewise_r", 2.0))
        piecewise_thresholds = list(
            getattr(cfg, "piecewise_thresholds", [0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 12.2])
        )
        outside_region_penalty = float(getattr(cfg, "outside_region_penalty", 20.0))

        if len(piecewise_thresholds) == 0:
            piecewise_thresholds = [0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 12.2]
        scaled_thresholds = [piecewise_r * float(v) for v in piecewise_thresholds]

        reward = torch.zeros(self.sim_env.num_envs, device=self.device)
        wait_mask = ~self.child_has_dropped
        drop_ids = self.step_drop_event_mask.nonzero(as_tuple=False).squeeze(-1)

        direction_reward = torch.zeros_like(reward)
        if direction_w > 0.0:
            current_dist_xy = torch.norm(
                self.target_position[:, 0:2] - self.obs_dict["robot_position"][:, 0:2], dim=1
            )
            progress_delta = self.previous_distance_xy - current_dist_xy
            valid_mask = wait_mask
            direction_reward[valid_mask] = direction_w * progress_delta[valid_mask]

        predicted_error_reward = torch.zeros_like(reward)
        (
            predicted_drop_error_xy_all,
            _predicted_fall_time_all,
            _attitude_deg_all,
            _omega_xy_all,
        ) = self._compute_drop_decision_features(
            target_relative_position=quat_rotate_inverse(
                self.obs_dict["robot_vehicle_orientation"],
                (self.target_position - self.obs_dict["robot_position"]),
            ),
            body_linvel=self.obs_dict["robot_body_linvel"],
            euler_angles=ssa(self.obs_dict["robot_euler_angles"]),
            body_angvel=self.obs_dict["robot_body_angvel"],
        )
        if pred_error_w > 0.0:
            pred_error_improvement = torch.clamp(
                self.prev_predicted_drop_error_xy - predicted_drop_error_xy_all,
                min=-pred_error_clip,
                max=pred_error_clip,
            )
            predicted_error_reward[wait_mask] = pred_error_w * pred_error_improvement[wait_mask]

        reward = reward + direction_reward
        reward = reward + predicted_error_reward

        score_reward = torch.zeros_like(reward)
        impulse_penalty = torch.zeros_like(reward)
        outside_region_penalty_reward = torch.zeros_like(reward)
        if drop_ids.numel() > 0:
            landing_error_xy = self.step_landing_error_xy[drop_ids]
            impulse_metric = self.step_impulse_metric[drop_ids]
            safe_d0 = max(score_d0, 1e-6)
            safe_p = max(score_p, 1e-6)
            raw_score = score_max * torch.exp(-torch.pow(landing_error_xy / safe_d0, safe_p))

            # Drop-only diagnostics (window accumulators for 10-epoch reporting).
            num_drop_samples = int(raw_score.numel())
            if num_drop_samples > 0:
                self.window_drop_sample_count += num_drop_samples
                self.window_landing_error_xy_sum += float(landing_error_xy.sum().item())
                release_to_target_xy = self.step_release_to_target_xy[drop_ids]
                self.window_release_to_target_xy_sum += float(release_to_target_xy.sum().item())
                self.window_release_to_target_xy_min = min(
                    self.window_release_to_target_xy_min,
                    float(release_to_target_xy.min().item()),
                )
                self.window_release_to_target_xy_max = max(
                    self.window_release_to_target_xy_max,
                    float(release_to_target_xy.max().item()),
                )
                roll_deg = torch.rad2deg(self.step_drop_roll[drop_ids])
                pitch_deg = torch.rad2deg(self.step_drop_pitch[drop_ids])
                self.window_drop_roll_deg_sum += float(roll_deg.sum().item())
                self.window_drop_roll_deg_min = min(
                    self.window_drop_roll_deg_min, float(roll_deg.min().item())
                )
                self.window_drop_roll_deg_max = max(
                    self.window_drop_roll_deg_max, float(roll_deg.max().item())
                )
                self.window_drop_pitch_deg_sum += float(pitch_deg.sum().item())
                self.window_drop_pitch_deg_min = min(
                    self.window_drop_pitch_deg_min, float(pitch_deg.min().item())
                )
                self.window_drop_pitch_deg_max = max(
                    self.window_drop_pitch_deg_max, float(pitch_deg.max().item())
                )
                attitude_total_deg = torch.rad2deg(self.step_drop_attitude_theta[drop_ids])
                self.window_drop_attitude_total_deg_sum += float(attitude_total_deg.sum().item())
                self.window_drop_attitude_total_deg_sq_sum += float(
                    torch.square(attitude_total_deg).sum().item()
                )
                self.window_impulse_metric_sum += float(impulse_metric.sum().item())
                self.window_impulse_metric_sq_sum += float(
                    torch.square(impulse_metric).sum().item()
                )
                attitude_total_deg_mean_step = float(attitude_total_deg.mean().item())
                att_ema_alpha = float(min(max(self.attitude_total_ema_alpha, 0.0), 1.0))
                if self.attitude_total_deg_ema is None:
                    self.attitude_total_deg_ema = attitude_total_deg_mean_step
                else:
                    self.attitude_total_deg_ema = (
                        att_ema_alpha * self.attitude_total_deg_ema
                        + (1.0 - att_ema_alpha) * attitude_total_deg_mean_step
                    )
                landing_error_xy_mean_step = float(landing_error_xy.mean().item())
                release_to_target_xy_mean_step = float(release_to_target_xy.mean().item())
                ema_alpha = float(min(max(self.landing_error_ema_alpha, 0.0), 1.0))
                if self.landing_error_xy_ema is None:
                    self.landing_error_xy_ema = landing_error_xy_mean_step
                else:
                    self.landing_error_xy_ema = (
                        ema_alpha * self.landing_error_xy_ema
                        + (1.0 - ema_alpha) * landing_error_xy_mean_step
                    )
                self.extras["drop_sample_count_step"] = num_drop_samples
                self.extras["landing_error_xy_drop_mean_step"] = landing_error_xy_mean_step
                self.extras["release_to_target_xy_drop_mean_step"] = release_to_target_xy_mean_step

            score_term = score_reward_w * raw_score
            score_reward[drop_ids] = score_term
            impulse_penalty[drop_ids] = -impulse_lambda * impulse_metric
            reward[drop_ids] = (
                score_reward[drop_ids]
                + impulse_penalty[drop_ids]
            )
            outside_mask = torch.zeros_like(landing_error_xy, dtype=torch.bool)
            if len(scaled_thresholds) > 0:
                outside_mask = landing_error_xy > scaled_thresholds[-1]
            if outside_mask.any():
                penalty_ids = drop_ids[outside_mask]
                outside_error = landing_error_xy[outside_mask] - scaled_thresholds[-1]
                outside_region_penalty_reward[penalty_ids] = torch.clamp(
                    -outside_error, min=-outside_region_penalty, max=0.0
                )
                score_reward[penalty_ids] = 0.0
                reward[penalty_ids] = (
                    outside_region_penalty_reward[penalty_ids] + impulse_penalty[penalty_ids]
                )

        # Soft lower-bound altitude penalty (no hard state overwrite).
        # Penalize when z < (spawn_height_reference - altitude_tolerance).
        # Fallback to fixed target-z baseline if reference is unavailable.
        if (
            hasattr(self, "spawn_height_reference")
            and self.spawn_height_reference.shape[0] == self.sim_env.num_envs
        ):
            lower_bound_z = self.spawn_height_reference - altitude_tolerance
        else:
            altitude_target_z = float(getattr(cfg, "altitude_target_z", 10.0))
            lower_bound_z = altitude_target_z - altitude_tolerance
        altitude_penalty = torch.zeros_like(reward)
        if altitude_low_penalty_w > 0.0:
            z = self.obs_dict["robot_position"][:, 2]
            below_depth = torch.clamp(lower_bound_z - z, min=0.0)
            altitude_penalty = -altitude_low_penalty_w * below_depth
            reward = reward + altitude_penalty

        self.previous_position = self.obs_dict["robot_position"].clone()
        self.previous_distance = torch.norm(
            self.target_position - self.obs_dict["robot_position"], dim=1
        )
        self.previous_distance_xy = torch.norm(
            self.target_position[:, 0:2] - self.obs_dict["robot_position"][:, 0:2], dim=1
        )
        self.previous_velocity = self.obs_dict["robot_linvel"].clone()
        self.previous_actions = self.actions.clone()
        self.prev_predicted_drop_error_xy[:] = predicted_drop_error_xy_all

        improvement_reward = (
            score_reward
            + impulse_penalty
            + outside_region_penalty_reward
        )
        noise_reduction_reward = torch.zeros_like(reward)
        safety_reward = altitude_penalty
        action_smoothness_penalty = torch.zeros_like(reward)
        hover_reward = torch.zeros_like(reward)
        threshold_reward = torch.zeros_like(reward)
        anchor_reward = torch.zeros_like(reward)

        return (
            reward,
            improvement_reward,
            direction_reward,
            noise_reduction_reward,
            safety_reward,
            action_smoothness_penalty,
            hover_reward,
            threshold_reward,
            anchor_reward,
        )

    def _update_best_point(self, total_score):
        if self.log_step_scores:
            env_id = min(self.log_step_env_id, total_score.shape[0] - 1)
            step_best_score_val = float(total_score[env_id].item())
            if step_best_score_val > self.best_total_score:
                self.best_total_score = step_best_score_val
                self.best_position = self.obs_dict["robot_position"][env_id].detach().cpu()
                logger.info(
                    "New best total_score=%s position=%s",
                    self.best_total_score,
                    self.best_position.tolist(),
                )
            return

        step_best_score, step_best_idx = torch.max(total_score, dim=0)
        step_best_score_val = float(step_best_score.item())
        if step_best_score_val > self.best_total_score:
            self.best_total_score = step_best_score_val
            self.best_position = (
                self.obs_dict["robot_position"][int(step_best_idx.item())].detach().cpu()
            )
            logger.info(
                "New best total_score=%s position=%s",
                self.best_total_score,
                self.best_position.tolist(),
            )
            self._write_best_point()

    def _write_best_point(self):
        if self.best_position is None:
            return
        self._resolve_best_point_path()
        with open(self.best_point_path, "w", encoding="utf-8") as handle:
            handle.write(f"score={self.best_total_score}\n")
            handle.write(f"position={self.best_position.tolist()}\n")

    def _record_step_score(self, total_score):
        if self._episode_log_written:
            return
        env_id = self.log_step_env_id
        if env_id < 0 or env_id >= total_score.shape[0]:
            return
        pos = self.obs_dict["robot_position"][env_id].detach().cpu().tolist()
        score = float(total_score[env_id].item())
        self._episode_step_log.append((self._episode_step_idx, pos, score))
        self._episode_step_idx += 1

    def _write_best_point_log(self):
        if self.best_position is None:
            return
        self._resolve_best_point_path()
        with open(self.best_point_path, "w", encoding="utf-8") as handle:
            handle.write(f"score={self.best_total_score}\n")
            handle.write(f"position={self.best_position.tolist()}\n")
            handle.write("step,x,y,z,total_score\n")
            for step_idx, pos, score in self._episode_step_log:
                handle.write(
        f"{step_idx},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f},{score:.6f}\n"
                )
            handle.write(f"best_score={self.best_total_score}\n")
            handle.write(f"best_position={self.best_position.tolist()}\n")

    def _init_tensorboard_writer(self):
        """Initialize TensorBoard writer lazily on first use."""
        runs_dir = os.environ.get("AERIAL_GYM_RUNS_DIR")
        experiment_name = os.environ.get("AERIAL_GYM_EXPERIMENT_NAME")
        
        if runs_dir and experiment_name:
            # Now rl_games has created the directory, find it
            run_dir = None
            if os.path.isdir(runs_dir):
                candidates = [
                    os.path.join(runs_dir, name)
                    for name in os.listdir(runs_dir)
                    if name.startswith(experiment_name)
                ]
                if candidates:
                    run_dir = max(candidates, key=os.path.getmtime)
            
            if run_dir is None:
                run_dir = os.path.join(runs_dir, experiment_name)
            
            summary_dir = os.path.join(run_dir, "summaries")
            self.writer = SummaryWriter(summary_dir)
            # Ensure reward-weight snapshot is present in the actual run directory.
            self._write_reward_weight_snapshot(run_dir=run_dir)
        else:
            self.writer = None

    def _populate_extras(self, improvement_reward, direction_reward, noise_reduction_reward):
        """Export only compact task metrics: release attitude and landing precision."""
        del improvement_reward, direction_reward, noise_reduction_reward

        # Drop-only window diagnostics (aggregated since last console print).
        drop_samples_window = int(getattr(self, "window_drop_sample_count", 0))
        if drop_samples_window > 0:
            landing_error_xy_mean_drop_window = (
                float(self.window_landing_error_xy_sum) / drop_samples_window
            )
            release_attitude_deg_mean_drop_window = (
                float(self.window_drop_attitude_total_deg_sum) / drop_samples_window
            )
            attitude_total_deg_var_window = max(
                float(self.window_drop_attitude_total_deg_sq_sum) / drop_samples_window
                - release_attitude_deg_mean_drop_window * release_attitude_deg_mean_drop_window,
                0.0,
            )
            release_attitude_deg_std_drop_window = float(np.sqrt(attitude_total_deg_var_window))
        else:
            landing_error_xy_mean_drop_window = 0.0
            release_attitude_deg_mean_drop_window = 0.0
            release_attitude_deg_std_drop_window = 0.0

        landing_error_xy_ema = (
            float(self.landing_error_xy_ema) if self.landing_error_xy_ema is not None else 0.0
        )
        release_attitude_deg_ema = (
            float(self.attitude_total_deg_ema) if self.attitude_total_deg_ema is not None else 0.0
        )

        exported_extras = {}
        if "episode_rewards" in self.extras:
            exported_extras["episode_rewards"] = self.extras["episode_rewards"]
        if "episode_lengths" in self.extras:
            exported_extras["episode_lengths"] = self.extras["episode_lengths"]
        if drop_samples_window > 0:
            exported_extras["performance/landing_error_xy_mean_drop_window"] = (
                landing_error_xy_mean_drop_window
            )
            exported_extras["performance/release_attitude_deg_mean_drop_window"] = (
                release_attitude_deg_mean_drop_window
            )
        if self.landing_error_xy_ema is not None:
            exported_extras["performance/landing_error_xy_ema"] = landing_error_xy_ema
        if self.attitude_total_deg_ema is not None:
            exported_extras["performance/release_attitude_deg_ema"] = release_attitude_deg_ema
        exported_extras["performance/score_d0_current"] = float(self.current_score_d0)
        exported_extras["performance/score_d0_curriculum_stage"] = int(self.score_d0_curriculum_stage)
        
        # Write diagnostics to TensorBoard (with lazy initialization)
        if not self._writer_initialized:
            self._init_tensorboard_writer()
            self._writer_initialized = True
        
        if self.writer is not None:
            if drop_samples_window > 0:
                self.writer.add_scalar(
                    "performance/landing_error_xy_mean_drop_window",
                    landing_error_xy_mean_drop_window,
                    self.num_task_steps,
                )
                self.writer.add_scalar(
                    "performance/release_attitude_deg_mean_drop_window",
                    release_attitude_deg_mean_drop_window,
                    self.num_task_steps,
                )
            if self.landing_error_xy_ema is not None:
                self.writer.add_scalar(
                    "performance/landing_error_xy_ema",
                    landing_error_xy_ema,
                    self.num_task_steps,
                )
            if self.attitude_total_deg_ema is not None:
                self.writer.add_scalar(
                    "performance/release_attitude_deg_ema",
                    release_attitude_deg_ema,
                    self.num_task_steps,
                )
            self.writer.add_scalar(
                "performance/score_d0_current",
                float(self.current_score_d0),
                self.num_task_steps,
            )
            self.writer.add_scalar(
                "performance/score_d0_curriculum_stage",
                int(self.score_d0_curriculum_stage),
                self.num_task_steps,
            )
        
        # Console stats: print once every 10 epochs.
        curr_epoch = int(getattr(self, "_train_epoch", 0))
        if curr_epoch > 0 and curr_epoch % 10 == 0 and curr_epoch != self._last_console_stats_epoch:
            self._update_score_d0_curriculum()
            drop_count_window = int(getattr(self, "window_drop_end_count", 0))
            no_drop_count_window = int(getattr(self, "window_no_drop_done_count", 0))
            drop_decision_total = drop_count_window + no_drop_count_window
            drop_rate_text = (
                f"{(100.0 * drop_count_window / drop_decision_total):.1f}%"
                if drop_decision_total > 0
                else "N/A"
            )
            landing_error_mean_text = (
                f"{landing_error_xy_mean_drop_window:.3f}" if drop_samples_window > 0 else "N/A"
            )
            release_attitude_mean_text = (
                f"{release_attitude_deg_mean_drop_window:.3f}" if drop_samples_window > 0 else "N/A"
            )
            landing_error_ema_text = (
                f"{landing_error_xy_ema:.3f}" if self.landing_error_xy_ema is not None else "N/A"
            )
            release_attitude_ema_text = (
                f"{release_attitude_deg_ema:.3f}" if self.attitude_total_deg_ema is not None else "N/A"
            )
            release_attitude_std_text = (
                f"{release_attitude_deg_std_drop_window:.3f}" if drop_samples_window > 0 else "N/A"
            )
            logger.warning(
                f"[Epoch {curr_epoch}] "
                f"drop_count={drop_count_window}, "
                f"drop_rate={drop_rate_text}, "
                f"score_d0={float(self.current_score_d0):.2f}, "
                f"landing_error_xy(mean/ema)={landing_error_mean_text}/{landing_error_ema_text}, "
                f"release_attitude_deg(mean/ema/std)="
                f"{release_attitude_mean_text}/{release_attitude_ema_text}/{release_attitude_std_text}"
            )
            self._last_console_stats_epoch = curr_epoch
            self.window_done_total = 0
            self.window_done_non_crash = 0
            self.window_done_speed_sum = 0.0
            self.window_done_hover_reward_count = 0
            self.window_drop_end_count = 0
            self.window_no_drop_done_count = 0
            self.window_drop_sample_count = 0
            self.window_landing_error_xy_sum = 0.0
            self.window_release_to_target_xy_sum = 0.0
            self.window_release_to_target_xy_min = float("inf")
            self.window_release_to_target_xy_max = 0.0
            self.window_drop_roll_deg_sum = 0.0
            self.window_drop_roll_deg_min = float("inf")
            self.window_drop_roll_deg_max = -float("inf")
            self.window_drop_pitch_deg_sum = 0.0
            self.window_drop_pitch_deg_min = float("inf")
            self.window_drop_pitch_deg_max = -float("inf")
            self.window_drop_attitude_total_deg_sum = 0.0
            self.window_drop_attitude_total_deg_sq_sum = 0.0
            self.window_impulse_metric_sum = 0.0
            self.window_impulse_metric_sq_sum = 0.0

        self.extras = exported_extras
        self.infos["extras"] = exported_extras

    def close(self):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")
        # Compatibility: older code expected delete_env() on sim_env, but current
        # EnvManager may not expose this API. Release resources safely when possible.
        try:
            if hasattr(self.sim_env, "delete_env") and callable(getattr(self.sim_env, "delete_env")):
                self.sim_env.delete_env()
        except Exception as e:
            logger.warning(f"sim_env.delete_env() failed during close: {e}")

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs, device=self.device))
        return self.get_return_tuple()


    def reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).view(-1)
        if env_ids.numel() == 0:
            return
        # Ensure each new episode starts from the default full-payload rigid-body model.
        self._restore_mother_rigidbody(env_ids)
            
        # --- Capture minimum distance-to-target for terminated environments ---
        try:
            min_dist_vals = self.episode_min_J[env_ids].detach().cpu().numpy()
            valid_mask = min_dist_vals < 999.0
            if np.any(valid_mask):
                self.min_J_buffer.extend(min_dist_vals[valid_mask])
            self.episode_min_J[env_ids] = 1000.0
        except Exception:
            # Silent fail is safer during training loop than crashing
            pass

        if self.log_step_scores and not self._episode_log_written:
            if (env_ids == self.log_step_env_id).any().item():
                self._episode_step_log = []
                self._episode_step_idx = 0
        self._set_fixed_obstacle_count()
        if self.fixed_env_enabled:
            self._apply_fixed_target(env_ids)
            self._apply_fixed_obstacles(env_ids)
        else:
            use_fixed_target = bool(getattr(self.task_config, "target_use_fixed", False))
            if use_fixed_target:
                fixed_target = torch.tensor(
                    getattr(self.task_config, "target_fixed_position", [0.0, 0.0, 0.0]),
                    device=self.device,
                    dtype=torch.float32,
                )
                if fixed_target.shape != (3,):
                    raise ValueError("target_fixed_position must be a 3D vector [x, y, z].")
                use_env_center = bool(getattr(self.task_config, "target_fixed_use_env_center", True))
                if use_env_center:
                    # Use task-level local bounds to avoid world/local frame mismatch.
                    local_center_xy = 0.5 * (self.env_bounds_min[0:2] + self.env_bounds_max[0:2])
                    target_fixed_z = float(getattr(self.task_config, "target_fixed_z", 0.0))
                    self.target_position[env_ids, 0] = local_center_xy[0] + fixed_target[0]
                    self.target_position[env_ids, 1] = local_center_xy[1] + fixed_target[1]
                    self.target_position[env_ids, 2] = target_fixed_z + fixed_target[2]
                else:
                    self.target_position[env_ids] = fixed_target.view(1, 3).expand(env_ids.shape[0], -1)
            else:
                target_xy_min = float(getattr(self.task_config, "target_xy_min", -20.0))
                target_xy_max = float(getattr(self.task_config, "target_xy_max", 20.0))
                target_fixed_z = float(getattr(self.task_config, "target_fixed_z", 0.0))
                sampled_xy = torch.empty((env_ids.shape[0], 2), device=self.device)
                sampled_xy.uniform_(target_xy_min, target_xy_max)
                self.target_position[env_ids, 0:2] = sampled_xy
                self.target_position[env_ids, 2] = target_fixed_z
        # Optionally fix spawn XY to reduce training variance.
        use_fixed_spawn_xy = bool(getattr(self.task_config, "spawn_use_fixed_xy", False))
        if use_fixed_spawn_xy:
            spawn_fixed_xy = torch.tensor(
                getattr(self.task_config, "spawn_fixed_xy", [0.0, 0.0]),
                device=self.device,
                dtype=torch.float32,
            )
            if spawn_fixed_xy.shape != (2,):
                raise ValueError("spawn_fixed_xy must be a 2D vector [x, y].")
            spawn_use_env_center = bool(getattr(self.task_config, "spawn_fixed_use_env_center", True))
            if spawn_use_env_center:
                local_center_xy = 0.5 * (self.env_bounds_min[0:2] + self.env_bounds_max[0:2])
                self.obs_dict["robot_position"][env_ids, 0] = local_center_xy[0] + spawn_fixed_xy[0]
                self.obs_dict["robot_position"][env_ids, 1] = local_center_xy[1] + spawn_fixed_xy[1]
            else:
                self.obs_dict["robot_position"][env_ids, 0:2] = spawn_fixed_xy.view(1, 2).expand(
                    env_ids.shape[0], -1
                )
        else:
            # Optional random-X spawn mode with fixed Y.
            use_random_spawn_x = bool(getattr(self.task_config, "spawn_use_random_x", False))
            if use_random_spawn_x:
                x_min = float(getattr(self.task_config, "spawn_random_x_min", -24.0))
                x_max = float(getattr(self.task_config, "spawn_random_x_max", -12.0))
                if x_max < x_min:
                    x_min, x_max = x_max, x_min

                sampled_x = torch.empty((env_ids.shape[0],), device=self.device)
                sampled_x.uniform_(x_min, x_max)

                fixed_y = float(getattr(self.task_config, "spawn_random_fixed_y", 0.0))
                random_use_env_center = bool(
                    getattr(self.task_config, "spawn_random_use_env_center", True)
                )
                if random_use_env_center:
                    local_center_xy = 0.5 * (self.env_bounds_min[0:2] + self.env_bounds_max[0:2])
                    self.obs_dict["robot_position"][env_ids, 0] = local_center_xy[0] + sampled_x
                    self.obs_dict["robot_position"][env_ids, 1] = local_center_xy[1] + fixed_y
                else:
                    self.obs_dict["robot_position"][env_ids, 0] = sampled_x
                    self.obs_dict["robot_position"][env_ids, 1] = fixed_y

        # Spawn mother-ship altitude.
        use_random_spawn_z = bool(getattr(self.task_config, "spawn_use_random_z", False))
        if use_random_spawn_z:
            z_min, z_max = self._get_spawn_z_sampling_range()
            sampled_z = torch.empty((env_ids.shape[0],), device=self.device)
            if abs(z_max - z_min) < 1e-9:
                sampled_z.fill_(z_min)
            else:
                sampled_z.uniform_(z_min, z_max)
            self.obs_dict["robot_position"][env_ids, 2] = sampled_z
        else:
            spawn_fixed_z = float(getattr(self.task_config, "spawn_fixed_z", 15.0))
            self.obs_dict["robot_position"][env_ids, 2] = spawn_fixed_z
        self.sim_env.IGE_env.write_to_sim()

        # Debug marker visualization is disabled.
        
        # Record spawn position and compute d_max for distance reward
        self.spawn_position[env_ids] = self.obs_dict["robot_position"][env_ids].clone()
        self.spawn_height_reference[env_ids] = self.spawn_position[env_ids, 2]
        dist_spawn_to_target = torch.norm(
            self.target_position[env_ids] - self.spawn_position[env_ids], dim=1
        )
        dist_spawn_to_target_xy = torch.norm(
            self.target_position[env_ids, 0:2] - self.spawn_position[env_ids, 0:2], dim=1
        )
        self.d_max[env_ids] = torch.clamp(dist_spawn_to_target, min=0.1)  # Avoid division by zero
        self.initial_distance_xy[env_ids] = torch.clamp(dist_spawn_to_target_xy, min=0.1)
        
        # Initialize previous distance for improvement reward
        self.previous_distance[env_ids] = dist_spawn_to_target
        self.previous_distance_xy[env_ids] = dist_spawn_to_target_xy
        
        # Initialize previous velocity for acceleration penalty (NEW)
        self.previous_velocity[env_ids] = self.obs_dict["robot_linvel"][env_ids]
        
        # Reset GMM force buffers
        self.main_wind_direction[env_ids] = 0.0
        self.main_wind_speed[env_ids] = 0.0
        self.main_wind_vector[env_ids] = 0.0
        self.local_wind_max_speed[env_ids] = 0.0
        self.gmm_force_direction[env_ids] = 0.0
        self.gmm_target_direction[env_ids] = 0.0
        self.gmm_force_update_counter[env_ids] = 0
        self.dryden_wind_state[env_ids] = 0.0
        self.dryden_sigma[env_ids] = 0.0
        self.dryden_tau[env_ids] = 1.0
        self.task_external_force_tensor[env_ids] = 0.0
        self.task_external_torque_tensor[env_ids] = 0.0
        self.child_landing_position[env_ids] = 0.0
        self.child_landing_xy_distance[env_ids] = float("inf")
        self.child_drop_triggered[env_ids] = False
        self.child_has_dropped[env_ids] = False
        self.step_drop_event_mask[env_ids] = False
        self.step_impulse_metric[env_ids] = 0.0
        self.step_drop_attitude_theta[env_ids] = 0.0
        self.step_drop_roll[env_ids] = 0.0
        self.step_drop_pitch[env_ids] = 0.0
        self.step_drop_yaw[env_ids] = 0.0
        self.step_drop_delta_theta[env_ids] = 0.0
        self.step_drop_delta_v[env_ids] = 0.0
        self.step_drop_delta_omega[env_ids] = 0.0
        self.step_drop_heading_error[env_ids] = 0.0
        self.step_landing_error_xy[env_ids] = 0.0
        self.step_release_to_target_xy[env_ids] = 0.0
        confidence_history_init = float(
            max(getattr(self.drop_model_config, "confidence_error_scale", 3.0), 1e-6)
        )
        self.pred_drop_error_history[env_ids] = confidence_history_init
        self.release_confidence[env_ids] = 0.0
        self.release_confidence_error_component[env_ids] = 0.0
        self.release_confidence_risk_component[env_ids] = 0.0
        self.release_confidence_stability_component[env_ids] = 0.0
        self.confidence_gate_blocked_mask[env_ids] = False
        self.step_drop_trace_norm_t[env_ids] = 0.0
        self.step_drop_trace_error_xy[env_ids] = 0.0
        self.step_drop_trace_pos_x[env_ids] = 0.0
        self.step_drop_trace_pos_y[env_ids] = 0.0
        self.step_drop_trace_pos_z[env_ids] = 0.0
        self.step_drop_trace_fall_steps[env_ids] = 0
        self.step_drop_trace_fall_time[env_ids] = 0.0
        
        # Reset success counter
        self.success_counter[env_ids] = 0.0
        
        # Reset previous state for reward calculation
        self.previous_position[env_ids] = self.obs_dict["robot_position"][env_ids]
        self.previous_distance[env_ids] = dist_spawn_to_target
        self.previous_distance_xy[env_ids] = dist_spawn_to_target_xy
        self.prev_predicted_drop_error_xy[env_ids] = dist_spawn_to_target_xy
        self.previous_actions[env_ids] = 0.0

        # Reset hover time counter
        self.hover_time_counter[env_ids] = 0.0
        
        # Reset arrival tracker
        self.has_arrived[env_ids] = False
        
        # ==== Wind field initialization ====
        # 1. Resample per-episode main wind (Layer 1)
        self._resample_main_wind(env_ids)
        if self.dryden_enabled:
            # 2. Dryden mode: sample per-env (sigma, tau), state starts from zero.
            self._resample_dryden_parameters(env_ids)
        else:
            # 2. Local-gust mode.
            self._resample_local_wind_speed(env_ids)
            local_horizontal_only = bool(
                getattr(self.gmm_force_config, "local_wind_horizontal_only", True)
            )
            local_dirs = self._sample_unit_directions(
                env_ids.shape[0], horizontal_only=local_horizontal_only
            )
            self.gmm_force_direction[env_ids] = local_dirs
            self.gmm_target_direction[env_ids] = local_dirs

        self._compute_base_observation()
        reset_base_obs = self.base_task_observations[env_ids]
        self.obs_frame_buffer[env_ids] = reset_base_obs.unsqueeze(1).repeat(
            1, self.obs_frame_stack, 1
        )
        
        # Initialize logging/diagnostic buffers that are still used downstream.
        if not hasattr(self, "hold_counter"):
            self.hold_counter = torch.zeros(self.sim_env.num_envs, device=self.device)
            self.max_hold_counter = torch.zeros(self.sim_env.num_envs, device=self.device)
            self.failure_log_path = os.path.join(self.task_config.log_dir if hasattr(self.task_config, "log_dir") else ".", "failure_log.txt")
            # Create/Clear log file
            with open(self.failure_log_path, "w") as f:
                f.write("dist,speed,max_hold_steps\n")

        # Failure Log (before reset) - Only for actually failed envs (not successful ones)
        # Check success_buf (updated in step before reset_idx)
        if hasattr(self, "success_buf"):
            failed_mask = (self.success_buf[env_ids] == 0)
            if failed_mask.any():
                failed_ids = env_ids[failed_mask]
                
                # Get stats
                dists = torch.norm(self.target_position[failed_ids] - self.obs_dict["robot_position"][failed_ids], dim=1)
                speeds = torch.norm(self.obs_dict["robot_linvel"][failed_ids], dim=1)
                holds = self.max_hold_counter[failed_ids]
                
                # Log to file
                try:
                    with open(self.failure_log_path, "a") as f:
                        for d, s, h in zip(dists.cpu().numpy(), speeds.cpu().numpy(), holds.cpu().numpy()):
                            line = f"{d:.4f},{s:.4f},{int(h)}\n"
                            f.write(line)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception as e:
                    # Fail silently or log error only
                    pass
        else:
            # First run, initialize buffers if not done (though 'if not hasattr' above handles logic buffers)
            # Ensure success_buf exists if we rely on it later? 
            # It's usually created in create_sim -> allocate_buffers.
            pass

        # Reset hold/logging buffers
        self.hold_counter[env_ids] = 0.0
        self.max_hold_counter[env_ids] = 0.0
        if hasattr(self, "hover_good_min"):
            self.hover_good_min[env_ids] = float("inf")
            self.hover_good_max[env_ids] = -float("inf")
        
        # Keep existing info snapshots (used by external evaluators) and only refresh base keys.
        if not isinstance(self.infos, dict):
            self.infos = {}
        self.infos["successes"] = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool
        )
        self.infos["crashes"] = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool
        )
        self.infos["timeouts"] = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        return

    def render(self):
        return self.sim_env.render()

    def logging_sanity_check(self, infos):
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        threshold = int(self.early_crash_config.threshold_steps)
        env_list_for_toc = (time_at_crash < threshold).nonzero(as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")
            # Handle early crash with retry logic
            self._handle_early_crash(env_list_for_toc)

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical("Success and timeout are occuring at the same time")
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(successes, timeouts))}"
            )
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, timeouts))}"
            )
        return

    def _handle_early_crash(self, env_ids):
        """Handle environments that crashed too early by retrying with position offset."""
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        self._early_crash_retries[env_ids] += 1

        max_retries = int(self.early_crash_config.max_retries)
        exceeded_mask = self._early_crash_retries[env_ids] > max_retries
        exceeded_envs = env_ids[exceeded_mask]
        retry_envs = env_ids[~exceeded_mask]

        if exceeded_envs.numel() > 0:
            logger.error(
                f"Envs {exceeded_envs.tolist()} exceeded max retries ({max_retries}) for early crash."
            )
            if self.early_crash_config.fallback_to_center:
                self._move_to_safe_position(exceeded_envs)
            else:
                logger.error("fallback_to_center is disabled. Skipping recovery.")

        if retry_envs.numel() > 0:
            logger.warning(
                f"Retrying reset for envs: {retry_envs.tolist()} "
                f"(attempt {self._early_crash_retries[retry_envs].tolist()})"
            )
            self._reset_with_position_offset(retry_envs)

    def _move_to_safe_position(self, env_ids):
        """Move drone to the center of the environment as a safe fallback."""
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        center = (self.env_bounds_min + self.env_bounds_max) / 2.0
        center_expanded = center.view(1, 3).expand(env_ids.shape[0], -1)

        # Set robot position to center
        self.obs_dict["robot_position"][env_ids] = center_expanded

        # Reset velocity to zero
        self.obs_dict["robot_linvel"][env_ids] = 0.0
        self.obs_dict["robot_angvel"][env_ids] = 0.0

        # Write changes to simulation
        self.sim_env.IGE_env.write_to_sim()

        # Reset retry counter for these environments
        self._early_crash_retries[env_ids] = 0

        logger.info(f"Moved envs {env_ids.tolist()} to safe center position: {center.tolist()}")

    def _reset_with_position_offset(self, env_ids):
        """Reset environment with a random position offset to avoid repeated collision."""
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        margin = float(self.early_crash_config.safe_spawn_margin)

        # Generate random offset in range [-margin, +margin]
        offset = (torch.rand(env_ids.shape[0], 3, device=self.device) - 0.5) * 2.0 * margin

        # Get current position and apply offset
        current_pos = self.obs_dict["robot_position"][env_ids].clone()
        new_pos = current_pos + offset

        # Clamp to stay within environment bounds with margin
        bounds_min = self.env_bounds_min + margin
        bounds_max = self.env_bounds_max - margin
        new_pos = torch.clamp(new_pos, bounds_min, bounds_max)

        # Apply new position
        self.obs_dict["robot_position"][env_ids] = new_pos

        # Reset velocity to zero
        self.obs_dict["robot_linvel"][env_ids] = 0.0
        self.obs_dict["robot_angvel"][env_ids] = 0.0

        # Write changes to simulation
        self.sim_env.IGE_env.write_to_sim()

        logger.info(
            f"Reset envs {env_ids.tolist()} with position offset. "
            f"Original: {current_pos[0].tolist()}, New: {new_pos[0].tolist()}"
        )

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        return

    def process_image_observation(self):
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        if self.task_config.vae_config.use_vae:
            self.image_latents[:] = self.vae_model.encode(image_obs)

    def _compute_drop_decision_features(
        self,
        target_relative_position,
        body_linvel,
        euler_angles,
        body_angvel,
    ):
        gravity = float(getattr(self.drop_model_config, "child_gravity", 9.81))
        gravity = max(gravity, 1e-6)
        z = torch.clamp(self.obs_dict["robot_position"][:, 2], min=0.05)
        predicted_fall_time = torch.sqrt(torch.clamp(2.0 * z / gravity, min=1e-6))
        predicted_drop_error_xy = torch.norm(
            target_relative_position[:, 0:2] - body_linvel[:, 0:2] * predicted_fall_time.unsqueeze(1),
            dim=1,
        )
        attitude_deg = torch.rad2deg(
            torch.sqrt(
                torch.clamp(
                    euler_angles[:, 0] * euler_angles[:, 0]
                    + euler_angles[:, 1] * euler_angles[:, 1],
                    min=0.0,
                )
            )
        )
        omega_xy = torch.norm(body_angvel[:, 0:2], dim=1)
        return predicted_drop_error_xy, predicted_fall_time, attitude_deg, omega_xy

    def _compute_release_confidence(self):
        """Compute heuristic release confidence in [0, 1] for confidence-gated DROP."""
        target_relative_position = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        body_linvel = self.obs_dict["robot_body_linvel"]
        body_angvel = self.obs_dict["robot_body_angvel"]
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        predicted_drop_error_xy, _, attitude_deg, omega_xy = (
            self._compute_drop_decision_features(
                target_relative_position=target_relative_position,
                body_linvel=body_linvel,
                euler_angles=euler_angles,
                body_angvel=body_angvel,
            )
        )

        self.pred_drop_error_history[:] = torch.roll(
            self.pred_drop_error_history, shifts=-1, dims=1
        )
        self.pred_drop_error_history[:, -1] = predicted_drop_error_xy
        predicted_error_std = torch.std(
            self.pred_drop_error_history, dim=1, unbiased=False
        )

        error_scale = float(
            max(getattr(self.drop_model_config, "confidence_error_scale", 3.0), 1e-6)
        )
        attitude_limit = float(
            max(getattr(self.drop_model_config, "max_release_attitude_deg", 10.0), 1e-6)
        )
        omega_limit = float(
            max(getattr(self.drop_model_config, "max_release_omega_xy", 0.3), 1e-6)
        )
        preferred_attitude_deg = float(
            getattr(self.drop_model_config, "preferred_release_attitude_deg", 7.5)
        )
        preferred_attitude_band_deg = float(
            max(getattr(self.drop_model_config, "preferred_release_attitude_band_deg", 2.5), 1e-6)
        )
        stability_scale = float(
            max(getattr(self.drop_model_config, "confidence_stability_scale", 1.0), 1e-6)
        )

        c_error = torch.exp(-torch.square(predicted_drop_error_xy / error_scale))
        c_attitude = torch.exp(
            -torch.square((attitude_deg - preferred_attitude_deg) / preferred_attitude_band_deg)
        )
        c_attitude = torch.where(
            attitude_deg <= attitude_limit,
            c_attitude,
            torch.zeros_like(c_attitude),
        )
        c_omega = torch.exp(-torch.square(omega_xy / omega_limit))
        c_risk = c_attitude * c_omega
        c_stability = torch.exp(-torch.square(predicted_error_std / stability_scale))
        confidence = torch.clamp(c_error * c_risk * c_stability, 0.0, 1.0)

        self.release_confidence[:] = confidence
        self.release_confidence_error_component[:] = c_error
        self.release_confidence_risk_component[:] = c_risk
        self.release_confidence_stability_component[:] = c_stability
        return confidence

    def _compute_base_observation(self):
        target_relative_position = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        body_linvel = self.obs_dict["robot_body_linvel"]
        body_angvel = self.obs_dict["robot_body_angvel"]
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])

        self.base_task_observations[:, 0:3] = target_relative_position
        self.base_task_observations[:, 3:6] = body_linvel
        self.base_task_observations[:, 6] = euler_angles[:, 0]
        self.base_task_observations[:, 7] = euler_angles[:, 1]
        self.base_task_observations[:, 8] = euler_angles[:, 2]
        self.base_task_observations[:, 9:12] = body_angvel
        self.base_task_observations[:, 12] = self.obs_dict["robot_position"][:, 2]
        return target_relative_position, body_linvel, body_angvel, euler_angles

    def _compute_predicted_recoil_terms(self):
        cfg = self.drop_impact_config
        if cfg is None or not bool(getattr(cfg, "enable_impact", True)):
            zeros = torch.zeros((self.sim_env.num_envs, 3), device=self.device, dtype=torch.float32)
            return zeros, zeros

        current_mass = self.obs_dict["robot_mass"].view(-1, 1).clamp_min(1e-6)
        m_child_cfg = float(getattr(cfg, "child_mass", 1.0))
        payload_mass = float(self._payload_mass_reference)
        if payload_mass <= 1e-6:
            payload_mass = m_child_cfg
        payload_mass = max(payload_mass, 1e-6)
        residual_mass = float(getattr(cfg, "dropped_payload_residual_mass", 1e-4))
        payload_removed_mass = max(payload_mass - max(residual_mass, 0.0), 0.0)
        dyn_mass = torch.clamp(current_mass - payload_removed_mass, min=1e-6)

        eject_speed = float(getattr(cfg, "eject_speed", 0.5))
        eject_dir_body = torch.tensor(
            getattr(cfg, "eject_direction_body", [0.0, 0.0, -1.0]),
            device=self.device,
            dtype=torch.float32,
        )
        if torch.norm(eject_dir_body) < 1e-6:
            eject_dir_body = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        eject_dir_body = eject_dir_body / torch.norm(eject_dir_body)
        v_eject_body = eject_speed * eject_dir_body.view(1, 3).expand(self.sim_env.num_envs, -1)

        sim_dt = float(self.obs_dict["dt"]) if "dt" in self.obs_dict else 0.01
        sim_dt = max(sim_dt, 1e-5)
        num_substeps = max(self.physics_steps_per_env_step_mean, 1.0)
        control_dt = max(sim_dt * num_substeps, 1e-5)
        delta_v_target_body = -(payload_mass / dyn_mass) * v_eject_body
        recoil_force_body = (dyn_mass / control_dt) * delta_v_target_body
        recoil_force_world = quat_rotate(self.obs_dict["robot_orientation"], recoil_force_body)

        if hasattr(self, "fixed_payload_offset_body"):
            payload_offset_body = self.fixed_payload_offset_body.view(1, 3).expand(
                self.sim_env.num_envs, -1
            )
        else:
            payload_offset_body = torch.tensor(
                getattr(cfg, "payload_offset_body", [0.0, 0.0, 0.0]),
                device=self.device,
                dtype=torch.float32,
            ).view(1, 3).expand(self.sim_env.num_envs, -1)

        if torch.norm(payload_offset_body[0]).item() > 1e-9:
            r_body = payload_offset_body - self._robot_com_body
            recoil_tau_body = torch.cross(r_body, recoil_force_body, dim=1)
            recoil_tau_world = quat_rotate(self.obs_dict["robot_orientation"], recoil_tau_body)
        else:
            recoil_tau_world = torch.zeros_like(recoil_force_world)
        return recoil_force_world, recoil_tau_world

    def _compute_privileged_observation(self):
        wind_world = self._compute_gmm_wind_vector(self.obs_dict["robot_position"])
        robot_mass = self.obs_dict["robot_mass"].view(-1, 1)
        robot_inertia = self.obs_dict["robot_inertia"]
        inertia_diag = torch.stack(
            [robot_inertia[:, 0, 0], robot_inertia[:, 1, 1], robot_inertia[:, 2, 2]], dim=1
        )
        recoil_force_world, recoil_tau_world = self._compute_predicted_recoil_terms()

        self.privileged_observations[:, 0:3] = wind_world
        self.privileged_observations[:, 3:4] = robot_mass
        self.privileged_observations[:, 4:7] = inertia_diag
        self.privileged_observations[:, 7:10] = recoil_force_world
        self.privileged_observations[:, 10:13] = recoil_tau_world
        return self.privileged_observations

    def _refresh_stacked_observations(self):
        self.obs_frame_buffer[:] = torch.roll(self.obs_frame_buffer, shifts=-1, dims=1)
        self.obs_frame_buffer[:, -1, :] = self.base_task_observations
        actor_obs = self.obs_frame_buffer.reshape(self.sim_env.num_envs, -1)
        self.task_obs["observations"][:] = actor_obs
        privileged_obs = self._compute_privileged_observation()
        self.task_obs["states"][:, 0:self.actor_observation_space_dim] = actor_obs
        self.task_obs["states"][:, self.actor_observation_space_dim :] = privileged_obs

    def process_obs_for_task(self):
        self._compute_base_observation()
        self._refresh_stacked_observations()

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def step(self, actions):
        # Recoil torque is event-based and should only persist for one physics step.
        self.task_external_torque_tensor.zero_()
        # Update local turbulence model.
        if self.dryden_enabled:
            self._update_dryden_turbulence()
        else:
            self._update_gmm_force_direction()
        
        # Apply GMM physical forces to robot
        self._apply_gmm_physical_forces()
        self.step_drop_event_mask[:] = False
        self.step_impulse_metric[:] = 0.0
        self.step_drop_attitude_theta[:] = 0.0
        self.step_drop_roll[:] = 0.0
        self.step_drop_pitch[:] = 0.0
        self.step_drop_yaw[:] = 0.0
        self.step_drop_delta_theta[:] = 0.0
        self.step_drop_delta_v[:] = 0.0
        self.step_drop_delta_omega[:] = 0.0
        self.step_drop_heading_error[:] = 0.0
        self.step_landing_error_xy[:] = 0.0
        self.step_release_to_target_xy[:] = 0.0
        self.step_drop_trace_norm_t[:] = 0.0
        self.step_drop_trace_error_xy[:] = 0.0
        self.step_drop_trace_pos_x[:] = 0.0
        self.step_drop_trace_pos_y[:] = 0.0
        self.step_drop_trace_pos_z[:] = 0.0
        self.step_drop_trace_fall_steps[:] = 0
        self.step_drop_trace_fall_time[:] = 0.0

        if actions.shape[1] < 4:
            raise ValueError("Action tensor must include at least 4 mother-control dimensions.")

        mother_actions = actions[:, 0:4]
        if actions.shape[1] >= 5:
            drop_switch = actions[:, 4]
        else:
            drop_switch = torch.full(
                (actions.shape[0],), -1.0, device=self.device, dtype=actions.dtype
            )

        drop_threshold = float(getattr(self.drop_model_config, "drop_threshold", 0.0))
        requested_drop_mask = drop_switch > drop_threshold
        allow_multiple_drops = bool(getattr(self.drop_model_config, "allow_multiple_drops", False))
        if allow_multiple_drops:
            candidate_drop_mask = requested_drop_mask
        else:
            candidate_drop_mask = requested_drop_mask & (~self.child_has_dropped)

        euler_angles_for_gate = ssa(self.obs_dict["robot_euler_angles"])
        attitude_deg_for_gate = torch.rad2deg(
            torch.sqrt(
                torch.clamp(
                    euler_angles_for_gate[:, 0] * euler_angles_for_gate[:, 0]
                    + euler_angles_for_gate[:, 1] * euler_angles_for_gate[:, 1],
                    min=0.0,
                )
            )
        )
        omega_xy_for_gate = torch.norm(self.obs_dict["robot_body_angvel"][:, 0:2], dim=1)
        attitude_limit = float(
            max(getattr(self.drop_model_config, "max_release_attitude_deg", 10.0), 1e-6)
        )
        omega_limit = float(
            max(getattr(self.drop_model_config, "max_release_omega_xy", 0.3), 1e-6)
        )
        hard_release_ready = (
            (attitude_deg_for_gate <= attitude_limit) & (omega_xy_for_gate <= omega_limit)
        )

        confidence_gate_enabled = bool(
            getattr(self.drop_model_config, "confidence_gate_enable", False)
        )
        self.confidence_gate_blocked_mask[:] = False
        if confidence_gate_enabled:
            confidence = self._compute_release_confidence()
            confidence_threshold = float(
                getattr(self.drop_model_config, "confidence_threshold", 0.45)
            )
            gate_ready = confidence > confidence_threshold
            min_steps = int(getattr(self.drop_model_config, "confidence_min_steps", 0))
            if min_steps > 0:
                min_step_ready = self.sim_env.sim_steps >= min_steps
            else:
                min_step_ready = torch.ones_like(requested_drop_mask, dtype=torch.bool)
            allowed_drop_mask = gate_ready & min_step_ready & hard_release_ready
            drop_event_mask = candidate_drop_mask & allowed_drop_mask
            self.confidence_gate_blocked_mask[:] = candidate_drop_mask & (~allowed_drop_mask)
        else:
            drop_event_mask = candidate_drop_mask & hard_release_ready
            self.confidence_gate_blocked_mask[:] = candidate_drop_mask & (~hard_release_ready)
        self.extras["confidence_gate_blocked_count"] = int(
            self.confidence_gate_blocked_mask.sum().item()
        )
        self.extras["release_confidence_mean"] = (
            float(self.release_confidence.mean().item()) if confidence_gate_enabled else 0.0
        )
        self.extras["release_confidence_min"] = (
            float(self.release_confidence.min().item()) if confidence_gate_enabled else 0.0
        )
        self.extras["release_confidence_max"] = (
            float(self.release_confidence.max().item()) if confidence_gate_enabled else 0.0
        )
        self.child_drop_triggered[:] = drop_event_mask

        if bool(getattr(self.drop_model_config, "enable_drop_model", True)) and drop_event_mask.any():
            dropped_env_ids = drop_event_mask.nonzero(as_tuple=False).squeeze(-1)
            add_child_eject_velocity = bool(
                getattr(self.drop_impact_config, "add_child_eject_velocity", True)
            )
            child_release_pos, child_release_vel = self._compute_child_release_kinematics(
                dropped_env_ids,
                include_eject_velocity=add_child_eject_velocity,
            )
            release_to_target_xy = self._compute_release_to_target_xy_local(
                dropped_env_ids, child_release_pos
            )
            euler_before = self.obs_dict["robot_euler_angles"][dropped_env_ids].clone()
            roll_pitch_before = euler_before[:, 0:2]
            yaw_before = euler_before[:, 2]
            linvel_before = self.obs_dict["robot_linvel"][dropped_env_ids].clone()
            omega_before = self.obs_dict["robot_body_angvel"][dropped_env_ids].clone()
            omega_drop_xy_before = torch.norm(omega_before[:, 0:2], dim=1)
            self._apply_drop_impact(dropped_env_ids)
        else:
            dropped_env_ids = None
            child_release_pos = None
            child_release_vel = None
            release_to_target_xy = None
            roll_pitch_before = None
            yaw_before = None
            linvel_before = None
            omega_before = None
            omega_drop_xy_before = None

        # Store mother control actions for smoothness penalties.
        self.actions = mother_actions

        transformed_action = self.action_transformation_function(mother_actions)
        logger.debug(f"raw_action: {mother_actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)

        if dropped_env_ids is not None:
            roll_pitch_after = self.obs_dict["robot_euler_angles"][dropped_env_ids, 0:2]
            linvel_after = self.obs_dict["robot_linvel"][dropped_env_ids]
            omega_after = self.obs_dict["robot_body_angvel"][dropped_env_ids]
            delta_v = torch.norm(linvel_after - linvel_before, dim=1)
            delta_theta = torch.norm(roll_pitch_after - roll_pitch_before, dim=1)
            delta_omega = torch.norm(omega_after - omega_before, dim=1)
            theta_drop = torch.norm(roll_pitch_before, dim=1)
            alpha = float(getattr(self.drop_reward_config, "impulse_alpha", 1.0))
            beta = float(getattr(self.drop_reward_config, "impulse_beta", 0.5))
            # Impact metric uses translational and angular jumps at release.
            impulse_metric = alpha * delta_v + beta * delta_omega
            self.step_impulse_metric[dropped_env_ids] = impulse_metric
            self.step_drop_attitude_theta[dropped_env_ids] = theta_drop
            self.step_drop_roll[dropped_env_ids] = roll_pitch_before[:, 0]
            self.step_drop_pitch[dropped_env_ids] = roll_pitch_before[:, 1]
            self.step_drop_yaw[dropped_env_ids] = yaw_before
            self.step_drop_delta_theta[dropped_env_ids] = delta_theta
            self.step_drop_delta_v[dropped_env_ids] = delta_v
            self.step_drop_delta_omega[dropped_env_ids] = delta_omega
            # Keep buffer name for compatibility; value now is DROP-pre ||[wx, wy]||.
            self.step_drop_heading_error[dropped_env_ids] = omega_drop_xy_before
            self.step_drop_event_mask[dropped_env_ids] = True

            self._simulate_child_free_fall(
                dropped_env_ids, init_pos=child_release_pos, init_vel=child_release_vel
            )
            self.step_landing_error_xy[dropped_env_ids] = self.child_landing_xy_distance[dropped_env_ids]
            self.step_release_to_target_xy[dropped_env_ids] = release_to_target_xy
            self.child_has_dropped[dropped_env_ids] = True
            self.extras["drop_count"] = int(dropped_env_ids.numel())
            self.extras["child_landing_xy_distance_mean"] = float(
                self.child_landing_xy_distance[dropped_env_ids].mean().item()
            )
            self.extras["release_to_target_xy_mean"] = float(release_to_target_xy.mean().item())
            self.extras["release_to_target_xy_min"] = float(release_to_target_xy.min().item())
            self.extras["release_to_target_xy_max"] = float(release_to_target_xy.max().item())
            self.extras["impulse_metric_mean"] = float(impulse_metric.mean().item())
            self.extras["drop_attitude_theta_mean"] = float(theta_drop.mean().item())
            self.extras["drop_heading_error_mean"] = float(omega_drop_xy_before.mean().item())
        else:
            self.extras["drop_count"] = 0
            self.extras["child_landing_xy_distance_mean"] = 0.0
            self.extras["release_to_target_xy_mean"] = 0.0
            self.extras["release_to_target_xy_min"] = 0.0
            self.extras["release_to_target_xy_max"] = 0.0
            self.extras["impulse_metric_mean"] = 0.0
            self.extras["drop_attitude_theta_mean"] = 0.0
            self.extras["drop_heading_error_mean"] = 0.0

        (
            self.rewards[:],
            improvement_reward,
            direction_reward,
            noise_reduction_reward,
            safety_reward,
            action_smoothness_penalty,
            hover_reward,
            threshold_reward,
            anchor_reward,
        ) = self._compute_reward_and_scores()
        confidence_gate_penalty = float(
            getattr(
                self.drop_reward_config,
                "blocked_drop_penalty",
                getattr(self.drop_model_config, "confidence_gate_penalty", 0.0),
            )
        )
        if confidence_gate_penalty > 0.0 and self.confidence_gate_blocked_mask.any():
            self.rewards[self.confidence_gate_blocked_mask] = (
                self.rewards[self.confidence_gate_blocked_mask] - confidence_gate_penalty
            )
            self.extras["confidence_gate_penalty_mean"] = float(
                confidence_gate_penalty
                * self.confidence_gate_blocked_mask.float().mean().item()
            )
        else:
            self.extras["confidence_gate_penalty_mean"] = 0.0
        if self.task_config.return_state_before_reset is True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )
        # End episode immediately after DROP reward is computed.
        # Use truncation channel so it is not mixed into crash statistics.
        drop_done_mask = self.step_drop_event_mask
        if drop_done_mask.any():
            self.truncations[drop_done_mask] = 1
        step_drop_end_count = int(drop_done_mask.sum().item())
        self.extras["drop_episode_end_count"] = step_drop_end_count
        self.window_drop_end_count += step_drop_end_count
        done_mask_pre_reward = (self.terminations > 0) | (self.truncations > 0)
        no_drop_done_mask = done_mask_pre_reward & (~self.child_has_dropped)
        step_no_drop_done_count = int(no_drop_done_mask.sum().item())
        no_drop_crash_mask = no_drop_done_mask & (self.terminations > 0)
        no_drop_timeout_mask = no_drop_done_mask & (self.terminations == 0) & (self.truncations > 0)
        crash_no_drop_penalty = float(
            getattr(
                self.drop_reward_config,
                "crash_no_drop_penalty",
                getattr(self.drop_reward_config, "outside_region_penalty", 20.0),
            )
        )
        missed_drop_no_drop_penalty = float(
            getattr(self.drop_reward_config, "missed_drop_no_drop_penalty", 8.0)
        )
        reasonable_no_drop_reward = float(
            getattr(self.drop_reward_config, "reasonable_no_drop_reward", 2.0)
        )
        reasonable_no_drop_threshold = float(
            getattr(self.drop_reward_config, "reasonable_no_drop_pred_error_threshold", 5.0)
        )
        no_drop_eval_max_xy_dist = float(
            getattr(self.drop_reward_config, "no_drop_eval_max_xy_dist", 10.0)
        )
        if crash_no_drop_penalty > 0.0 and no_drop_crash_mask.any():
            self.rewards[no_drop_crash_mask] = (
                self.rewards[no_drop_crash_mask] - crash_no_drop_penalty
            )

        reasonable_no_drop_count = 0
        missed_drop_no_drop_count = 0
        no_drop_pred_error_mean = 0.0
        if no_drop_timeout_mask.any():
            timeout_env_ids = no_drop_timeout_mask.nonzero(as_tuple=False).squeeze(-1)
            add_child_eject_velocity = bool(
                getattr(self.drop_impact_config, "add_child_eject_velocity", True)
            )
            pred_pos, pred_vel = self._compute_child_release_kinematics(
                timeout_env_ids,
                include_eject_velocity=add_child_eject_velocity,
            )
            _, pred_error_xy = self._predict_child_landing_xy(
                timeout_env_ids,
                init_pos=pred_pos,
                init_vel=pred_vel,
            )
            timeout_dist_xy = torch.norm(
                self.target_position[timeout_env_ids, 0:2]
                - self.obs_dict["robot_position"][timeout_env_ids, 0:2],
                dim=1,
            )
            far_mask = timeout_dist_xy > no_drop_eval_max_xy_dist
            reasonable_mask = (pred_error_xy > reasonable_no_drop_threshold) & (~far_mask)
            missed_mask = (~reasonable_mask)
            reasonable_env_ids = timeout_env_ids[reasonable_mask]
            missed_env_ids = timeout_env_ids[missed_mask]

            if reasonable_no_drop_reward != 0.0 and reasonable_env_ids.numel() > 0:
                self.rewards[reasonable_env_ids] = (
                    self.rewards[reasonable_env_ids] + reasonable_no_drop_reward
                )
            if missed_drop_no_drop_penalty > 0.0 and missed_env_ids.numel() > 0:
                self.rewards[missed_env_ids] = (
                    self.rewards[missed_env_ids] - missed_drop_no_drop_penalty
                )

            reasonable_no_drop_count = int(reasonable_env_ids.numel())
            missed_drop_no_drop_count = int(missed_env_ids.numel())
            no_drop_pred_error_mean = float(pred_error_xy.mean().item())

        self.extras["no_drop_done_count"] = step_no_drop_done_count
        self.extras["reasonable_no_drop_count"] = reasonable_no_drop_count
        self.extras["missed_drop_no_drop_count"] = missed_drop_no_drop_count
        self.extras["no_drop_pred_landing_error_xy_mean"] = no_drop_pred_error_mean
        self.window_no_drop_done_count += step_no_drop_done_count

        position = self.obs_dict["robot_position"]
        dist_to_target = torch.norm(self.target_position - position, dim=1)
        in_goal_mask = dist_to_target <= self.success_config.success_radius
        self.has_arrived = self.has_arrived | in_goal_mask

        linvel = self.obs_dict["robot_linvel"]
        linvel_magnitude = torch.norm(linvel, dim=1)
        cond_vel = linvel_magnitude <= self.success_config.stability_velocity_threshold
        is_stable = in_goal_mask & cond_vel

        self.hold_counter = torch.where(
            is_stable,
            self.hold_counter + 1.0,
            torch.zeros_like(self.hold_counter)
        )
        
        # Update max hold counter for logging
        self.max_hold_counter = torch.max(self.max_hold_counter, self.hold_counter)
        
        # Success-based terminal shaping is disabled for drop-decision reward design.
        has_succeeded = torch.zeros_like(in_goal_mask, dtype=torch.bool)
        self.success_buf[:] = 0
        
        # TensorBoard Logging
        self.extras["improvement_reward"] = improvement_reward.mean()
        self.extras["direction_reward"] = direction_reward.mean()
        self.extras["step_reward"] = self.rewards.mean()
        self.extras["max_hold_steps"] = self.max_hold_counter.mean()
        
        # Accumulate episode statistics
        self.episode_sums += self.rewards
        self.episode_lengths += 1
        self.episode_pot_sums += improvement_reward.detach()
        self.episode_safe_sums += safety_reward.detach()
        self.episode_smooth_sums += action_smoothness_penalty.detach()
        self.episode_hover_sums += hover_reward.detach()
        self.episode_threshold_sums += threshold_reward.detach()
        self.episode_anchor_sums += anchor_reward.detach()
        
        # Check for done envs (terminations or truncations)
        dones = (self.terminations > 0) | (self.truncations > 0)
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).flatten()

            done_count = int(done_indices.numel())
            self.window_done_total += done_count
            crash_mask = (self.obs_dict["crashes"][done_indices] > 0)
            non_crash_mask = ~crash_mask
            if non_crash_mask.any():
                end_speeds = torch.norm(self.obs_dict["robot_linvel"][done_indices], dim=1)
                self.window_done_speed_sum += float(end_speeds[non_crash_mask].sum().item())
                self.window_done_non_crash += int(non_crash_mask.sum().item())

            # --- Capture final arrival status only ---
            try:
                d_pos = self.obs_dict["robot_position"][done_indices]
                dist_to_tgt = torch.norm(self.target_position[done_indices] - d_pos, dim=1)
                arrival_threshold = 2.0
                final_arrived = (dist_to_tgt <= arrival_threshold)
                final_arrived_vals = final_arrived.detach().cpu().numpy()
                term_reward_val = self.reward_params.get("terminal_reward", 0.0)
                if term_reward_val > 0.0:
                    arrival_bonus = final_arrived.float() * term_reward_val
                    self.rewards[done_indices] += arrival_bonus
                self.final_arrival_buffer.extend(final_arrived_vals.astype(float))
                self.recent_arrival_buffer.extend(final_arrived_vals)
            except Exception as e:
                logger.error(f"Failed to compute final arrival stats: {e}")
                import traceback
                traceback.print_exc()

            
            # Log episode rewards and lengths for rl_games
            # [REMOVED] Buggy terminal hover reward that was causing negative sums
            # if hasattr(self, "hover_good_max"):
            #     terminal_hover_reward = 10.0 * self.hover_good_max[done_indices]
            #     self.rewards[done_indices] += terminal_hover_reward
            #     self.episode_sums[done_indices] += terminal_hover_reward
            #     self.episode_hover_sums[done_indices] += terminal_hover_reward
            #     self.window_done_hover_reward_count += int((terminal_hover_reward > 0).sum().item())

            self.extras["episode_rewards"] = self.episode_sums[done_indices].cpu().numpy().tolist()
            self.extras["episode_lengths"] = self.episode_lengths[done_indices].cpu().numpy().tolist()

            # Reset buffers for done envs
            self.episode_sums[done_indices] = 0
            self.episode_lengths[done_indices] = 0
            self.episode_pot_sums[done_indices] = 0
            self.episode_safe_sums[done_indices] = 0
            self.episode_smooth_sums[done_indices] = 0
            self.episode_hover_sums[done_indices] = 0
            self.episode_threshold_sums[done_indices] = 0
            self.episode_anchor_sums[done_indices] = 0

        # Populate infos for sanity check and external access
        self.infos["successes"] = self.success_buf
        self.infos["crashes"] = self.obs_dict["crashes"]
        self.infos["timeouts"] = self.truncations.int() # Truncations are boolean/uint8, consistent with others

        # We only mark success at the end of episode for success rate calculation to be simple
        # OR we can track if it *ever* succeeded during the episode.
        # Let's use the standard approach: Success if condition met at termination OR if persistent success achieved.
        # To make it robust: If has_succeeded is true, we consider this episode a success.
        
        successes = has_succeeded.float()
        
        # Ensure crashes override success
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(self.terminations > 0, torch.zeros_like(timeouts), timeouts)

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations
        self.infos["drops"] = self.step_drop_event_mask
        # Snapshots for evaluation scripts: keep per-env DROP data before auto-reset clears tensors.
        self.infos["dones_snapshot"] = ((self.terminations > 0) | (self.truncations > 0)).clone()
        self.infos["crashes_snapshot"] = self.terminations.clone()
        self.infos["timeouts_snapshot"] = self.truncations.clone()
        self.infos["drops_snapshot"] = self.step_drop_event_mask.clone()
        self.infos["drop_roll_snapshot"] = self.step_drop_roll.clone()
        self.infos["drop_pitch_snapshot"] = self.step_drop_pitch.clone()
        self.infos["drop_yaw_snapshot"] = self.step_drop_yaw.clone()
        self.infos["drop_delta_theta_snapshot"] = self.step_drop_delta_theta.clone()
        self.infos["drop_delta_v_snapshot"] = self.step_drop_delta_v.clone()
        self.infos["drop_delta_omega_snapshot"] = self.step_drop_delta_omega.clone()
        self.infos["impulse_metric_snapshot"] = self.step_impulse_metric.clone()
        self.infos["landing_error_xy_snapshot"] = self.step_landing_error_xy.clone()
        self.infos["release_to_target_xy_snapshot"] = self.step_release_to_target_xy.clone()
        self.infos["drop_trace_norm_t_snapshot"] = self.step_drop_trace_norm_t.clone()
        self.infos["drop_trace_error_xy_snapshot"] = self.step_drop_trace_error_xy.clone()
        self.infos["drop_trace_pos_x_snapshot"] = self.step_drop_trace_pos_x.clone()
        self.infos["drop_trace_pos_y_snapshot"] = self.step_drop_trace_pos_y.clone()
        self.infos["drop_trace_pos_z_snapshot"] = self.step_drop_trace_pos_z.clone()
        self.infos["drop_trace_fall_steps_snapshot"] = self.step_drop_trace_fall_steps.clone()
        self.infos["drop_trace_fall_time_snapshot"] = self.step_drop_trace_fall_time.clone()

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        
        # Update recent_episodes deque for success rate calculation
        # Identify environments that are done (success, crash, or timeout)
        done_envs = (self.terminations > 0) | (self.truncations > 0)
        done_indices = done_envs.nonzero(as_tuple=False).squeeze(-1)
        
        if done_indices.numel() > 0:
            for idx in done_indices:
                if successes[idx] > 0:
                    self.recent_episodes.append("success")
                elif self.terminations[idx] > 0:
                    self.recent_episodes.append("crash")
                elif self.truncations[idx] > 0:
                    self.recent_episodes.append("timeout")
                
                # Update arrival stats for done envs
                self.recent_arrivals.append(bool(self.has_arrived[idx].item()))

        # Reset early crash retry counter for environments that didn't crash this step
        non_crash_envs = (self.terminations == 0).nonzero(as_tuple=False).squeeze(-1)
        if non_crash_envs.numel() > 0:
            self._early_crash_retries[non_crash_envs] = 0

        # Simplified logging (no more total_score from old system)
        self._populate_extras(improvement_reward, direction_reward, noise_reduction_reward)
        if self.log_step_scores and not self._episode_log_written:
            env_id = min(self.log_step_env_id, self.truncations.shape[0] - 1)
            done = (self.terminations[env_id] > 0) | (self.truncations[env_id] > 0)
            if bool(done.item()):
                self._write_best_point_log()
                self._episode_log_written = True

        reset_envs = self.sim_env.post_reward_calculation_step()

        if len(reset_envs) > 0:
            # self.reset_idx(reset_envs) # Redundant: EnvManager already called it!
            pass
        
        # EVAL mode: track if we've passed the early crash threshold
        if self._eval_mode and not self._eval_init_phase_complete:
            env_id = self.log_step_env_id if self.log_step_scores else 0
            threshold = int(self.early_crash_config.threshold_steps)
            if self.sim_env.sim_steps[env_id] >= threshold:
                self._eval_init_phase_complete = True
                logger.info(f"EVAL mode: init phase complete (passed {threshold} steps)")
        
        # EVAL mode: exit on crash or timeout after init phase is complete
        if self._eval_mode and self._eval_init_phase_complete and len(reset_envs) > 0:
            env_id = self.log_step_env_id if self.log_step_scores else 0
            if env_id in reset_envs:
                # Check if it's a crash or timeout (not success)
                is_crash = self.terminations[env_id] > 0
                is_timeout = self.truncations[env_id] > 0
                is_success = self.infos.get("successes", torch.zeros_like(self.terminations))[env_id] > 0
                
                if (is_crash or is_timeout) and not is_success:
                    # Save best point before exiting
                    if self.log_step_scores and not self._episode_log_written:
                        self._write_best_point_log()
                        self._episode_log_written = True
                    else:
                        self._write_best_point()
                    
                    reason = "crash" if is_crash else "timeout"
                    logger.info(f"EVAL mode: {reason} detected, exiting.")
                    # sys.exit(0)  <-- DISABLED to allow comparison script to continue
        
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        self.num_task_steps += 1
        self.process_image_observation()
        self._draw_env0_debug_markers()
        
        # Merge extras into infos so rl_games can see them
        self.infos.update(self.extras)

        if self.task_config.return_state_before_reset is False:
            return_tuple = self.get_return_tuple()
        return return_tuple
