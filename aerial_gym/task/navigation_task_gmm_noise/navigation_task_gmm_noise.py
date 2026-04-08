import os
import sys
import json
import numpy as np
import torch
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
        self.use_wind_estimation_features = bool(
            getattr(self.task_config, "use_wind_estimation_features", False)
        )
        self.drop_obstacle_obs_dim = 3  # obstacle xyz (or -1e3 sentinel when absent)
        self.base_observation_space_dim = 12 + self.drop_obstacle_obs_dim
        self.obs_linvel_history_frames = int(
            getattr(self.task_config, "obs_linvel_history_frames", 4)
        )
        if self.obs_linvel_history_frames < 1:
            self.obs_linvel_history_frames = 1
        self.wind_aug_prev_cmd_dim = 4
        self.wind_aug_delta_v_dim = 3
        self.wind_aug_linvel_hist_dim = 3 * self.obs_linvel_history_frames
        self.augmented_observation_space_dim = (
            self.base_observation_space_dim
            + self.wind_aug_prev_cmd_dim
            + self.wind_aug_delta_v_dim
            + self.wind_aug_linvel_hist_dim
        )
        self.wind_aug_prev_cmd_start = self.base_observation_space_dim
        self.wind_aug_prev_cmd_end = self.wind_aug_prev_cmd_start + self.wind_aug_prev_cmd_dim
        self.wind_aug_delta_v_start = self.wind_aug_prev_cmd_end
        self.wind_aug_delta_v_end = self.wind_aug_delta_v_start + self.wind_aug_delta_v_dim
        self.wind_aug_linvel_hist_start = self.wind_aug_delta_v_end
        self.wind_aug_linvel_hist_end = (
            self.wind_aug_linvel_hist_start + self.wind_aug_linvel_hist_dim
        )
        expected_obs_dim = (
            self.augmented_observation_space_dim
            if self.use_wind_estimation_features
            else self.base_observation_space_dim
        )
        if int(getattr(self.task_config, "observation_space_dim", expected_obs_dim)) != expected_obs_dim:
            logger.warning(
                f"Overriding observation_space_dim to {expected_obs_dim} "
                f"(use_wind_estimation_features={self.use_wind_estimation_features})."
            )
        self.task_config.observation_space_dim = expected_obs_dim

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
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
        }
        # Observation-history buffers for optional non-privileged wind-estimation features.
        self.obs_prev_cmd_body = torch.zeros(
            (self.sim_env.num_envs, 4), device=self.device, requires_grad=False
        )
        self.obs_prev_body_linvel = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.obs_body_linvel_history = torch.zeros(
            (self.sim_env.num_envs, self.obs_linvel_history_frames, 3),
            device=self.device,
            requires_grad=False,
        )
        self.current_cmd_body_for_obs = torch.zeros(
            (self.sim_env.num_envs, 4), device=self.device, requires_grad=False
        )

        self.num_task_steps = 0
        self.infos = {}

        self.noise_config = self.task_config.noise_config
        self.num_noise_sources = int(self.noise_config.num_sources)

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
        # Optional single obstacle around target used for drop-risk scoring.
        self.drop_obstacle_enable = bool(
            getattr(self.drop_reward_config, "drop_obstacle_enable", False)
        )
        self.drop_obstacle_spawn_prob = float(
            getattr(self.drop_reward_config, "drop_obstacle_spawn_prob", 0.5)
        )
        self.drop_obstacle_center_radius_min = float(
            getattr(self.drop_reward_config, "drop_obstacle_center_radius_min", 1.0)
        )
        self.drop_obstacle_center_radius_max = float(
            getattr(self.drop_reward_config, "drop_obstacle_center_radius_max", 2.0)
        )
        self.drop_obstacle_radius_min = float(
            getattr(self.drop_reward_config, "drop_obstacle_radius_min", 1.0)
        )
        self.drop_obstacle_radius_max = float(
            getattr(self.drop_reward_config, "drop_obstacle_radius_max", 3.0)
        )
        self.drop_obstacle_height = float(
            getattr(self.drop_reward_config, "drop_obstacle_height", 0.4)
        )
        self.drop_obstacle_absent_obs_value = float(
            getattr(self.drop_reward_config, "drop_obstacle_absent_obs_value", -1e3)
        )
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
        self.drop_obstacle_exists = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.drop_obstacle_position = torch.full(
            (self.sim_env.num_envs, 3),
            self.drop_obstacle_absent_obs_value,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.drop_obstacle_radius = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.step_drop_hit_obstacle = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.drop_obstacle_asset_file = str(
            getattr(
                self.drop_reward_config, "drop_obstacle_asset_file", "drop_box_0p2_0p2_0p4.urdf"
            )
        )
        self.drop_obstacle_asset_index = -1
        self.drop_obstacle_instance_enabled = False
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
        
        # Success counter for continuous success check (NEW)
        self.success_counter = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        # Episode outcome tracking for success rate calculation (last 100 episodes)
        self.recent_episodes = deque(maxlen=100)
        
        # Distance tracking for reward shaping
        self.previous_distance = torch.zeros(self.sim_env.num_envs, device=self.device)
        self.initial_distance_xy = torch.zeros(self.sim_env.num_envs, device=self.device)
        self.previous_distance_xy = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        # Velocity tracking for acceleration penalty
        self.previous_velocity = torch.zeros(self.sim_env.num_envs, 3, device=self.device)
        
        # Hover time counter for cumulative hover reward (NEW)
        self.hover_time_counter = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        self._init_noise_buffers()
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
        self._resolve_drop_obstacle_asset_index()
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
        self.window_raw_score_hist = self._init_score_histogram()
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
        
        
        self.reset_idx(torch.arange(self.sim_env.num_envs, device=self.device))

    @staticmethod
    def _format_score_label(score_value):
        score_f = float(score_value)
        if score_f.is_integer():
            return str(int(score_f))
        return f"{score_f:g}"

    @staticmethod
    def _score_label_to_tag(score_label):
        return str(score_label).replace("-", "neg_").replace(".", "p")

    def _get_piecewise_scores(self):
        scores = list(
            getattr(
                self.drop_reward_config,
                "piecewise_scores",
                [20.0, 16.0, 13.0, 10.0, 8.0, 6.0, 3.0, 1.0, 0.0],
            )
        )
        if len(scores) == 0:
            scores = [20.0, 16.0, 13.0, 10.0, 8.0, 6.0, 3.0, 1.0, 0.0]
        return [float(v) for v in scores]

    def _init_score_histogram(self):
        return {
            self._format_score_label(score): 0
            for score in self._get_piecewise_scores()
        }

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

    def _resample_drop_obstacles(self, env_ids):
        """Sample one optional obstacle per env around the target for DROP scoring."""
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        self.drop_obstacle_exists[env_ids] = False
        self.drop_obstacle_position[env_ids] = self.drop_obstacle_absent_obs_value
        self.drop_obstacle_radius[env_ids] = 0.0

        if not self.drop_obstacle_enable:
            return

        spawn_prob = float(min(max(self.drop_obstacle_spawn_prob, 0.0), 1.0))
        spawn_mask = torch.rand((env_ids.shape[0],), device=self.device) < spawn_prob
        if not spawn_mask.any():
            return

        active_env_ids = env_ids[spawn_mask]
        n_active = active_env_ids.shape[0]
        if n_active == 0:
            return

        center_r_min = max(
            0.0, min(self.drop_obstacle_center_radius_min, self.drop_obstacle_center_radius_max)
        )
        center_r_max = max(
            center_r_min,
            max(self.drop_obstacle_center_radius_min, self.drop_obstacle_center_radius_max),
        )
        if abs(center_r_max - center_r_min) < 1e-9:
            radial_dist = torch.full((n_active,), center_r_min, device=self.device)
        else:
            # Uniform-in-area sampling on annulus [r_min, r_max].
            rand_u = torch.rand((n_active,), device=self.device)
            radial_dist = torch.sqrt(
                rand_u * (center_r_max * center_r_max - center_r_min * center_r_min)
                + center_r_min * center_r_min
            )
        angles = torch.rand((n_active,), device=self.device) * (2.0 * torch.pi)
        offsets_xy = torch.stack(
            [radial_dist * torch.cos(angles), radial_dist * torch.sin(angles)], dim=1
        )
        obstacle_xy = self.target_position[active_env_ids, 0:2] + offsets_xy
        lower_xy = self.env_bounds_min[0:2].view(1, 2)
        upper_xy = self.env_bounds_max[0:2].view(1, 2)
        obstacle_xy = torch.max(torch.min(obstacle_xy, upper_xy), lower_xy)

        obstacle_r_min = max(
            0.0, min(self.drop_obstacle_radius_min, self.drop_obstacle_radius_max)
        )
        obstacle_r_max = max(
            obstacle_r_min,
            max(self.drop_obstacle_radius_min, self.drop_obstacle_radius_max),
        )
        if abs(obstacle_r_max - obstacle_r_min) < 1e-9:
            obstacle_radius = torch.full((n_active,), obstacle_r_min, device=self.device)
        else:
            obstacle_radius = torch.empty((n_active,), device=self.device)
            obstacle_radius.uniform_(obstacle_r_min, obstacle_r_max)

        obstacle_z = torch.full(
            (n_active,),
            0.5 * max(self.drop_obstacle_height, 0.0),
            device=self.device,
            dtype=torch.float32,
        )
        self.drop_obstacle_position[active_env_ids, 0:2] = obstacle_xy
        self.drop_obstacle_position[active_env_ids, 2] = obstacle_z
        self.drop_obstacle_radius[active_env_ids] = obstacle_radius
        self.drop_obstacle_exists[active_env_ids] = True

    def _resolve_drop_obstacle_asset_index(self):
        """Resolve the sim asset slot used for the instantiated DROP obstacle."""
        self.drop_obstacle_asset_index = -1
        self.drop_obstacle_instance_enabled = False

        if "obstacle_position" not in self.obs_dict:
            logger.warning("DROP obstacle instancing disabled: obstacle tensors not found.")
            return

        global_asset_dicts = getattr(self.sim_env, "global_asset_dicts", None)
        if not global_asset_dicts or len(global_asset_dicts) == 0:
            logger.warning("DROP obstacle instancing disabled: global_asset_dicts unavailable.")
            return
        env0_assets = global_asset_dicts[0]
        if env0_assets is None or len(env0_assets) == 0:
            logger.warning("DROP obstacle instancing disabled: no env assets loaded.")
            return

        target_file = os.path.basename(self.drop_obstacle_asset_file)
        resolved_index = -1
        for idx, asset_info in enumerate(env0_assets):
            filename = os.path.basename(str(asset_info.get("filename", "")))
            if filename != target_file:
                continue
            # Prefer explicit object assets if duplicated filenames exist.
            asset_type = str(asset_info.get("asset_type", ""))
            if asset_type == "objects":
                resolved_index = idx
                break
            if resolved_index < 0:
                resolved_index = idx

        if resolved_index < 0:
            logger.warning(
                "DROP obstacle instancing disabled: asset '%s' not found in env assets.",
                target_file,
            )
            return

        num_assets_tensor = int(self.obs_dict["obstacle_position"].shape[1])
        if resolved_index >= num_assets_tensor:
            logger.warning(
                "DROP obstacle instancing disabled: resolved index %d out of tensor range %d.",
                resolved_index,
                num_assets_tensor,
            )
            return

        self.drop_obstacle_asset_index = resolved_index
        self.drop_obstacle_instance_enabled = True
        logger.info(
            "DROP obstacle actor slot resolved: file=%s, slot=%d",
            target_file,
            self.drop_obstacle_asset_index,
        )

    def _sync_drop_obstacle_instances(self, env_ids):
        """Apply sampled DROP obstacle states to the instantiated sim obstacle actor."""
        if env_ids.numel() == 0:
            return
        if not self.drop_obstacle_instance_enabled:
            return
        if "obstacle_position" not in self.obs_dict:
            return

        idx = int(self.drop_obstacle_asset_index)
        num_assets = int(self.obs_dict["obstacle_position"].shape[1])
        if idx < 0 or idx >= num_assets:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        id_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)

        # Default: hide obstacle actor for these envs.
        self.obs_dict["obstacle_position"][env_ids, idx, 0:3] = -1000.0
        self.obs_dict["obstacle_orientation"][env_ids, idx, 0:4] = id_quat.view(1, 4)
        self.obs_dict["obstacle_linvel"][env_ids, idx, :].zero_()
        self.obs_dict["obstacle_angvel"][env_ids, idx, :].zero_()

        active_mask = self.drop_obstacle_exists[env_ids]
        if bool(active_mask.any().item()):
            active_env_ids = env_ids[active_mask]
            self.obs_dict["obstacle_position"][active_env_ids, idx, 0:3] = (
                self.drop_obstacle_position[active_env_ids]
            )
            self.obs_dict["obstacle_orientation"][active_env_ids, idx, 0:4] = id_quat.view(1, 4)
            self.obs_dict["obstacle_linvel"][active_env_ids, idx, :].zero_()
            self.obs_dict["obstacle_angvel"][active_env_ids, idx, :].zero_()

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

    def _init_noise_buffers(self):
        num_envs = self.sim_env.num_envs
        num_sources = self.num_noise_sources
        
        # Initialize previous state buffers for reward calculation
        self.previous_position = torch.zeros(
            (num_envs, 3), device=self.device, requires_grad=False
        )
        self.previous_actions = torch.zeros(
            (num_envs, 4), device=self.device, requires_grad=False
        )

        self.noise_centers = torch.zeros(
            (num_envs, num_sources, 3), device=self.device, requires_grad=False
        )
        self.noise_sigmas = torch.zeros(
            (num_envs, num_sources, 3), device=self.device, requires_grad=False
        )
        self.noise_weights = torch.zeros(
            (num_envs, num_sources), device=self.device, requires_grad=False
        )
        self.position_noise = torch.zeros((num_envs, 3), device=self.device, requires_grad=False)

        self.noise_sigma_min = torch.tensor(
            self.noise_config.sigma_min, device=self.device, requires_grad=False
        )
        self.noise_sigma_max = torch.tensor(
            self.noise_config.sigma_max, device=self.device, requires_grad=False
        )

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
            "noise_centers",
            "noise_sigmas",
            "noise_weights",
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

        noise_centers = torch.tensor(
            preset["noise_centers"], device=self.device, dtype=torch.float32
        )
        noise_sigmas = torch.tensor(
            preset["noise_sigmas"], device=self.device, dtype=torch.float32
        )
        noise_weights = torch.tensor(
            preset["noise_weights"], device=self.device, dtype=torch.float32
        )

        expected_sources = int(self.num_noise_sources)
        if noise_centers.shape != (expected_sources, 3):
            raise ValueError("Fixed noise_centers must be shaped as (num_sources, 3).")
        if noise_sigmas.shape != (expected_sources, 3):
            raise ValueError("Fixed noise_sigmas must be shaped as (num_sources, 3).")
        if noise_weights.shape != (expected_sources,):
            raise ValueError("Fixed noise_weights must be shaped as (num_sources,).")

        weight_sum = float(noise_weights.sum().item())
        if weight_sum <= 0.0:
            raise ValueError("Fixed noise_weights must sum to a positive value.")
        noise_weights = noise_weights / weight_sum

        self.fixed_target_position = target_position
        self.fixed_obstacle_positions = obstacle_positions
        self.fixed_noise_centers = noise_centers
        self.fixed_noise_sigmas = noise_sigmas
        self.fixed_noise_weights = noise_weights
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

    def _apply_fixed_noise(self, env_ids):
        num_envs = env_ids.shape[0]
        centers = self.fixed_noise_centers.view(1, -1, 3).expand(num_envs, -1, -1)
        sigmas = self.fixed_noise_sigmas.view(1, -1, 3).expand(num_envs, -1, -1)
        weights = self.fixed_noise_weights.view(1, -1).expand(num_envs, -1)
        self.noise_centers[env_ids] = centers
        self.noise_sigmas[env_ids] = sigmas
        self.noise_weights[env_ids] = weights

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

    def _resample_noise_sources(self, env_ids):
        if self.num_noise_sources <= 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        num_envs = env_ids.shape[0]
        num_sources = self.num_noise_sources

        # Stratified Noise Sampling Strategy
        # Group 1: Global roaming sources
        #   - Randomly distributed across the entire environment.
        # Group 2: Local target-near sources
        #   - Spawned within 1.0m to 3.0m radius of the target.
        # NOTE: No source is placed exactly at the target point.
        
        # --- Group 1: Global sources ---
        num_global = min(3, num_sources)
        bounds_min_global = self.env_bounds_min.view(1, 1, 3).expand(num_envs, num_global, 3)
        bounds_max_global = self.env_bounds_max.view(1, 1, 3).expand(num_envs, num_global, 3)
        centers_global = torch_rand_float_tensor(bounds_min_global, bounds_max_global)

        # --- Group 2: Local target-near sources ---
        num_local = num_sources - num_global
        if num_local > 0:
            # Generate random directions
            random_dirs = torch.randn((num_envs, num_local, 3), device=self.device)
            random_dirs = torch.nn.functional.normalize(random_dirs, dim=2)
            
            # Generate random distances between 1.0m and 3.0m
            # dist = min + rand * (max - min)
            local_dist_min = 1.0
            local_dist_max = 3.0
            random_dists = torch.rand((num_envs, num_local, 1), device=self.device) * (local_dist_max - local_dist_min) + local_dist_min
            
            # Calculate offsets: direction * distance
            offsets = random_dirs * random_dists
            
            # Add to target position (need to reshape target to broadcast)
            # self.target_position: (num_envs, 3) -> (num_envs, 1, 3)
            # CRITICAL FIX: Must index target_position with env_ids to match current batch!
            centers_local = self.target_position[env_ids].unsqueeze(1) + offsets
            
            # Clamp to environment bounds just in case target is near wall
            centers_local = torch.max(centers_local, self.env_bounds_min.view(1, 1, 3))
            centers_local = torch.min(centers_local, self.env_bounds_max.view(1, 1, 3))
            
            # Concatenate all groups: global + local
            centers = torch.cat([centers_global, centers_local], dim=1)
        else:
            centers = centers_global

        sigma_min = self.noise_sigma_min.view(1, 1, 3).expand(num_envs, num_sources, 3)
        sigma_max = self.noise_sigma_max.view(1, 1, 3).expand(num_envs, num_sources, 3)
        sigmas = torch_rand_float_tensor(sigma_min, sigma_max)

        weight_min = float(self.noise_config.weight_min)
        weight_max = float(self.noise_config.weight_max)
        weights = torch.rand((num_envs, num_sources), device=self.device) * (
            weight_max - weight_min
        ) + weight_min
        weights = weights / weights.sum(dim=1, keepdim=True)

        self.noise_centers[env_ids] = centers
        self.noise_sigmas[env_ids] = sigmas
        self.noise_weights[env_ids] = weights

    def _estimate_noise_range(self, env_ids):
        """
        Monte Carlo estimation of min and max noise intensity in the environment.
        Used for normalizing noise intensity in the Unified Cost Function.
        """
        if not self.noise_config.enable_noise or self.num_noise_sources <= 0:
            self.estimated_n_min[env_ids] = 0.0
            self.estimated_n_max[env_ids] = 1.0
            return

        num_samples = int(self.reward_params.get("n_min_max_sample_size", 1000))
        num_envs_reset = env_ids.shape[0]
        
        # 1. Sample random positions inside environment bounds
        # shape: (num_envs_reset, num_samples, 3)
        sample_positions = torch_rand_float_tensor(
            self.env_bounds_min.view(1, 1, 3).expand(num_envs_reset, num_samples, 3),
            self.env_bounds_max.view(1, 1, 3).expand(num_envs_reset, num_samples, 3)
        )
        
        # 2. Compute noise intensity for all samples
        # Expand noise params to match samples
        # centers: (num_envs_reset, 1, num_sources, 3)
        centers = self.noise_centers[env_ids].unsqueeze(1)
        sigmas = self.noise_sigmas[env_ids].unsqueeze(1)
        weights = self.noise_weights[env_ids].unsqueeze(1)
        
        # sample_positions: (num_envs_reset, num_samples, 1, 3)
        pos = sample_positions.unsqueeze(2)
        
        # Vectorized GMM computation
        deltas = pos - centers
        scaled = (deltas / sigmas).pow(2).sum(dim=-1)
        mixture = torch.exp(-0.5 * scaled)
        intensity = (weights * mixture).sum(dim=2) # (num_envs_reset, num_samples)
        
        # 3. Find min and max for each environment
        n_min, _ = intensity.min(dim=1)
        n_max, _ = intensity.max(dim=1)
        
        # Avoid division by zero if flat
        n_max = torch.max(n_max, n_min + 1e-6)
        
        self.estimated_n_min[env_ids] = n_min
        self.estimated_n_max[env_ids] = n_max

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

    def _compute_position_noise(self):
        # DEBUG: Print noise status every 50 steps
        # if hasattr(self, 'num_task_steps') and self.num_task_steps % 50 == 0:
        #     print(f"[NOISE DEBUG] enable_noise={self.noise_config.enable_noise}, num_sources={self.num_noise_sources}")
        
        if not self.noise_config.enable_noise or self.num_noise_sources <= 0:
            self.position_noise.zero_()
            # if hasattr(self, 'num_task_steps') and self.num_task_steps % 50 == 0:
            #     print(f"[NOISE DEBUG] Position noise is DISABLED (returning zeros)")
            return self.position_noise

        position = self.obs_dict["robot_position"]
        deltas = position.unsqueeze(1) - self.noise_centers
        scaled = (deltas / self.noise_sigmas).pow(2).sum(dim=-1)
        mixture = torch.exp(-0.5 * scaled)
        mixture = (self.noise_weights * mixture).sum(dim=1)

        noise = torch.randn_like(position) * mixture.unsqueeze(1) * self.noise_config.noise_scale
        self.position_noise[:] = noise
        
        # if hasattr(self, 'num_task_steps') and self.num_task_steps % 50 == 0:
        #     print(f"[NOISE DEBUG] Position noise magnitude: {torch.norm(noise[0]).item():.4f}")
        
        return self.position_noise
    
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
        This local gust update is independent from GMM source count so that
        time-varying gusts can remain enabled even when spatial GMM sources
        are disabled (num_noise_sources == 0).
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

    def _compute_gmm_mixture(self, position, env_ids=None):
        """
        Compute GMM mixture intensity at given position.
        
        Args:
            position: (num_envs or subset, 3) - positions to evaluate
            env_ids: (optional) indices of environments to compute for. 
                     If provided, noise params are sliced.
            
        Returns:
            mixture_intensity: (num_envs or subset,) - GMM intensity
        """
        # position: (N, 3)
        # noise_centers: (num_envs, num_sources, 3)
        
        if env_ids is not None:
            centers = self.noise_centers[env_ids]
            sigmas = self.noise_sigmas[env_ids]
            weights = self.noise_weights[env_ids]
        else:
            centers = self.noise_centers
            sigmas = self.noise_sigmas
            weights = self.noise_weights
        
        deltas = position.unsqueeze(1) - centers  # (N, num_sources, 3)
        scaled = (deltas / sigmas).pow(2).sum(dim=-1)  # (N, num_sources)
        mixture = torch.exp(-0.5 * scaled)  # Gaussian PDF
        weighted_mixture = (weights * mixture).sum(dim=1)  # (N,)
        
        return weighted_mixture

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
        elif self.num_noise_sources > 0:
            mixture_intensity = self._compute_gmm_mixture(position, env_ids=env_ids)
            if env_ids is not None:
                weight_sum = self.noise_weights[env_ids].sum(dim=1).clamp_min(1e-6)
                directions = self.gmm_force_direction[env_ids]
            else:
                weight_sum = self.noise_weights.sum(dim=1).clamp_min(1e-6)
                directions = self.gmm_force_direction

            if getattr(self.gmm_force_config, "normalize_intensity", True):
                intensity = torch.clamp(mixture_intensity / weight_sum, 0.0, 1.0)
            else:
                intensity = torch.clamp(mixture_intensity, min=0.0)

            if env_ids is not None:
                local_v_max = self.local_wind_max_speed[env_ids]
            else:
                local_v_max = self.local_wind_max_speed
            local_wind = intensity.unsqueeze(1) * local_v_max.unsqueeze(1) * directions
        else:
            # Spatial GMM disabled: keep a time-varying but spatially uniform local gust.
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

    def _predict_child_landing_xy(self, env_ids, init_pos=None, init_vel=None):
        """Predict child landing position/distance without mutating DROP buffers."""
        if env_ids.numel() == 0:
            empty_pos = torch.empty((0, 3), device=self.device)
            empty_dist = torch.empty((0,), device=self.device)
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
                active[landed_local] = False

            flying_now = ~landed_now
            if flying_now.any():
                child_pos[local_ids[flying_now]] = next_pos[flying_now]

        # Fallback in case max_steps is reached before touching ground.
        if active.any():
            remain_local = active.nonzero(as_tuple=False).squeeze(-1)
            landing_pos[remain_local, 0:2] = child_pos[remain_local, 0:2]
            landing_pos[remain_local, 2] = torch.clamp(child_pos[remain_local, 2], min=0.0)

        landing_dist = torch.norm(
            landing_pos[:, 0:2] - self.target_position[env_ids, 0:2], dim=1
        )
        return landing_pos, landing_dist

    def _simulate_child_free_fall(self, env_ids, init_pos=None, init_vel=None):
        """Simulate child payload free fall with a = g + (c_child/m_child)*(v_w - v_child)."""
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        landing_pos, landing_dist = self._predict_child_landing_xy(
            env_ids=env_ids,
            init_pos=init_pos,
            init_vel=init_vel,
        )
        self.child_landing_position[env_ids] = landing_pos
        self.child_landing_xy_distance[env_ids] = landing_dist




    def _compute_reward_and_scores(self):
        """
        Drop-decision reward (direct replacement):
        - WAIT: reward = R_progress
                where R_progress = direction_reward_weight * (d_prev_xy - d_curr_xy)
                (only before DROP, positive when getting closer to target in XY)
        - DROP: reward = R_score + R_drop
                where:
                  R_score = score_reward_weight * score_max * exp(-(landing_error_xy/score_d0)^score_p)
                  R_drop = -impulse_penalty_weight * impulse_metric
                           + posture_reward_weight * exp(-(theta_drop/attitude_theta0)^2)
                           + angvel_reward_weight * exp(-(omega_drop_xy/attitude_theta0)^2)
        where impulse_metric = alpha * Delta_v + beta * Delta_omega.
        """
        cfg = self.drop_reward_config
        altitude_tolerance = float(getattr(cfg, "altitude_tolerance", 0.5))
        altitude_low_penalty_w = float(getattr(cfg, "altitude_low_penalty_weight", 1.0))
        direction_w = float(getattr(cfg, "direction_reward_weight", 0.02))
        direction_min_target_dist = float(getattr(cfg, "direction_min_target_dist", 0.1))
        score_reward_w = float(
            getattr(
                cfg,
                "score_reward_weight",
                float(getattr(cfg, "landing_reward_weight", 1.0))
                * float(getattr(cfg, "piecewise_score_scale", 0.1)),
            )
        )
        impulse_lambda = float(getattr(cfg, "impulse_penalty_weight", 0.1))
        attitude_theta0 = float(getattr(cfg, "attitude_theta0", 0.12))
        attitude_w = float(getattr(cfg, "attitude_reward_weight", 0.1))
        score_max = float(getattr(cfg, "score_max", 20.0))
        score_d0 = float(getattr(cfg, "score_d0", 3.6))
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

        # Pre-DROP distance-progress shaping (XY plane):
        # R_progress = w_dir * (d_prev_xy - d_curr_xy)
        direction_reward = torch.zeros_like(reward)
        if direction_w > 0.0:
            current_dist_xy = torch.norm(
                self.target_position[:, 0:2] - self.obs_dict["robot_position"][:, 0:2], dim=1
            )
            progress_delta = self.previous_distance_xy - current_dist_xy
            # I_far gate is disabled: keep direction-progress reward active for all pre-DROP steps.
            valid_mask = wait_mask
            direction_reward[valid_mask] = direction_w * progress_delta[valid_mask]

        reward = reward + direction_reward

        # Score term is kept independent from R_drop_core by design.
        score_reward = torch.zeros_like(reward)
        drop_core_reward = torch.zeros_like(reward)
        impulse_penalty = torch.zeros_like(reward)
        attitude_reward = torch.zeros_like(reward)
        outside_region_penalty_reward = torch.zeros_like(reward)
        if drop_ids.numel() > 0:
            landing_error_xy = self.step_landing_error_xy[drop_ids]
            impulse_metric = self.step_impulse_metric[drop_ids]
            obstacle_hit_mask = torch.zeros_like(landing_error_xy, dtype=torch.bool)
            if self.drop_obstacle_enable:
                has_obstacle = self.drop_obstacle_exists[drop_ids]
                if has_obstacle.any():
                    landing_xy = self.child_landing_position[drop_ids, 0:2]
                    obstacle_xy = self.drop_obstacle_position[drop_ids, 0:2]
                    obstacle_radius = self.drop_obstacle_radius[drop_ids]
                    dist_to_obstacle_xy = torch.norm(landing_xy - obstacle_xy, dim=1)
                    obstacle_hit_mask = has_obstacle & (dist_to_obstacle_xy <= obstacle_radius)
            self.step_drop_hit_obstacle[drop_ids] = obstacle_hit_mask
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
            theta_drop = self.step_drop_attitude_theta[drop_ids]
            omega_drop_xy = self.step_drop_heading_error[drop_ids]
            theta_scale = max(attitude_theta0, 1e-6)
            # Only grant attitude bonus when drop accuracy score is positive.
            posture_bonus = attitude_w * torch.exp(-torch.square(theta_drop / theta_scale))
            # Use the same weight/scale as posture bonus for DROP-pre angular-rate stability.
            angvel_bonus = attitude_w * torch.exp(-torch.square(omega_drop_xy / theta_scale))
            attitude_bonus = posture_bonus + angvel_bonus
            attitude_bonus = torch.where(raw_score > 0.0, attitude_bonus, torch.zeros_like(attitude_bonus))
            score_reward[drop_ids] = score_term
            impulse_penalty[drop_ids] = -impulse_lambda * impulse_metric
            attitude_reward[drop_ids] = attitude_bonus
            drop_core_reward[drop_ids] = impulse_penalty[drop_ids] + attitude_reward[drop_ids]
            reward[drop_ids] = (
                score_reward[drop_ids]
                + drop_core_reward[drop_ids]
            )
            outside_mask = torch.zeros_like(landing_error_xy, dtype=torch.bool)
            if len(scaled_thresholds) > 0:
                outside_mask = landing_error_xy > scaled_thresholds[-1]
            penalty_mask = outside_mask | obstacle_hit_mask
            if outside_region_penalty > 0.0 and penalty_mask.any():
                penalty_ids = drop_ids[penalty_mask]
                outside_region_penalty_reward[penalty_ids] = -outside_region_penalty
                score_reward[penalty_ids] = 0.0
                impulse_penalty[penalty_ids] = 0.0
                attitude_reward[penalty_ids] = 0.0
                drop_core_reward[penalty_ids] = 0.0
                reward[penalty_ids] = outside_region_penalty_reward[penalty_ids]

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

        # Keep legacy buffers/outputs alive for downstream logging code compatibility.
        self.current_J = torch.zeros_like(reward)
        self.previous_potential = torch.zeros_like(reward)
        self.previous_position = self.obs_dict["robot_position"].clone()
        self.previous_distance = torch.norm(
            self.target_position - self.obs_dict["robot_position"], dim=1
        )
        self.previous_distance_xy = torch.norm(
            self.target_position[:, 0:2] - self.obs_dict["robot_position"][:, 0:2], dim=1
        )
        self.previous_velocity = self.obs_dict["robot_linvel"].clone()
        self.previous_actions = self.actions.clone()

        improvement_reward = (
            score_reward
            + impulse_penalty
            + attitude_reward
            + outside_region_penalty_reward
        )
        noise_reduction_reward = torch.zeros_like(reward)
        noise_intensity = self._compute_gmm_mixture(self.obs_dict["robot_position"])
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
            noise_intensity,
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

    def _populate_extras(self, improvement_reward, direction_reward, noise_reduction_reward, noise_intensity):
        """Simplified extras logging for unified reward function"""
        # Calculate diagnostic metrics
        crash_rate_instant = self.obs_dict["crashes"].float().mean().item()
        avg_z_pos = self.obs_dict["robot_position"][:, 2].mean().item()
        drop_count = int(self.extras.get("drop_count", 0))
        child_dist_to_goal_xy = float(self.extras.get("child_landing_xy_distance_mean", 0.0))
        release_to_target_xy_step = float(self.extras.get("release_to_target_xy_mean", 0.0))
        release_to_target_xy_step_min = float(self.extras.get("release_to_target_xy_min", 0.0))
        release_to_target_xy_step_max = float(self.extras.get("release_to_target_xy_max", 0.0))
        
        # Calculate hover metrics
        avg_hover_time = self.hover_time_counter.mean().item()
        
        # Calculate success rate from completed episodes
        if len(self.recent_episodes) > 0:
            total_episodes = len(self.recent_episodes)
            success_count = self.recent_episodes.count('success')
            crash_count = self.recent_episodes.count('crash')
            timeout_count = self.recent_episodes.count('timeout')
            
            success_rate = success_count / total_episodes
            crash_rate_episodes = crash_count / total_episodes
            timeout_rate = timeout_count / total_episodes
        else:
            success_rate = 0.0
            crash_rate_episodes = 0.0
            timeout_rate = 0.0

        # Drop-only window diagnostics (aggregated since last console print).
        drop_samples_window = int(getattr(self, "window_drop_sample_count", 0))
        if drop_samples_window > 0:
            landing_error_xy_mean_drop_window = (
                float(self.window_landing_error_xy_sum) / drop_samples_window
            )
            release_to_target_xy_mean_drop_window = (
                float(self.window_release_to_target_xy_sum) / drop_samples_window
            )
            release_to_target_xy_min_drop_window = float(self.window_release_to_target_xy_min)
            release_to_target_xy_max_drop_window = float(self.window_release_to_target_xy_max)
            drop_roll_deg_mean_window = float(self.window_drop_roll_deg_sum) / drop_samples_window
            drop_roll_deg_min_window = float(self.window_drop_roll_deg_min)
            drop_roll_deg_max_window = float(self.window_drop_roll_deg_max)
            drop_pitch_deg_mean_window = float(self.window_drop_pitch_deg_sum) / drop_samples_window
            drop_pitch_deg_min_window = float(self.window_drop_pitch_deg_min)
            drop_pitch_deg_max_window = float(self.window_drop_pitch_deg_max)
            attitude_total_deg_mean_window = (
                float(self.window_drop_attitude_total_deg_sum) / drop_samples_window
            )
            attitude_total_deg_var_window = max(
                float(self.window_drop_attitude_total_deg_sq_sum) / drop_samples_window
                - attitude_total_deg_mean_window * attitude_total_deg_mean_window,
                0.0,
            )
            attitude_total_deg_std_window = float(np.sqrt(attitude_total_deg_var_window))
            impulse_metric_mean_window = float(self.window_impulse_metric_sum) / drop_samples_window
            impulse_metric_var_window = max(
                float(self.window_impulse_metric_sq_sum) / drop_samples_window
                - impulse_metric_mean_window * impulse_metric_mean_window,
                0.0,
            )
            impulse_metric_std_window = float(np.sqrt(impulse_metric_var_window))
        else:
            landing_error_xy_mean_drop_window = 0.0
            release_to_target_xy_mean_drop_window = 0.0
            release_to_target_xy_min_drop_window = 0.0
            release_to_target_xy_max_drop_window = 0.0
            drop_roll_deg_mean_window = 0.0
            drop_roll_deg_min_window = 0.0
            drop_roll_deg_max_window = 0.0
            drop_pitch_deg_mean_window = 0.0
            drop_pitch_deg_min_window = 0.0
            drop_pitch_deg_max_window = 0.0
            attitude_total_deg_mean_window = 0.0
            attitude_total_deg_std_window = 0.0
            impulse_metric_mean_window = 0.0
            impulse_metric_std_window = 0.0
        landing_error_xy_ema = (
            float(self.landing_error_xy_ema) if self.landing_error_xy_ema is not None else 0.0
        )
        attitude_total_deg_ema = (
            float(self.attitude_total_deg_ema) if self.attitude_total_deg_ema is not None else 0.0
        )
        
        extras = {
            "improvement_reward": float(improvement_reward.mean().item()),
            "direction_reward": float(direction_reward.mean().item()),
            "noise_reduction_reward": float(noise_reduction_reward.mean().item()),
            "noise_intensity": float(noise_intensity.mean().item()),
            "hover_time": avg_hover_time,
            "total_reward": float(self.rewards.mean().item()),
            "episode_length": float(self.task_config.episode_len_steps),
            # Episode-based metrics
            "metrics/crash_rate": crash_rate_episodes,
            "metrics/timeout_rate": timeout_rate,
            # Diagnostic metrics for TensorBoard
            "info/crash_rate_instant": crash_rate_instant,
            "info/avg_z_position": avg_z_pos,
            # Drop-quality diagnostics (window aggregated)
            "performance/landing_error_xy_mean_drop_window": landing_error_xy_mean_drop_window,
            "performance/release_to_target_xy_mean_drop_window": release_to_target_xy_mean_drop_window,
            "performance/landing_error_xy_ema": landing_error_xy_ema,
            "performance/impulse_metric_mean": impulse_metric_mean_window,
            "performance/impulse_metric_std": impulse_metric_std_window,
            "performance/attitude_total_deg_ema": attitude_total_deg_ema,
            "performance/attitude_total_deg_std_drop_window": attitude_total_deg_std_window,
            "curriculum/spawn_z_min": float(self.spawn_z_curriculum_min_current),
            "curriculum/spawn_z_alpha": float(self.spawn_z_curriculum_alpha_current),
        }
        if drop_count > 0:
            extras["performance/release_to_target_xy"] = release_to_target_xy_step
        learning_rate = os.environ.get("AERIAL_GYM_LR")
        if learning_rate is not None:
            extras["learning_rate"] = float(learning_rate)
        
        # Write diagnostics to TensorBoard (with lazy initialization)
        if not self._writer_initialized:
            self._init_tensorboard_writer()
            self._writer_initialized = True
        
        if self.writer is not None:
            self.writer.add_scalar(
                "performance/landing_error_xy_mean_drop_window",
                landing_error_xy_mean_drop_window,
                self.num_task_steps,
            )
            self.writer.add_scalar(
                "performance/release_to_target_xy_mean_drop_window",
                release_to_target_xy_mean_drop_window,
                self.num_task_steps,
            )
            self.writer.add_scalar(
                "performance/landing_error_xy_ema",
                landing_error_xy_ema,
                self.num_task_steps,
            )
            self.writer.add_scalar(
                "performance/impulse_metric_mean",
                impulse_metric_mean_window,
                self.num_task_steps,
            )
            self.writer.add_scalar(
                "performance/impulse_metric_std",
                impulse_metric_std_window,
                self.num_task_steps,
            )
            self.writer.add_scalar(
                "performance/attitude_total_deg_ema",
                attitude_total_deg_ema,
                self.num_task_steps,
            )
            self.writer.add_scalar(
                "performance/attitude_total_deg_std_drop_window",
                attitude_total_deg_std_window,
                self.num_task_steps,
            )
            if drop_count > 0:
                self.writer.add_scalar(
                    "performance/release_to_target_xy", release_to_target_xy_step, self.num_task_steps
                )

        if self.writer is None and not self._writer_initialized:
             self._init_tensorboard_writer()
             self._writer_initialized = True
             
        if self.writer:
            if hasattr(self, "current_raw_signal"):
                # Mean Magnitude (Scalar summary)
                mean_signal = self.current_raw_signal.abs().mean().item()
                self.writer.add_scalar("Performance/mean_abs_signal", mean_signal, self.num_task_steps)
                # print(f"DEBUG: Wrote signal histogram. Mean signal: {mean_signal:.4f}")
        
        # Console stats: print once every 10 epochs.
        curr_epoch = int(getattr(self, "_train_epoch", 0))
        if curr_epoch > 0 and curr_epoch % 10 == 0 and curr_epoch != self._last_console_stats_epoch:
            logger.warning(
                f"[Epoch {curr_epoch}] crash_rate_ep={crash_rate_episodes:.1%}, "
                f"timeout_rate_ep={timeout_rate:.1%}, "
                f"release_to_target_xy(mean/min/max)="
                f"{release_to_target_xy_mean_drop_window:.3f}/"
                f"{release_to_target_xy_min_drop_window:.3f}/"
                f"{release_to_target_xy_max_drop_window:.3f}, "
                f"landing_error_xy_mean={landing_error_xy_mean_drop_window:.3f}, "
                f"landing_error_xy_ema={landing_error_xy_ema:.3f}, "
                f"release_roll_deg(mean/min/max)="
                f"{drop_roll_deg_mean_window:.3f}/"
                f"{drop_roll_deg_min_window:.3f}/"
                f"{drop_roll_deg_max_window:.3f}, "
                f"release_pitch_deg(mean/min/max)="
                f"{drop_pitch_deg_mean_window:.3f}/"
                f"{drop_pitch_deg_min_window:.3f}/"
                f"{drop_pitch_deg_max_window:.3f}, "
                f"impulse_metric(mean/std)="
                f"{impulse_metric_mean_window:.3f}/"
                f"{impulse_metric_std_window:.3f}, "
                f"attitude_total_deg_ema={attitude_total_deg_ema:.3f}, "
                f"attitude_total_deg_std_window={attitude_total_deg_std_window:.3f}"
            )
            self._last_console_stats_epoch = curr_epoch
            self.window_done_total = 0
            self.window_done_non_crash = 0
            self.window_done_speed_sum = 0.0
            self.window_done_hover_reward_count = 0
            self.window_drop_end_count = 0
            self.window_no_drop_done_count = 0
            self.window_drop_sample_count = 0
            self.window_raw_score_hist = self._init_score_histogram()
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

        
        self.infos["extras"] = extras

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
            
        # --- Capture Final J for Terminated Environments (Stats) ---
        # Calculate J for the envs about to be reset.
        try:
            # 1. Get current state for reset envs
            d_pos = self.obs_dict["robot_position"][env_ids]
            
            # 2. Compute current noise directly (obs_dict key may not exist)
            current_noise = self._compute_gmm_mixture(d_pos, env_ids)
            
            # 3. Normalize noise
            if hasattr(self, "estimated_n_min"):
                n_range = self.estimated_n_max[env_ids] - self.estimated_n_min[env_ids] + 1e-6
                n_hat = (current_noise - self.estimated_n_min[env_ids]) / n_range
                n_hat = torch.clamp(n_hat, 0.0, 1.0)
            else:
                n_hat = torch.zeros_like(current_noise)
                
            # 4. Calculate Distance cost
            dist_to_tgt = torch.norm(self.target_position[env_ids] - d_pos, dim=1)
            
            # 5. Compute J = w_d * (d/d0)^2 + w_n * n_hat
            w_d = self.reward_params["potential_w_d"]
            w_n = self.reward_params["potential_w_n"]
            d0 = self.reward_params["potential_d0"]
            
            final_J = w_d * (dist_to_tgt / d0).pow(2) + w_n * n_hat
            
            # 6. Add to buffer (CPU side)
            final_J_vals = final_J.detach().cpu().numpy()
            self.final_J_buffer.extend(final_J_vals)
            
            # --- Capture Min J for Terminated Environments ---
            min_J_vals = self.episode_min_J[env_ids].detach().cpu().numpy()
            
            # Filter out placeholder values (e.g., from initial reset before any steps)
            valid_mask = min_J_vals < 999.0
            if np.any(valid_mask):
                self.min_J_buffer.extend(min_J_vals[valid_mask])
                
            # Reset Min J tracker for these envs
            self.episode_min_J[env_ids] = 1000.0
            
        except Exception as e:
            # Silent fail is safer during training loop than crashing
            pass

        if self.log_step_scores and not self._episode_log_written:
            if (env_ids == self.log_step_env_id).any().item():
                self._episode_step_log = []
                self._episode_step_idx = 0
        self._set_fixed_obstacle_count()
        if self.fixed_env_enabled:
            self._apply_fixed_target(env_ids)
            self._apply_fixed_noise(env_ids)
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
        self._resample_drop_obstacles(env_ids)
        self._sync_drop_obstacle_instances(env_ids)

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
        self.step_drop_hit_obstacle[env_ids] = False
        
        # Reset success counter
        self.success_counter[env_ids] = 0.0
        
        # Reset previous state for reward calculation
        self.previous_position[env_ids] = self.obs_dict["robot_position"][env_ids]
        self.previous_distance[env_ids] = dist_spawn_to_target
        self.previous_distance_xy[env_ids] = dist_spawn_to_target_xy
        self.previous_actions[env_ids] = 0.0
        self.current_cmd_body_for_obs[env_ids] = 0.0
        self.obs_prev_cmd_body[env_ids] = 0.0
        self.obs_prev_body_linvel[env_ids] = self.obs_dict["robot_body_linvel"][env_ids]
        reset_body_linvel = self.obs_dict["robot_body_linvel"][env_ids]
        self.obs_body_linvel_history[env_ids] = reset_body_linvel.unsqueeze(1).repeat(
            1, self.obs_linvel_history_frames, 1
        )

        # Reset hover time counter
        self.hover_time_counter[env_ids] = 0.0
        
        # Reset arrival tracker
        self.has_arrived[env_ids] = False
        
        # ==== Unified Cost Function Initialization ====
        # 1. Resample noise env parameters
        self._resample_noise_sources(env_ids)
        # 1.1 Resample per-episode main wind (Layer 1)
        self._resample_main_wind(env_ids)
        if self.dryden_enabled:
            # 1.15 Dryden mode: sample per-env (sigma, tau), state starts from zero.
            self._resample_dryden_parameters(env_ids)
        else:
            # 1.15 Legacy local-gust mode.
            self._resample_local_wind_speed(env_ids)
            local_horizontal_only = bool(
                getattr(self.gmm_force_config, "local_wind_horizontal_only", True)
            )
            local_dirs = self._sample_unit_directions(
                env_ids.shape[0], horizontal_only=local_horizontal_only
            )
            self.gmm_force_direction[env_ids] = local_dirs
            self.gmm_target_direction[env_ids] = local_dirs
        
        # 2. Estimate noise range for normalization (min/max)
        if not hasattr(self, "estimated_n_min"): # Initialize buffers if missing (first run)
            self.estimated_n_min = torch.zeros(self.sim_env.num_envs, device=self.device)
            self.estimated_n_max = torch.ones(self.sim_env.num_envs, device=self.device)
            self.previous_potential = torch.zeros(self.sim_env.num_envs, device=self.device)
            
            # Scheme B Buffers
            self.J_best_in_goal = torch.full((self.sim_env.num_envs,), float('inf'), device=self.device)
            self.J_ema = torch.zeros(self.sim_env.num_envs, device=self.device)
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

        # Reset Scheme B buffers
        self.J_best_in_goal[env_ids] = float('inf')
        self.hold_counter[env_ids] = 0.0
        self.max_hold_counter[env_ids] = 0.0
        if hasattr(self, "hover_good_min"):
            self.hover_good_min[env_ids] = float("inf")
            self.hover_good_max[env_ids] = -float("inf")
            
        self._estimate_noise_range(env_ids)
        
        # 3. Compute initial potential J_0
        initial_noise_level = self._compute_gmm_mixture(self.previous_position[env_ids], env_ids)
        
        # Normalize noise: n_hat = (n - min) / (max - min)
        n_hat = (initial_noise_level - self.estimated_n_min[env_ids]) / (
            self.estimated_n_max[env_ids] - self.estimated_n_min[env_ids] + 1e-6
        )
        n_hat = torch.clamp(n_hat, 0.0, 1.0)
        
        # Distance component
        d = self.previous_distance[env_ids]
        d0 = self.reward_params["potential_d0"]
        w_d = self.reward_params["potential_w_d"]
        w_n = self.reward_params["potential_w_n"]
        
        # J = w_d * (d/d0)^2 + w_n * n_hat
        J_0 = w_d * (d / d0).pow(2) + w_n * n_hat
        self.previous_potential[env_ids] = J_0
        if hasattr(self, "J_ema"):
            self.J_ema[env_ids] = J_0
        
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

    def process_obs_for_task(self):
        target_relative_position = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        self.task_obs["observations"][:, 0:3] = target_relative_position

        self.task_obs["observations"][:, 3:6] = self.obs_dict["robot_body_linvel"]
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        self.task_obs["observations"][:, 6] = euler_angles[:, 0]
        self.task_obs["observations"][:, 7] = euler_angles[:, 1]
        self.task_obs["observations"][:, 8:11] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 11] = self.obs_dict["robot_position"][:, 2]
        self.task_obs["observations"][:, 12:15] = self.drop_obstacle_position
        if self.use_wind_estimation_features:
            dt = float(self.obs_dict["dt"]) if "dt" in self.obs_dict else 0.01
            dt = max(dt, 1e-4)
            body_linvel = self.obs_dict["robot_body_linvel"]
            delta_v = (body_linvel - self.obs_prev_body_linvel) / dt
            self.task_obs["observations"][
                :, self.wind_aug_prev_cmd_start : self.wind_aug_prev_cmd_end
            ] = self.obs_prev_cmd_body
            self.task_obs["observations"][
                :, self.wind_aug_delta_v_start : self.wind_aug_delta_v_end
            ] = delta_v

            if self.obs_linvel_history_frames > 1:
                linvel_stack = torch.cat(
                    (self.obs_body_linvel_history[:, 1:, :], body_linvel.unsqueeze(1)),
                    dim=1,
                )
            else:
                linvel_stack = body_linvel.unsqueeze(1)
            linvel_stack_flat = linvel_stack.reshape(self.sim_env.num_envs, -1)
            self.task_obs["observations"][
                :, self.wind_aug_linvel_hist_start : self.wind_aug_linvel_hist_end
            ] = linvel_stack_flat

    def _update_obs_history_buffers(self):
        if not self.use_wind_estimation_features:
            return
        self.obs_prev_cmd_body[:] = self.current_cmd_body_for_obs
        self.obs_prev_body_linvel[:] = self.obs_dict["robot_body_linvel"]
        self.obs_body_linvel_history[:] = torch.roll(
            self.obs_body_linvel_history, shifts=-1, dims=1
        )
        self.obs_body_linvel_history[:, -1, :] = self.obs_dict["robot_body_linvel"]

    def get_return_tuple(self):
        self.process_obs_for_task()
        self._update_obs_history_buffers()
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
        self.step_drop_hit_obstacle[:] = False

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
            drop_event_mask = requested_drop_mask
        else:
            drop_event_mask = requested_drop_mask & (~self.child_has_dropped)
        self.child_drop_triggered[:] = drop_event_mask

        if bool(getattr(self.drop_model_config, "enable_drop_model", True)) and drop_event_mask.any():
            dropped_env_ids = drop_event_mask.nonzero(as_tuple=False).squeeze(-1)
            child_release_pos = self.obs_dict["robot_position"][dropped_env_ids].clone()
            child_release_vel = self.obs_dict["robot_linvel"][dropped_env_ids].clone()
            add_child_eject_velocity = bool(
                getattr(self.drop_impact_config, "add_child_eject_velocity", True)
            )
            if add_child_eject_velocity:
                child_release_vel += self._compute_eject_velocity_world(dropped_env_ids)
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
        self.current_cmd_body_for_obs[:, 0:4] = transformed_action[:, 0:4]
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
            noise_intensity,
            safety_reward,
            action_smoothness_penalty,
            hover_reward,
            threshold_reward,
            anchor_reward,
        ) = self._compute_reward_and_scores()
        if dropped_env_ids is not None:
            self.extras["drop_hit_obstacle_count"] = int(
                self.step_drop_hit_obstacle[dropped_env_ids].sum().item()
            )
        else:
            self.extras["drop_hit_obstacle_count"] = 0

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
        if crash_no_drop_penalty > 0.0 and no_drop_crash_mask.any():
            self.rewards[no_drop_crash_mask] = (
                self.rewards[no_drop_crash_mask] - crash_no_drop_penalty
            )

        reasonable_no_drop_count = 0
        missed_drop_no_drop_count = 0
        no_drop_pred_error_mean = 0.0
        if no_drop_timeout_mask.any():
            timeout_env_ids = no_drop_timeout_mask.nonzero(as_tuple=False).squeeze(-1)
            pred_pos = self.obs_dict["robot_position"][timeout_env_ids].clone()
            pred_vel = self.obs_dict["robot_linvel"][timeout_env_ids].clone()
            add_child_eject_velocity = bool(
                getattr(self.drop_impact_config, "add_child_eject_velocity", True)
            )
            if add_child_eject_velocity:
                pred_vel += self._compute_eject_velocity_world(timeout_env_ids)
            _, pred_error_xy = self._predict_child_landing_xy(
                timeout_env_ids,
                init_pos=pred_pos,
                init_vel=pred_vel,
            )
            reasonable_mask = pred_error_xy > reasonable_no_drop_threshold
            missed_mask = ~reasonable_mask
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

        # Scheme B: Dynamic Stability & Terminal Reward
        # 1. Update Best Potential in Goal
        position = self.obs_dict["robot_position"]
        dist_to_target = torch.norm(self.target_position - position, dim=1)
        in_goal_mask = dist_to_target <= self.success_config.success_radius
        
        # Update Arrival Tracker (Metric: "Have I ever been there?")
        self.has_arrived = self.has_arrived | in_goal_mask
        
        # J_t is stored in self.previous_potential (updated in _compute_reward_and_scores)
        J_t = self.previous_potential
        if not hasattr(self, "J_ema"):
            self.J_ema = J_t.clone()
        ema_alpha = getattr(self.success_config, "stability_potential_ema_alpha", 0.9)
        self.J_ema = ema_alpha * self.J_ema + (1.0 - ema_alpha) * J_t
        J_for_success = self.J_ema
        
        # Update J_best_in_goal where in_goal_mask is True
        current_best = self.J_best_in_goal
        new_best = torch.min(current_best, J_for_success)
        self.J_best_in_goal = torch.where(in_goal_mask, new_best, current_best)
        
        # 2. Check Stability Conditions
        # Cond 1: Position (d <= 2.0m) -> already in_goal_mask
        
        # Cond 2: Velocity (v <= v_hold)
        linvel = self.obs_dict["robot_linvel"]
        linvel_magnitude = torch.norm(linvel, dim=1)
        cond_vel = linvel_magnitude <= self.success_config.stability_velocity_threshold
        
        # Cond 3: Potential Quality (J_t <= J_best + delta)
        cond_pot = J_for_success <= (self.J_best_in_goal + self.success_config.stability_potential_delta)
        
        # Stable?
        is_stable = in_goal_mask & cond_vel & cond_pot
        
        # 3. Update Hold Counter
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

            # --- Capture Final J and Arrival Status ---
            # Reuse already-computed J values (avoid dimension mismatch)
            try:
                final_J = self.current_J[done_indices]
                final_J_vals = final_J.detach().cpu().numpy()
                self.final_J_buffer.extend(final_J_vals)
                
                # Calculate distance for arrival check
                d_pos = self.obs_dict["robot_position"][done_indices]
                dist_to_tgt = torch.norm(self.target_position[done_indices] - d_pos, dim=1)
                
                # Check if distance <= 2.0 (3D Euclidean distance)
                arrival_threshold = 2.0
                final_arrived = (dist_to_tgt <= arrival_threshold)
                final_arrived_vals = final_arrived.detach().cpu().numpy()
                
                # --- Apply Terminal Reward (NEW) ⭐ ---
                # Retrieve terminal reward config (default to 0.0 if not set yet)
                term_reward_val = self.reward_params.get("terminal_reward", 0.0)
                
                if term_reward_val > 0.0:
                    # Create bonus tensor: +term_reward where arrived, 0 otherwise
                    # CAUTION: 'rewards' tensor is (num_envs,), so we update specific indices
                    arrival_bonus = final_arrived.float() * term_reward_val
                    self.rewards[done_indices] += arrival_bonus
                
                # Legacy buffer for compatibility
                self.final_arrival_buffer.extend(final_arrived_vals.astype(float))
                
                # New sliding window buffer for recent arrival rate
                self.recent_arrival_buffer.extend(final_arrived_vals)
                

                
            except Exception as e:
                # Log error for debugging
                logger.error(f"Failed to compute final J and arrival stats: {e}")
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
        self.infos["drop_hit_obstacle_snapshot"] = self.step_drop_hit_obstacle.clone()
        self.infos["impulse_metric_snapshot"] = self.step_impulse_metric.clone()
        self.infos["landing_error_xy_snapshot"] = self.step_landing_error_xy.clone()
        self.infos["release_to_target_xy_snapshot"] = self.step_release_to_target_xy.clone()

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
        self._populate_extras(improvement_reward, direction_reward, noise_reduction_reward, noise_intensity)
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
