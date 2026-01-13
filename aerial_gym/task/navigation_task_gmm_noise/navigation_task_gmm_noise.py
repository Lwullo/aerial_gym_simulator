import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gym.spaces import Dict, Box

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

        self.score_config = self.task_config.score_config
        self.score_w1 = float(self.score_config.w1)
        self.score_w2 = float(self.score_config.w2)
        self.score_w3 = float(self.score_config.w3)
        self.score_c = float(self.score_config.c)
        self.score_r_obs = float(self.score_config.r_obs)

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
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

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
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_transformation_function = self.task_config.action_transformation_function

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
        }

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
        
        # Buffer for score gradient calculation
        self.previous_total_score = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        
        # GMM Physical Force buffers
        self.gmm_force_config = self.task_config.gmm_force_config
        self.gmm_force_direction = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.gmm_force_update_counter = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        
        # Sliding average buffer for total_score (window size: 10)
        self.score_window_size = 10
        self.score_history = torch.zeros(
            (self.sim_env.num_envs, self.score_window_size), device=self.device, requires_grad=False
        )
        self.score_history_idx = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        self.score_history_filled = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool
        )

        # Cache score configuration for performance
        sc = self.task_config.score_config
        self.score_w1 = float(sc.w1)
        self.score_w2 = float(sc.w2)
        self.score_w3 = float(sc.w3)
        self.score_c = float(sc.c)
        self.score_r_obs = float(sc.r_obs)
        
        self._init_noise_buffers()
        self._load_fixed_env_preset()
        
        self.best_point_path = self._resolve_best_point_path()
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

        # TensorBoard writer for direct metric logging
        runs_dir = os.environ.get("AERIAL_GYM_RUNS_DIR", "runs")
        experiment_name = os.environ.get("AERIAL_GYM_EXPERIMENT_NAME", "gmm_noise_diagnostics")
        tb_log_dir = os.path.join(runs_dir, f"{experiment_name}_tb")
        self._tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")

        # Ensure assets are placed within the fixed bounds.
        self.sim_env.reset()
        self.reset_idx(torch.arange(self.sim_env.num_envs, device=self.device))

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
        if not self.noise_config.enable_noise or self.num_noise_sources <= 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        num_envs = env_ids.shape[0]
        num_sources = self.num_noise_sources

        bounds_min = self.env_bounds_min.view(1, 1, 3).expand(num_envs, num_sources, 3)
        bounds_max = self.env_bounds_max.view(1, 1, 3).expand(num_envs, num_sources, 3)
        centers = torch_rand_float_tensor(bounds_min, bounds_max)

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

    def _draw_env0_debug_markers(self):
        viewer_ctrl = getattr(self.sim_env.IGE_env, "viewer", None)
        if viewer_ctrl is None or viewer_ctrl.viewer is None:
            return

        gym = self.sim_env.IGE_env.gym
        env_handles = getattr(self.sim_env.IGE_env, "env_handles", None)
        if not env_handles:
            return
        env_handle = env_handles[0]

        if hasattr(gym, "clear_lines"):
            gym.clear_lines(viewer_ctrl.viewer)

        target = self.target_position[0].detach().cpu().numpy()
        noise_centers = self.noise_centers[0].detach().cpu().numpy()

        lines = []
        colors = []

        axis_half_len = 0.25  # axis length = 0.5
        axes = np.eye(3, dtype=np.float32)
        for i in range(3):
            start = target - axis_half_len * axes[i]
            end = target + axis_half_len * axes[i]
            lines.append([start, end])
            colors.append([0.0, 1.0, 0.0])

        radius = 0.3
        segments = 24
        angles = np.linspace(0.0, 2.0 * np.pi, segments + 1, dtype=np.float32)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        unit_xy = np.stack([cos_a, sin_a, np.zeros_like(cos_a)], axis=1)
        unit_xz = np.stack([cos_a, np.zeros_like(cos_a), sin_a], axis=1)
        unit_yz = np.stack([np.zeros_like(cos_a), cos_a, sin_a], axis=1)

        for center in noise_centers:
            for unit_circle in (unit_xy, unit_xz, unit_yz):
                pts = center + radius * unit_circle
                for idx in range(segments):
                    lines.append([pts[idx], pts[idx + 1]])
                    colors.append([1.0, 0.0, 0.0])

        if not lines:
            return

        line_vertices = np.array(lines, dtype=np.float32).reshape(-1, 3)
        line_colors = np.array(colors, dtype=np.float32)
        gym.add_lines(viewer_ctrl.viewer, env_handle, line_colors.shape[0], line_vertices, line_colors)

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
    
    def _update_gmm_force_direction(self):
        """Update random force direction periodically based on force_update_steps."""
        if not self.gmm_force_config.enable_physical_force:
            return
        
        # Increment counter for all environments
        self.gmm_force_update_counter += 1
        
        # Find environments that need direction update
        update_mask = self.gmm_force_update_counter >= self.gmm_force_config.force_update_steps
        
        if update_mask.any():
            num_to_update = update_mask.sum().item()
            
            # Generate random unit vectors for environments needing update
            random_dirs = torch.randn((num_to_update, 3), device=self.device)
            random_dirs = random_dirs / torch.norm(random_dirs, dim=1, keepdim=True)
            
            # Update directions and reset counters
            self.gmm_force_direction[update_mask] = random_dirs
            self.gmm_force_update_counter[update_mask] = 0
    
    def _apply_gmm_physical_forces(self):
        """Apply GMM-based physical forces to the robot."""
        # DEBUG: Print force status every 50 steps
        # if hasattr(self, 'num_task_steps') and self.num_task_steps % 50 == 0:
        #     print(f"[FORCE DEBUG] enable_physical_force={self.gmm_force_config.enable_physical_force}, num_sources={self.num_noise_sources}")
        
        if not self.gmm_force_config.enable_physical_force or self.num_noise_sources <= 0:
            # if hasattr(self, 'num_task_steps') and self.num_task_steps % 50 == 0:
            #     print(f"[FORCE DEBUG] GMM physical force is DISABLED")
            return
        
        # Compute GMM mixture intensity at current position
        position = self.obs_dict["robot_position"]
        deltas = position.unsqueeze(1) - self.noise_centers
        scaled = (deltas / self.noise_sigmas).pow(2).sum(dim=-1)
        mixture = torch.exp(-0.5 * scaled)
        mixture_intensity = (self.noise_weights * mixture).sum(dim=1)  # Shape: (num_envs,)
        
        # Convert intensity to force magnitude: F = Mixture(P) × mass × g × k
        mass = self.gmm_force_config.drone_mass
        g = self.gmm_force_config.gravity
        k = self.gmm_force_config.disturbance_coefficient
        force_magnitude = mixture_intensity * mass * g * k  # Shape: (num_envs,)
        
        # DEBUG: Print GMM force diagnostics for environment 0
        # if hasattr(self, 'num_task_steps'):
        #     if self.num_task_steps % 10 == 0:  # Print every 10 steps to reduce spam
        #         env_id = 0
        #         print(f"[GMM Force Debug - Step {self.num_task_steps}]")
        #         print(f"  Position: {position[env_id].cpu().numpy()}")
        #         print(f"  Mixture Intensity: {mixture_intensity[env_id].item():.4f}")
        #         print(f"  Force Magnitude: {force_magnitude[env_id].item():.2f} N")
        #         print(f"  Force Direction: {self.gmm_force_direction[env_id].cpu().numpy()}")
        #         print(f"  Force Vector: {(force_magnitude[env_id] * self.gmm_force_direction[env_id]).cpu().numpy()}")
        
        # Apply force in random direction
        # robot_force_tensor shape: (num_envs, num_robot_rigid_bodies, 3)
        # We apply force only to the base link (index 0)
        force_vector = force_magnitude.unsqueeze(1) * self.gmm_force_direction  # Shape: (num_envs, 3)
        
        # Access robot force tensor from global dict (only base link)
        robot_force_tensor = self.obs_dict.get("robot_force_tensor", None)
        if robot_force_tensor is not None:
            # Apply to base link (first rigid body)
            robot_force_tensor[:, 0, :] = force_vector
        else:
            # Fallback: directly access global force tensor
            # Robot rigid bodies start at index 0
            global_force_tensor = self.sim_env.IGE_env.global_tensor_dict["global_force_tensor"]
            num_rigid_bodies_per_env = self.sim_env.IGE_env.num_rigid_bodies_per_env
            
            for env_id in range(self.sim_env.num_envs):
                base_idx = env_id * num_rigid_bodies_per_env
                global_force_tensor[base_idx] = force_vector[env_id]



    def _compute_score_components(self, position):
        dist = torch.norm(position - self.target_position, dim=1)
        s_dist = 1.0 - dist / self.d_max
        s_dist = torch.clamp(s_dist, 0.0, 1.0)

        deltas = position.unsqueeze(1) - self.noise_centers
        scaled = (deltas / self.noise_sigmas).pow(2).sum(dim=-1)
        mixture = torch.exp(-0.5 * scaled)
        mixture = (mixture * self.noise_weights).sum(dim=1)
        g_noise = mixture.pow(2)
        s_noise = 1.0 - g_noise
        s_noise = torch.clamp(s_noise, 0.0, 1.0)

        # Use LiDAR minimum distance for s_obs (handles rotated obstacles)
        if "depth_range_pixels" in self.obs_dict:
            depth_pixels = self.obs_dict["depth_range_pixels"]
            # Get minimum distance from LiDAR depth image
            min_depth = depth_pixels.view(depth_pixels.shape[0], -1).min(dim=1).values
            # Clamp to valid range
            lidar_max_range = self.reward_params.get("lidar_max_range", 10.0)
            d_obs = torch.clamp(min_depth, min=0.01, max=lidar_max_range)
        else:
            # Fallback to spherical approximation if LiDAR not available
            obs_pos = self.obs_dict["obstacle_position"]
            bounds_min = self.env_bounds_min.view(1, 1, 3)
            bounds_max = self.env_bounds_max.view(1, 1, 3)
            in_bounds = ((obs_pos >= bounds_min) & (obs_pos <= bounds_max)).all(dim=2)
            obs_deltas = position.unsqueeze(1) - obs_pos
            obs_dist = torch.norm(obs_deltas, dim=-1)
            large = torch.tensor(1e6, device=self.device)
            obs_dist = torch.where(in_bounds, obs_dist, large)
            d_obs_surface = torch.clamp(obs_dist - self.score_r_obs, min=0.0)
            d_obs = d_obs_surface.min(dim=1).values
        
        s_obs = 1.0 - torch.exp(-self.score_c * d_obs.pow(2))
        s_obs = torch.clamp(s_obs, 0.0, 1.0)

        total_score = self.score_w1 * s_dist + self.score_w2 * s_noise + self.score_w3 * s_obs
        return s_dist, s_noise, s_obs, total_score

    def _compute_reward_and_scores(self):
        position = self.obs_dict["robot_position"]
        s_dist, s_noise, s_obs, total_score = self._compute_score_components(position)
        
        # Update sliding average window
        for env_id in range(self.sim_env.num_envs):
            idx = self.score_history_idx[env_id].item()
            self.score_history[env_id, idx] = total_score[env_id]
            self.score_history_idx[env_id] = (idx + 1) % self.score_window_size
            if idx == self.score_window_size - 1:
                self.score_history_filled[env_id] = True
        
        # Compute averaged score (use mean of available history)
        averaged_score = torch.where(
            self.score_history_filled,  # Fixed: removed .unsqueeze(1)
            self.score_history.mean(dim=1),
            self.score_history.sum(dim=1) / (self.score_history_idx + 1).float()
        )
        
        # 1. Goal Reward (Position based)
        goal_pos = self.target_position
        dist_to_goal = torch.norm(position - goal_pos, dim=1)
        prev_dist_to_goal = torch.norm(self.previous_position - goal_pos, dim=1)
        
        pos_reward_magnitude = self.reward_params["pos_reward_magnitude"]
        pos_reward_exponent = self.reward_params["pos_reward_exponent"]
        pos_reward = pos_reward_magnitude * (1.0 / (1.0 + dist_to_goal)).pow(pos_reward_exponent)
        
        # 2. Getting Closer Reward
        getting_closer = (prev_dist_to_goal - dist_to_goal) > 0
        getting_closer_reward = getting_closer.float() * self.reward_params["getting_closer_reward_multiplier"]
        
        # 3. Very Close to Goal Reward
        very_close = dist_to_goal < 0.5
        very_close_reward = very_close.float() * self.reward_params["very_close_to_goal_reward_magnitude"]
        
        # 4. Action Penalty
        # Action difference penalty
        action_diff = self.actions - self.previous_actions
        x_diff_penalty = -self.reward_params["x_action_diff_penalty_magnitude"] * (action_diff[:, 0].abs().pow(self.reward_params["x_action_diff_penalty_exponent"]))
        z_diff_penalty = -self.reward_params["z_action_diff_penalty_magnitude"] * (action_diff[:, 2].abs().pow(self.reward_params["z_action_diff_penalty_exponent"]))
        yawrate_diff_penalty = -self.reward_params["yawrate_action_diff_penalty_magnitude"] * (action_diff[:, 3].abs().pow(self.reward_params["yawrate_action_diff_penalty_exponent"]))
        
        # Absolute action penalty
        x_abs_penalty = -self.reward_params["x_absolute_action_penalty_magnitude"] * (self.actions[:, 0].abs().pow(self.reward_params["x_absolute_action_penalty_exponent"]))
        z_abs_penalty = -self.reward_params["z_absolute_action_penalty_magnitude"] * (self.actions[:, 2].abs().pow(self.reward_params["z_absolute_action_penalty_exponent"]))
        yawrate_abs_penalty = -self.reward_params["yawrate_absolute_action_penalty_magnitude"] * (self.actions[:, 3].abs().pow(self.reward_params["yawrate_absolute_action_penalty_exponent"]))

        action_penalty = x_diff_penalty + z_diff_penalty + yawrate_diff_penalty + x_abs_penalty + z_abs_penalty + yawrate_abs_penalty

        # 5. Collision Penalty
        collision_penalty = self.reward_params["collision_penalty"]
        collision_mask = (self.obs_dict["crashes"] > 0).float()
        collision_reward = collision_penalty * collision_mask
        
        # Total Reward
        reward = pos_reward + getting_closer_reward + very_close_reward + action_penalty + collision_reward
        
        # Update previous state
        self.previous_total_score = averaged_score.clone()
        self.previous_position = position.clone()
        self.previous_actions = self.actions.clone() # Ensure self.previous_actions is updated in step() or init()
        
        return reward, total_score, s_dist, s_noise, s_obs, pos_reward, very_close_reward

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

    def _populate_extras(
        self, total_score, reward, reward_goal, reward_noise, s_dist, s_noise, s_obs
    ):
        # Calculate diagnostic metrics
        crash_rate = self.obs_dict["crashes"].float().mean().item()
        avg_z_pos = self.obs_dict["robot_position"][:, 2].mean().item()
        
        extras = {
            "total_score": float(total_score.mean().item()),
            "total_reward": float(reward.mean().item()),
            "score_s_dist": float(s_dist.mean().item()),
            "score_s_noise": float(s_noise.mean().item()),
            "score_s_obs": float(s_obs.mean().item()),
            "score_w1": self.score_w1,
            "score_w2": self.score_w2,
            "score_w3": self.score_w3,
            "best_total_score": float(self.best_total_score),
            "score_gradient_weight": float(self.reward_params.get("score_gradient_weight", 10.0)),
            "episode_length": float(self.task_config.episode_len_steps),
            # Diagnostic metrics for TensorBoard
            "info/crash_rate": crash_rate,
            "info/avg_z_position": avg_z_pos,
        }
        learning_rate = os.environ.get("AERIAL_GYM_LR")
        if learning_rate is not None:
            extras["learning_rate"] = float(learning_rate)
        
        # Periodic terminal logging (every 1000 steps)
        if self.num_task_steps % 1000 == 0:
            logger.info(f"[Step {self.num_task_steps}] crash_rate={crash_rate:.3f}, avg_z={avg_z_pos:.2f}m, reward={reward.mean().item():.2f}")
        
        # Direct TensorBoard logging (every 100 steps to reduce overhead)
        if self.num_task_steps % 100 == 0:
            step = self.num_task_steps
            self._tb_writer.add_scalar("info/crash_rate", crash_rate, step)
            self._tb_writer.add_scalar("info/avg_z_position", avg_z_pos, step)
            self._tb_writer.add_scalar("reward/total", reward.mean().item(), step)
            self._tb_writer.add_scalar("score/total", total_score.mean().item(), step)
            self._tb_writer.add_scalar("score/s_dist", s_dist.mean().item(), step)
            self._tb_writer.add_scalar("score/s_noise", s_noise.mean().item(), step)
            self._tb_writer.add_scalar("score/s_obs", s_obs.mean().item(), step)
        
        self.infos["extras"] = extras

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs, device=self.device))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).view(-1)
        if env_ids.numel() == 0:
            return
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
            target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
            self.target_position[env_ids] = torch_interpolate_ratio(
                min=self.obs_dict["env_bounds_min"][env_ids],
                max=self.obs_dict["env_bounds_max"][env_ids],
                ratio=target_ratio[env_ids],
            )
            if self.noise_config.resample_on_reset:
                self._resample_noise_sources(env_ids)
        if (env_ids == 0).any().item():
            self._draw_env0_debug_markers()
        
        # Initialize previous score to reasonable default (avoids first-step spike)
        # Typical score is around 1.0-2.0, using 1.5 as stable initial value
        self.previous_total_score[env_ids] = 1.5
        
        # Reset sliding average buffers
        self.score_history[env_ids] = 0.0
        self.score_history_idx[env_ids] = 0
        self.score_history_filled[env_ids] = False
        
        # Reset GMM force buffers
        self.gmm_force_direction[env_ids] = 0.0
        self.gmm_force_update_counter[env_ids] = 0
        
        # Reset previous state for reward calculation
        self.previous_position[env_ids] = self.obs_dict["robot_position"][env_ids]
        self.previous_actions[env_ids] = 0.0
        
        self.infos = {}
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
        pos_noise = self._compute_position_noise()
        noisy_position = self.obs_dict["robot_position"] + pos_noise

        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"], (self.target_position - noisy_position)
        )
        perturbed_vec_to_tgt = vec_to_tgt + 0.1 * 2 * (torch.rand_like(vec_to_tgt - 0.5))
        dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)
        perturbed_unit_vec_to_tgt = perturbed_vec_to_tgt / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 0:3] = perturbed_unit_vec_to_tgt
        self.task_obs["observations"][:, 3] = dist_to_tgt
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        perturbed_euler_angles = euler_angles + 0.1 * (torch.rand_like(euler_angles) - 0.5)
        self.task_obs["observations"][:, 4] = perturbed_euler_angles[:, 0]
        self.task_obs["observations"][:, 5] = perturbed_euler_angles[:, 1]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        if self.task_config.vae_config.use_vae:
            self.task_obs["observations"][:, 17:] = self.image_latents

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
        # Update GMM force direction periodically
        self._update_gmm_force_direction()
        
        # Apply GMM physical forces to robot
        self._apply_gmm_physical_forces()
        
        # Store actions for reward calculation
        self.actions = actions
        
        transformed_action = self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)

        (
            self.rewards[:],
            total_score,
            s_dist,
            s_noise,
            s_obs,
            reward_goal,
            reward_noise,
        ) = self._compute_reward_and_scores()

        if self.task_config.return_state_before_reset is True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        successes = self.truncations * (
            torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0
        )
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(self.terminations > 0, torch.zeros_like(timeouts), timeouts)

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )

        # Reset early crash retry counter for environments that didn't crash this step
        non_crash_envs = (self.terminations == 0).nonzero(as_tuple=False).squeeze(-1)
        if non_crash_envs.numel() > 0:
            self._early_crash_retries[non_crash_envs] = 0

        if self.log_step_scores and not self._episode_log_written:
            self._record_step_score(total_score)
        self._update_best_point(total_score)
        self._populate_extras(total_score, self.rewards, reward_goal, reward_noise, s_dist, s_noise, s_obs)
        if self.log_step_scores and not self._episode_log_written:
            env_id = min(self.log_step_env_id, self.truncations.shape[0] - 1)
            done = (self.terminations[env_id] > 0) | (self.truncations[env_id] > 0)
            if bool(done.item()):
                self._write_best_point_log()
                self._episode_log_written = True

        reset_envs = self.sim_env.post_reward_calculation_step()
        
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
                    print("done")
                    sys.exit(0)
        
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        self.num_task_steps += 1
        self.process_image_observation()

        if self.task_config.return_state_before_reset is False:
            return_tuple = self.get_return_tuple()
        return return_tuple
