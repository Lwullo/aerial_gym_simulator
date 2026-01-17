import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gym.spaces import Dict, Box
from collections import deque

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
        
        # Spawn position and d_max buffers (for distance reward calculation)
        self.spawn_position = torch.zeros(
            (num_envs, 3), device=self.device, requires_grad=False
        )
        self.d_max = torch.zeros(
            num_envs, device=self.device, requires_grad=False
        )
        
        # GMM Physical Force buffers
        self.gmm_force_config = self.task_config.gmm_force_config
        self.gmm_force_direction = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.gmm_target_direction = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.gmm_force_update_counter = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        
        # Success counter for continuous success check (NEW)
        self.success_counter = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        # Episode outcome tracking for success rate calculation (last 100 episodes)
        self.recent_episodes = deque(maxlen=100)
        
        # Distance tracking for improvement reward
        self.previous_distance = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        # Velocity tracking for acceleration penalty
        self.previous_velocity = torch.zeros(self.sim_env.num_envs, 3, device=self.device)
        
        # Hover time counter for cumulative hover reward (NEW)
        self.hover_time_counter = torch.zeros(self.sim_env.num_envs, device=self.device)
        
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

        # TensorBoard writer for motor thrust logging (lazily initialized on first use)
        self._writer_initialized = False
        self.writer = None



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

        # Stratified Noise Sampling Strategy ⭐
        # Total sources: 5 (defined in config)
        # Group 1: Global Roaming (Indices 0, 1, 2)
        #   - Randomly distributed across the entire environment
        #   - Provide general wind/disturbance throughout the flight
        # Group 2: Target Guardians (Indices 3, 4)
        #   - Spawned within 1.0m to 3.0m radius of the target
        #   - Create complex turbulence near the goal to test fine control
        
        # --- Group 1: Global Sources (First 3) ---
        num_global = 3
        bounds_min_global = self.env_bounds_min.view(1, 1, 3).expand(num_envs, num_global, 3)
        bounds_max_global = self.env_bounds_max.view(1, 1, 3).expand(num_envs, num_global, 3)
        centers_global = torch_rand_float_tensor(bounds_min_global, bounds_max_global)

        # --- Group 2: Local Target Sources (Last 2) ---
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
            
            # Concatenate global and local centers
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
        """
        Update GMM force direction smoothly using a low-pass filter.
        New direction = 0.9 * old_direction + 0.1 * target_direction
        Prevents sudden "kicks" to the drone during hover.
        """
        if not self.gmm_force_config.enable_physical_force:
            return

        self.gmm_force_update_counter += 1
        
        # Periodically update the random target direction (not the actual force direction)
        update_mask = self.gmm_force_update_counter >= self.gmm_force_config.force_update_steps
        if update_mask.any():
            env_ids = update_mask.nonzero(as_tuple=False).squeeze(-1)
            
            # Generate new random target directions for these envs
            random_dirs = torch.randn((len(env_ids), 3), device=self.device)
            random_dirs = torch.nn.functional.normalize(random_dirs, dim=1)
            self.gmm_target_direction[env_ids] = random_dirs
            
            # Reset counter
            self.gmm_force_update_counter[env_ids] = 0
            
        # Smoothly interpolate current direction towards target direction (Low-pass filter)
        # alpha = 0.1 (smoothing factor)
        alpha = 0.1
        self.gmm_force_direction = (1.0 - alpha) * self.gmm_force_direction + alpha * self.gmm_target_direction
        
        # Re-normalize to ensure unit magnitude
        self.gmm_force_direction = torch.nn.functional.normalize(self.gmm_force_direction, dim=1)
    
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
        
        # Apply force in random direction as additive disturbance
        # robot_force_tensor shape: (num_envs, num_robot_rigid_bodies, 3)
        # We apply force only to the base link (index 0)
        force_vector = force_magnitude.unsqueeze(1) * self.gmm_force_direction  # Shape: (num_envs, 3)
        
        # Access robot force tensor from global dict (only base link)
        robot_force_tensor = self.obs_dict.get("robot_force_tensor", None)
        if robot_force_tensor is not None:
            # CRITICAL FIX: Add disturbance force instead of replacing controller force
            # This preserves the control forces while adding GMM-based perturbation
            robot_force_tensor[:, 0, :] += force_vector  # Additive disturbance ✅
        else:
            # Fallback: directly access global force tensor
            # Robot rigid bodies start at index 0
            global_force_tensor = self.sim_env.IGE_env.global_tensor_dict["global_force_tensor"]
            num_rigid_bodies_per_env = self.sim_env.IGE_env.num_rigid_bodies_per_env
            for env_id in range(self.sim_env.num_envs):
                base_idx = env_id * num_rigid_bodies_per_env
                global_force_tensor[base_idx] = force_vector[env_id]

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




    def _compute_reward_and_scores(self):
        """
        Unified reward function design (no explicit stage switching):
        - Navigation behavior emerges from improvement + direction rewards
        - Optimization behavior emerges from noise_reduction reward
        - Hovering behavior emerges from hover_bonus when position_quality is good
        """
        position = self.obs_dict["robot_position"]
        linvel = self.obs_dict["robot_linvel"]
        
        # ==== Core Metrics ====
        dist_to_target = torch.norm(position - self.target_position, dim=1)
        linvel_magnitude = torch.norm(linvel, dim=1)
        current_noise = self._compute_gmm_mixture(position)
        
        # ==== Unified Potential Function Reward (J(p)) ⭐ ====
        # Calculate current potential J_t
        if not hasattr(self, "estimated_n_min"):
             # Fallback if uninitialized (should imply first step or error)
             n_hat = torch.zeros_like(current_noise)
        else:
            n_hat = (current_noise - self.estimated_n_min) / (
                self.estimated_n_max - self.estimated_n_min + 1e-6
            )
            n_hat = torch.clamp(n_hat, 0.0, 1.0)
            
        d0 = self.reward_params["potential_d0"]
        w_d = self.reward_params["potential_w_d"]
        w_n = self.reward_params["potential_w_n"]
        
        # J_t = w_d * (d/d0)^2 + w_n * n_hat
        J_t = w_d * (dist_to_target / d0).pow(2) + w_n * n_hat
        
        # Reward = k_J * tanh( (J_prev - J_t) / s_J )
        k_J = self.reward_params["potential_kj"]
        s_J = self.reward_params["potential_sj"]
        
        potential_diff = self.previous_potential - J_t
        
        # Use tanh to saturate reward and normalize small gradients
        potential_improvement_reward = k_J * torch.tanh(potential_diff / s_J)
        
        # Update previous potential for next step
        self.previous_potential = J_t.clone()
        
        # For logging compatibility (keep variable names for return, but zero them or use proxies)
        improvement_reward = potential_improvement_reward # Borrow this variable slot for logging
        noise_reduction_reward = torch.zeros_like(potential_improvement_reward)
        
        # 3. Direction Alignment Reward (auxiliary navigation guidance)
        vec_to_target = self.target_position - position
        distance_safe = torch.clamp(dist_to_target, min=0.01)
        direction_to_target = vec_to_target / distance_safe.unsqueeze(1)
        
        velocity = self.obs_dict["robot_linvel"]
        speed = torch.norm(velocity, dim=1, keepdim=True)
        speed_safe = torch.clamp(speed, min=0.01)
        
        velocity_direction = velocity / speed_safe
        alignment = (direction_to_target * velocity_direction).sum(dim=1)
        
        direction_reward = self.reward_params["direction_alignment_reward_magnitude"] * torch.clamp(
            alignment, min=0.0
        )
        
        # Set to zero if speed is very low (stationary)
        moving_mask = (speed.squeeze(1) > 0.05).float()
        direction_reward = direction_reward * moving_mask
        
        
        # ==== Action Smoothness Penalties (Action-based) ⭐ ====
        # 1. Action Magnitude Penalty (Energy/Effort): -k_a * ||u_t||^2
        # self.actions are already normalized (usually -1 to 1)
        k_a = self.reward_params["action_magnitude_penalty_weight"]
        action_norm_sq = torch.sum(self.actions.pow(2), dim=1)
        action_magnitude_penalty = -k_a * action_norm_sq

        # 2. Action Change Penalty (Smoothness/Jitter): -k_Delta_a * ||u_t - u_{t-1}||^2
        k_Delta_a = self.reward_params["action_change_penalty_weight"]
        action_diff_norm_sq = torch.sum((self.actions - self.previous_actions).pow(2), dim=1)
        action_change_penalty = -k_Delta_a * action_diff_norm_sq
        
        # Combined smoothness penalty
        action_smoothness_penalty = action_magnitude_penalty + action_change_penalty
        
        # ==== Safety Reward (Penalty Log-Barrier) (NEW) ⭐⭐ ====
        # Based on depth map for obstacle avoidance
        depth_pixels = self.obs_dict["depth_range_pixels"].squeeze(1)  # (num_envs, H, W)
        
        # 1. Handle Invalid/Zero Depth -> Max Range
        # Assumption: 0 means invalid/too far. Using 10.0m as default max range for this env.
        max_range = 10.0
        depth_pixels = torch.where(depth_pixels <= 0.0, torch.tensor(max_range, device=self.device), depth_pixels)
        
        # Denormalize depth (pixels are 0-1, need meters)
        # Assumes sensor config has normalize_range=True (default)
        distances = depth_pixels.view(self.sim_env.num_envs, -1) * max_range
        
        # 2. Penalty Log-Barrier: R = k * min(log(d) - log(threshold), 0)
        # Only penalize if distance < threshold. Reward is 0 if safe.
        threshold = self.reward_params.get("safety_dist_threshold", torch.tensor(1.0, device=self.device))
        
        # Clamp minimal distance for stability in log calculation
        safe_distances = torch.clamp(distances, min=self.reward_params["min_safe_distance_clamp"])
        
        log_dist = torch.log(safe_distances)
        log_threshold = torch.log(threshold)
        
        # Calculate penalty for each pixel (ray)
        # Using mean over pixels as before
        pixel_penalties = torch.clamp(log_dist - log_threshold, max=0.0)
        safety_reward = self.reward_params["safety_reward_magnitude"] * pixel_penalties.mean(dim=1)
        
        # ==== Collision Penalty ====
        collision_penalty = self.reward_params["collision_penalty"]
        collision_mask = (self.obs_dict["crashes"] > 0).float()
        collision_reward = collision_penalty * collision_mask
        
        # ==== Total Unified Reward (no explicit stage switching) ====
        reward = (
            improvement_reward           # Now carries the Unified Potential Reward
            # + noise_reduction_reward    # Removed (is 0.0)
            + direction_reward            # 2.0 × alignment (navigation guidance)
            + action_smoothness_penalty   # Action-based smoothness (-k_a, -k_da)
            + safety_reward               # 2.0 × mean(log(distances)) (obstacle avoidance)
            + collision_reward            # -100
        )
        
        # ==== Update Previous State ====
        self.previous_position = position.clone()
        self.previous_distance = dist_to_target.clone()
        self.previous_actions = self.actions.clone()
        self.previous_velocity = linvel.clone()
        
        # For logging purposes
        noise_intensity = current_noise
        
        return reward, improvement_reward, direction_reward, noise_reduction_reward, noise_intensity

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
            logger.info(f"TensorBoard writer initialized at: {summary_dir}")
        else:
            self.writer = None
            logger.warning("TensorBoard writer not initialized (env vars not set)")

    def _populate_extras(self, improvement_reward, direction_reward, noise_reduction_reward, noise_intensity):
        """Simplified extras logging for unified reward function"""
        # Calculate diagnostic metrics
        crash_rate_instant = self.obs_dict["crashes"].float().mean().item()
        avg_z_pos = self.obs_dict["robot_position"][:, 2].mean().item()
        
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
        
        # Get motor thrust values from environment 0 for monitoring
        try:
            motor_thrusts = self.sim_env.robot_manager.robot.control_allocator.motor_model.current_motor_thrust
            motor_thrust_env0 = motor_thrusts[0]  # Get thrusts from environment 0 (shape: [4])
            motor_thrust_0 = float(motor_thrust_env0[0].item())
            motor_thrust_1 = float(motor_thrust_env0[1].item())
            motor_thrust_2 = float(motor_thrust_env0[2].item())
            motor_thrust_3 = float(motor_thrust_env0[3].item())
        except Exception as e:
            # If motor thrust data is unavailable, use default values
            motor_thrust_0 = motor_thrust_1 = motor_thrust_2 = motor_thrust_3 = 0.0
            if self.num_task_steps == 0:
                logger.warning(f"Could not access motor thrust data: {e}")
        
        extras = {
            "improvement_reward": float(improvement_reward.mean().item()),
            "direction_reward": float(direction_reward.mean().item()),
            "noise_reduction_reward": float(noise_reduction_reward.mean().item()),
            "noise_intensity": float(noise_intensity.mean().item()),
            "hover_time": avg_hover_time,
            "total_reward": float(self.rewards.mean().item()),
            "episode_length": float(self.task_config.episode_len_steps),
            # Episode-based metrics
            "metrics/success_rate": success_rate,
            "metrics/crash_rate": crash_rate_episodes,
            "metrics/timeout_rate": timeout_rate,
            # Diagnostic metrics for TensorBoard
            "info/crash_rate_instant": crash_rate_instant,
            "info/avg_z_position": avg_z_pos,
            # Motor thrust metrics from environment 0
            "motor_thrust_0": motor_thrust_0,
            "motor_thrust_1": motor_thrust_1,
            "motor_thrust_2": motor_thrust_2,
            "motor_thrust_3": motor_thrust_3,
        }
        learning_rate = os.environ.get("AERIAL_GYM_LR")
        if learning_rate is not None:
            extras["learning_rate"] = float(learning_rate)
        
        # Write motor thrust data to TensorBoard (with lazy initialization)
        if not self._writer_initialized:
            self._init_tensorboard_writer()
            self._writer_initialized = True
        
        if self.writer is not None:
            self.writer.add_scalar("motor_thrust/motor_0", motor_thrust_0, self.num_task_steps)
            self.writer.add_scalar("motor_thrust/motor_1", motor_thrust_1, self.num_task_steps)
            self.writer.add_scalar("motor_thrust/motor_2", motor_thrust_2, self.num_task_steps)
            self.writer.add_scalar("motor_thrust/motor_3", motor_thrust_3, self.num_task_steps)

        
        # Periodic terminal logging (every 1000 steps) - YELLOW COLOR with hover and noise monitoring
        if self.num_task_steps % 1000 == 0:
            # Calculate attitude angles
            euler = self.obs_dict["robot_euler_angles"]
            avg_tilt_deg = torch.norm(euler[:, :2], dim=1).mean().item() * 57.2958
            
            logger.warning(
                f"[Step {self.num_task_steps}] success={success_rate:.1%}, crash={crash_rate_episodes:.1%}, "
                f"timeout={timeout_rate:.1%}, hover_time={avg_hover_time:.1f}, "
                f"noise={noise_intensity.mean().item():.3f}, tilt={avg_tilt_deg:.1f}°, "
                f"reward={self.rewards.mean().item():.2f}"
            )

        
        self.infos["extras"] = extras

    def close(self):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs, device=self.device))
        return self.get_return_tuple()


    def reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).view(-1)
        if env_ids.numel() == 0:
            return
            
        # Record episode outcomes for success rate tracking (before reset)
        if hasattr(self, 'infos') and 'successes' in self.infos:
            for env_id in env_ids:
                env_id_int = int(env_id.item())
                if self.infos["successes"][env_id_int] > 0:
                    self.recent_episodes.append('success')
                elif self.infos["crashes"][env_id_int] > 0:
                    self.recent_episodes.append('crash')
                else:  # timeout
                    self.recent_episodes.append('timeout')
        
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
        
        # Record spawn position and compute d_max for distance reward
        self.spawn_position[env_ids] = self.obs_dict["robot_position"][env_ids].clone()
        dist_spawn_to_target = torch.norm(
            self.target_position[env_ids] - self.spawn_position[env_ids], dim=1
        )
        self.d_max[env_ids] = torch.clamp(dist_spawn_to_target, min=0.1)  # Avoid division by zero
        
        # Initialize previous distance for improvement reward
        self.previous_distance[env_ids] = dist_spawn_to_target
        
        # Initialize previous velocity for acceleration penalty (NEW)
        self.previous_velocity[env_ids] = self.obs_dict["robot_linvel"][env_ids]
        
        # Reset GMM force buffers
        self.gmm_force_direction[env_ids] = 0.0
        self.gmm_target_direction[env_ids] = 0.0
        self.gmm_force_update_counter[env_ids] = 0
        
        # Reset success counter
        self.success_counter[env_ids] = 0.0
        
        # Reset previous state for reward calculation
        self.previous_position[env_ids] = self.obs_dict["robot_position"][env_ids]
        self.previous_distance[env_ids] = dist_spawn_to_target
        self.previous_actions[env_ids] = 0.0
        
        # Reset hover time counter
        self.hover_time_counter[env_ids] = 0.0
        
        # ==== Unified Cost Function Initialization ====
        # 1. Resample noise env parameters
        self._resample_noise_sources(env_ids)
        
        # 2. Estimate noise range for normalization (min/max)
        if not hasattr(self, "estimated_n_min"): # Initialize buffers if missing (first run)
            self.estimated_n_min = torch.zeros(self.sim_env.num_envs, device=self.device)
            self.estimated_n_max = torch.ones(self.sim_env.num_envs, device=self.device)
            self.previous_potential = torch.zeros(self.sim_env.num_envs, device=self.device)
            
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
            improvement_reward,
            direction_reward,
            noise_reduction_reward,
            noise_intensity,
        ) = self._compute_reward_and_scores()

        if self.task_config.return_state_before_reset is True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # New success condition: Continuous hover stability
        position = self.obs_dict["robot_position"]
        dist_to_target = torch.norm(self.target_position - position, dim=1)
        is_in_range = dist_to_target < self.success_config.success_radius
        
        linvel = self.obs_dict["robot_linvel"]
        linvel_magnitude = torch.norm(linvel, dim=1)
        is_velocity_low = linvel_magnitude < self.success_config.max_velocity
        
        euler = self.obs_dict["robot_euler_angles"]
        max_angle_rad = torch.deg2rad(torch.tensor(self.success_config.max_roll_pitch_deg, device=self.device))
        is_attitude_stable = (euler[:, 0].abs() < max_angle_rad) & (euler[:, 1].abs() < max_angle_rad)
        
        # Check if current step meets hover requirements
        is_hovering = is_in_range * is_velocity_low * is_attitude_stable
        
        # Update continuous success counter
        # Increment if hovering, reset to 0 if not
        self.success_counter = torch.where(
            is_hovering,
            self.success_counter + 1.0,
            torch.zeros_like(self.success_counter)
        )
        
        # Determine success: Counter exceeds threshold
        # Once successful, it stays successful for logging purposes, but we don't necessarily terminate immediately
        # unless you want early termination. For now, let's keep running to train stability.
        # But for the "success" metric, we flag it if threshold is met.
        has_succeeded = self.success_counter >= self.success_config.min_success_steps
        
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

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )

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
