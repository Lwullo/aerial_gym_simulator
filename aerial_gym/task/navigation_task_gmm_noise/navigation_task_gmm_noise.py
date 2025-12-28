import torch

from aerial_gym.task.navigation_task.navigation_task import NavigationTask
from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task_gmm_noise")


class NavigationTaskGmmNoise(NavigationTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        super().__init__(task_config, seed, num_envs, headless, device, use_warp)

        self.noise_config = self.task_config.noise_config
        self.num_noise_sources = int(self.noise_config.num_sources)

        self.env_bounds_min = torch.tensor(
            self.task_config.env_bounds_min, device=self.device, requires_grad=False
        )
        self.env_bounds_max = torch.tensor(
            self.task_config.env_bounds_max, device=self.device, requires_grad=False
        )

        self._apply_fixed_env_bounds()
        self._set_fixed_obstacle_count()
        self._init_noise_buffers()

        # Ensure assets are placed within the fixed bounds.
        self.sim_env.reset()
        self.reset_idx(torch.arange(self.sim_env.num_envs, device=self.device))

    def _apply_fixed_env_bounds(self):
        env = self.sim_env.IGE_env
        bounds_min = self.env_bounds_min.view(1, 3).expand_as(env.env_lower_bound_min)
        bounds_max = self.env_bounds_max.view(1, 3).expand_as(env.env_upper_bound_max)
        env.env_lower_bound_min[:] = bounds_min
        env.env_lower_bound_max[:] = bounds_min
        env.env_upper_bound_min[:] = bounds_max
        env.env_upper_bound_max[:] = bounds_max
        env.env_lower_bound[:] = bounds_min
        env.env_upper_bound[:] = bounds_max

    def _set_fixed_obstacle_count(self):
        self.curriculum_level = int(self.task_config.num_obstacles_in_env)
        self.obs_dict["curriculum_level"] = self.curriculum_level
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = 0.0

    def _init_noise_buffers(self):
        num_envs = self.sim_env.num_envs
        num_sources = self.num_noise_sources

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

    def _resample_noise_sources(self, env_ids):
        if not self.noise_config.enable_noise or self.num_noise_sources <= 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        num_envs = env_ids.shape[0]
        num_sources = self.num_noise_sources

        bounds_min = (
            self.env_bounds_min.view(1, 1, 3).expand(num_envs, num_sources, 3)
        )
        bounds_max = (
            self.env_bounds_max.view(1, 1, 3).expand(num_envs, num_sources, 3)
        )
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

    def _compute_position_noise(self):
        if not self.noise_config.enable_noise or self.num_noise_sources <= 0:
            self.position_noise.zero_()
            return self.position_noise

        position = self.obs_dict["robot_position"]
        deltas = position.unsqueeze(1) - self.noise_centers
        scaled = (deltas / self.noise_sigmas).pow(2).sum(dim=-1)
        mixture = torch.exp(-0.5 * scaled)
        mixture = (self.noise_weights * mixture).sum(dim=1)

        noise = torch.randn_like(position) * mixture.unsqueeze(1) * self.noise_config.noise_scale
        self.position_noise[:] = noise
        return self.position_noise

    def reset_idx(self, env_ids):
        self._set_fixed_obstacle_count()
        super().reset_idx(env_ids)
        if self.noise_config.resample_on_reset:
            self._resample_noise_sources(env_ids)
        return

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        return

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
