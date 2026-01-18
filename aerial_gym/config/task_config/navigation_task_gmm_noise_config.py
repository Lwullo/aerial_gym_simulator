import torch
import os
from aerial_gym import AERIAL_GYM_DIRECTORY

# Dynamic override for preset assets
# This must happen before task_config is used by the simulator
def apply_preset_overrides():
    preset_id_str = os.environ.get("AERIAL_GYM_PRESET_ID", "-1")
    try:
        preset_id = -1
    except:
        preset_id = -1
        
    if preset_id < 0:
        return

    from aerial_gym.config.asset_config.env_object_config import (
        panel_asset_params,
        object_asset_params,
    )
    
    # Preset 0: Panels
    if preset_id == -1:
        panel_asset_params.num_assets = 6
        panel_asset_params.file = "panel.urdf"
        object_asset_params.num_assets = 0
        eval_mode = os.environ.get("AERIAL_GYM_EVAL_MODE", "").strip().lower()
        if eval_mode in ("1", "true", "yes", "y", "t"):
            panel_asset_params.keep_in_env = False
    # Preset 1: Rods
    elif preset_id == 1:
        panel_asset_params.num_assets = 0
        object_asset_params.num_assets = 12
        object_asset_params.file = "cuboidal_rod.urdf"
    # Preset 2: Cubes
    elif preset_id == 2:
        panel_asset_params.num_assets = 0
        object_asset_params.num_assets = 15
        object_asset_params.file = "small_cube.urdf"

apply_preset_overrides()


def _read_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "env_with_obstacles"
    robot_name = "lmf2"
    controller_name = "lmf2_velocity_control"
    args = {}
    num_envs = 256
    use_warp = True
    headless = True
    device = "cuda:0"
    observation_space_dim = 13 + 4 + 64  # root_state + action_dim + latent_dims
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 1200  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    # fixed bounds for the rectangular space
    env_bounds_min = [0.0, 0.0, 0.0]
    env_bounds_max = [10.0, 10.0, 10.0]

    # fixed number of obstacles to keep in the environment (excluding keep_in_env assets)
    num_obstacles_in_env = 11
    target_min_ratio = [0.2, 0.2, 0.3]  # Z updated to 3.0m (0.3)
    target_max_ratio = [0.8, 0.8, 0.7]  # Z updated to 7.0m (0.7)

    reward_parameters = {
        # Unified Cost Function Reward (J_prev - J_t) with Tanh Smoothing ⭐
        "potential_kj": 1.0,               # Max reward per step (k_J)
        "potential_sj": 0.1,               # Sensitivity scale (s_J), small change normalization
        "potential_w_d": 1.0,              # Weight for distance cost (w_d)
        "potential_w_n": 1.0,              # Weight for noise cost (w_n)
        "potential_d0": 2.0,               # Reference distance (d0) for normalization
        "n_min_max_sample_size": 1000,     # Number of samples to estimate noise range on reset
        
        # Direction alignment reward (velocity direction alignment with goal) ⭐
        "direction_alignment_reward_magnitude": 0.0,  # 速度导向，暂时禁用 (2.0 -> 0.0)
        
        # Safety reward (based on depth map for obstacle avoidance) (NEW) ⭐
        "safety_reward_magnitude": 0.1,             # Weight for safety reward (Penalty only)
        "safety_dist_threshold": 1.0,               # Distance threshold for safety penalty (meters)
        "min_safe_distance_clamp": 0.1,             # Min distance clamp to prevent log(0) (meters)
        
        # Action Smoothness Penalties (Action-based) ⭐
        "action_magnitude_penalty_weight": 0.05,    # k_a (energy/effort penalty)
        "action_change_penalty_weight": 0.1,        # k_Delta_a (smoothness/jitter penalty)
        
        # Collision penalty
        "collision_penalty": -20.0,
    }

    class vae_config:
        use_vae = True
        latent_dims = 64
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        model_folder = AERIAL_GYM_DIRECTORY
        image_res = (270, 480)
        interpolation_mode = "nearest"
        return_sampled_latent = True

    class noise_config:
        enable_noise = True
        num_sources = 5
        sigma_min = [5.0, 5.0, 2.5]
        sigma_max = [15.0, 15.0, 7.5]
        weight_min = 0.1
        weight_max = 1.0
        noise_scale = 1.0
        resample_on_reset = True

    class gmm_force_config:
        """GMM-based physical force disturbance (NOW ENABLED for realistic training)"""
        enable_physical_force = True  # Re-enabled for realistic disturbance
        disturbance_coefficient = 0.05  # k = 0.05 (reduced for stability)
        force_update_steps = 5  # Update random direction every N steps
        drone_mass = 12.04  # kg (CORRECTED to match actual robot mass from URDF)
        gravity = 9.81  # m/s²
        
    # Score config removed - formulas integrated into reward function
    # Distance reward uses s_dist formula: 1 - dist/d_max
    # Noise reward uses gradient: max(0, noise_prev - noise_curr)
    # Obstacle safety (s_obs) removed per user request

    class adaptive_grid_config:
        """自适应网格细化配置"""
        # 第一阶段：粗网格
        coarse_step = 1.0           # 初始步长 (m)
        
        # 第二阶段：局部细化
        gradient_threshold = 0.15   # 触发细化的梯度阈值
        top_percent_refine = 0.10   # 对 Top 10% 高分区域细化
        refine_step = 0.25          # 细化步长 (m)
        refine_radius = 1.5         # 以高梯度点为中心的细化半径 (m)
        
        # 第三阶段：最终精化
        polish_top_n = 3            # 对 Top N 个候选点进行最终精化
        polish_step = 0.05          # 精化步长 (m)
        polish_radius = 0.5         # 精化搜索范围 (m)

    class success_config:
        """Success condition configuration (Scheme B)"""
        success_reward = 0.0         # Terminal Reward (DISABLED per request)
        success_radius = 2.0          # Target zone radius (m)
        
        # Stability criteria
        stability_velocity_threshold = 0.35  # v_hold (m/s)
        stability_potential_delta = 0.03     # delta_J (potential tolerance)
        min_success_steps = 60               # N_hold (0.6s)
    
    class early_crash_config:
        max_retries = 5              # 最大重试次数
        threshold_steps = 5          # 判定"过早碰撞"的步数阈值
        safe_spawn_margin = 1.0      # 与障碍物的安全边距（米）
        fallback_to_center = True    # 超过重试次数后移到中心

    # Fixed presets removed - target and noise now randomly generated
    # Target position: random within target_min/max_ratio
    # Noise sources: random GMM parameters within configured ranges

    class curriculum:
        min_level = 15
        max_level = 50
        check_after_log_instances = 2048
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level

    def action_transformation_function(action):
        clamped_action = torch.clamp(action, -1.0, 1.0)
        # Reduced XY speed for safety and stability
        max_speed = torch.tensor([0.8, 0.8, 0.5], device=clamped_action.device)  # X/Y: 0.8, Z: 0.5 (reduced from 1.0)
        max_yawrate = torch.pi / 6  # 30°/s (0.524 rad/s) - Reduced from 45°/s for stability
        processed_action = clamped_action.clone()
        processed_action[:, 0:3] = max_speed * processed_action[:, 0:3]
        processed_action[:, 3] = max_yawrate * processed_action[:, 3]
        return processed_action

