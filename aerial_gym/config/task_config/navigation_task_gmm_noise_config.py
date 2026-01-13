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
    episode_len_steps = 500  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    # fixed bounds for the rectangular space
    env_bounds_min = [0.0, 0.0, 0.0]
    env_bounds_max = [20.0, 20.0, 10.0]

    # fixed number of obstacles to keep in the environment (excluding keep_in_env assets)
    num_obstacles_in_env = 44
    target_min_ratio = [0.90, 0.1, 0.1]  # target ratio w.r.t environment bounds in x,y,z
    target_max_ratio = [0.94, 0.90, 0.90]  # target ratio w.r.t environment bounds in x,y,z

    reward_parameters = {
        "pos_reward_magnitude": 5.0,
        "pos_reward_exponent": 1.0 / 3.5,
        "very_close_to_goal_reward_magnitude": 5.0,
        "very_close_to_goal_reward_exponent": 2.0,
        "getting_closer_reward_multiplier": 3.0,
        "x_action_diff_penalty_magnitude": 0.1,
        "x_action_diff_penalty_exponent": 2.0,
        "z_action_diff_penalty_magnitude": 0.8,
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.8,
        "yawrate_action_diff_penalty_exponent": 3.33,
        "x_absolute_action_penalty_magnitude": 0.1,
        "x_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 0.5,
        "z_absolute_action_penalty_exponent": 1.0,
        "yawrate_absolute_action_penalty_magnitude": 0.1,
        "yawrate_absolute_action_penalty_exponent": 2.0,
        "collision_penalty": -200.0,
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
        """GMM physical force configuration"""
        enable_physical_force = False  # TEMPORARY: Disabled for testing Lee controller
        drone_mass = 12.0  # kg
        gravity = 9.8  # m/s^2
        disturbance_coefficient = 0.2  # k ∈ [0.1, 0.2] - restored to original value
        force_update_steps = 5  # Update random direction every N steps
        
    class score_config:
        w1 = 0.6    #目标距离项权重 (s_dist)
        w2 = 0.7    #噪声项权重 (s_noise)
        w3 = 1.0   #障碍物安全项权重 (s_obs)
        c  = 1.5
        r_obs = 1.0
        grid_step = 0.2

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

    class early_crash_config:
        max_retries = 5              # 最大重试次数
        threshold_steps = 5          # 判定"过早碰撞"的步数阈值
        safe_spawn_margin = 0.5      # 与障碍物的安全边距（米）
        fallback_to_center = True    # 超过重试次数后移到中心

    fixed_env_presets = [
        {
            # Preset 0: 面板 (panel.urdf) - 6个大型障碍物
            "name": "preset_0_panels",
            "target_position": [9.2, 5.0, 2.0],  # Z reduced for 4.5m bounds
            "obstacle_type": "panels",  # 使用 panels 资产类型
            "obstacle_urdf": "panel.urdf",  # URDF 尺寸: 0.1×1.2×3.0m
            "r_obs": 1.62,  # 外接球半径用于暴力脚本
            "noise_centers": [
                [2.5, 2.5, 1.5],
                [5.0, 5.0, 2.0],
                [7.5, 2.5, 1.5],
                [8.0, 8.0, 2.5],
                [3.0, 8.0, 2.0],
            ],
            "noise_sigmas": [
                [6.0, 6.0, 2.5],
                [8.0, 8.0, 3.0],
                [7.0, 9.0, 2.5],
                [10.0, 10.0, 3.5],
                [12.0, 7.0, 3.0],
            ],
            "noise_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "obstacle_positions": [
                [2.0, 2.0, 1.5],  # Panel center Z ≤ 2.0 (3m height / 2 + margin)
                [5.0, 2.0, 1.8],
                [8.0, 2.0, 1.5],
                [2.0, 7.0, 1.8],
                [5.0, 7.0, 1.5],
                [8.0, 7.0, 2.0],
            ],
        },
        {
            # Preset 1: 细长杆 (cuboidal_rod.urdf) - 12个中型障碍物
            "name": "preset_1_rods",
            "target_position": [8.9, 6.2, 1.0],  # Z reduced for 4.5m bounds
            "obstacle_type": "objects",  # 使用 objects 资产类型
            "obstacle_urdf": "cuboidal_rod.urdf",  # URDF 尺寸: 0.1×0.1×2.0m
            "r_obs": 1.0,  # 外接球半径用于暴力脚本
            "noise_centers": [
                [2.0, 5.0, 1.5],
                [5.0, 5.0, 2.5],
                [8.0, 5.0, 3.0],
                [4.0, 1.5, 1.5],
                [6.0, 8.5, 2.0],
            ],
            "noise_sigmas": [
                [6.0, 6.0, 2.5],
                [8.0, 8.0, 3.0],
                [7.0, 9.0, 2.5],
                [10.0, 10.0, 3.5],
                [12.0, 7.0, 3.0],
            ],
            "noise_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "obstacle_positions": [
                [1.5, 1.5, 1.5],  # Rod center Z ≤ 3.0 (2m height / 2 + margin)
                [3.5, 1.5, 2.5],
                [6.0, 1.5, 1.5],
                [8.5, 1.5, 2.0],
                [1.5, 5.0, 2.0],
                [5.0, 5.0, 1.5],
                [8.5, 5.0, 2.5],
                [1.5, 8.5, 2.0],
                [3.5, 8.5, 2.5],
                [6.0, 8.5, 1.5],
                [8.5, 8.5, 2.5],
                [5.0, 3.0, 2.0],
            ],
        },
        {
            # Preset 2: 立方体 (small_cube.urdf) - 15个小型障碍物
            "name": "preset_2_cubes",
            "target_position": [9.0, 2.0, 3.0],  # Z reduced for 4.5m bounds
            "obstacle_type": "objects",  # 使用 objects 资产类型
            "obstacle_urdf": "small_cube.urdf",  # URDF 尺寸: 0.4×0.4×0.4m
            "r_obs": 0.35,  # 外接球半径用于暴力脚本
            "noise_centers": [
                [1.5, 5.0, 2.0],
                [8.5, 5.0, 2.5],
                [5.0, 1.5, 1.5],
                [5.0, 8.5, 3.0],
                [5.0, 5.0, 2.0],
            ],
            "noise_sigmas": [
                [6.0, 6.0, 2.5],
                [8.0, 8.0, 3.0],
                [7.0, 9.0, 2.5],
                [10.0, 10.0, 3.5],
                [12.0, 7.0, 3.0],
            ],
            "noise_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "obstacle_positions": [
                [1.5, 1.5, 1.0],
                [3.5, 1.5, 1.5],
                [5.5, 1.5, 2.0],
                [7.5, 1.5, 2.5],
                [1.5, 3.5, 1.2],
                [3.5, 3.5, 1.8],
                [5.5, 3.5, 2.5],
                [7.5, 3.5, 3.0],
                [1.5, 5.5, 1.5],
                [3.5, 5.5, 2.0],
                [5.5, 5.5, 2.8],
                [7.5, 5.5, 3.5],
                [2.5, 7.5, 2.0],
                [5.0, 7.5, 2.5],
                [7.5, 7.5, 3.2],
            ],
        },
    ]

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
        max_speed = torch.tensor([1.0, 1.0, 0.8], device=clamped_action.device)  # Y-axis increased from 0.5 to 1.0
        max_yawrate = torch.pi / 3  # [rad/s]
        processed_action = clamped_action.clone()
        processed_action[:, 0:3] = max_speed * processed_action[:, 0:3]
        processed_action[:, 3] = max_yawrate * processed_action[:, 3]
        return processed_action
