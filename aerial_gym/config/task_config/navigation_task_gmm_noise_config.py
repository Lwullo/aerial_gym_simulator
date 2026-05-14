import torch
import os
from aerial_gym import AERIAL_GYM_DIRECTORY

# Dynamic override for preset assets
# This must happen before task_config is used by the simulator
def apply_preset_overrides():
    preset_id_str = os.environ.get("AERIAL_GYM_PRESET_ID", "-1")
    try:
        preset_id = int(preset_id_str)
    except (TypeError, ValueError):
        preset_id = -1
        
    if preset_id < 0:
        return

    from aerial_gym.config.asset_config.env_object_config import (
        panel_asset_params,
        object_asset_params,
    )
    
    # Preset 0: Panels
    if preset_id == 0:
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
    else:
        raise ValueError(
            f"Unsupported AERIAL_GYM_PRESET_ID={preset_id}. Expected one of: 0, 1, 2."
        )

apply_preset_overrides()


def disable_non_wall_assets():
    """Disable non-wall assets for this navigation task."""
    from aerial_gym.config.asset_config.env_object_config import (
        panel_asset_params,
        thin_asset_params,
        tree_asset_params,
        object_asset_params,
        tile_asset_params,
    )

    panel_asset_params.num_assets = 0
    thin_asset_params.num_assets = 0
    tree_asset_params.num_assets = 0
    object_asset_params.num_assets = 0
    tile_asset_params.num_assets = 0

    panel_asset_params.keep_in_env = False
    thin_asset_params.keep_in_env = False
    tree_asset_params.keep_in_env = False
    object_asset_params.keep_in_env = False
    tile_asset_params.keep_in_env = False


disable_non_wall_assets()


def _read_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class task_config:
    seed = 42
    sim_name = "base_sim"
    env_name = "env_with_obstacles"
    # robot_name = "base_quadrotor"
    # controller_name = "lee_attitude_control"
    robot_name = "lmf2"
    controller_name = "lmf2_velocity_control"
    args = {}
    num_envs = 256
    use_warp = True
    headless = True
    device = "cuda:0"
    base_observation_dim = 13
    frame_stack = 6
    observation_space_dim = 78
    privileged_observation_space_dim = 13
    critic_observation_space_dim = 91
    use_central_value = True
    critic_use_privileged_obs = True
    action_space_dim = 5  # [vx_cmd, vy_cmd, vz_cmd, yawrate_cmd, drop_switch]
    episode_len_steps = 1000  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    # fixed bounds for the rectangular space (centered at world origin)
    env_bounds_min = [-25.0, -25.0, 0.0]
    env_bounds_max = [25.0, 25.0, 15.0]

    # fixed number of obstacles to keep in the environment (excluding keep_in_env assets)
    num_obstacles_in_env = 0

    # Target sampling: XY in [-20, 20], fixed ground Z=0
    target_xy_min = -20.0
    target_xy_max = 20.0
    target_fixed_z = 0.0
    # Fixed-target training switch (if True, target is not randomized at reset).
    # When target_fixed_use_env_center=True, target_fixed_position is interpreted
    # as an offset from each environment center (recommended for parallel envs).
    target_use_fixed = True
    target_fixed_use_env_center = True
    target_fixed_position = [0.0, 0.0, 0.0]

    # Mother-ship spawn altitude:
    # If spawn_use_random_z=True, z ~ U(spawn_random_z_min, spawn_random_z_max).
    # Otherwise use spawn_fixed_z.
    spawn_fixed_z = 10.0
    spawn_use_random_z = True
    spawn_random_z_min = 2.0
    spawn_random_z_max = 13.0
    # Optional spawn-Z curriculum (linear by training epoch):
    # z_min(epoch) transitions from start_min -> end_min, while z_max stays spawn_random_z_max.
    # This helps policy first learn release timing at higher altitude, then generalize downward.
    spawn_z_curriculum_enable = True
    spawn_z_curriculum_start_min = 8.0
    spawn_z_curriculum_end_min = 2.0
    spawn_z_curriculum_warmup_epochs = 0
    spawn_z_curriculum_full_epochs = 2000
    # Fixed spawn XY (fallback mode).
    spawn_use_fixed_xy = False
    spawn_fixed_use_env_center = True
    spawn_fixed_xy = [-12.0, 0.0]
    # Randomized spawn X with fixed Y:
    # x ~ U(spawn_random_x_min, spawn_random_x_max), y = spawn_random_fixed_y
    # If spawn_random_use_env_center=True, values are offsets to each env center.
    spawn_use_random_x = True
    spawn_random_use_env_center = True
    spawn_random_x_min = -24.0
    spawn_random_x_max = -12.0
    spawn_random_fixed_y = 0.0

    # Legacy ratio-based target sampling params (kept for compatibility).
    target_min_ratio = [0.2, 0.2, 0.3]  # Z updated to 3.0m (0.3)
    target_max_ratio = [0.8, 0.8, 0.7]  # Z updated to 7.0m (0.7)

    reward_parameters = {
        # Direction alignment reward ⭐
        "direction_alignment_reward_magnitude": 0.1,
        
        # Action Smoothness Penalties (Action-based) ⭐
        "action_magnitude_penalty_weight": 0.05,    # k_a (energy/effort penalty)
        "action_change_penalty_weight": 0.1,        # k_Delta_a (smoothness/jitter penalty)
    }

    class vae_config:
        use_vae = False
        latent_dims = 64
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        model_folder = AERIAL_GYM_DIRECTORY
        image_res = (270, 480)
        interpolation_mode = "nearest"
        return_sampled_latent = True

    class gmm_force_config:
        """Two-layer wind field:
        w_total = w_main + w_local
        - w_main: per-env constant over an episode
        - w_local: time-varying local gust
        """
        enable_physical_force = True
        # Layer 1: main wind (sampled per environment at reset, fixed within episode)
        main_wind_speed_min = 1.0  # m/s
        main_wind_speed_max = 3.0  # m/s
        main_wind_horizontal_only = False  # allow 3D wind direction (z component enabled)
        # Layer 2: local gust disturbance
        force_update_steps = 5  # update local disturbance direction every N steps
        local_wind_max_speed_min = 0.2  # m/s, per-env sampled lower bound of |w_local|
        local_wind_max_speed_max = 0.3  # m/s, per-env sampled upper bound of |w_local|
        local_wind_max_speed = 0.3  # legacy fallback alias
        local_wind_horizontal_only = True
        # Legacy alias kept for backward compatibility (fallback only).
        max_wind_speed = 4.0
        drag_coefficient = 4.0  # N/(m/s), c_drag in F_drag = c_drag * (v_w - v_uav)
        # Legacy parameters kept for compatibility with older debug scripts.
        disturbance_coefficient = 0.05
        drone_mass = 12.04
        gravity = 9.81

    class dryden_config:
        # Simplified Dryden turbulence (three independent first-order filters / OU-like):
        # w_dryden[k+1] = a * w_dryden[k] + b * xi, xi~N(0, I)
        # a = exp(-dt / tau), b = sigma * sqrt(1 - a^2)
        enable_dryden = True
        # Per-axis turbulence std (m/s), sampled once per env per episode.
        sigma_min = [0.2, 0.2, 0.15]
        sigma_max = [0.3, 0.3, 0.25]
        # Per-axis time constants tau (s), sampled once per env per episode.
        tau_min = [0.8, 0.8, 0.8]
        tau_max = [2.0, 2.0, 2.0]
        # If True, force vertical turbulence component to zero.
        horizontal_only = False
        # Optional clamp to avoid extreme gust tails (in units of sigma).
        clip_sigma = 3.0

    class drop_model_config:
        enable_drop_model = True
        drop_threshold = 0.7  # drop_switch > threshold triggers drop
        allow_multiple_drops = False
        # Optional confidence-gated release:
        # If enabled, a requested DROP is executed only when the heuristic
        # release confidence is above confidence_threshold.
        confidence_gate_enable = False
        confidence_threshold = 0.45
        confidence_error_scale = 2.0
        confidence_stability_scale = 1.0
        max_release_attitude_deg = 45.0
        max_release_omega_xy = 0.3
        preferred_release_attitude_deg = 7.5
        preferred_release_attitude_band_deg = 2.5
        confidence_history_len = 5
        confidence_min_steps = 3
        confidence_gate_penalty = 0.0
        child_gravity = 9.81
        # Child free-fall wind model:
        # a = g + (c_child / m_child) * (v_w - v_child)
        child_drag_coefficient = 1.0  # c_child, unit: N/(m/s)
        child_mass = 1.0  # m_child, unit: kg
        child_wind_scale = 1.0  # legacy fallback alias for (c_child / m_child)
        max_child_sim_steps = 5000

    class drop_impact_config:
        enable_impact = True
        # Child initial velocity at release:
        # v_child0 = v_mother + R_WB * v_eject_body
        add_child_eject_velocity = True
        # Recoil implementation mode:
        # True  -> apply one-step recoil force/torque (no direct velocity impulse).
        # False -> legacy direct delta-v / delta-omega impulse injection.
        use_force_recoil = True
        child_mass = 1.0
        mother_mass = 11.04
        eject_speed = 0.5  # m/s
        eject_direction_body = [0.0, 0.0, -1.0]  # release direction: downward
        # Fixed release mount for recoil torque: choose one mount once and keep fixed for the run.
        random_fixed_payload_mount = False
        fixed_payload_mount_index = 0
        payload_mount_points_body = [
            [0.0919, 0.0919, -0.13],
            [0.0919, -0.0919, -0.13],
            [-0.0919, -0.0919, -0.13],
            [-0.0919, 0.0919, -0.13],
        ]
        # Legacy fallback offset when mount points are not provided.
        payload_offset_body = [0.0, 0.0, 0.0]
        # Random angular kick (legacy stochastic term) can be disabled.
        enable_random_angular_kick = False
        angular_sigma_base = 0.1  # rad/s
        angular_sigma_scale = 0.2  # sigma = base * (1 + scale * |omega|)
        max_delta_omega = 0.5  # rad/s clamp for one drop event

    class drop_reward_config:
        # Mother altitude soft constraint:
        # Penalize only when z < (altitude_target_z - altitude_tolerance).
        # penalty = -altitude_low_penalty_weight * ((altitude_target_z - altitude_tolerance) - z)
        altitude_target_z = 10.0
        altitude_tolerance = 0.5
        altitude_low_penalty_weight = 1.0
        # Distance-progress shaping before DROP:
        # R_dir = direction_reward_weight * (d_prev_xy - d_curr_xy)
        # Positive when moving closer to target, negative when moving away.
        # Active only when child has not dropped yet.
        direction_reward_weight = 0.4
        # Predicted-release-error shaping before DROP:
        # reward the improvement of "if released now" landing error instead of
        # penalizing the absolute error every step.
        pred_error_shaping_weight = 0.5
        pred_error_improvement_clip = 1.0
        # DROP accuracy reward (continuous):
        # R_score = score_reward_weight * score_max * exp(- (landing_error_xy / score_d0)^score_p)
        score_max = 20.0
        score_d0 = 4.0
        score_p = 2.0
        # Performance-driven score_d0 curriculum:
        # start wider for easier early learning, then tighten once the policy
        # can reliably release and reduce the landing error.
        score_d0_curriculum_enable = True
        score_d0_curriculum_values = [4.0, 3.0, 2.5, 2.0, 1.5]
        score_d0_curriculum_min_drop_rate = [0.15, 0.35, 0.55, 0.65]
        score_d0_curriculum_max_error_ema = [8.0, 5.0, 3.0, 1.2]
        # Keep outer region threshold for hard override penalty (outside -> -outside_region_penalty).
        piecewise_r = 1.0
        piecewise_thresholds = [0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 8.0]
        # If landing_error_xy is outside the outermost scored region (d > max threshold),
        # use linear penalty beyond the threshold and clamp to this minimum.
        outside_region_penalty = 20.0
        blocked_drop_penalty = 2.0
        # Risk-aware no-DROP terminal reward:
        # crash+no_drop is still bad; timeout+no_drop is acceptable when the predicted
        # landing error is already too large for a reasonable release.
        crash_no_drop_penalty = 20.0
        missed_drop_no_drop_penalty = 8.0
        reasonable_no_drop_reward = 0.0
        reasonable_no_drop_pred_error_threshold = 5.0
        no_drop_eval_max_xy_dist = 10.0
        # Unified score weight for continuous score:
        # R_score = score_reward_weight * score_max * exp(-(d_xy/score_d0)^score_p)
        score_reward_weight = 1.0
        # Impulse metric: alpha * Delta_v + beta * Delta_omega
        impulse_alpha = 1.0
        impulse_beta = 0.8
        impulse_penalty_weight = 0.5
        # EMA for drop-only landing_error_xy diagnostics
        landing_error_ema_alpha = 0.9
        # EMA for drop-only attitude-total diagnostics (deg)
        attitude_total_ema_alpha = 0.9
        
    # Score config removed - formulas integrated into reward function
    # Distance reward uses s_dist formula: 1 - dist/d_max
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
        success_reward = 50.0         # Terminal Reward when Scheme B success triggers (disabled)
        success_radius = 3.0          # Target zone radius (m)
        
        # Stability criteria
        stability_velocity_threshold = 0.35  # v_hold (m/s)
        stability_potential_delta = 0.15     # delta_J (potential tolerance)
        stability_potential_ema_alpha = 0.9  # EMA smoothing for J in success check
        min_success_steps = 15               # N_hold (steps)
    
    class early_crash_config:
        max_retries = 5              # 最大重试次数
        threshold_steps = 5          # 判定"过早碰撞"的步数阈值
        safe_spawn_margin = 1.0      # 与障碍物的安全边距（米）
        fallback_to_center = True    # 超过重试次数后移到中心

    # Fixed presets removed - target is randomized at reset
    # Target position: random within target_min/max_ratio

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
        # Mother velocity-command limits [vx, vy, vz] in m/s
        max_speed = torch.tensor([1.2, 1.2, 1.2], device=clamped_action.device)
        max_yawrate = torch.pi / 6  # 30°/s (0.524 rad/s) - Reduced from 45°/s for stability
        processed_action = clamped_action.clone()
        processed_action[:, 0:3] = max_speed * processed_action[:, 0:3]
        processed_action[:, 3] = max_yawrate * processed_action[:, 3]
        return processed_action
