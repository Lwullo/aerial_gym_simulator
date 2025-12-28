import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


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
    episode_len_steps = 100  # real physics time for simulation is this value multiplied by sim.dt

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

    class score_config:
        w1 = 1.0    #障碍物安全项权重
        w2 = 0.7    #目标距离项权重
        w3 = 0.5   #噪声项权重
        c  = 1.5
        r_obs = 2.0
        grid_step = 0.8

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
        max_speed = torch.tensor([1.2, 0.5, 0.8], device=clamped_action.device)
        max_yawrate = torch.pi / 3  # [rad/s]
        processed_action = clamped_action.clone()
        processed_action[:, 0:3] = max_speed * processed_action[:, 0:3]
        processed_action[:, 3] = max_yawrate * processed_action[:, 3]
        return processed_action
