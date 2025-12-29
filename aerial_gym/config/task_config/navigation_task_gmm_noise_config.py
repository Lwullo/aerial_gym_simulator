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
    episode_len_steps = 500  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    # fixed bounds for the rectangular space
    env_bounds_min = [0.0, 0.0, 0.0]
    env_bounds_max = [10.0, 10.0, 6.0]

    # fixed number of obstacles to keep in the environment (excluding keep_in_env assets)
    num_obstacles_in_env = 15 #障碍物数量更改
    # -1 disables fixed presets; set to 0/1/2 to use a fixed environment for validation.
    preset_id = -1
    target_min_ratio = [0.90, 0.1, 0.1]  # target ratio w.r.t environment bounds in x,y,z
    target_max_ratio = [0.94, 0.90, 0.90]  # target ratio w.r.t environment bounds in x,y,z

    reward_parameters = {
        "goal_reward_weight": 1.0,
        "noise_reward_weight": 0.7,
        "body_rate_penalty_weight": 0.01,
        "collision_penalty": -50.0,
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
        w1 = 0.6    #目标距离项权重 (s_dist)
        w2 = 0.7    #噪声项权重 (s_noise)
        w3 = 1.0   #障碍物安全项权重 (s_obs)
        c  = 1.5
        r_obs = 1.0
        grid_step = 0.8

    fixed_env_presets = [
        {
            "name": "preset_0_grid",
            "target_position": [9.2, 5.0, 1.5],
            "noise_centers": [
                [2.5, 2.5, 2.0],
                [5.0, 5.0, 3.0],
                [7.5, 2.5, 1.5],
                [8.0, 8.0, 4.5],
                [3.0, 8.0, 2.5],
            ],
            "noise_sigmas": [
                [6.0, 6.0, 3.0],
                [8.0, 8.0, 4.0],
                [7.0, 9.0, 3.5],
                [10.0, 10.0, 5.0],
                [12.0, 7.0, 4.5],
            ],
            "noise_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "obstacle_positions": [
                [1.5, 1.5, 1.0],
                [3.5, 1.5, 2.0],
                [5.5, 1.5, 3.0],
                [7.5, 1.5, 4.0],
                [1.5, 3.5, 1.0],
                [3.5, 3.5, 2.0],
                [5.5, 3.5, 3.0],
                [7.5, 3.5, 4.0],
                [1.5, 5.5, 1.0],
                [3.5, 5.5, 2.0],
                [5.5, 5.5, 3.0],
                [7.5, 5.5, 4.0],
                [1.5, 7.5, 1.0],
                [3.5, 7.5, 2.0],
                [5.5, 7.5, 3.0],
                [7.5, 7.5, 4.0],
                [1.5, 9.0, 1.0],
                [3.5, 9.0, 2.0],
                [5.5, 9.0, 3.0],
                [7.5, 9.0, 4.0],
            ],
        },
        {
            "name": "preset_1_corridors",
            "target_position": [8.5, 9.0, 2.5],
            "noise_centers": [
                [2.0, 5.0, 2.0],
                [5.0, 5.0, 3.5],
                [8.0, 5.0, 4.5],
                [4.0, 1.5, 1.5],
                [6.0, 8.5, 2.5],
            ],
            "noise_sigmas": [
                [6.0, 6.0, 3.0],
                [8.0, 8.0, 4.0],
                [7.0, 9.0, 3.5],
                [10.0, 10.0, 5.0],
                [12.0, 7.0, 4.5],
            ],
            "noise_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "obstacle_positions": [
                [1.0, 2.0, 1.5],
                [2.0, 2.0, 3.5],
                [3.0, 2.0, 1.5],
                [4.0, 2.0, 3.5],
                [5.0, 2.0, 1.5],
                [6.0, 2.0, 3.5],
                [7.0, 2.0, 1.5],
                [8.0, 2.0, 3.5],
                [9.0, 2.0, 1.5],
                [9.5, 2.0, 3.5],
                [1.0, 8.0, 2.0],
                [2.0, 8.0, 4.0],
                [3.0, 8.0, 2.0],
                [4.0, 8.0, 4.0],
                [5.0, 8.0, 2.0],
                [6.0, 8.0, 4.0],
                [7.0, 8.0, 2.0],
                [8.0, 8.0, 4.0],
                [9.0, 8.0, 2.0],
                [9.5, 8.0, 4.0],
            ],
        },
        {
            "name": "preset_2_perimeter",
            "target_position": [9.0, 2.0, 4.5],
            "noise_centers": [
                [1.5, 5.0, 3.0],
                [8.5, 5.0, 3.0],
                [5.0, 1.5, 2.0],
                [5.0, 8.5, 4.0],
                [5.0, 5.0, 2.5],
            ],
            "noise_sigmas": [
                [6.0, 6.0, 3.0],
                [8.0, 8.0, 4.0],
                [7.0, 9.0, 3.5],
                [10.0, 10.0, 5.0],
                [12.0, 7.0, 4.5],
            ],
            "noise_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "obstacle_positions": [
                [1.0, 1.5, 2.0],
                [1.0, 3.0, 2.0],
                [1.0, 4.5, 2.0],
                [1.0, 6.0, 2.0],
                [1.0, 7.5, 2.0],
                [9.0, 2.0, 3.5],
                [9.0, 3.5, 3.5],
                [9.0, 5.0, 3.5],
                [9.0, 6.5, 3.5],
                [9.0, 8.0, 3.5],
                [2.0, 1.0, 1.5],
                [3.5, 1.0, 1.5],
                [5.0, 1.0, 1.5],
                [6.5, 1.0, 1.5],
                [8.0, 1.0, 1.5],
                [1.5, 9.0, 4.0],
                [3.0, 9.0, 4.0],
                [4.5, 9.0, 4.0],
                [6.0, 9.0, 4.0],
                [7.5, 9.0, 4.0],
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
        max_speed = torch.tensor([1.0, 0.5, 0.8], device=clamped_action.device)
        max_yawrate = torch.pi / 3  # [rad/s]
        processed_action = clamped_action.clone()
        processed_action[:, 0:3] = max_speed * processed_action[:, 0:3]
        processed_action[:, 3] = max_yawrate * processed_action[:, 3]
        return processed_action
