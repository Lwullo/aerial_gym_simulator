import numpy as np
import torch

class control:
    """
    Control parameters
    controller:
        lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
        lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
        lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
    kP: gains for position
    kV: gains for velocity
    kR: gains for attitude
    kOmega: gains for angular velocity
    """

    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = np.pi / 3.0
    scale_pos=0.5
    scale_vel=0.5
    scale_rot=1.5
    scale_angvel=3.0

    K_pos_tensor_max =torch.tensor([0.7, 0.7, 1.0]) * scale_pos  # “当前位置”和“目标位置”之间的误差，输出一个期望的速度指令给K_vel
    K_pos_tensor_min =torch.tensor([0.7, 0.7, 1.0]) * scale_pos# used for lee_position_control only
#   K_pos_tensor_max =torch.tensor([2.0, 3.0, 1.0]) * scale_pos  # “当前位置”和“目标位置”之间的误差，输出一个期望的速度指令给K_vel
#     K_pos_tensor_min =torch.tensor([2.0, 3.0, 1.0]) * scale_pos# used for lee_position_control only

    # K_vel_tensor_max = [
    #     0.33,
    #     0.33,
    #     0.13,
    # ]  # used for lee_position_control, lee_velocity_control only
    # K_vel_tensor_min = [0.27, 0.27, 0.17]

    K_vel_tensor_max =torch.tensor([
        5.0,
        5.0,
        1.3,
    ])*scale_vel # k_vel是“当前速度”和“期望速度”之间的误差，输出一个期望的姿态指令给K_rot
    K_vel_tensor_min = torch.tensor([2.7,2.7,1.7])*scale_vel

    K_rot_tensor_max =torch.tensor([
        1.85,
        1.85,
        0.4,
    ])*scale_rot # 期望姿态”和“无人机当前姿态”之间的误差。输出一个期望的角速度指令给K_angvel
    K_rot_tensor_min =torch.tensor( [1.6, 1.6, 0.25])*scale_rot

    K_angvel_tensor_max = torch.tensor([
        0.5,
        0.5,
        0.09,
    ])*scale_angvel  # 期望角速度”和“无人机当前角速度”之间的误差，输出一个期望的推力指令，直接给电机
    K_angvel_tensor_min = torch.tensor([0.4, 0.4, 0.075])*scale_angvel

    randomize_params = False
