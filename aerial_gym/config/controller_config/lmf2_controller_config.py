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
    max_inclination_angle_rad = np.pi / 6.0  # 30° (reduced from 60° for stability)
    max_yaw_rate = np.pi / 3.0
    scale_pos=0.5
    scale_vel=1.0  # Set to 1.0 since using pre-scaled values
    scale_rot=1.0  # Set to 1.0 since using pre-scaled values
    scale_angvel=1.0  # Set to 1.0 since using pre-scaled values

    K_pos_tensor_max = torch.tensor([0.7, 0.7, 1.0]) * scale_pos  # "当前位置"和"目标位置"之间的误差，输出一个期望的速度指令给K_vel
    K_pos_tensor_min = torch.tensor([0.7, 0.7, 1.0]) * scale_pos  # used for lee_position_control only

    # # Reference gains for faster velocity tracking
    # K_vel_tensor_max = torch.tensor([4.8, 4.8, 4.8])  # 从5.2降低
    # K_vel_tensor_min = torch.tensor([4.8, 4.8, 4.8])

    # K_rot_tensor_max = torch.tensor([6.5, 6.5, 1.5])  # 从7.0降低
    # K_rot_tensor_min = torch.tensor([6.5, 6.5, 1.5])

    # K_angvel_tensor_max = torch.tensor([4.5, 4.5, 0.5])
    # K_angvel_tensor_min = torch.tensor([4.5, 4.5, 0.5])
    K_vel_tensor_max = torch.tensor([1.9172, 1.9172, 0.7233])
    K_vel_tensor_min = torch.tensor([1.9172, 1.9172, 0.7233])

    K_rot_tensor_max = torch.tensor([1.7822, 1.7822, 0.4297])
    K_rot_tensor_min = torch.tensor([1.7822, 1.7822, 0.4297])

    K_angvel_tensor_max = torch.tensor([1.7354, 1.7354, 0.1018])
    K_angvel_tensor_min = torch.tensor([1.7354, 1.7354, 0.1018])

    randomize_params = False
