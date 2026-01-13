import torch
from aerial_gym.utils.math import *


from aerial_gym.control.controllers.base_lee_controller import *
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("velocity_controller")


class LeeVelocityController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Lee attitude controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired thrust, roll, pitch and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        self.reset_commands()
        self.accel[:] = self.compute_acceleration(
            setpoint_position=self.robot_position,
            setpoint_velocity=command_actions[:, 0:3],
        )
        forces = (self.accel[:] - self.gravity) * self.mass
        
        # DEBUG: Print every 100 steps for env 0
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            
        # if self.debug_counter % 100 == 0:
        #     print(f"\n[VEL_CTRL DEBUG] Step {self.debug_counter}:")
        #     print(f"  Cmd velocity: {command_actions[0, 0:3].cpu().numpy()}")
        #     print(f"  Current velocity: {self.robot_linvel[0].cpu().numpy()}")
        #     print(f"  Calculated accel: {self.accel[0].cpu().numpy()}")
        #     print(f"  Forces (world): {forces[0].cpu().numpy()}")
        
        # Thrust command is the projection of world-frame forces onto body Z-axis
        # Lee controller uses attitude tilt to produce horizontal motion
        self.wrench_command[:, 2] = torch.sum(
            forces * quat_to_rotation_matrix(self.robot_orientation)[:, :, 2], dim=1
        )
        
        # if self.debug_counter % 100 == 0:
        #     print(f"  Thrust_z: {self.wrench_command[0, 2].item():.2f} N")
        #     print(f"  Wrench [fx,fy,fz]: {self.wrench_command[0, 0:3].cpu().numpy()}")

        # after calculating forces, we calculate the desired euler angles
        self.desired_quat[:] = calculate_desired_orientation_for_position_velocity_control(
            forces, self.robot_euler_angles[:, 2], self.buffer_tensor
        )
        
        # DEBUG: Print attitude tracking
        # if self.debug_counter % 100 == 0:
        #     # Calculate desired pitch from forces (approximate)
        #     # desired_pitch ≈ atan2(fx, fz)
        #     desired_pitch_rad = torch.atan2(forces[0, 0], forces[0, 2]).item()
        #     desired_pitch_deg = desired_pitch_rad * 180 / 3.14159
            
        #     actual_euler = self.robot_euler_angles[0].cpu().numpy()
        #     actual_euler_deg = actual_euler * 180 / 3.14159
            
        #     print(f"  Desired pitch (deg): {desired_pitch_deg:.2f} (from force direction)")
        #     print(f"  Actual attitude (deg):  roll={actual_euler_deg[0]:.2f}, pitch={actual_euler_deg[1]:.2f}, yaw={actual_euler_deg[2]:.2f}")
        #     print(f"  Pitch tracking error (deg): {desired_pitch_deg - actual_euler_deg[1]:.2f}")

        self.euler_angle_rates[:, :2] = 0.0
        self.euler_angle_rates[:, 2] = command_actions[:, 3]
        self.desired_body_angvel[:] = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor
        )

        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )
        
        # DEBUG: Print torques
        # if self.debug_counter % 100 == 0:
        #     print(f"  Torque command: {self.wrench_command[0, 3:6].cpu().numpy()}")
        #     print(f"  K_rot: {self.K_rot_tensor_current[0].cpu().numpy()}")
        #     print(f"  K_angvel: {self.K_angvel_tensor_current[0].cpu().numpy()}")

        return self.wrench_command
