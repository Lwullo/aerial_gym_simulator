from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    args = get_args()
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        # robot_name="base_quadrotor",
        robot_name="lmf2",
        # controller_name="lee_attitude_control",
        controller_name = "lmf2_velocity_control",  # 使用速度控制器来模拟RL训练环境
        args=None,
        device="cuda:0",
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    # RL action_transformation_function 的实际输出范围:
    # vx, vy: max ±0.8 m/s, vz: max ±0.5 m/s, yaw_rate: max ±π/6 rad/s
    max_speed_xy = 0.8
    max_speed_z = 0.5
    max_yaw_rate = torch.pi / 6
    # Metrics for stability evaluation
    avg_jitter = 0.0
    avg_vel_error = 0.0
    avg_z_pos = 1.0 # Current altitude
    count = 0
    
    print("\n" + "="*90)
    print(f"{'Step':>8} | {'Altitude (m)':>12} | {'Jitter (rad/s)':>15} | {'Vel Error (m/s)':>15} | {'Status'}")
    print("-" * 90)

    for i in range(10000):
        if i % 1000 == 0 and i > 0:
            status = "STABLE" if (avg_jitter / count < 0.5) else "JITTERY/UNSTABLE"
            print(f"{i:>8} | {avg_z_pos/count:>12.2f} | {avg_jitter/count:>15.4f} | {avg_vel_error/count:>15.4f} | {status}")
            
            logger.info(f"Step {i}, changing target setpoint.")
            # 只在RL实际使用的速度范围内随机给定速度指令
            actions[:, 0:2] = max_speed_xy * (torch.rand_like(actions[:, 0:2]) * 2 - 1)
            actions[:, 2] = max_speed_z * (torch.rand_like(actions[:, 2]) * 2 - 1)
            actions[:, 3] = max_yaw_rate * (torch.rand_like(actions[:, 3]) * 2 - 1)
            
            # Reset metrics for next interval
            avg_jitter = 0.0
            avg_vel_error = 0.0
            count = 0
            # env_manager.reset() # 不需要每次随机速度都重置，这样能看连续性

        env_manager.step(actions=actions)
        
        # Calculate real-time metrics
        with torch.no_grad():
            obs = env_manager.get_obs()
            # Jitter: Magnitude of body angular velocity
            angvel = obs["robot_body_angvel"]
            jitter = torch.norm(angvel, dim=1).mean().item()
            avg_jitter += jitter
            
            # Velocity Error: Commanded vs Actual
            actual_vel = obs["robot_body_linvel"]
            # Commanded vel is the first 3 components of actions (multiplied by speed factors)
            cmd_vel = actions[:, :3].clone()
            # cmd_vel[:, 0:2] *= max_speed_xy (这里actions已经是乘过的了，见下面逻辑)
            # 注意：actions在赋值时已经乘过max_speed了
            vel_error = torch.norm(actual_vel - cmd_vel, dim=1).mean().item()
            avg_vel_error += vel_error

            # Altitude monitoring
            current_z = obs["robot_position"][:, 2].mean().item()
            avg_z_pos += current_z
            
            count += 1
            
            # Ground avoidance and auto-reset
            if current_z < 0.2:
                print(f"[{i}] WARNING: Drone too low ({current_z:.2f}m). Resetting env to prevent flip.")
                env_manager.reset()
                # Clear action to hover for a moment after reset
                actions[:] = 0.0
            
            # Early warning for catastrophic failure
            if jitter > 10.0:
                print(f"CRITICAL: Excessive Jitter ({jitter:.2f}) at step {i}!")
