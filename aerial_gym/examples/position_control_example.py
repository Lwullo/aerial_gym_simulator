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
    for i in range(10000):
        if i % 1000 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            # 只在RL实际使用的速度范围内随机给定速度指令
            actions[:, 0:2] = max_speed_xy * (torch.rand_like(actions[:, 0:2]) * 2 - 1)
            actions[:, 2] = max_speed_z * (torch.rand_like(actions[:, 2]) * 2 - 1)
            actions[:, 3] = max_yaw_rate * (torch.rand_like(actions[:, 3]) * 2 - 1)
            env_manager.reset()
        env_manager.step(actions=actions)
