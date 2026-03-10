#!/usr/bin/env python3
"""
LMF2 Velocity Controller Auto-Tuner
=====================================
自动诊断并调整 lmf2_velocity_control 控制器增益，直到实现稳定悬停和速度跟踪。
最终结果写回 lmf2_controller_config.py。

用法：
    python aerial_gym/debug/auto_tune_lmf2_controller.py
"""

# IMPORTANT: isaacgym MUST be imported before torch
import isaacgym  # noqa: F401

import sys
import os
import torch
import numpy as np
import time

from aerial_gym.utils.logging import CustomLogger
logger = CustomLogger(__name__)
logger.setLevel("WARNING")


# =============================================
# 调参配置
# =============================================
NUM_ENVS = 32
DEVICE = "cuda:0"

FLIP_HEIGHT = 0.1         # 高度低于此值视为翻倒/坠毁 (meters)
FLIP_TILT_DEG = 55.0      # 倾角超过此值视为翻倒
MAX_CRASH_RATE = 0.05     # 接受标准：超过95%环境稳定(5%容错)

HOVER_STEPS = 400         # 悬停测试步数  (4秒, dt=0.01)
VEL_STEPS = 500           # 速度跟踪测试步数 (5秒)

# K_vel 候选缩放因子 (从高到低)
K_VEL_CANDIDATES = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.22, 0.16, 0.12]
# =============================================


def get_controller(env_manager):
    """尝试从 env_manager 中获取底层控制器对象。"""
    try:
        return env_manager.robot_manager.robot.controller
    except AttributeError:
        return None


def set_controller_k_vel(controller, new_scale: float, base_k_vel: list, num_envs: int, device: str):
    """直接修改控制器内部的 K_linvel_tensor_current。"""
    k_val = torch.tensor([v * new_scale for v in base_k_vel], device=device, dtype=torch.float32)
    k_exp = k_val.unsqueeze(0).expand(num_envs, -1).clone()
    controller.K_linvel_tensor_current = k_exp
    controller.K_linvel_tensor_max = k_exp.clone()
    controller.K_linvel_tensor_min = k_exp.clone()
    return k_val.cpu().numpy().tolist()


def compute_tilt_deg(orientations: torch.Tensor) -> torch.Tensor:
    """从四元数 [qx, qy, qz, qw] 计算倾斜角度（度）。"""
    x = orientations[:, 0]
    y = orientations[:, 1]
    z_dot = 1 - 2 * (x * x + y * y)
    z_dot = torch.clamp(z_dot, -1.0, 1.0)
    return torch.rad2deg(torch.acos(z_dot))


def run_stability_test(env_manager, actions: torch.Tensor, num_steps: int, label: str) -> float:
    """
    运行稳定性测试，返回崩溃率 (0.0 ~ 1.0)。
    崩溃标准：高度 < FLIP_HEIGHT 或 倾角 > FLIP_TILT_DEG。
    """
    num_envs = actions.shape[0]
    crashed = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
    step_of_first_crash = [None] * num_envs

    env_manager.reset()

    for step in range(num_steps):
        env_manager.step(actions=actions)

        try:
            pos = env_manager.global_tensor_dict["robot_position_tensors"]
            quat = env_manager.global_tensor_dict["robot_orientation_tensors"]
        except (KeyError, AttributeError):
            continue

        tilt = compute_tilt_deg(quat)
        height_crash = pos[:, 2] < FLIP_HEIGHT
        tilt_crash = tilt > FLIP_TILT_DEG
        new_crashes = (height_crash | tilt_crash) & ~crashed

        # 记录首次崩溃步数
        for i in new_crashes.nonzero(as_tuple=True)[0]:
            step_of_first_crash[i.item()] = step

        crashed |= (height_crash | tilt_crash)

    crash_rate = crashed.float().mean().item()
    crash_count = crashed.sum().item()

    if crash_count > 0:
        first_crashes = [s for s in step_of_first_crash if s is not None]
        mean_survive = np.mean(first_crashes) if first_crashes else 0
        logger.warning(f"  [{label}] ❌ 崩溃率={crash_rate:.0%} ({crash_count}/{num_envs}), 平均存活 {mean_survive:.0f} 步 ({mean_survive*0.01:.1f}s)")
    else:
        logger.warning(f"  [{label}] ✅ 全部 {num_envs} 个环境稳定通过 ({num_steps} 步)")

    return crash_rate


def write_gains_to_config(k_vel_list: list):
    """将最优增益写回 lmf2_controller_config.py。"""
    config_path = os.path.join(
        os.path.dirname(__file__),
        "../../config/controller_config/lmf2_controller_config.py"
    )
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        logger.warning(f"❌ 配置文件不存在: {config_path}")
        return False

    with open(config_path, "r") as f:
        content = f.read()

    # 将 K_vel_tensor_max/min 替换为新值
    k_str = f"[{k_vel_list[0]:.4f}, {k_vel_list[1]:.4f}, {k_vel_list[2]:.4f}]"
    import re
    content = re.sub(
        r"K_vel_tensor_max\s*=\s*torch\.tensor\([^\)]+\)",
        f"K_vel_tensor_max = torch.tensor({k_str})  # [AUTO-TUNED]",
        content
    )
    content = re.sub(
        r"K_vel_tensor_min\s*=\s*torch\.tensor\([^\)]+\)",
        f"K_vel_tensor_min = torch.tensor({k_str})  # [AUTO-TUNED]",
        content
    )

    with open(config_path, "w") as f:
        f.write(content)

    logger.warning(f"✅ 配置已写回: {config_path}")
    logger.warning(f"   K_vel_tensor_max/min = {k_str}")
    return True


def main():
    logger.warning("=" * 60)
    logger.warning("  LMF2 速度控制器自动调参工具")
    logger.warning("=" * 60)
    logger.warning(f"  测试环境数: {NUM_ENVS}, 设备: {DEVICE}")
    logger.warning(f"  崩溃判定: 高度 < {FLIP_HEIGHT}m 或 倾角 > {FLIP_TILT_DEG}°")
    logger.warning(f"  通过标准: 崩溃率 < {MAX_CRASH_RATE:.0%}")
    logger.warning("")

    # 构建仿真环境 (headless)
    logger.warning("正在初始化仿真环境 (headless 模式)...")
    from aerial_gym.sim.sim_builder import SimBuilder
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="lmf2",
        controller_name="lmf2_velocity_control",
        args=None,
        device=DEVICE,
        num_envs=NUM_ENVS,
        headless=True,
        use_warp=True,   # lmf2 使用 LiDAR 传感器，必须启用 warp
    )
    logger.warning("✅ 仿真环境初始化完成。")
    logger.warning("")

    controller = get_controller(env_manager)
    if controller is None:
        logger.warning("❌ 无法获取控制器对象，无法进行自动调参。")
        return

    # 读取当前基准 K_vel 值 (来自 config)
    base_K_vel = controller.K_linvel_tensor_current[0].cpu().tolist()
    logger.warning(f"当前 K_vel 基准值: {[f'{v:.2f}' for v in base_K_vel]}")
    logger.warning("")

    best_scale = None

    # =============================================
    # 阶段 1: 悬停测试 (zero velocity)
    # =============================================
    logger.warning("━" * 60)
    logger.warning("阶段 1: 悬停测试 (发送零速度指令)")
    logger.warning("━" * 60)

    hover_actions = torch.zeros((NUM_ENVS, 4), device=DEVICE)
    hover_crash = run_stability_test(env_manager, hover_actions, HOVER_STEPS, "悬停@K_vel=1.0x")

    if hover_crash > MAX_CRASH_RATE:
        logger.warning("")
        logger.warning("❌ 悬停不稳！开始搜索稳定的 K_vel 缩放因子...")
        logger.warning("")
        for scale in K_VEL_CANDIDATES:
            new_K = set_controller_k_vel(controller, scale, base_K_vel, NUM_ENVS, DEVICE)
            label = f"悬停@K_vel={scale:.2f}x → {[f'{v:.3f}' for v in new_K]}"
            cr = run_stability_test(env_manager, hover_actions, HOVER_STEPS, label)
            if cr <= MAX_CRASH_RATE:
                best_scale = scale
                logger.warning(f"\n  ✅ 找到稳定悬停增益: K_vel 缩放 = {scale:.2f}")
                break
        if best_scale is None:
            logger.warning("❌ 所有候选 K_vel 值都无法实现稳定悬停。请检查 URDF 或 K_rot/K_angvel。")
            return
    else:
        best_scale = 1.0
        logger.warning("✅ 当前增益悬停稳定，继续速度跟踪测试。")

    # =============================================
    # 阶段 2: 速度跟踪测试 (RL 实际动作范围)
    # =============================================
    logger.warning("")
    logger.warning("━" * 60)
    logger.warning("阶段 2: 速度跟踪测试 (模拟 RL 输出范围 ±0.8m/s)")
    logger.warning("━" * 60)

    # 使用找到的最稳定 K_vel
    set_controller_k_vel(controller, best_scale, base_K_vel, NUM_ENVS, DEVICE)

    # 测试多个速度方向
    vel_tests = [
        ("vx=+0.8", torch.tensor([[0.8, 0.0, 0.0, 0.0]] * NUM_ENVS, device=DEVICE)),
        ("vx=-0.8", torch.tensor([[-0.8, 0.0, 0.0, 0.0]] * NUM_ENVS, device=DEVICE)),
        ("vy=+0.8", torch.tensor([[0.0, 0.8, 0.0, 0.0]] * NUM_ENVS, device=DEVICE)),
        ("vz=+0.5", torch.tensor([[0.0, 0.0, 0.5, 0.0]] * NUM_ENVS, device=DEVICE)),
        ("随机方向", (torch.rand((NUM_ENVS, 4), device=DEVICE) * 2 - 1) * torch.tensor([0.8, 0.8, 0.5, 0.524], device=DEVICE)),
    ]

    vel_pass = True
    for name, vel_actions in vel_tests:
        cr = run_stability_test(env_manager, vel_actions, VEL_STEPS, f"速度跟踪 {name}")
        if cr > MAX_CRASH_RATE:
            logger.warning(f"  ⚠️  {name} 测试失败，尝试进一步降低 K_vel...")
            vel_pass = False
            # 进一步降低 K_vel
            current_idx = K_VEL_CANDIDATES.index(best_scale) if best_scale in K_VEL_CANDIDATES else 0
            for scale in K_VEL_CANDIDATES[current_idx + 1:]:
                set_controller_k_vel(controller, scale, base_K_vel, NUM_ENVS, DEVICE)
                cr2 = run_stability_test(env_manager, vel_actions, VEL_STEPS, f"  ↳ {name}@{scale:.2f}x")
                if cr2 <= MAX_CRASH_RATE:
                    best_scale = scale
                    vel_pass = True
                    logger.warning(f"  ✅ {name} 稳定通过，使用 K_vel 缩放 = {scale:.2f}")
                    break
            if not vel_pass:
                logger.warning(f"  ❌ 无法使 {name} 稳定。考虑降低 max_inclination_angle_rad 或调整 K_rot。")
                break

    # =============================================
    # 输出结果并写回配置
    # =============================================
    logger.warning("")
    logger.warning("=" * 60)
    final_K_vel = set_controller_k_vel(controller, best_scale, base_K_vel, NUM_ENVS, DEVICE)
    logger.warning(f"  🎯 最优 K_vel 缩放因子: {best_scale:.2f}")
    logger.warning(f"  🎯 最优 K_vel 值: {[f'{v:.4f}' for v in final_K_vel]}")
    
    if vel_pass:
        logger.warning("  📊 结论: ✅ 控制器稳定，可用于 RL 训练")
    else:
        logger.warning("  📊 结论: ⚠️  部分速度方向仍不稳定，建议同时调低 K_angvel 和增大 K_rot")

    logger.warning("=" * 60)
    logger.warning("")
    logger.warning("正在将最优增益写回配置文件...")
    write_gains_to_config(final_K_vel)

    logger.warning("")
    logger.warning("调参完成！下次运行 position_control_example.py 即使用新参数。")


if __name__ == "__main__":
    main()
