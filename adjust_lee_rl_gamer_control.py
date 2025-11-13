"""Tune Lee controller gains with rl-games PPO and multi-env Isaac Gym.
CUDA_VISIBLE_DEVICES=0 python -m aerial_gym.examples.adjust_lee_rl_gamer_control   --device cuda:0   --sim-envs 128   --num-workers 1   --eval-horizon 800   --max-epochs 200   --metric-log logs/stage2_1.0kg_metrics.jsonl   --history-path logs/stage2_1.0kg_eval.jsonl   --checkpoint-dir checkpoints/stage2_1.0kg_fixbug   --load-checkpoint checkpoints/stage2_1.0kg/lee_gain_tune/nn/lee_gain_tune_run.pth 


"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys

# 确保 PyTorch 在 Isaac Gym 之后导入，避免 gymdeps 检查失败
from isaacgym import gymapi  # noqa: F401

import gym
from gym import spaces
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# 兼容旧版依赖（如 networkx）仍引用 NumPy 早期别名的情况
if not hasattr(np, "int"):  # pragma: no cover - 仅针对旧依赖补丁
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz_tensor,
    ssa,
)


from tqdm import tqdm

gym.logger.set_level(gym.logger.ERROR)

DEBUG_LOG = False          # 开/关总开关
DEBUG_EVERY_N_STEPS = 1  # 每 N 步采样一次，别太频繁
DEBUG_ENV_IDX = 0         # 只看第 0 个环境，避免 1000+ env 同时打印
DT = 0.01                 # 你的仿真步长

# ---------------------------- 常量与配置 ---------------------------- #
'''
CUDA_VISIBLE_DEVICES=0 python -m aerial_gym.examples.adjust_lee_rl_gamer_control \
  --device cuda:0 \
  --sim-envs 8192 \
  --num-workers 1 \
  --eval-horizon 900 \
  --max-epochs 200 \
  --metric-log logs/stage2_1.0kg_metrics.jsonl \
  --history-path logs/stage2_1.0kg_eval.jsonl \
  --checkpoint-dir checkpoints/stage2_1.0kg \
  --load-checkpoint "checkpoints/stage1_0.5kg/lee_gain_tune/nn/lee_gain_tune_run.pth"
'''
# -- 基础环境配置 --
SIM_NAME = "base_sim"
ENV_NAME = "empty_env"
ROBOT_NAME = "lmf1"
CONTROLLER_NAME = "lmf2_velocity_control"

# -- 强化学习训练默认值 --
DEFAULT_EVAL_HORIZON = 2500     # rollout 步数，越大评估越长
DEFAULT_TOTAL_TIMESTEPS = 65536   # 估算训练总步数，用于推断 max_epochs
DEFAULT_NUM_WORKERS = 1        # rl-games 并行 actor 数量
DEFAULT_SIM_NUM_ENVS = 256     # Isaac Gym 中的并行无人机数量
DEFAULT_MAX_EPOCHS = 128       # rl-games 训练的最大 epoch 数
DEFAULT_EVAL_ROLLOUTS = 2      # 每次评估重复 rollout 次数，降低指标噪声

# -- 仿真物理/控制参数 --
DEFAULT_THRUST_MARGIN = 2.5     # 推力裕度，决定最大推力倍数
DEFAULT_HEADLESS = False         # True=关闭查看器，False=开启可视化
DEFAULT_USE_WARP = True         # Warp 物理加速开关
DEFAULT_THRUST_LOG = False      # 是否打印推力调节日志

# -- 训练日志与检查点 --
DEFAULT_EXPERIMENT_NAME = "lee_gain_tune"
DEFAULT_HISTORY_PATH = "rlg_eval_history.jsonl"
DEFAULT_METRIC_LOG_PATH = "rlg_metric_log.jsonl"
DEFAULT_CHECKPOINT_DIR = "rlg_checkpoints"

# -- 任务场景配置 --
RELEASE_PLAN: List[Tuple[int, str]] = []


OBS_DIM = 8

PARAM_SPECS = [
    ("K_pos_xyz", 3, [6, 20]),
    ("K_vel_xyz", 3, [6, 20]),
    ("K_rot_xyz", 3, [7,20]),
    ("K_angvel_xyz", 3, [1, 10]),
]


LOSS_WEIGHTS = {
    "pos_rmse": 1.2,
    "att_rmse": 0.6,
    "force_mean": 0.2,
    "final_error": 2.0,
    "settle_steps": 0.05,
    "pos_peak": 2.0,
    "att_peak": 1.6,
    "pos_osc": 0.8,
    "att_osc": 0.7,
}
# Settling detection tuned for smooth gradients instead of二元跳变
SETTLE_THRESHOLD = 0.95          # 稳态误差上界
SETTLE_WINDOW = 25               # 阈值内连续步数
SETTLE_BURN_IN = 40              # burn-in 时长，避免初始值破坏判定
SETTLE_HYSTERESIS = 0.15         # 需要从高误差下降多少才认为稳定
OSC_BURN_IN = 60                 # 计算振荡惩罚时忽略的步数，避免初期噪声
# 若 custom/best_reward 连续上升、pos/att rmse 明显降后：
# 再切回 final_error=1.2~1.5 做精调

# 训练监控提示：
# - Policy 表现：reward = -loss，best_payload 会记录 pos/att RMSE、力均值、最终误差，趋势向好代表收敛。
# - 损失曲线：losses/a_loss、losses/c_loss、losses/entropy 写入 metric_log，可配合 plot_loss_curves() 评估训练。
# - 方差与稳定性：history_path 记录每次评估指标，可离线分析 reward 波动或绘制箱线图。
# - 策略评估：best_lee_gains_rlg.json 与 checkpoint_dir 中的权重可用于重放、恢复意外中断的训练。

# 子机参数（质量 / 偏移）
PAYLOAD_LAYOUT: List[Tuple[str, float, torch.Tensor]] = []


def adjust_motor_thrust_limits(env_manager, margin: float = 1.0, log: bool = False) -> None:
    """
    根据给定裕度调整电机推力上限。首次调用时会缓存原始推力上限，之后均基于原始值缩放。
    """
    robot = env_manager.robot_manager.robot
    motor_model = robot.control_allocator.motor_model
    if not hasattr(motor_model, "_baseline_max_thrust"):
        motor_model._baseline_max_thrust = motor_model.max_thrust.clone().detach()

    baseline = motor_model._baseline_max_thrust
    new_limit = (baseline * float(margin)).clone()
    motor_model.max_thrust = new_limit

    if log:
        base_val = float(baseline[0, 0].item())
        new_val = float(new_limit[0, 0].item())
        print(
            f"[adjust_motor_thrust_limits] max_thrust: {base_val:.2f} -> {new_val:.2f} (margin={margin})"
        )


def _ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ---------------------------- 数据结构 ---------------------------- #

@dataclass
class Payload:
    name: str
    mass: float
    offset: torch.Tensor


class MetricRecorder:
    """Append scalar metrics to a JSONL file for post-run analysis."""

    def __init__(self, path: Optional[str]):
        self.path = path
        if self.path:
            _ensure_parent_dir(self.path)
            # 创建/清空文件头部，防止旧内容混淆
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def log(self, tag: str, value: float, step: Optional[int] = None, walltime: Optional[float] = None):
        if self.path is None or value is None:
            return
        entry = {
            "tag": tag,
            "value": float(value),
            "step": int(step) if step is not None else None,
            "walltime": float(walltime) if walltime is not None else None,
            "timestamp": time.time(),
        }
        with open(self.path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


class ResultsTracker:
    """记录评估指标，并在出现更优策略时保存，同时推送到 TensorBoard。"""

    def __init__(
        self,
        best_path: str,
        history_path: Optional[str] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        self.best_path = best_path
        self.history_path = history_path
        _ensure_parent_dir(self.best_path)
        if self.history_path:
            _ensure_parent_dir(self.history_path)
        self.best_payload: Optional[Dict[str, object]] = None
        self.writer = writer
        self.eval_count = 0

    def update(self, gains: np.ndarray, metrics: Dict[str, float]):
        entry = {
            "reward": float(-metrics["loss"]),
            "gains": gains.astype(float).tolist(),
            "timestamp": time.time(),
            **{k: float(v) for k, v in metrics.items()},
        }
        is_best = self.best_payload is None or entry["reward"] > self.best_payload["reward"]
        entry["is_best"] = is_best
        self._append_history(entry)
        self.eval_count += 1
        self._log_eval_metrics(entry)
        if is_best:
            self.best_payload = entry
            try:
                with open(self.best_path, "w", encoding="utf-8") as fp:
                    json.dump(entry, fp, indent=2, ensure_ascii=False)
            except OSError as exc:
                print(f"[adjust_lee_rl_gamer_control] 写入 {self.best_path} 失败: {exc}")
            self._log_best_metrics(entry)

    def _append_history(self, payload: Dict[str, object]):
        if not self.history_path:
            return
        try:
            with open(self.history_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[adjust_lee_rl_gamer_control] 写入 {self.history_path} 失败: {exc}")

    def attach_writer(self, writer: Optional[SummaryWriter]):
        self.writer = writer

    def _log_eval_metrics(self, entry: Dict[str, object]):
        if not self.writer:
            return
        step = self.eval_count
        scalar_map = [
            ("eval/reward", "reward"),
            ("eval/loss", "loss"),
            ("eval/pos_rmse", "pos_rmse"),
            ("eval/att_rmse", "att_rmse"),
            ("eval/force_mean", "force_mean"),
            ("eval/final_error", "final_error"),
            ("eval/settle_steps", "settle_steps"),
            ("eval/pos_peak", "pos_peak"),
            ("eval/att_peak", "att_peak"),
            ("eval/pos_osc", "pos_osc"),
            ("eval/att_osc", "att_osc"),
            ("eval/thrust_saturation_mean", "thrust_saturation_mean"),
            ("eval/thrust_max_norm", "thrust_max_norm"),
        ]
        for tag, key in scalar_map:
            value = entry.get(key)
            if value is None:
                continue
            try:
                self.writer.add_scalar(tag, float(value), step)
            except Exception as exc:
                print(f"[ResultsTracker] 写入 TensorBoard {tag} 失败: {exc}")

    def _log_best_metrics(self, entry: Dict[str, object]):
        if not self.writer:
            return
        step = self.eval_count
        scalar_map = [
            ("custom/best_reward", "reward"),
            ("custom/best_pos_rmse", "pos_rmse"),
            ("custom/best_att_rmse", "att_rmse"),
            ("custom/best_final_error", "final_error"),
            ("custom/best_settle_steps", "settle_steps"),
            ("custom/best_pos_peak", "pos_peak"),
            ("custom/best_att_peak", "att_peak"),
            ("custom/best_pos_osc", "pos_osc"),
            ("custom/best_att_osc", "att_osc"),
            ("custom/best_thrust_saturation", "thrust_saturation_mean"),
            ("custom/best_thrust_max_norm", "thrust_max_norm"),
        ]
        for tag, key in scalar_map:
            value = entry.get(key)
            if value is None:
                continue
            try:
                self.writer.add_scalar(tag, float(value), step)
            except Exception as exc:
                print(f"[ResultsTracker] 写入 TensorBoard {tag} 失败: {exc}")


class ProgressObserver(AlgoObserver):
    """
    自定义 observer：使用 tqdm 展示训练进度，并记录损失曲线以便会后绘图。
    """

    def __init__(self, total_epochs: int, tracker: ResultsTracker, metric_recorder: Optional[MetricRecorder] = None):
        super().__init__()
        self.requested_total = int(total_epochs) if total_epochs and total_epochs > 0 else None
        self.tracker = tracker
        self.metric_recorder = metric_recorder
        self.pbar = None
        self.algo = None
        self.loss_history = {}
        self._wrapped_writer = False
        self._original_add_scalar = None
        self._last_epoch = 0
        self._last_logged_best_epoch = 0

    def before_init(self, base_name, config, experiment_name):
        if self.requested_total is None:
            max_epochs = config.get("max_epochs", -1)
            if isinstance(max_epochs, int) and max_epochs > 0:
                self.requested_total = max_epochs

    def after_init(self, algo):
        self.algo = algo
        algo_max_epochs = getattr(algo, "max_epochs", -1)
        total = None
        if isinstance(algo_max_epochs, int) and algo_max_epochs > 0:
            total = algo_max_epochs
        elif self.requested_total:
            total = self.requested_total

        self.pbar = tqdm(total=total, desc="rl-games training", unit="epoch")
        self._wrap_writer()

    def _wrap_writer(self):
        if self._wrapped_writer or self.algo is None:
            return
        writer = getattr(self.algo, "writer", None)
        if writer is None or not hasattr(writer, "add_scalar"):
            return

        original_add_scalar = writer.add_scalar

        def add_scalar(tag, scalar_value, *args, **kwargs):
            global_step = None
            walltime = None
            if args:
                global_step = args[0]
            if len(args) > 1:
                walltime = args[1]
            global_step = kwargs.pop("global_step", global_step)
            walltime = kwargs.pop("walltime", walltime)
            kwargs.pop("new_style", None)
            kwargs.pop("double_precision", None)
            value = scalar_value
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            try:
                value_f = float(value)
            except Exception:
                value_f = None

            if value_f is not None and tag in ("losses/a_loss", "losses/c_loss", "losses/entropy"):
                epoch = getattr(self.algo, "epoch_num", 0)
                self.loss_history.setdefault(tag, []).append((epoch, value_f))

            call_kwargs = {}
            if global_step is not None:
                call_kwargs["global_step"] = global_step
            if walltime is not None:
                call_kwargs["walltime"] = walltime
            call_kwargs.update(kwargs)

            try:
                result = original_add_scalar(tag, value, **call_kwargs)
            except TypeError:
                result = original_add_scalar(tag, value)

            if self.metric_recorder and value_f is not None:
                step_int = None
                if global_step is not None:
                    try:
                        step_int = int(global_step)
                    except Exception:
                        step_int = None
                self.metric_recorder.log(tag, value_f, step_int, walltime)
            return result

        writer.add_scalar = add_scalar
        self._original_add_scalar = original_add_scalar
        self._wrapped_writer = True

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.pbar is None:
            return
        if epoch_num <= self._last_epoch:
            return

        delta = epoch_num - self._last_epoch
        self.pbar.update(delta)
        self._last_epoch = epoch_num

        postfix = {}
        if self.loss_history.get("losses/a_loss"):
            postfix["actor_loss"] = f"{self.loss_history['losses/a_loss'][-1][1]:.4f}"
        if self.loss_history.get("losses/c_loss"):
            postfix["critic_loss"] = f"{self.loss_history['losses/c_loss'][-1][1]:.4f}"
        if self.loss_history.get("losses/entropy"):
            postfix["entropy"] = f"{self.loss_history['losses/entropy'][-1][1]:.4f}"

        if self.tracker and self.tracker.best_payload:
            postfix["best_reward"] = f"{self.tracker.best_payload['reward']:.4f}"
            self._log_best_metrics(epoch_num)

        if postfix:
            self.pbar.set_postfix(postfix)

    def close(self):
        if getattr(self, "pbar", None) is not None:
            self.pbar.close()
            self.pbar = None
        if self._wrapped_writer and self.algo and getattr(self.algo, "writer", None):
            self.algo.writer.add_scalar = self._original_add_scalar
            self._wrapped_writer = False
            self._original_add_scalar = None

    def _log_best_metrics(self, epoch_num: int):
        if not self.tracker or not self.tracker.best_payload:
            return
        if epoch_num <= self._last_logged_best_epoch:
            return
        writer = getattr(self.algo, "writer", None)
        if writer is None:
            return
        best = self.tracker.best_payload
        writer.add_scalar("custom/best_reward", best["reward"], epoch_num)
        writer.add_scalar("custom/best_pos_rmse", best["pos_rmse"], epoch_num)
        writer.add_scalar("custom/best_final_error", best["final_error"], epoch_num)
        if "thrust_saturation_mean" in best:
            writer.add_scalar("custom/best_thrust_saturation", best["thrust_saturation_mean"], epoch_num)
        if "thrust_max_norm" in best:
            writer.add_scalar("custom/best_thrust_max_norm", best["thrust_max_norm"], epoch_num)
        if self.metric_recorder:
            self.metric_recorder.log("custom/best_reward", best["reward"], epoch_num, None)
            self.metric_recorder.log("custom/best_pos_rmse", best["pos_rmse"], epoch_num, None)
            self.metric_recorder.log("custom/best_final_error", best["final_error"], epoch_num, None)
            if "thrust_saturation_mean" in best:
                self.metric_recorder.log(
                    "custom/best_thrust_saturation", best["thrust_saturation_mean"], epoch_num, None
                )
            if "thrust_max_norm" in best:
                self.metric_recorder.log(
                    "custom/best_thrust_max_norm", best["thrust_max_norm"], epoch_num, None
                )
        self._last_logged_best_epoch = epoch_num

    def get_loss_history(self) -> Dict[str, List[Tuple[int, float]]]:
        return {tag: list(values) for tag, values in self.loss_history.items()}

    def __del__(self):
        self.close()


# ---------------------------- 负载管理器 ---------------------------- #

class SimplePayloadManager:
    """
    与 adjust_lee_control 中一致：负责更新总质量 / 惯量，并在释放时施加冲击。
    """

    def __init__(self, env_manager, payloads: List[Payload], device: torch.device, thrust_logging: bool):
        self.env_manager = env_manager
        self.device = device
        self.payloads: Dict[str, Payload] = {p.name: p for p in payloads}
        self.attached = {p.name: True for p in payloads}
        self.thrust_logging = thrust_logging

        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles = list(env_manager.IGE_env.env_handles)
        self.robot_handles = list(env_manager.robot_manager.robot_handles)

        self.body_props_per_env = [
            self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            for env_handle, robot_handle in zip(self.env_handles, self.robot_handles)
        ]

        base_prop = self.body_props_per_env[0][0]
        self.base_mass = base_prop.mass
        self.base_com = torch.tensor(
            [base_prop.com.x, base_prop.com.y, base_prop.com.z],
            device=self.device,
            dtype=torch.float32,
        )
        self.base_inertia = torch.tensor(
            [
                [base_prop.inertia.x.x, base_prop.inertia.x.y, base_prop.inertia.x.z],
                [base_prop.inertia.y.x, base_prop.inertia.y.y, base_prop.inertia.y.z],
                [base_prop.inertia.z.x, base_prop.inertia.z.y, base_prop.inertia.z.z],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.pending_impulses: List[Dict[str, torch.Tensor]] = []
        self.apply_to_sim()

    # def _compute_total_properties(self):
    #     total_mass = torch.tensor(self.base_mass, device=self.device)
    #     numerator_com = self.base_mass * self.base_com.clone()

    #     for name, payload in self.payloads.items():
    #         if not self.attached[name]:
    #             continue
    #         total_mass += payload.mass
    #         numerator_com += payload.mass * payload.offset.to(self.device)

    #     combined_com = torch.zeros(3, device=self.device)
    #     if total_mass.item() > 1e-6:
    #         combined_com[:] = numerator_com / total_mass

    #     identity = torch.eye(3, device=self.device)
    #     inertia_total = self.base_inertia.clone()
    #     d_base = self.base_com - combined_com
    #     inertia_total += self.base_mass * (
    #         torch.dot(d_base, d_base) * identity - torch.outer(d_base, d_base)
    #     )

    #     for name, payload in self.payloads.items():
    #         if not self.attached[name]:
    #             continue
    #         r = payload.offset.to(self.device) - combined_com
    #         inertia_total += payload.mass * (torch.dot(r, r) * identity - torch.outer(r, r))

    #     return total_mass, inertia_total, combined_com

    # def apply_to_sim(self):
    #     total_mass, inertia_total, combined_com = self._compute_total_properties()

    #     inertia_np = inertia_total.detach().cpu().numpy()
    #     for env_handle, robot_handle, props in zip(
    #         self.env_handles, self.robot_handles, self.body_props_per_env
    #     ):
    #         base_prop = props[0]
    #         base_prop.mass = float(total_mass.item())
    #         base_prop.com = torch_to_vec3(combined_com)
    #         base_prop.inertia = tensor33_to_mat33(inertia_np)
    #         self.gym.set_actor_rigid_body_properties(
    #             env_handle, robot_handle, props, recomputeInertia=False
    #         )

    #     env = self.env_manager
    #     env.robot_manager.robot_mass = float(total_mass.item())
    #     env.robot_manager.robot_masses.fill_(float(total_mass.item()))
    #     env.IGE_env.global_tensor_dict["robot_mass"].fill_(float(total_mass.item()))

    #     inertia_tensor = torch.tensor(inertia_np, device=self.device, dtype=torch.float32)
    #     env.robot_manager.robot_inertia = inertia_tensor
    #     env.robot_manager.robot_inertias[:] = inertia_tensor
    #     env.IGE_env.global_tensor_dict["robot_inertia"][:] = inertia_tensor

    #     controller_mass = torch.full((env.num_envs, 1), float(total_mass.item()), device=self.device)
    #     env.robot_manager.robot.controller.mass = controller_mass

    #     adjust_motor_thrust_limits(
    #         env_manager=self.env_manager,
    #         margin=DEFAULT_THRUST_MARGIN,
    #         log=self.thrust_logging,
    #     )

    # def release(self, name: str):
    #     if name not in self.payloads or not self.attached[name]:
    #         return

    #     payload = self.payloads[name]
    #     prev_mass = self.current_mass()
    #     offset = payload.offset.to(self.device)
    #     self.attached[name] = False
    #     self.apply_to_sim()

    #     new_mass = self.current_mass()
    #     root_state = self.env_manager.IGE_env.global_tensor_dict["vec_root_tensor"]
    #     base_states = root_state[:, 0] if root_state.ndim == 3 else root_state
    #     if new_mass > 1e-5:
    #         scale = prev_mass / new_mass
    #         base_states[:, 7:10] *= scale
    #     self.env_manager.IGE_env.write_to_sim()

    #     gravity_vec = self.env_manager.IGE_env.global_tensor_dict["gravity"][0].to(self.device)
    #     torque_impulse = -payload.mass * torch.cross(offset, gravity_vec)
    #     if torch.linalg.norm(torque_impulse).item() > 1e-6:
    #         self.pending_impulses.append(
    #             {"torque": torque_impulse, "steps": torch.tensor(45, device=self.device)}
    #         )

    # def consume_impulse(self) -> torch.Tensor:
    #     if not self.pending_impulses:
    #         return torch.zeros(3, device=self.device)

    #     total = torch.zeros(3, device=self.device)
    #     remaining: List[Dict[str, torch.Tensor]] = []
    #     for item in self.pending_impulses:
    #         total += item["torque"]
    #         steps_left = item["steps"] - 1
    #         if steps_left.item() > 0:
    #             item["steps"] = steps_left
    #             remaining.append(item)
    #     self.pending_impulses = remaining
    #     return total

    def current_mass(self) -> float:
        env = self.env_manager
        return float(env.robot_manager.robot_mass)

    def reset(self):
        self.attached = {name: True for name in self.payloads}
        self.pending_impulses.clear()
        self.apply_to_sim()


# ---------------------------- 工具函数 ---------------------------- #

def torch_to_vec3(t: torch.Tensor) -> gymapi.Vec3:
    return gymapi.Vec3(float(t[0].item()), float(t[1].item()), float(t[2].item()))


def tensor33_to_mat33(m: np.ndarray) -> gymapi.Mat33:
    mat = gymapi.Mat33()
    mat.x = gymapi.Vec3(*m[0])
    mat.y = gymapi.Vec3(*m[1])
    mat.z = gymapi.Vec3(*m[2])
    return mat


# ---------------------------- 单环境定义 ---------------------------- #

class LeeGainTuningEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        device: str = None,
        eval_horizon: int = DEFAULT_EVAL_HORIZON,
        eval_rollouts: int = DEFAULT_EVAL_ROLLOUTS,
        release_plan: List[Tuple[int, str]] = None,
        tracker: ResultsTracker = None,
        **kwargs,
    ):
        super().__init__()
        device_override = kwargs.pop("device", None)
        self.headless = bool(kwargs.pop("headless", DEFAULT_HEADLESS))
        self.use_warp = bool(kwargs.pop("use_warp", DEFAULT_USE_WARP))
        sim_num_envs = kwargs.pop("sim_num_envs", None)
        if sim_num_envs is None:
            sim_num_envs = kwargs.pop("num_envs", 1)
        self.sim_num_envs = max(1, int(sim_num_envs))
        if device_override is not None:
            device = device_override
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if isinstance(device, torch.device):
            device = device.type + (f":{device.index}" if device.index is not None else "")
        self.device_str = device
        self.device = torch.device(self.device_str)
        self.eval_horizon = eval_horizon
        self.eval_rollouts = max(1, int(eval_rollouts))
        self.release_plan = release_plan or []
        self.tracker = tracker

        self.param_count = sum(spec[1] for spec in PARAM_SPECS)
        self.obs_dim = OBS_DIM
        self.action_space = spaces.Box(-np.ones(self.param_count), np.ones(self.param_count), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.obs_dim, dtype=np.float32),
            high=np.inf * np.ones(self.obs_dim, dtype=np.float32),
        )

        self.sim_builder = SimBuilder()
        self.env_manager = self.sim_builder.build_env(
            sim_name=SIM_NAME,
            env_name=ENV_NAME,
            robot_name=ROBOT_NAME,
            controller_name=CONTROLLER_NAME,
            args=None,
            device=self.device_str,
            num_envs=self.sim_num_envs,
            headless=self.headless,
            use_warp=self.use_warp,
        )
        payloads = [Payload(name=n, mass=m, offset=offset) for n, m, offset in PAYLOAD_LAYOUT]
        self.payload_manager = (
            SimplePayloadManager(
                self.env_manager,
                payloads,
                self.device,
                thrust_logging=DEFAULT_THRUST_LOG,
            )
            if payloads and self.release_plan
            else None
        )
        self.controller = self.env_manager.robot_manager.robot.controller
        num_actions = self.env_manager.robot_manager.robot.num_actions
        self.target_actions = torch.zeros(
            (self.env_manager.num_envs, num_actions), device=self.device, dtype=torch.float32
        )
        self.release_lookup = {step: name for step, name in self.release_plan}
        self._closed = False
        self._recent_metrics = np.zeros(6, dtype=np.float32)
        self.episode_len = kwargs.pop("episode_len", 5)
        self.curr_step = 0
        self.prev_metrics: Optional[Dict[str, float]] = None

    def reset(self):
        self.env_manager.reset()
        if self.payload_manager is not None:
            self.payload_manager.reset()
        self.env_manager.reset_tensors()
        self.target_actions.zero_()
        self.curr_step = 0
        self.prev_metrics = None
        return self._formatted_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not np.isfinite(action).all():
            action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, -0.95, 0.95)

        gains = self._denormalize(action)
        metrics = self._evaluate(gains)
        if not np.all(np.isfinite(list(metrics.values()))):
            metrics = {
                "pos_rmse": 1e3,
                "att_rmse": 1e3,
                "force_mean": 1e3,
                "final_error": 1e3,
                "loss": 1e6,
            }

        if self.tracker is not None:
            self.tracker.update(gains, metrics)
        reward = self._compute_segment_reward(metrics)
        self._recent_metrics = np.array(
            [
                metrics["pos_rmse"],
                metrics["att_rmse"],
                metrics["force_mean"],
                metrics["final_error"],
                metrics["thrust_saturation_mean"],
                metrics["thrust_max_norm"],
            ],
            dtype=np.float32,
        )
        obs = self._formatted_obs()
        self.curr_step += 1
        done = self.curr_step >= self.episode_len
        if done:
            self.curr_step = 0
        return obs, float(reward), done, metrics

    # -------------------- 评估逻辑（复用 adjust_lee_control） -------------------- #

    def _evaluate(self, gains: np.ndarray) -> Dict[str, float]:
        self._apply_gains(self.controller, gains)
        metrics_acc: Optional[Dict[str, float]] = None
        for _ in range(self.eval_rollouts):
            self.env_manager.reset()
            if self.payload_manager is not None:
                self.payload_manager.reset()
            self.env_manager.reset_tensors()
            self.target_actions.zero_()
            metrics = self._rollout()
            if metrics_acc is None:
                metrics_acc = {k: float(v) for k, v in metrics.items()}
            else:
                for key, value in metrics.items():
                    metrics_acc[key] = metrics_acc.get(key, 0.0) + float(value)
        assert metrics_acc is not None
        if self.eval_rollouts == 1:
            return metrics_acc
        return {k: v / float(self.eval_rollouts) for k, v in metrics_acc.items()}

    def _apply_gains(self, controller, gains: np.ndarray):
        ptr = 0

        def _assign(attr: str, value_row: torch.Tensor):
            current = getattr(controller, attr)
            expanded = value_row.expand_as(current).clone()
            setattr(controller, attr, expanded)

        for name, dim, bounds in PARAM_SPECS:
            segment = gains[ptr : ptr + dim]
            ptr += dim
            actual = segment
            tensor = torch.tensor(actual, device=self.device, dtype=torch.float32).view(1, -1)

            if name.startswith("K_pos"):
                _assign("K_pos_tensor_current", tensor)
                _assign("K_pos_tensor_min", tensor)
                _assign("K_pos_tensor_max", tensor)
            elif name.startswith("K_vel"):
                _assign("K_linvel_tensor_current", tensor)
                _assign("K_linvel_tensor_min", tensor)
                _assign("K_linvel_tensor_max", tensor)
            elif name.startswith("K_rot"):
                _assign("K_rot_tensor_current", tensor)
                _assign("K_rot_tensor_min", tensor)
                _assign("K_rot_tensor_max", tensor)
            elif name.startswith("K_angvel"):
                _assign("K_angvel_tensor_current", tensor)
                _assign("K_angvel_tensor_min", tensor)
                _assign("K_angvel_tensor_max", tensor)

            if DEBUG_LOG:
                gains_dict = {
                    "K_pos": controller.K_pos_tensor_current[0].detach().cpu().tolist(),
                    "K_vel": controller.K_linvel_tensor_current[0].detach().cpu().tolist(),
                    "K_rot": controller.K_rot_tensor_current[0].detach().cpu().tolist(),
                    "K_angvel": controller.K_angvel_tensor_current[0].detach().cpu().tolist(),
                }
                print(
                    f"[GAINS] K_pos={gains_dict['K_pos']}, K_vel={gains_dict['K_vel']}, "
                    f"K_rot={gains_dict['K_rot']}, K_angvel={gains_dict['K_angvel']}"
                )


    def _rollout(self) -> Dict[str, float]:
        env_manager = self.env_manager
        payload_manager = self.payload_manager
        release_lookup = self.release_lookup if payload_manager is not None else {}
        crash_detected = False

        pos_errors = []
        att_errors = []
        control_effort = []
        pos_error_deltas: List[float] = []
        att_error_deltas: List[float] = []
        thrust_sat_history: List[float] = []
        thrust_max_history: List[float] = []
        pos_peak = 0.0
        att_peak = 0.0
        settle_step = None
        best_window_mean: Optional[float] = None
        seen_high_error = False
        motor_model = env_manager.robot_manager.robot.control_allocator.motor_model

        for step in range(self.eval_horizon):
            obs = env_manager.get_obs()
            orientations = obs["robot_orientation"]
            # 读取上一步速度（用于差分加速度），字段名按你的 env 保持一致
            # vec_root_tensor: [pos(0:3), quat(3:7), lin_vel(7:10), ang_vel(10:13), ...]
            root = env_manager.IGE_env.global_tensor_dict["vec_root_tensor"]
            root_this = root[:, 0] if root.ndim == 3 else root       # [num_envs, 13+]
            v_prev = root_this[:, 7:10].clone()
            w_prev = root_this[:, 10:13].clone()
            

            if payload_manager is not None and step in release_lookup:
                payload_manager.release(release_lookup[step])

            env_manager.reset_tensors()

            if payload_manager is not None:
                torque_world = payload_manager.consume_impulse()
                if torch.linalg.norm(torque_world).item() > 0.0:
                    torque_body = quat_rotate_inverse(
                        orientations,
                        torque_world.unsqueeze(0).expand(env_manager.num_envs, -1),
                    )
                    env_manager.IGE_env.global_tensor_dict["robot_torque_tensor"][:, 0, :] += torque_body

            env_manager.step(actions=self.target_actions)
            with torch.no_grad():
                thrusts = motor_model.current_motor_thrust
                max_thrust = motor_model.max_thrust
                normalized = torch.where(max_thrust > 1e-6, thrusts / max_thrust, torch.zeros_like(thrusts))
                thrust_sat_history.append(float((normalized >= 0.999).float().mean().item()))
                thrust_max_history.append(float(normalized.max().item()))
            reset_envs = env_manager.post_reward_calculation_step()
            reset_count = int(reset_envs.numel()) if isinstance(reset_envs, torch.Tensor) else len(reset_envs)
            if reset_count > 0:
                crash_detected = True
                break

            obs_after = env_manager.get_obs()
            
            pos = obs_after["robot_position"]
            att = ssa(get_euler_xyz_tensor(obs_after["robot_orientation"]))
            forces = env_manager.IGE_env.global_tensor_dict["robot_force_tensor"]

            if (
                not torch.isfinite(pos).all()
                or not torch.isfinite(att).all()
                or not torch.isfinite(forces).all()
            ):
                crash_detected = True
                break

            pos_errors.append(torch.norm(pos, dim=1).mean())
            att_errors.append(torch.norm(att, dim=1).mean())
            control_effort.append(torch.norm(forces, dim=-1).mean())
            if len(pos_errors) >= 2:
                delta = torch.abs(pos_errors[-1] - pos_errors[-2]).item()
                if step >= OSC_BURN_IN:
                    pos_error_deltas.append(float(delta))
            if len(att_errors) >= 2:
                delta_att = torch.abs(att_errors[-1] - att_errors[-2]).item()
                if step >= OSC_BURN_IN:
                    att_error_deltas.append(float(delta_att))
            if len(pos_errors) >= SETTLE_WINDOW:
                window_tensor = torch.stack(pos_errors[-SETTLE_WINDOW:])
                window_mean = float(window_tensor.mean().item())
                if best_window_mean is None or window_mean < best_window_mean:
                    best_window_mean = window_mean
                if window_mean > (SETTLE_THRESHOLD + SETTLE_HYSTERESIS):
                    seen_high_error = True
                if (
                    settle_step is None
                    and step >= SETTLE_BURN_IN
                    and seen_high_error
                    and window_mean < SETTLE_THRESHOLD
                ):
                    settle_step = step - SETTLE_WINDOW // 2
            pos_peak = max(pos_peak, float(pos_errors[-1].item()))
            att_peak = max(att_peak, float(att_errors[-1].item()))

        if crash_detected or not pos_errors:
            return {
                "pos_rmse": 1e3,
                "att_rmse": 1e3,
                "force_mean": 1e3,
                "final_error": 1e3,
                "thrust_saturation_mean": 1.0,
                "thrust_max_norm": 1.0,
                "settle_steps": float(self.eval_horizon),
                "pos_peak": 1e3,
                "att_peak": 1e3,
                "loss": 1e6,
            }

        pos_rmse = torch.stack(pos_errors).mean().item()
        att_rmse = torch.stack(att_errors).mean().item()
        force_mean = torch.stack(control_effort).mean().item()
        final_error = pos_errors[-1].item()
        pos_osc = float(np.mean(pos_error_deltas)) if pos_error_deltas else 0.0
        att_osc = float(np.mean(att_error_deltas)) if att_error_deltas else 0.0
        thrust_sat_mean = float(np.mean(thrust_sat_history)) if thrust_sat_history else 0.0
        thrust_max_norm = float(np.max(thrust_max_history)) if thrust_max_history else 0.0
        if settle_step is None:
            if best_window_mean is not None:
                hysteresis_span = max(SETTLE_HYSTERESIS, 1e-6)
                progress = np.clip(
                    (SETTLE_THRESHOLD + SETTLE_HYSTERESIS - best_window_mean) / hysteresis_span,
                    0.0,
                    1.0,
                )
                dynamic_range = max(self.eval_horizon - SETTLE_BURN_IN, 1.0)
                settle_steps = float(self.eval_horizon - progress * dynamic_range)
            else:
                settle_steps = float(self.eval_horizon)
        else:
            settle_steps = float(max(settle_step, SETTLE_BURN_IN))

        loss = (
            LOSS_WEIGHTS["pos_rmse"] * pos_rmse
            + LOSS_WEIGHTS["att_rmse"] * att_rmse
            + LOSS_WEIGHTS["force_mean"] * force_mean
            + LOSS_WEIGHTS["final_error"] * final_error
            + LOSS_WEIGHTS["settle_steps"] * settle_steps
            + LOSS_WEIGHTS["pos_peak"] * pos_peak
            + LOSS_WEIGHTS["att_peak"] * att_peak
            + LOSS_WEIGHTS["pos_osc"] * pos_osc
            + LOSS_WEIGHTS["att_osc"] * att_osc
        )

        return {
            "pos_rmse": pos_rmse,
            "att_rmse": att_rmse,
            "force_mean": force_mean,
            "final_error": final_error,
            "thrust_saturation_mean": thrust_sat_mean,
            "thrust_max_norm": thrust_max_norm,
            "settle_steps": settle_steps,
            "pos_peak": pos_peak,
            "att_peak": att_peak,
            "pos_osc": pos_osc,
            "att_osc": att_osc,
            "loss": loss,
        }

    def _compute_segment_reward(self, metrics: Dict[str, float]) -> float:
        if self.prev_metrics is None:
            reward = -metrics["loss"]
        else:
            pos_gain = self.prev_metrics["pos_rmse"] - metrics["pos_rmse"]
            att_gain = self.prev_metrics["att_rmse"] - metrics["att_rmse"]
            settle_gain = self.prev_metrics["settle_steps"] - metrics["settle_steps"]
            pos_osc_gain = self.prev_metrics.get("pos_osc", 0.0) - metrics.get("pos_osc", 0.0)
            att_osc_gain = self.prev_metrics.get("att_osc", 0.0) - metrics.get("att_osc", 0.0)
            reward = (
                1.2 * pos_gain
                + 0.8 * att_gain
                + 0.01 * settle_gain
                + 1.0 * pos_osc_gain
                + 0.8 * att_osc_gain
                - 0.0015 * metrics["force_mean"]
                - 0.001 * (metrics["pos_peak"] + metrics["att_peak"])
            )
            reward = float(np.clip(reward, -10.0, 10.0))
        self.prev_metrics = {k: float(metrics[k]) for k in metrics.keys()}
        return reward

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        gains = []
        ptr = 0
        for _, dim, bounds in PARAM_SPECS:
            segment = action[ptr : ptr + dim]
            ptr += dim
            low, high = bounds
            gains.append(((segment + 1.0) * 0.5) * (high - low) + low)
        return np.concatenate(gains, dtype=np.float32)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.sim_builder.delete_env()
        except Exception:
            pass
        self.env_manager = None
        self.payload_manager = None

    def __del__(self):
        self.close()

    def _formatted_obs(self) -> np.ndarray:
        metrics = np.tanh(
            self._recent_metrics / np.array([10, 10, 10, 10, 1, 1], dtype=np.float32)
        )
        noise_dim = self.obs_dim - metrics.shape[0]
        noise = (
            np.random.randn(noise_dim).astype(np.float32)
            if noise_dim > 0
            else np.empty(0, dtype=np.float32)
        )
        return np.concatenate([metrics, noise]).astype(np.float32)


# ---------------------------- 自定义 VecEnv ---------------------------- #

class LeeGainVecEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        creator = env_configurations.configurations[config_name]["env_creator"]
        self.envs = [creator(**kwargs) for _ in range(num_actors)]
        self.num_actors = num_actors

        sample_env = self.envs[0]
        self.action_space = sample_env.action_space
        self.observation_space = sample_env.observation_space

    # def step(self, actions):
    #     actions = np.asarray(actions)
    #     obs_list = []
    #     reward_list = []
    #     done_list = []
    #     info_list = []

    #     for env, act in zip(self.envs, actions):
    #         obs, reward, done, info = env.step(act)
    #         if done:
    #             obs_reset = env.reset()
    #             obs_list.append(obs_reset)
    #         else:
    #             obs_list.append(obs)
    #         reward_list.append(reward)
    #         done_list.append(done)
    #         info_list.append(info)

    #     obs_arr = np.stack(obs_list).astype(np.float32)
    #     reward_arr = np.asarray(reward_list, dtype=np.float32)
    #     done_arr = np.asarray(done_list, dtype=np.uint8)
    #     return obs_arr, reward_arr, done_arr, info_list
    def step(self, actions):
        actions = np.asarray(actions)
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        # --- 这是修复后的逻辑 ---
        # 无论 done 是 True 还是 False，
        # 我们都必须先把 env.step() 返回的真实 (obs, reward, done) 记录下来
        for env, act in zip(self.envs, actions):
            obs, reward, done, info = env.step(act)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        # --- 修复结束 ---

        obs_arr = np.stack(obs_list).astype(np.float32)
        reward_arr = np.asarray(reward_list, dtype=np.float32)
        done_arr = np.asarray(done_list, dtype=np.uint8)
        return obs_arr, reward_arr, done_arr, info_list
    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs).astype(np.float32)

    def reset_done(self):
        return self.reset()

    def get_number_of_agents(self):
        """
        获取代理(agent)的数量
        返回:
            int: 代理的数量，这里固定返回1
        """
        return 1

    def get_env_info(self):
        return {"action_space": self.action_space, "observation_space": self.observation_space}

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    def __del__(self):
        self.close()


# 注册环境
def register_env(tracker: ResultsTracker, eval_horizon: int, eval_rollouts: int, release_plan):
    env_configurations.register(
        "lee_gain_tune",
        {
            "env_creator": lambda **kwargs: LeeGainTuningEnv(
                tracker=tracker,
                eval_horizon=eval_horizon,
                eval_rollouts=eval_rollouts,
                release_plan=release_plan,
                **kwargs,
            ),
            "vecenv_type": "LEE-GAIN-VEC",
        },
    )

    vecenv.register(
        "LEE-GAIN-VEC",
        lambda config_name, num_actors, **kwargs: LeeGainVecEnv(config_name, num_actors, **kwargs),
    )


# ---------------------------- 可视化 ---------------------------- #

def plot_loss_curves(loss_history: Dict[str, List[Tuple[int, float]]]):
    if not loss_history:
        print("训练未产生可用损失记录，跳过绘图。")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未检测到 matplotlib，无法绘制损失曲线。可运行 `pip install matplotlib` 安装后重试。")
        return

    tag_to_label = {
        "losses/a_loss": "actor loss",
        "losses/c_loss": "critic loss",
        "losses/entropy": "entropy",
    }

    plt.figure(figsize=(8, 5))
    for tag, series in sorted(loss_history.items()):
        if not series:
            continue
        epochs = [epoch for epoch, _ in series]
        values = [value for _, value in series]
        plt.plot(epochs, values, label=tag_to_label.get(tag, tag))

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("rl-games Training Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    try:
        plt.show()
    except Exception as exc:
        output_path = "rl_games_loss_curve.png"
        plt.savefig(output_path)
        print(f"无法直接显示图像，已将曲线保存到 {output_path}: {exc}")


def plot_eval_history(history_path: Optional[str]):
    if not history_path or not os.path.exists(history_path):
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未检测到 matplotlib，无法绘制评估历史。")
        return

    records = []
    try:
        with open(history_path, "r", encoding="utf-8") as fp:
            for line in fp:
                entry = line.strip()
                if not entry:
                    continue
                try:
                    records.append(json.loads(entry))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        print(f"读取评估历史失败: {exc}")
        return

    if not records:
        print("评估历史为空，跳过绘图。")
        return

    indices = list(range(1, len(records) + 1))
    metric_keys = [
        "reward",
        "pos_rmse",
        "att_rmse",
        "force_mean",
        "final_error",
        "thrust_saturation_mean",
        "thrust_max_norm",
    ]
    metric_keys = [k for k in metric_keys if all(k in r for r in records)]
    if not metric_keys:
        return

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(8, 2.5 * len(metric_keys)), sharex=True)
    if len(metric_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, metric_keys):
        values = [float(r[key]) for r in records]
        ax.plot(indices, values, marker="o", label=key)
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Evaluation index")
    fig.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        output_path = "rl_games_eval_history.png"
        fig.savefig(output_path)
        print(f"无法直接显示评估曲线，已保存到 {output_path}: {exc}")


# ---------------------------- RL Games 配置 ---------------------------- #

def create_config(num_workers: int, max_epochs: int, sim_envs: int) -> Dict:
    num_actors = 1
    horizon = max(16, sim_envs // 4)
    batch_size = num_actors * horizon
    minibatch = max(32, batch_size // 4)
    minibatch = min(minibatch, batch_size)
    while batch_size % minibatch != 0 and minibatch > num_actors:
        minibatch -= num_actors
    if batch_size % minibatch != 0:
        minibatch = num_actors

    config = {
        "params": {
            "seed": 42,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": -0.7},
                        "fixed_sigma": False,
                    }
                },
                "mlp": {
                    "units": [128, 64],
                    "activation": "elu",
                    "initializer": {"name": "default", "scale": 2},
                },
            },
            "config": {
                "name": "lee_gain_tune_run",
                "env_name": "lee_gain_tune",
                "env_config": {},
                "num_actors": num_actors,
                "horizon_length": horizon,
                "batch_size": batch_size,
                "minibatch_size": minibatch,
                "mini_epochs": 6,
                "ppo": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 5e-5,
                "lr_schedule": "adaptive",
                "grad_norm": 1.0,
                "entropy_coef": 0.0001,
                "critic_coef": 2.0,
                "clip_value": False,
                "e_clip": 0.15,
                "kl_threshold": 0.01,
                "truncate_grads": True,
                "normalize_advantage": True,
                "normalize_input": False,
                "normalize_value": True,
                "bounds_loss_coef": 0.0001,
                "max_epochs": max_epochs,
                "save_best_after": 2,
                "save_frequency": 5,
                "keep_checkpoints": 5,
                "save_on_exit": True,
                "score_to_win": 1e9,
                "player": {"render": False, "deterministic": True},
                "reward_shaper": {"scale_value": 1.0},
                "print_stats": False,
            },
        }
    }
    return config


# ---------------------------- 主流程 ---------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="使用 rl-games 调优 Lee 控制器增益")
    parser.add_argument("--device", default=None, help="设备字符串，如 cuda:0 或 cpu")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="rl-games 并行 actor 数量")
    parser.add_argument("--num-envs", type=int, default=None, help=argparse.SUPPRESS)  # 兼容旧脚本参数
    parser.add_argument("--sim-envs", type=int, default=DEFAULT_SIM_NUM_ENVS, help="Isaac Gym 并行无人机数量")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, help="训练总步数（用于估算 max_epochs）")
    parser.add_argument("--eval-horizon", type=int, default=DEFAULT_EVAL_HORIZON, help="单次评估的仿真步数")
    parser.add_argument(
        "--eval-rollouts",
        type=int,
        default=DEFAULT_EVAL_ROLLOUTS,
        help="每次评估重复 rollout 的次数（数值越大越平滑，但耗时更久）",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="rl-games 的最大 epoch 数（覆盖 total-timesteps 推算）",
    )
    parser.add_argument("--best-path", default="best_lee_gains_rlg.json", help="最佳结果保存路径")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help=f"取消 headless 以打开查看器窗口（默认{'开启 headless' if DEFAULT_HEADLESS else '关闭 headless'}）",
    )
    parser.add_argument(
        "--disable-warp",
        action="store_true",
        help=f"禁用 Warp 加速（默认{'启用' if DEFAULT_USE_WARP else '禁用'}）",
    )
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH, help="评估指标历史输出 (JSONL)")
    parser.add_argument("--metric-log", default=DEFAULT_METRIC_LOG_PATH, help="训练损失/指标日志 (JSONL)")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="rl-games 检查点保存目录")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="rl-games experiment_name")
    parser.add_argument("--load-checkpoint", default=None, help="恢复训练时载入的 checkpoint 路径")
    return parser.parse_args()


def main():
    args = parse_args()

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("runs", args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=log_dir)
    try:
        tracker = ResultsTracker(args.best_path, history_path=args.history_path, writer=tensorboard_writer)
        metric_recorder = MetricRecorder(args.metric_log)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        num_workers = args.num_workers
        if args.num_envs is not None:
            num_workers = args.num_envs
        num_workers = max(1, int(num_workers))
        sim_envs = max(1, int(args.sim_envs))
        headless = DEFAULT_HEADLESS
        if args.no_headless:
            headless = False
        use_warp = DEFAULT_USE_WARP and not args.disable_warp

        experiment_name = args.experiment_name or DEFAULT_EXPERIMENT_NAME
        eval_rollouts = max(1, int(args.eval_rollouts))

        register_env(tracker, args.eval_horizon, eval_rollouts, RELEASE_PLAN)

        derived_epochs = max(32, args.total_timesteps // num_workers)
        if args.max_epochs and args.max_epochs > 0:
            max_epochs = args.max_epochs
        else:
            max_epochs = max(DEFAULT_MAX_EPOCHS, derived_epochs)
        config = create_config(num_workers, max_epochs, sim_envs)
        env_cfg = config["params"]["config"]["env_config"]
        env_cfg["device"] = device_str
        env_cfg["sim_num_envs"] = sim_envs
        env_cfg["headless"] = headless
        env_cfg["use_warp"] = use_warp
        env_cfg["eval_rollouts"] = eval_rollouts
        cfg = config["params"]["config"]
        cfg["experiment_name"] = experiment_name
        cfg["full_experiment_name"] = experiment_name
        cfg["train_dir"] = args.checkpoint_dir

        observer = ProgressObserver(max_epochs, tracker, metric_recorder=metric_recorder)
        runner = Runner(algo_observer=observer)

        try:
            runner.load(config)
            runner.reset()
            run_args = {"train": True}
            if args.load_checkpoint:
                run_args["checkpoint"] = args.load_checkpoint
            runner.run(run_args)
        finally:
            observer.close()

        if tracker.best_payload:
            print("\n========== 最佳增益（rl-games） ==========")
            print(json.dumps(tracker.best_payload, indent=2))
        else:
            print("\n未找到有效的增益结果，请检查训练是否成功。")

        loss_history = observer.get_loss_history()
        plot_loss_curves(loss_history)
        plot_eval_history(args.history_path)
    finally:
        tensorboard_writer.close()


if __name__ == "__main__":
    main()
