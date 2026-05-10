"""
Paper-style comparative evaluation for DROP task:
1) PPO policy (loaded from checkpoint)
2) Proposed policy: MLP+GRU (loaded from checkpoint)
3) PPO-GRU baseline: GRU-only (optional checkpoint)
4) MPC controller

Outputs:
- Raw CSVs, summary CSVs
- PNG + PDF figures (paper style)
"""

import argparse
import csv
import logging
import math
import os
import sys
import distutils
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root import path is available when script is launched directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import isaacgym
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("eval_drop_paper_comparison")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


task_config = None
NavigationTaskGmmNoise = None


def _ensure_eval_env_modules_loaded():
    global task_config, NavigationTaskGmmNoise
    if task_config is not None and NavigationTaskGmmNoise is not None:
        return
    from aerial_gym.config.task_config.navigation_task_gmm_noise_config import task_config as _task_config
    from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import (
        NavigationTaskGmmNoise as _NavigationTaskGmmNoise,
    )

    task_config = _task_config
    NavigationTaskGmmNoise = _NavigationTaskGmmNoise

DEFAULT_BASELINE_CHECKPOINT = (
    "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/"
    "runs/PPO/nn/last_ppo_mlp_drop_v1_ep_2010_rew_19.82835.pth"
)
DEFAULT_GRU_CHECKPOINT = (
    "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/"
    "runs/PPO-GRU-plus/nn/last_gmm_noise_run_ep_10000_rew__27.005426_.pth"
)
DEFAULT_RNN_CHECKPOINT = ""
DEFAULT_OUTPUT_ROOT = (
    "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/result"
)
BASE_TASK_OBS_DIM = 15
BASE_TASK_OBS_WITH_DROP_DECISION_DIM = 18
AUG_TASK_OBS_DIM = 34
AUG_TASK_OBS_WITH_DROP_DECISION_DIM = 37
CASE_SEED_STRIDE = 10007


@dataclass
class MethodSummary:
    method: str
    seed: int
    done_total: int
    drop_done: int
    no_drop_done: int


@dataclass
class MethodSpec:
    name: str
    actor: Optional[nn.Module]
    rms: Optional["RunningMeanStd"]
    obs_dim: int
    use_rnn: bool = False
    rnn_hidden_size: int = 64
    kind: str = "ppo"
    controller: Optional["MPCDropController"] = None
    force_drop_step: Optional[int] = None


@dataclass
class EvalCondition:
    name: str
    display_name: str
    overrides: Dict[str, Any]
    fixed_obstacle: bool = False
    fixed_obstacle_offset_xy: Tuple[float, float] = (1.5, 0.0)
    fixed_obstacle_radius: float = 0.2


@dataclass
class ExperimentSpec:
    name: str
    display_name: str
    conditions: List[EvalCondition]


class RunningMeanStd:
    def __init__(self, mean: torch.Tensor, var: torch.Tensor, count: torch.Tensor, device: str):
        self.mean = mean.to(device=device, dtype=torch.float32)
        self.var = var.to(device=device, dtype=torch.float32)
        self.count = count
        self.device = device

    def normalize(
        self, obs: torch.Tensor, epsilon: float = 1e-8, clip_range: float = 5.0
    ) -> torch.Tensor:
        # Match rl_games RunningMeanStd inference behavior:
        # y = clamp((obs - mean) / sqrt(var + eps), -5, 5)
        y = (obs - self.mean) / torch.sqrt(torch.clamp(self.var, min=epsilon))
        return torch.clamp(y, min=-clip_range, max=clip_range)


class RNNBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(x, h)


class PPOActorGRU(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        rnn_hidden_size: int = 64,
        mlp_units: Optional[List[int]] = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        mlp_units = mlp_units or [256, 128, 64]
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for out_dim in mlp_units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ELU())
            in_dim = out_dim
        self.actor_mlp = nn.Sequential(*layers)
        self.rnn = RNNBlock(input_size=in_dim, hidden_size=rnn_hidden_size)
        self.layer_norm = nn.LayerNorm(rnn_hidden_size) if use_layer_norm else nn.Identity()
        self.mu = nn.Linear(rnn_hidden_size, act_dim)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.actor_mlp(obs)
        x = x.unsqueeze(0)  # [1, N, 64]
        x, h_next = self.rnn(x, h)
        x = x.squeeze(0)
        x = self.layer_norm(x)
        mu = self.mu(x)
        return mu, h_next


class PPOActorRNNOnly(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        rnn_hidden_size: int = 64,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.rnn = RNNBlock(input_size=obs_dim, hidden_size=rnn_hidden_size)
        self.layer_norm = nn.LayerNorm(rnn_hidden_size) if use_layer_norm else nn.Identity()
        self.mu = nn.Linear(rnn_hidden_size, act_dim)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs.unsqueeze(0)  # [1, N, obs_dim]
        x, h_next = self.rnn(x, h)
        x = x.squeeze(0)
        x = self.layer_norm(x)
        mu = self.mu(x)
        return mu, h_next


class PPOActorMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, mlp_units: Optional[List[int]] = None):
        super().__init__()
        mlp_units = mlp_units or [256, 128, 64]
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for out_dim in mlp_units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ELU())
            in_dim = out_dim
        self.actor_mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(in_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu(self.actor_mlp(obs))


class MPCDropController:
    """
    Lightweight receding-horizon controller for fair comparison:
    - Input: same observation vector used by RL policies.
    - Objective: minimize predicted landing error + release-attitude instability.
    """

    def __init__(
        self,
        obs_dim: int,
        desired_altitude: float = 10.0,
        max_speed_xy: float = 1.2,
        max_speed_z: float = 1.2,
        drop_error_threshold: float = 1.0,
        drop_attitude_deg_threshold: float = 12.0,
        drop_angular_vel_threshold: float = 1.2,
        drop_dist_xy_threshold: float = 6.0,
        drop_min_z: float = 2.0,
    ):
        self.obs_dim = int(obs_dim)
        self.desired_altitude = float(desired_altitude)
        self.max_speed_xy = float(max_speed_xy)
        self.max_speed_z = float(max_speed_z)
        self.drop_error_threshold = float(drop_error_threshold)
        self.drop_attitude_deg_threshold = float(drop_attitude_deg_threshold)
        self.drop_angular_vel_threshold = float(drop_angular_vel_threshold)
        self.drop_dist_xy_threshold = float(drop_dist_xy_threshold)
        self.drop_min_z = float(drop_min_z)
        self.gravity = 9.81

        speeds = torch.tensor([0.0, 0.3, 0.6, 0.9, 1.2], dtype=torch.float32)
        angles_deg = torch.tensor([-20.0, 0.0, 20.0], dtype=torch.float32)
        self._candidate_speeds = speeds
        self._candidate_angles = angles_deg * math.pi / 180.0

    def _candidate_velocities_xy(self, target_xy: torch.Tensor) -> torch.Tensor:
        # target_xy: [N, 2], returns [N, K, 2]
        eps = 1e-6
        dist = torch.norm(target_xy, dim=1, keepdim=True)
        base_dir = target_xy / (dist + eps)
        ux = base_dir[:, 0:1]
        uy = base_dir[:, 1:2]

        c = torch.cos(self._candidate_angles.to(target_xy.device)).view(1, -1)
        s = torch.sin(self._candidate_angles.to(target_xy.device)).view(1, -1)
        dir_x = c * ux - s * uy
        dir_y = s * ux + c * uy
        dirs = torch.stack((dir_x, dir_y), dim=2)  # [N, A, 2]

        spd = self._candidate_speeds.to(target_xy.device).view(1, -1, 1)  # [1, S, 1]
        dirs = dirs.unsqueeze(2)  # [N, A, 1, 2]
        vel = dirs * spd.unsqueeze(1)  # [N, A, S, 2]
        return vel.reshape(target_xy.shape[0], -1, 2)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[1] < 12:
            raise RuntimeError(
                f"MPC requires >=12 observation dims, got {obs.shape[1]}"
            )

        target_rel = obs[:, 0:3]
        body_linvel = obs[:, 3:6]
        roll = obs[:, 6]
        pitch = obs[:, 7]
        body_angvel = obs[:, 8:11]
        z = obs[:, 11]

        target_xy = target_rel[:, 0:2]
        vel_xy = body_linvel[:, 0:2]
        dist_xy = torch.norm(target_xy, dim=1)

        t_fall = torch.sqrt(torch.clamp(2.0 * torch.clamp(z, min=0.1) / self.gravity, min=1e-4))
        cand_vel_xy = self._candidate_velocities_xy(target_xy)
        pred_error_xy = torch.norm(
            target_xy.unsqueeze(1) - cand_vel_xy * t_fall.view(-1, 1, 1), dim=2
        )
        smooth_penalty = torch.norm(cand_vel_xy - vel_xy.unsqueeze(1), dim=2)
        costs = pred_error_xy + 0.15 * smooth_penalty
        best_idx = torch.argmin(costs, dim=1)
        desired_xy = cand_vel_xy[torch.arange(obs.shape[0], device=obs.device), best_idx]

        desired_vz = torch.clamp(
            0.8 * (self.desired_altitude - z), min=-self.max_speed_z, max=self.max_speed_z
        )

        action = torch.zeros((obs.shape[0], 5), device=obs.device, dtype=obs.dtype)
        action[:, 0] = torch.clamp(desired_xy[:, 0] / self.max_speed_xy, -1.0, 1.0)
        action[:, 1] = torch.clamp(desired_xy[:, 1] / self.max_speed_xy, -1.0, 1.0)
        action[:, 2] = torch.clamp(desired_vz / self.max_speed_z, -1.0, 1.0)
        action[:, 3] = 0.0

        pred_release_err = torch.norm(target_xy - vel_xy * t_fall.view(-1, 1), dim=1)
        attitude_deg = torch.rad2deg(torch.sqrt(torch.clamp(roll * roll + pitch * pitch, min=0.0)))
        omega_xy = torch.norm(body_angvel[:, 0:2], dim=1)
        drop_mask = (
            (pred_release_err <= self.drop_error_threshold)
            & (attitude_deg <= self.drop_attitude_deg_threshold)
            & (omega_xy <= self.drop_angular_vel_threshold)
            & (dist_xy <= self.drop_dist_xy_threshold)
            & (z >= self.drop_min_z)
        )
        action[:, 4] = torch.where(
            drop_mask,
            torch.ones_like(action[:, 4]),
            -torch.ones_like(action[:, 4]),
        )
        return action


def _ppo_checkpoint_has_actor_mlp(model_state: Dict[str, torch.Tensor]) -> bool:
    return "a2c_network.actor_mlp.0.weight" in model_state


def _ppo_checkpoint_has_rnn(model_state: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("a2c_network.rnn.") for k in model_state.keys())


def _ppo_checkpoint_has_layer_norm(model_state: Dict[str, torch.Tensor]) -> bool:
    return "a2c_network.layer_norm.weight" in model_state


def _infer_actor_mlp_units(model_state: Dict[str, torch.Tensor]) -> List[int]:
    units: List[int] = []
    linear_indices: List[int] = []
    prefix = "a2c_network.actor_mlp."
    suffix = ".weight"
    for key, value in model_state.items():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        idx_text = key[len(prefix) : -len(suffix)]
        if not idx_text.isdigit() or len(value.shape) != 2:
            continue
        linear_indices.append(int(idx_text))

    for idx in sorted(linear_indices):
        units.append(int(model_state[f"{prefix}{idx}{suffix}"].shape[0]))
    return units


def _infer_ppo_dims(model_state: Dict[str, torch.Tensor]) -> Tuple[int, int, int, bool, List[int]]:
    actor_mlp_obs_key = "a2c_network.actor_mlp.0.weight"
    rnn_input_key = "a2c_network.rnn.rnn.weight_ih_l0"
    rnn_hidden_key = "a2c_network.rnn.rnn.weight_hh_l0"
    act_key = "a2c_network.mu.weight"

    if act_key not in model_state:
        raise RuntimeError("Failed to infer PPO action dimension: mu.weight missing.")

    act_dim = int(model_state[act_key].shape[0])
    has_actor_mlp = _ppo_checkpoint_has_actor_mlp(model_state)
    has_rnn = _ppo_checkpoint_has_rnn(model_state)
    actor_mlp_units = _infer_actor_mlp_units(model_state) if has_actor_mlp else []

    if has_actor_mlp:
        obs_dim = int(model_state[actor_mlp_obs_key].shape[1])
    elif has_rnn and rnn_input_key in model_state:
        # RNN-only checkpoint: observations feed directly into the recurrent block.
        obs_dim = int(model_state[rnn_input_key].shape[1])
    else:
        raise RuntimeError(
            "Failed to infer PPO observation dimension: neither actor_mlp nor RNN input weights found."
        )

    if has_rnn:
        if rnn_hidden_key not in model_state:
            raise RuntimeError("Failed to infer PPO RNN hidden size: recurrent hidden weights missing.")
        rnn_hidden_size = int(model_state[rnn_hidden_key].shape[1])
    else:
        rnn_hidden_size = 0

    return obs_dim, act_dim, rnn_hidden_size, has_actor_mlp, actor_mlp_units


def _infer_eval_obs_layout(obs_dims: List[int]) -> Tuple[int, bool, bool]:
    dims = [int(v) for v in obs_dims if int(v) > 0]
    if BASE_TASK_OBS_WITH_DROP_DECISION_DIM in dims and any(
        v in (AUG_TASK_OBS_DIM, AUG_TASK_OBS_WITH_DROP_DECISION_DIM) for v in dims
    ):
        raise RuntimeError(
            "Incompatible observation layouts: base+drop-decision (18D) cannot be mixed with augmented "
            "wind-estimation checkpoints (34D/37D) in the same evaluation run."
        )
    has_augmented_obs = any(
        v in (AUG_TASK_OBS_DIM, AUG_TASK_OBS_WITH_DROP_DECISION_DIM) for v in dims
    )
    has_drop_decision_features = any(
        v in (BASE_TASK_OBS_WITH_DROP_DECISION_DIM, AUG_TASK_OBS_WITH_DROP_DECISION_DIM)
        for v in dims
    )
    if has_augmented_obs:
        eval_obs_dim = (
            AUG_TASK_OBS_WITH_DROP_DECISION_DIM
            if has_drop_decision_features
            else AUG_TASK_OBS_DIM
        )
    else:
        eval_obs_dim = (
            BASE_TASK_OBS_WITH_DROP_DECISION_DIM
            if has_drop_decision_features
            else BASE_TASK_OBS_DIM
        )
    return eval_obs_dim, has_augmented_obs, has_drop_decision_features


def load_ppo_policy(checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(model_state, dict):
        raise RuntimeError("Unsupported checkpoint format: model state dict not found.")
    ppo_obs_dim, ppo_act_dim, rnn_hidden_size, has_actor_mlp, actor_mlp_units = _infer_ppo_dims(
        model_state
    )
    use_rnn = _ppo_checkpoint_has_rnn(model_state)
    if use_rnn:
        use_layer_norm = _ppo_checkpoint_has_layer_norm(model_state)
        if has_actor_mlp:
            actor = PPOActorGRU(
                obs_dim=ppo_obs_dim,
                act_dim=ppo_act_dim,
                rnn_hidden_size=rnn_hidden_size,
                mlp_units=actor_mlp_units,
                use_layer_norm=use_layer_norm,
            ).to(device)
        else:
            actor = PPOActorRNNOnly(
                obs_dim=ppo_obs_dim,
                act_dim=ppo_act_dim,
                rnn_hidden_size=rnn_hidden_size,
                use_layer_norm=use_layer_norm,
            ).to(device)
    else:
        actor = PPOActorMLP(
            obs_dim=ppo_obs_dim,
            act_dim=ppo_act_dim,
            mlp_units=actor_mlp_units,
        ).to(device)

    actor_state = {}
    for key, value in model_state.items():
        if key.startswith("a2c_network."):
            stripped = key.replace("a2c_network.", "", 1)
            actor_state[stripped] = value

    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        logger.warning(f"Actor missing keys: {missing}")
    if unexpected:
        logger.warning(f"Actor unexpected keys: {unexpected}")
    actor.eval()

    rms_mean = model_state.get("running_mean_std.running_mean", None)
    rms_var = model_state.get("running_mean_std.running_var", None)
    rms_count = model_state.get("running_mean_std.count", None)
    if rms_mean is None or rms_var is None or rms_count is None:
        logger.warning("running_mean_std not found in checkpoint model state. Using raw observations.")
        rms = None
    else:
        rms = RunningMeanStd(mean=rms_mean, var=rms_var, count=rms_count, device=device)

    return actor, rms, ppo_obs_dim, ppo_act_dim, use_rnn, rnn_hidden_size, has_actor_mlp


class SACActorMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int],
        act_dim: int,
        activation: str = "elu",
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
    ):
        super().__init__()
        if activation.lower() == "relu":
            act_layer = nn.ReLU
        elif activation.lower() == "elu":
            act_layer = nn.ELU
        elif activation.lower() == "tanh":
            act_layer = nn.Tanh
        else:
            raise ValueError(f"Unsupported SAC actor activation: {activation}")

        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_layer())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2 * act_dim))
        self.trunk = nn.Sequential(*layers)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        out = self.trunk(obs)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        if deterministic:
            return torch.tanh(mu)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return torch.tanh(mu + std * eps)


def _extract_sac_actor_state(ckpt_obj: Dict) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("actor", None), dict):
        return ckpt_obj["actor"]

    model_state = ckpt_obj.get("model", None) if isinstance(ckpt_obj, dict) else None
    if isinstance(model_state, dict):
        for prefix in ("sac_network.actor.", "actor."):
            if any(k.startswith(prefix) for k in model_state.keys()):
                return {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}

    if isinstance(ckpt_obj, dict) and any(k.startswith("trunk.") for k in ckpt_obj.keys()):
        return ckpt_obj

    raise RuntimeError("Unsupported SAC checkpoint format: actor state dict not found.")


def _infer_sac_dims(actor_state: Dict[str, torch.Tensor]) -> Tuple[int, List[int], int]:
    linear_layers = []
    for k, v in actor_state.items():
        m = re.fullmatch(r"trunk\.(\d+)\.weight", k)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        linear_layers.append((layer_idx, int(v.shape[1]), int(v.shape[0])))
    linear_layers.sort(key=lambda x: x[0])
    if len(linear_layers) < 2:
        raise RuntimeError("SAC actor state has insufficient linear layers to infer dimensions.")
    obs_dim = linear_layers[0][1]
    out_dim = linear_layers[-1][2]
    if out_dim % 2 != 0:
        raise RuntimeError(f"SAC actor output dim {out_dim} is invalid (must be 2 * action_dim).")
    hidden_dims = [layer[2] for layer in linear_layers[:-1]]
    act_dim = out_dim // 2
    return obs_dim, hidden_dims, act_dim


def load_sac_policy(
    checkpoint_path: str,
    device: str,
    activation: str = "elu",
    log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAC checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    actor_state = _extract_sac_actor_state(ckpt)
    sac_obs_dim, hidden_dims, sac_act_dim = _infer_sac_dims(actor_state)
    actor = SACActorMLP(
        obs_dim=sac_obs_dim,
        hidden_dims=hidden_dims,
        act_dim=sac_act_dim,
        activation=activation,
        log_std_bounds=log_std_bounds,
    ).to(device)

    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        logger.warning(f"SAC actor missing keys: {missing}")
    if unexpected:
        logger.warning(f"SAC actor unexpected keys: {unexpected}")
    actor.eval()
    return actor, sac_obs_dim, sac_act_dim


def _set_config_attr(root: Any, dotted_key: str, value: Any):
    parts = dotted_key.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def _apply_task_overrides(overrides: Dict[str, Any]):
    _ensure_eval_env_modules_loaded()
    for key, value in overrides.items():
        _set_config_attr(task_config, key, value)


def build_env(
    num_envs: int,
    seed: int,
    device: str,
    headless: bool,
    episode_len_steps: int,
    observation_space_dim: int = None,
    use_wind_estimation_features: bool = None,
    use_drop_decision_features: bool = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    _ensure_eval_env_modules_loaded()
    task_config.num_envs = num_envs
    task_config.seed = seed
    task_config.device = device
    task_config.headless = headless
    task_config.episode_len_steps = episode_len_steps
    if overrides:
        _apply_task_overrides(overrides)
    if use_wind_estimation_features is not None:
        task_config.use_wind_estimation_features = bool(use_wind_estimation_features)
    if use_drop_decision_features is not None:
        task_config.use_drop_decision_features = bool(use_drop_decision_features)
    if observation_space_dim is not None:
        task_config.observation_space_dim = int(observation_space_dim)
    env = NavigationTaskGmmNoise(task_config)
    env.reset()
    return env


def apply_runtime_overrides_to_env(env, overrides: Dict[str, Any]):
    """
    Update task config and env-cached fields without creating a new Isaac Foundation.
    """
    _ensure_eval_env_modules_loaded()
    if overrides:
        _apply_task_overrides(overrides)

    env.task_config = task_config
    env.noise_config = task_config.noise_config
    env.gmm_force_config = task_config.gmm_force_config

    env.dryden_config = getattr(task_config, "dryden_config", None)
    env.dryden_enabled = bool(
        env.dryden_config is not None and bool(getattr(env.dryden_config, "enable_dryden", False))
    )
    if hasattr(env, "dryden_wind_state"):
        env.dryden_wind_state.zero_()
    if hasattr(env, "dryden_sigma"):
        env.dryden_sigma.zero_()
    if hasattr(env, "dryden_tau"):
        env.dryden_tau.fill_(1.0)

    env.drop_model_config = getattr(task_config, "drop_model_config", None)
    env.drop_impact_config = getattr(task_config, "drop_impact_config", None)
    env.drop_reward_config = getattr(task_config, "drop_reward_config", None)

    dr = env.drop_reward_config
    env.drop_obstacle_enable = bool(getattr(dr, "drop_obstacle_enable", False))
    env.drop_obstacle_spawn_prob = float(getattr(dr, "drop_obstacle_spawn_prob", 0.5))
    env.drop_obstacle_center_radius_min = float(
        getattr(dr, "drop_obstacle_center_radius_min", 1.0)
    )
    env.drop_obstacle_center_radius_max = float(
        getattr(dr, "drop_obstacle_center_radius_max", 2.0)
    )
    env.drop_obstacle_radius_min = float(getattr(dr, "drop_obstacle_radius_min", 0.1))
    env.drop_obstacle_radius_max = float(getattr(dr, "drop_obstacle_radius_max", 0.1))
    env.drop_obstacle_height = float(getattr(dr, "drop_obstacle_height", 0.4))
    env.drop_obstacle_absent_obs_value = float(getattr(dr, "drop_obstacle_absent_obs_value", -1e3))


def parse_seed_list(seed_text: str) -> List[int]:
    seeds = []
    for token in seed_text.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds parsed from --seeds.")
    return seeds


def _to_numpy_bool(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(bool)


def _to_numpy_float(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64)


def _info_tensor(infos: Dict, key: str, fallback_key: str, default: torch.Tensor) -> torch.Tensor:
    if key in infos:
        return infos[key]
    if fallback_key and fallback_key in infos:
        return infos[fallback_key]
    return default


def _collect_step_arrays(env, infos: Dict) -> Dict[str, np.ndarray]:
    num_envs_local = env.sim_env.num_envs
    zeros_bool = torch.zeros(num_envs_local, device=env.device, dtype=torch.bool)
    zeros_float = torch.zeros(num_envs_local, device=env.device, dtype=torch.float32)
    trace_samples = int(getattr(env, "eval_drop_trace_num_samples", 64))
    zeros_trace = torch.zeros((num_envs_local, trace_samples), device=env.device, dtype=torch.float32)
    zeros_long = torch.zeros(num_envs_local, device=env.device, dtype=torch.long)
    dones_tensor = (env.terminations > 0) | (env.truncations > 0)

    drops_t = _info_tensor(infos, "drops_snapshot", "drops", zeros_bool)
    dones_t = _info_tensor(infos, "dones_snapshot", "", dones_tensor)
    crashes_t = _info_tensor(infos, "crashes_snapshot", "crashes", zeros_bool)
    timeouts_t = _info_tensor(infos, "timeouts_snapshot", "timeouts", zeros_bool)
    roll_t = _info_tensor(infos, "drop_roll_snapshot", "", zeros_float)
    pitch_t = _info_tensor(infos, "drop_pitch_snapshot", "", zeros_float)
    yaw_t = _info_tensor(infos, "drop_yaw_snapshot", "", zeros_float)
    dtheta_t = _info_tensor(infos, "drop_delta_theta_snapshot", "", zeros_float)
    domega_t = _info_tensor(infos, "drop_delta_omega_snapshot", "", zeros_float)
    impulse_t = _info_tensor(infos, "impulse_metric_snapshot", "", zeros_float)
    landerr_t = _info_tensor(infos, "landing_error_xy_snapshot", "", zeros_float)
    rel2tgt_t = _info_tensor(infos, "release_to_target_xy_snapshot", "", zeros_float)
    trace_norm_t = _info_tensor(infos, "drop_trace_norm_t_snapshot", "", zeros_trace)
    trace_error_t = _info_tensor(infos, "drop_trace_error_xy_snapshot", "", zeros_trace)
    trace_pos_x_t = _info_tensor(infos, "drop_trace_pos_x_snapshot", "", zeros_trace)
    trace_pos_y_t = _info_tensor(infos, "drop_trace_pos_y_snapshot", "", zeros_trace)
    trace_pos_z_t = _info_tensor(infos, "drop_trace_pos_z_snapshot", "", zeros_trace)
    trace_fall_steps_t = _info_tensor(infos, "drop_trace_fall_steps_snapshot", "", zeros_long)
    trace_fall_time_t = _info_tensor(infos, "drop_trace_fall_time_snapshot", "", zeros_float)

    return {
        "drops": _to_numpy_bool(drops_t),
        "dones": _to_numpy_bool(dones_t),
        "crashes": _to_numpy_bool(crashes_t),
        "timeouts": _to_numpy_bool(timeouts_t),
        "roll": _to_numpy_float(roll_t),
        "pitch": _to_numpy_float(pitch_t),
        "yaw": _to_numpy_float(yaw_t),
        "delta_theta": _to_numpy_float(dtheta_t),
        "delta_omega": _to_numpy_float(domega_t),
        "impulse": _to_numpy_float(impulse_t),
        "landing_error_xy": _to_numpy_float(landerr_t),
        "release_to_target_xy": _to_numpy_float(rel2tgt_t),
        "trace_norm_t": _to_numpy_float(trace_norm_t),
        "trace_error_xy": _to_numpy_float(trace_error_t),
        "trace_pos_x": _to_numpy_float(trace_pos_x_t),
        "trace_pos_y": _to_numpy_float(trace_pos_y_t),
        "trace_pos_z": _to_numpy_float(trace_pos_z_t),
        "trace_fall_steps": _to_numpy_float(trace_fall_steps_t),
        "trace_fall_time": _to_numpy_float(trace_fall_time_t),
    }


def _build_case_row(
    method: str,
    seed: int,
    case_seed: int,
    batch_idx: int,
    env_id: int,
    case_id: int,
    sim_step: int,
    done_reached: bool,
    dropped: bool,
    crashed: bool,
    timeout: bool,
    arrays: Dict[str, np.ndarray] = None,
) -> Dict[str, float]:
    row = {
        "method": method,
        "seed": seed,
        "case_seed": case_seed,
        "batch_idx": batch_idx,
        "case_id": case_id,
        "env_id": env_id,
        "sim_step": sim_step,
        "done_reached": int(done_reached),
        "dropped": int(dropped),
        "crashed": int(crashed),
        "timeout": int(timeout),
        "no_drop_done": int(done_reached and (not dropped)),
        "roll_deg": float("nan"),
        "pitch_deg": float("nan"),
        "yaw_deg": float("nan"),
        "landing_error_xy_m": float("nan"),
        "release_to_target_xy_m": float("nan"),
        "delta_theta_rad": float("nan"),
        "delta_theta_deg": float("nan"),
        "delta_omega_rad_s": float("nan"),
        "delta_omega_deg_s": float("nan"),
        "impulse_metric": float("nan"),
    }
    if dropped and arrays is not None:
        roll = float(arrays["roll"][env_id])
        pitch = float(arrays["pitch"][env_id])
        yaw = float(arrays["yaw"][env_id])
        dtheta = float(arrays["delta_theta"][env_id])
        domega = float(arrays["delta_omega"][env_id])
        row.update(
            {
                "roll_deg": float(np.degrees(roll)),
                "pitch_deg": float(np.degrees(pitch)),
                "yaw_deg": float(np.degrees(yaw)),
                "landing_error_xy_m": float(arrays["landing_error_xy"][env_id]),
                "release_to_target_xy_m": float(arrays["release_to_target_xy"][env_id]),
                "delta_theta_rad": dtheta,
                "delta_theta_deg": float(np.degrees(dtheta)),
                "delta_omega_rad_s": domega,
                "delta_omega_deg_s": float(np.degrees(domega)),
                "impulse_metric": float(arrays["impulse"][env_id]),
            }
        )
    return row


def _build_post_drop_trace_record(
    method: str,
    seed: int,
    case_seed: int,
    batch_idx: int,
    env_id: int,
    case_id: int,
    sim_step: int,
    crashed: bool,
    timeout: bool,
    arrays: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    return {
        "method": method,
        "seed": int(seed),
        "case_seed": int(case_seed),
        "batch_idx": int(batch_idx),
        "case_id": int(case_id),
        "env_id": int(env_id),
        "sim_step": int(sim_step),
        "crashed": int(crashed),
        "timeout": int(timeout),
        "trace_fall_steps": int(round(float(arrays["trace_fall_steps"][env_id]))),
        "trace_fall_time_s": float(arrays["trace_fall_time"][env_id]),
        "norm_t": np.asarray(arrays["trace_norm_t"][env_id], dtype=np.float32).copy(),
        "error_xy_m": np.asarray(arrays["trace_error_xy"][env_id], dtype=np.float32).copy(),
        "pos_x_m": np.asarray(arrays["trace_pos_x"][env_id], dtype=np.float32).copy(),
        "pos_y_m": np.asarray(arrays["trace_pos_y"][env_id], dtype=np.float32).copy(),
        "pos_z_m": np.asarray(arrays["trace_pos_z"][env_id], dtype=np.float32).copy(),
    }


def _run_method_on_case_batch(
    env,
    method: str,
    seed: int,
    case_seed: int,
    batch_idx: int,
    case_id_offset: int,
    valid_env_count: int,
    device: str,
    actor,
    rms=None,
    obs_dim: int = None,
    use_rnn: bool = False,
) -> Tuple[List[Dict[str, float]], MethodSummary]:
    # Reset to the same case seed before each method to ensure strict pairing.
    env.seed(case_seed)
    env.reset()
    num_envs = env.sim_env.num_envs
    episode_len_steps = int(getattr(env.task_config, "episode_len_steps", 1000))
    valid_env_count = int(min(max(valid_env_count, 0), num_envs))
    active_eval = np.zeros(num_envs, dtype=bool)
    active_eval[:valid_env_count] = True
    done_seen = np.logical_not(active_eval)

    if actor is None:
        raise ValueError(f"{method} evaluation requires a loaded actor.")
    if obs_dim is None:
        raise ValueError(f"{method} evaluation requires obs_dim.")
    hidden = None
    if use_rnn:
        hidden = torch.zeros((1, num_envs, 64), device=device, dtype=torch.float32)

    rows_by_env: List[Dict[str, float]] = [None] * valid_env_count
    done_total = 0
    drop_done = 0
    no_drop_done = 0
    sim_step = 0
    max_steps = max(int(episode_len_steps * 2), 2000)

    while (not np.all(done_seen)) and sim_step < max_steps:
        obs = env.task_obs["observations"][:, :obs_dim]
        if rms is not None:
            obs = rms.normalize(obs)
        with torch.no_grad():
            if use_rnn:
                mu, hidden_next = actor(obs, hidden)
                hidden = hidden_next
            else:
                mu = actor(obs)
            actions = torch.clamp(mu, -1.0, 1.0)

        _, _, _, _, infos = env.step(actions)
        sim_step += 1

        arrays = _collect_step_arrays(env, infos)
        done_new = arrays["dones"] & (~done_seen)
        done_idx = np.flatnonzero(done_new)

        if done_idx.size > 0:
            for idx in done_idx:
                if idx >= valid_env_count or rows_by_env[idx] is not None:
                    continue
                dropped = bool(arrays["drops"][idx])
                crashed = bool(arrays["crashes"][idx])
                timeout = bool(arrays["timeouts"][idx])
                case_id = case_id_offset + idx
                rows_by_env[idx] = _build_case_row(
                    method=method,
                    seed=seed,
                    case_seed=case_seed,
                    batch_idx=batch_idx,
                    env_id=int(idx),
                    case_id=int(case_id),
                    sim_step=sim_step,
                    done_reached=True,
                    dropped=dropped,
                    crashed=crashed,
                    timeout=timeout,
                    arrays=arrays,
                )
                done_total += 1
                if dropped:
                    drop_done += 1
                else:
                    no_drop_done += 1

        done_seen = done_seen | done_new

        if hidden is not None and done_idx.size > 0:
            hidden[:, done_idx, :] = 0.0

        if sim_step % 200 == 0:
            logger.info(
                f"[{method}][seed={seed}][case_seed={case_seed}] step={sim_step}, "
                f"done={done_total}/{valid_env_count}, drop_done={drop_done}, no_drop_done={no_drop_done}"
            )

    missing = [i for i in range(valid_env_count) if rows_by_env[i] is None]
    if missing:
        logger.warning(
            f"[{method}][seed={seed}][case_seed={case_seed}] "
            f"missing done envs in max_steps={max_steps}: {len(missing)}"
        )
        for idx in missing:
            rows_by_env[idx] = _build_case_row(
                method=method,
                seed=seed,
                case_seed=case_seed,
                batch_idx=batch_idx,
                env_id=int(idx),
                case_id=int(case_id_offset + idx),
                sim_step=sim_step,
                done_reached=False,
                dropped=False,
                crashed=False,
                timeout=False,
                arrays=None,
            )

    rows = [r for r in rows_by_env if r is not None]

    summary = MethodSummary(
        method=method,
        seed=seed,
        done_total=done_total,
        drop_done=drop_done,
        no_drop_done=no_drop_done,
    )
    return rows, summary


def _apply_fixed_drop_obstacle(
    env,
    valid_env_count: int,
    offset_xy: Tuple[float, float],
    radius: float,
):
    if valid_env_count <= 0:
        return
    if not hasattr(env, "drop_obstacle_position"):
        return

    env_ids = torch.arange(valid_env_count, device=env.device, dtype=torch.long)
    target_xy = env.target_position[env_ids, 0:2]
    offset = torch.tensor(offset_xy, device=env.device, dtype=torch.float32).view(1, 2)
    obstacle_xy = target_xy + offset

    lower_xy = env.env_bounds_min[0:2].view(1, 2)
    upper_xy = env.env_bounds_max[0:2].view(1, 2)
    obstacle_xy = torch.max(torch.min(obstacle_xy, upper_xy), lower_xy)

    env.drop_obstacle_exists[env_ids] = True
    env.drop_obstacle_position[env_ids, 0:2] = obstacle_xy
    env.drop_obstacle_position[env_ids, 2] = 0.5 * 0.4
    env.drop_obstacle_radius[env_ids] = float(radius)

    if hasattr(env, "_sync_drop_obstacle_instances"):
        env._sync_drop_obstacle_instances(env_ids)

    if "observations" in env.task_obs:
        env.task_obs["observations"][env_ids, 12:15] = env.drop_obstacle_position[env_ids]


def _make_post_reset_hook(condition: EvalCondition) -> Optional[Callable[[Any, int], None]]:
    if not condition.fixed_obstacle:
        return None

    def _hook(env, valid_env_count: int):
        _apply_fixed_drop_obstacle(
            env=env,
            valid_env_count=valid_env_count,
            offset_xy=condition.fixed_obstacle_offset_xy,
            radius=condition.fixed_obstacle_radius,
        )

    return _hook


def _run_method_on_case_batch_generic(
    env,
    method_spec: MethodSpec,
    seed: int,
    case_seed: int,
    batch_idx: int,
    case_id_offset: int,
    valid_env_count: int,
    device: str,
    post_reset_hook: Optional[Callable[[Any, int], None]] = None,
) -> Tuple[List[Dict[str, float]], MethodSummary, List[Dict[str, Any]]]:
    env.seed(case_seed)
    env.reset()
    # Eval-only safety: some backends keep non-zero per-env step counters across resets.
    # If not cleared, timeout truncation can occur much earlier than local sim_step,
    # which prevents late-stage force-drop from ever firing.
    if hasattr(env, "sim_env") and hasattr(env.sim_env, "sim_steps"):
        try:
            env.sim_env.sim_steps.zero_()
        except Exception:
            pass

    num_envs = env.sim_env.num_envs
    episode_len_steps = int(getattr(env.task_config, "episode_len_steps", 1000))
    valid_env_count = int(min(max(valid_env_count, 0), num_envs))
    if post_reset_hook is not None:
        post_reset_hook(env, valid_env_count)

    active_eval = np.zeros(num_envs, dtype=bool)
    active_eval[:valid_env_count] = True
    done_seen = np.logical_not(active_eval)

    rows_by_env: List[Dict[str, float]] = [None] * valid_env_count
    trace_records: List[Dict[str, Any]] = []
    done_total = 0
    drop_done = 0
    no_drop_done = 0
    sim_step = 0
    max_steps = max(int(episode_len_steps * 2), 2000)

    hidden = None
    if method_spec.kind == "ppo" and method_spec.use_rnn:
        hidden = torch.zeros(
            (1, num_envs, int(method_spec.rnn_hidden_size)),
            device=device,
            dtype=torch.float32,
        )

    while (not np.all(done_seen)) and sim_step < max_steps:
        obs_full = env.task_obs["observations"]
        obs = obs_full[:, : method_spec.obs_dim]

        with torch.no_grad():
            if method_spec.kind == "ppo":
                if method_spec.actor is None:
                    raise ValueError(f"{method_spec.name} requires a loaded actor")
                if method_spec.rms is not None:
                    obs = method_spec.rms.normalize(obs)
                if method_spec.use_rnn:
                    mu, hidden_next = method_spec.actor(obs, hidden)
                    hidden = hidden_next
                else:
                    mu = method_spec.actor(obs)
                actions = torch.clamp(mu, -1.0, 1.0)
            elif method_spec.kind == "mpc":
                if method_spec.controller is None:
                    raise ValueError("MPC method requires controller")
                actions = method_spec.controller.act(obs)
                actions = torch.clamp(actions, -1.0, 1.0)
                if method_spec.force_drop_step is not None and sim_step >= int(method_spec.force_drop_step):
                    # "Must drop unless crash": if a case is still active near timeout,
                    # force drop switch on this step.
                    active_idx = np.flatnonzero(~done_seen)
                    if active_idx.size > 0:
                        active_idx_t = torch.as_tensor(
                            active_idx, device=actions.device, dtype=torch.long
                        )
                        actions[active_idx_t, 4] = 1.0
            else:
                raise RuntimeError(f"Unsupported method kind: {method_spec.kind}")

        _, _, _, _, infos = env.step(actions)
        sim_step += 1

        arrays = _collect_step_arrays(env, infos)
        done_new = arrays["dones"] & (~done_seen)
        done_idx = np.flatnonzero(done_new)

        if done_idx.size > 0:
            for idx in done_idx:
                if idx >= valid_env_count or rows_by_env[idx] is not None:
                    continue
                dropped = bool(arrays["drops"][idx])
                crashed = bool(arrays["crashes"][idx])
                timeout = bool(arrays["timeouts"][idx])
                case_id = case_id_offset + idx
                rows_by_env[idx] = _build_case_row(
                    method=method_spec.name,
                    seed=seed,
                    case_seed=case_seed,
                    batch_idx=batch_idx,
                    env_id=int(idx),
                    case_id=int(case_id),
                    sim_step=sim_step,
                    done_reached=True,
                    dropped=dropped,
                    crashed=crashed,
                    timeout=timeout,
                    arrays=arrays,
                )
                if dropped:
                    trace_records.append(
                        _build_post_drop_trace_record(
                            method=method_spec.name,
                            seed=seed,
                            case_seed=case_seed,
                            batch_idx=batch_idx,
                            env_id=int(idx),
                            case_id=int(case_id),
                            sim_step=sim_step,
                            crashed=crashed,
                            timeout=timeout,
                            arrays=arrays,
                        )
                    )
                done_total += 1
                if dropped:
                    drop_done += 1
                else:
                    no_drop_done += 1

        done_seen = done_seen | done_new
        if hidden is not None and done_idx.size > 0:
            hidden[:, done_idx, :] = 0.0

    missing = [i for i in range(valid_env_count) if rows_by_env[i] is None]
    if missing:
        logger.warning(
            f"[{method_spec.name}][seed={seed}][case_seed={case_seed}] "
            f"missing done envs in max_steps={max_steps}: {len(missing)}"
        )
        for idx in missing:
            rows_by_env[idx] = _build_case_row(
                method=method_spec.name,
                seed=seed,
                case_seed=case_seed,
                batch_idx=batch_idx,
                env_id=int(idx),
                case_id=int(case_id_offset + idx),
                sim_step=sim_step,
                done_reached=False,
                dropped=False,
                crashed=False,
                timeout=False,
                arrays=None,
            )

    rows = [r for r in rows_by_env if r is not None]
    summary = MethodSummary(
        method=method_spec.name,
        seed=seed,
        done_total=done_total,
        drop_done=drop_done,
        no_drop_done=no_drop_done,
    )
    return rows, summary, trace_records


def evaluate_methods_single_seed(
    env,
    method_specs: List[MethodSpec],
    seed: int,
    episodes_target: int,
    device: str,
    post_reset_hook: Optional[Callable[[Any, int], None]] = None,
) -> Tuple[List[Dict[str, float]], List[MethodSummary], List[Dict[str, Any]]]:
    num_envs = env.sim_env.num_envs
    all_rows: List[Dict[str, float]] = []
    all_trace_records: List[Dict[str, Any]] = []
    summary_map: Dict[str, MethodSummary] = {
        m.name: MethodSummary(method=m.name, seed=seed, done_total=0, drop_done=0, no_drop_done=0)
        for m in method_specs
    }

    case_offset = 0
    batch_idx = 0
    while case_offset < episodes_target:
        valid_env_count = min(num_envs, episodes_target - case_offset)
        case_seed = int(seed + batch_idx * CASE_SEED_STRIDE)

        for method_spec in method_specs:
            rows_batch, summary_batch, trace_records_batch = _run_method_on_case_batch_generic(
                env=env,
                method_spec=method_spec,
                seed=seed,
                case_seed=case_seed,
                batch_idx=batch_idx,
                case_id_offset=case_offset,
                valid_env_count=valid_env_count,
                device=device,
                post_reset_hook=post_reset_hook,
            )
            all_rows.extend(rows_batch)
            all_trace_records.extend(trace_records_batch)
            agg = summary_map[method_spec.name]
            agg.done_total += summary_batch.done_total
            agg.drop_done += summary_batch.drop_done
            agg.no_drop_done += summary_batch.no_drop_done

        logger.info(
            f"[paired][seed={seed}] batch={batch_idx}, cases={case_offset}->{case_offset + valid_env_count - 1}"
        )
        case_offset += valid_env_count
        batch_idx += 1

    return all_rows, [summary_map[m.name] for m in method_specs], all_trace_records


def evaluate_paired_single_seed(
    env,
    seed: int,
    episodes_target: int,
    device: str,
    baseline_actor,
    baseline_rms,
    baseline_obs_dim: int,
    baseline_use_rnn: bool,
    gru_actor,
    gru_rms,
    gru_obs_dim: int,
    gru_use_rnn: bool,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], MethodSummary, MethodSummary]:
    num_envs = env.sim_env.num_envs
    baseline_rows_all: List[Dict[str, float]] = []
    gru_rows_all: List[Dict[str, float]] = []

    baseline_done_total = 0
    baseline_drop_done = 0
    baseline_no_drop_done = 0
    gru_done_total = 0
    gru_drop_done = 0
    gru_no_drop_done = 0

    case_offset = 0
    batch_idx = 0
    while case_offset < episodes_target:
        valid_env_count = min(num_envs, episodes_target - case_offset)
        case_seed = int(seed + batch_idx * CASE_SEED_STRIDE)

        baseline_rows_batch, baseline_summary_batch = _run_method_on_case_batch(
            env=env,
            method="ppo_baseline",
            seed=seed,
            case_seed=case_seed,
            batch_idx=batch_idx,
            case_id_offset=case_offset,
            valid_env_count=valid_env_count,
            device=device,
            actor=baseline_actor,
            rms=baseline_rms,
            obs_dim=baseline_obs_dim,
            use_rnn=baseline_use_rnn,
        )
        gru_rows_batch, gru_summary_batch = _run_method_on_case_batch(
            env=env,
            method="ppo_gru",
            seed=seed,
            case_seed=case_seed,
            batch_idx=batch_idx,
            case_id_offset=case_offset,
            valid_env_count=valid_env_count,
            device=device,
            actor=gru_actor,
            rms=gru_rms,
            obs_dim=gru_obs_dim,
            use_rnn=gru_use_rnn,
        )

        baseline_rows_all.extend(baseline_rows_batch)
        gru_rows_all.extend(gru_rows_batch)
        baseline_done_total += baseline_summary_batch.done_total
        baseline_drop_done += baseline_summary_batch.drop_done
        baseline_no_drop_done += baseline_summary_batch.no_drop_done
        gru_done_total += gru_summary_batch.done_total
        gru_drop_done += gru_summary_batch.drop_done
        gru_no_drop_done += gru_summary_batch.no_drop_done

        logger.info(
            f"[paired][seed={seed}] batch={batch_idx}, cases={case_offset}->{case_offset + valid_env_count - 1}, "
            f"PPO(drop/no_drop)={baseline_summary_batch.drop_done}/{baseline_summary_batch.no_drop_done}, "
            f"Proposed(drop/no_drop)={gru_summary_batch.drop_done}/{gru_summary_batch.no_drop_done}"
        )

        case_offset += valid_env_count
        batch_idx += 1

    baseline_summary = MethodSummary(
        method="ppo_baseline",
        seed=seed,
        done_total=baseline_done_total,
        drop_done=baseline_drop_done,
        no_drop_done=baseline_no_drop_done,
    )
    gru_summary = MethodSummary(
        method="ppo_gru",
        seed=seed,
        done_total=gru_done_total,
        drop_done=gru_drop_done,
        no_drop_done=gru_no_drop_done,
    )
    return baseline_rows_all, gru_rows_all, baseline_summary, gru_summary


def ensure_dirs(output_root: str):
    raw_dir = os.path.join(output_root, "raw")
    stats_dir = os.path.join(output_root, "stats")
    fig_dir = os.path.join(output_root, "figures")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return raw_dir, stats_dir, fig_dir


def write_rows_csv(path: str, rows: List[Dict[str, float]]):
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "seed", "env_id", "sim_step"])
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int_from_csv(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def _safe_float_from_csv(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except Exception:
        return default


def _normalize_saved_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "landing_error_xy_m" in row and "done_reached" in row:
        out = dict(row)
        out["seed"] = _safe_int_from_csv(out.get("seed", 0), default=0)
        out["case_seed"] = _safe_int_from_csv(out.get("case_seed", out["seed"]), default=out["seed"])
        out["batch_idx"] = _safe_int_from_csv(out.get("batch_idx", 0), default=0)
        out["case_id"] = _safe_int_from_csv(out.get("case_id", 0), default=0)
        out["env_id"] = _safe_int_from_csv(out.get("env_id", out["case_id"]), default=out["case_id"])
        out["sim_step"] = _safe_int_from_csv(out.get("sim_step", 0), default=0)
        out["done_reached"] = _safe_int_from_csv(out.get("done_reached", 0), default=0)
        out["dropped"] = _safe_int_from_csv(out.get("dropped", 0), default=0)
        out["crashed"] = _safe_int_from_csv(out.get("crashed", 0), default=0)
        out["timeout"] = _safe_int_from_csv(out.get("timeout", 0), default=0)
        out["no_drop_done"] = _safe_int_from_csv(
            out.get("no_drop_done", int(out["done_reached"] == 1 and out["dropped"] == 0)),
            default=int(out["done_reached"] == 1 and out["dropped"] == 0),
        )
        for metric_key in [
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "landing_error_xy_m",
            "release_to_target_xy_m",
            "delta_theta_rad",
            "delta_theta_deg",
            "delta_omega_rad_s",
            "delta_omega_deg_s",
            "impulse_metric",
        ]:
            out[metric_key] = _safe_float_from_csv(out.get(metric_key, float("nan")))
        return out

    method = str(row.get("method", "")).strip()
    seed = _safe_int_from_csv(row.get("seed", 0), default=0)
    case_index = _safe_int_from_csv(row.get("case_id", row.get("case_index", 0)), default=0)
    done_reached = _safe_int_from_csv(row.get("done_reached", 1), default=1)
    dropped = _safe_int_from_csv(row.get("dropped", 0), default=0)
    crashed = _safe_int_from_csv(row.get("crashed", row.get("termination_flag", 0)), default=0)
    timeout = _safe_int_from_csv(row.get("timeout", row.get("timeout_flag", 0)), default=0)
    normalized = {
        "method": method,
        "seed": seed,
        "case_seed": _safe_int_from_csv(row.get("case_seed", seed), default=seed),
        "batch_idx": _safe_int_from_csv(row.get("batch_idx", 0), default=0),
        "case_id": case_index,
        "env_id": _safe_int_from_csv(row.get("env_id", case_index), default=case_index),
        "sim_step": _safe_int_from_csv(row.get("sim_step", 0), default=0),
        "done_reached": done_reached,
        "dropped": dropped,
        "crashed": crashed,
        "timeout": timeout,
        "no_drop_done": int(done_reached == 1 and dropped == 0),
        "roll_deg": _safe_float_from_csv(row.get("roll_deg", row.get("drop_roll_deg", float("nan")))),
        "pitch_deg": _safe_float_from_csv(row.get("pitch_deg", row.get("drop_pitch_deg", float("nan")))),
        "yaw_deg": _safe_float_from_csv(row.get("yaw_deg", float("nan"))),
        "landing_error_xy_m": _safe_float_from_csv(
            row.get("landing_error_xy_m", row.get("drop_error_xy", float("nan")))
        ),
        "release_to_target_xy_m": _safe_float_from_csv(row.get("release_to_target_xy_m", float("nan"))),
        "delta_theta_rad": _safe_float_from_csv(row.get("delta_theta_rad", float("nan"))),
        "delta_theta_deg": _safe_float_from_csv(row.get("delta_theta_deg", float("nan"))),
        "delta_omega_rad_s": _safe_float_from_csv(row.get("delta_omega_rad_s", float("nan"))),
        "delta_omega_deg_s": _safe_float_from_csv(row.get("delta_omega_deg_s", float("nan"))),
        "impulse_metric": _safe_float_from_csv(row.get("impulse_metric", float("nan"))),
    }
    return normalized


def load_rows_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Saved evaluation CSV not found: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [_normalize_saved_row(dict(row)) for row in reader]


def _empty_post_drop_trace_bundle(num_samples: int = 0) -> Dict[str, np.ndarray]:
    return {
        "method": np.asarray([], dtype="<U16"),
        "seed": np.asarray([], dtype=np.int64),
        "case_seed": np.asarray([], dtype=np.int64),
        "batch_idx": np.asarray([], dtype=np.int64),
        "case_id": np.asarray([], dtype=np.int64),
        "env_id": np.asarray([], dtype=np.int64),
        "sim_step": np.asarray([], dtype=np.int64),
        "crashed": np.asarray([], dtype=np.int64),
        "timeout": np.asarray([], dtype=np.int64),
        "trace_fall_steps": np.asarray([], dtype=np.int64),
        "trace_fall_time_s": np.asarray([], dtype=np.float32),
        "norm_t": np.zeros((0, num_samples), dtype=np.float32),
        "error_xy_m": np.zeros((0, num_samples), dtype=np.float32),
        "pos_x_m": np.zeros((0, num_samples), dtype=np.float32),
        "pos_y_m": np.zeros((0, num_samples), dtype=np.float32),
        "pos_z_m": np.zeros((0, num_samples), dtype=np.float32),
    }


def write_post_drop_traces_npz(path: str, trace_records: List[Dict[str, Any]]) -> str:
    if not trace_records:
        np.savez_compressed(path, **_empty_post_drop_trace_bundle())
        return path
    num_samples = int(np.asarray(trace_records[0]["norm_t"]).shape[0])
    bundle = {
        "method": np.asarray([str(r["method"]) for r in trace_records]),
        "seed": np.asarray([int(r["seed"]) for r in trace_records], dtype=np.int64),
        "case_seed": np.asarray([int(r["case_seed"]) for r in trace_records], dtype=np.int64),
        "batch_idx": np.asarray([int(r["batch_idx"]) for r in trace_records], dtype=np.int64),
        "case_id": np.asarray([int(r["case_id"]) for r in trace_records], dtype=np.int64),
        "env_id": np.asarray([int(r["env_id"]) for r in trace_records], dtype=np.int64),
        "sim_step": np.asarray([int(r["sim_step"]) for r in trace_records], dtype=np.int64),
        "crashed": np.asarray([int(r["crashed"]) for r in trace_records], dtype=np.int64),
        "timeout": np.asarray([int(r["timeout"]) for r in trace_records], dtype=np.int64),
        "trace_fall_steps": np.asarray(
            [int(r["trace_fall_steps"]) for r in trace_records], dtype=np.int64
        ),
        "trace_fall_time_s": np.asarray(
            [float(r["trace_fall_time_s"]) for r in trace_records], dtype=np.float32
        ),
        "norm_t": np.stack(
            [np.asarray(r["norm_t"], dtype=np.float32) for r in trace_records], axis=0
        ).reshape(-1, num_samples),
        "error_xy_m": np.stack(
            [np.asarray(r["error_xy_m"], dtype=np.float32) for r in trace_records], axis=0
        ).reshape(-1, num_samples),
        "pos_x_m": np.stack(
            [np.asarray(r["pos_x_m"], dtype=np.float32) for r in trace_records], axis=0
        ).reshape(-1, num_samples),
        "pos_y_m": np.stack(
            [np.asarray(r["pos_y_m"], dtype=np.float32) for r in trace_records], axis=0
        ).reshape(-1, num_samples),
        "pos_z_m": np.stack(
            [np.asarray(r["pos_z_m"], dtype=np.float32) for r in trace_records], axis=0
        ).reshape(-1, num_samples),
    }
    np.savez_compressed(path, **bundle)
    return path


def load_post_drop_traces_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Saved post-drop trace file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        bundle = {k: data[k] for k in data.files}
    if "method" not in bundle:
        return _empty_post_drop_trace_bundle()
    return bundle


def build_summaries_from_rows(rows: List[Dict[str, Any]], methods: List[str]) -> List[MethodSummary]:
    summaries: List[MethodSummary] = []
    for method in methods:
        done_rows = [
            r for r in rows if str(r.get("method", "")) == method and int(r.get("done_reached", 0)) == 1
        ]
        seed = int(done_rows[0].get("seed", 0)) if done_rows else 0
        done_total = len(done_rows)
        drop_done = sum(int(r.get("dropped", 0)) for r in done_rows)
        no_drop_done = sum(int(r.get("no_drop_done", 0)) for r in done_rows)
        summaries.append(
            MethodSummary(
                method=method,
                seed=seed,
                done_total=done_total,
                drop_done=drop_done,
                no_drop_done=no_drop_done,
            )
        )
    return summaries


def metric_stats(values: np.ndarray):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(values.size),
    }


def save_summary_tables(stats_dir: str, rows: List[Dict[str, float]], summaries: List[MethodSummary]):
    methods = sorted(set([r["method"] for r in rows])) if rows else []
    metrics = [
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "landing_error_xy_m",
        "release_to_target_xy_m",
        "delta_theta_rad",
        "delta_theta_deg",
        "delta_omega_rad_s",
        "delta_omega_deg_s",
        "impulse_metric",
    ]

    mean_std_path = os.path.join(stats_dir, "summary_mean_std.csv")
    median_iqr_path = os.path.join(stats_dir, "summary_median_iqr.csv")
    no_drop_path = os.path.join(stats_dir, "no_drop_rate.csv")

    with open(mean_std_path, "w", newline="") as f_mean, open(median_iqr_path, "w", newline="") as f_med:
        w_mean = csv.writer(f_mean)
        w_med = csv.writer(f_med)
        w_mean.writerow(["method", "metric", "n", "mean", "std", "min", "max"])
        w_med.writerow(["method", "metric", "n", "median", "q1", "q3", "iqr"])

        for method in methods:
            method_rows = [r for r in rows if r["method"] == method]
            for metric in metrics:
                arr = np.array(
                    [float(r.get(metric, float("nan"))) for r in method_rows],
                    dtype=np.float64,
                )
                st = metric_stats(arr)
                w_mean.writerow(
                    [method, metric, st["n"], st["mean"], st["std"], st["min"], st["max"]]
                )
                w_med.writerow(
                    [method, metric, st["n"], st["median"], st["q1"], st["q3"], st["iqr"]]
                )

    with open(no_drop_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "done_total", "drop_done", "no_drop_done", "no_drop_rate"])
        for s in summaries:
            rate = float(s.no_drop_done) / float(max(s.done_total, 1))
            w.writerow([s.method, s.seed, s.done_total, s.drop_done, s.no_drop_done, rate])

    return mean_std_path, median_iqr_path, no_drop_path


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ema(x: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y


def _set_tight_ylim_from_smoothed(ax, curves: List[np.ndarray], q_low: float = 2.0, q_high: float = 98.0):
    valid = []
    for c in curves:
        if c is None:
            continue
        c = np.asarray(c, dtype=np.float64)
        c = c[np.isfinite(c)]
        if c.size > 0:
            valid.append(c)
    if not valid:
        return
    all_vals = np.concatenate(valid, axis=0)
    lo = float(np.percentile(all_vals, q_low))
    hi = float(np.percentile(all_vals, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi <= lo:
        center = 0.5 * (hi + lo)
        span = max(1e-3, abs(center) * 0.1 + 1e-2)
        ax.set_ylim(center - span, center + span)
        return
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def _row_metric_value(row: Dict[str, float], metric_key: str) -> float:
    if metric_key == "attitude_total_deg":
        roll = float(row.get("roll_deg", float("nan")))
        pitch = float(row.get("pitch_deg", float("nan")))
        if np.isfinite(roll) and np.isfinite(pitch):
            return float(math.sqrt(roll * roll + pitch * pitch))
        return float("nan")
    return float(row.get(metric_key, float("nan")))


def _canonical_method_key(method: str) -> str:
    text = str(method).strip()
    normalized = text.lower().replace("-", "_").replace("(", "").replace(")", "")
    normalized = normalized.replace(" ", "_")
    if normalized in {"ppo", "ppo_baseline", "ppobaseline"}:
        return "ppo"
    if normalized in {"ppo_gru", "ppo_gru_plus", "proposed"}:
        return "ppo_gru"
    if normalized in {"ppo_rnn", "rnn", "rnn_only", "ppo_rnn_only"}:
        return "ppo_rnn"
    if normalized == "mpc":
        return "mpc"
    return normalized


def method_display_name(method: str) -> str:
    """Paper-facing method names; raw CSV method keys are intentionally unchanged."""
    key = _canonical_method_key(method)
    if key == "ppo":
        return "PPO"
    if key == "ppo_gru":
        return "Proposed"
    if key == "ppo_rnn":
        return "PPO-GRU"
    if key == "mpc":
        return "MPC"
    return str(method).strip() or "Unknown"


def method_color(method: str) -> Optional[str]:
    key = _canonical_method_key(method)
    return {
        "ppo": "#1f77b4",
        "ppo_gru": "#d62728",
        "ppo_rnn": "#ff7f0e",
        "mpc": "#2ca02c",
    }.get(key)


def _short_experiment_label(experiment_display_name: str) -> str:
    text = str(experiment_display_name).strip()
    if ":" in text and text.lower().startswith("experiment"):
        sim_label = text.split(":", 1)[0].strip()
        # Paper order differs from simulation order:
        # simulation exp4 -> paper Experiment 3, simulation exp5 -> paper Experiment 4.
        if sim_label == "Experiment 4":
            return "Experiment 3"
        if sim_label == "Experiment 5":
            return "Experiment 4"
        return sim_label
    return text or "Experiment"


def _short_condition_label(condition_display_name: str) -> str:
    key = str(condition_display_name).strip().lower()
    return {
        "no_obstacle": "No Obstacle",
        "random_obstacle": "Random Obstacle",
        "weak_wind": "Low Wind",
        "medium_wind": "Medium Wind",
        "strong_wind": "High Wind",
    }.get(key, "")


def _short_figure_title_prefix(experiment_display_name: str, condition_display_name: str) -> str:
    exp_label = _short_experiment_label(experiment_display_name)
    condition_label = _short_condition_label(condition_display_name)
    if condition_label:
        return f"{exp_label}-{condition_label}"
    return exp_label


def paired_metric_arrays(rows: List[Dict[str, float]], metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    baseline_map: Dict[int, Dict[str, float]] = {}
    gru_map: Dict[int, Dict[str, float]] = {}
    for r in rows:
        case_id = int(r["case_id"])
        method = r["method"]
        if method == "ppo_baseline":
            baseline_map[case_id] = r
        elif method == "ppo_gru":
            gru_map[case_id] = r

    common_case_ids = sorted(set(baseline_map.keys()) & set(gru_map.keys()))
    baseline_vals = []
    gru_vals = []
    for cid in common_case_ids:
        rp = baseline_map[cid]
        rb = gru_map[cid]
        # Paired comparison only for cases where both methods actually dropped.
        if int(rp.get("dropped", 0)) != 1 or int(rb.get("dropped", 0)) != 1:
            continue
        vp = _row_metric_value(rp, metric_key)
        vb = _row_metric_value(rb, metric_key)
        if np.isfinite(vp) and np.isfinite(vb):
            baseline_vals.append(vp)
            gru_vals.append(vb)
    return np.asarray(baseline_vals, dtype=np.float64), np.asarray(gru_vals, dtype=np.float64)


def paired_metric_delta_array(rows: List[Dict[str, float]], metric_key: str) -> np.ndarray:
    y_baseline, y_gru = paired_metric_arrays(rows, metric_key)
    n = min(y_baseline.size, y_gru.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    # Delta is defined as: Proposed minus PPO.
    return y_gru[:n] - y_baseline[:n]


def single_method_metric_array(rows: List[Dict[str, float]], method: str, metric_key: str) -> np.ndarray:
    vals = []
    method_rows = [r for r in rows if r["method"] == method]
    method_rows.sort(key=lambda r: int(r.get("case_id", 0)))
    for r in method_rows:
        if int(r.get("dropped", 0)) != 1:
            continue
        v = _row_metric_value(r, metric_key)
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=np.float64)


def plot_trend_figure(
    rows: List[Dict[str, float]],
    metrics: List[Tuple[str, str, str]],
    suptitle: str,
    out_png: str,
    out_pdf: str,
):
    set_paper_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 6.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    c_baseline = method_color("ppo") or "#1f77b4"
    c_gru = method_color("ppo_gru") or "#d62728"

    for ax, (key, title, ylab) in zip(axes, metrics):
        y_baseline, y_gru = paired_metric_arrays(rows, key)
        n = min(y_baseline.size, y_gru.size)
        if n < 5:
            logger.warning(f"Skip metric plot due to insufficient paired finite samples: {key}")
            ax.set_title(f"{title} (insufficient samples)", fontweight="bold", pad=10)
            ax.set_xlabel("DROP Sample Index")
            ax.set_ylabel(ylab)
            ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
            continue
        x = np.arange(1, n + 1)
        y_baseline = y_baseline[:n]
        y_gru = y_gru[:n]
        y_baseline_ema = ema(y_baseline, alpha=0.97)
        y_gru_ema = ema(y_gru, alpha=0.97)

        ax.plot(x, y_baseline_ema, color=c_baseline, linewidth=2.8, label=method_display_name("ppo"))
        ax.plot(x, y_gru_ema, color=c_gru, linewidth=2.8, label=method_display_name("ppo_gru"))
        _set_tight_ylim_from_smoothed(ax, [y_baseline_ema, y_gru_ema])

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("DROP Sample Index")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
        leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95)
        leg.get_frame().set_linewidth(1.2)
        leg.get_frame().set_edgecolor("#222222")

    fig.suptitle(suptitle, fontsize=24, fontweight="bold")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_paired_delta_figure(
    rows: List[Dict[str, float]],
    metrics: List[Tuple[str, str, str]],
    suptitle: str,
    out_png: str,
    out_pdf: str,
):
    set_paper_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 6.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for i, (ax, (key, title, ylab)) in enumerate(zip(axes, metrics)):
        y = paired_metric_delta_array(rows=rows, metric_key=key)
        n = y.size
        if n < 5:
            logger.warning(f"Skip paired-delta plot due to insufficient samples: {key}")
            ax.set_title(f"{title} (insufficient samples)", fontweight="bold", pad=10)
            ax.set_xlabel("Paired DROP Sample Index")
            ax.set_ylabel(ylab)
            ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
            continue

        x = np.arange(1, n + 1)
        y_ema = ema(y, alpha=0.97)
        c = colors[i % len(colors)]

        # Show every paired case and its smoothed trend.
        ax.plot(x, y, color=c, linewidth=1.4, alpha=0.75, label="Per-case delta")
        ax.plot(x, y_ema, color="#111111", linewidth=2.0, alpha=0.95, label="EMA")
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.9)
        _set_tight_ylim_from_smoothed(ax, [y, y_ema])

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("Paired DROP Sample Index")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
        leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95)
        leg.get_frame().set_linewidth(1.2)
        leg.get_frame().set_edgecolor("#222222")

    fig.suptitle(suptitle, fontsize=24, fontweight="bold")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_single_method_trend_figure(
    rows: List[Dict[str, float]],
    method: str,
    metrics: List[Tuple[str, str, str]],
    suptitle: str,
    out_png: str,
    out_pdf: str,
):
    set_paper_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 6.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    c_main = "#1f77b4"
    method_label = method_display_name(method)

    for ax, (key, title, ylab) in zip(axes, metrics):
        y = single_method_metric_array(rows=rows, method=method, metric_key=key)
        n = y.size
        if n < 5:
            logger.warning(f"Skip single-method plot due to insufficient samples: {method}/{key}")
            ax.set_title(f"{title} (insufficient samples)", fontweight="bold", pad=10)
            ax.set_xlabel("DROP Sample Index")
            ax.set_ylabel(ylab)
            ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
            continue
        x = np.arange(1, n + 1)
        y_ema = ema(y, alpha=0.97)

        ax.plot(x, y_ema, color=c_main, linewidth=2.8, label=method_label)
        _set_tight_ylim_from_smoothed(ax, [y_ema])
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("DROP Sample Index")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
        leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95)
        leg.get_frame().set_linewidth(1.2)
        leg.get_frame().set_edgecolor("#222222")

    fig.suptitle(suptitle, fontsize=24, fontweight="bold")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_single_method_cdf_figure(
    rows: List[Dict[str, float]],
    method: str,
    metric_key: str,
    title: str,
    x_label: str,
    out_png: str,
    out_pdf: str,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0),
):
    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 6.2), constrained_layout=True)

    vals = single_method_metric_array(rows=rows, method=method, metric_key=metric_key)
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        logger.warning(f"Skip CDF plot due to insufficient samples: {method}/{metric_key}")
        ax.set_title(f"{title} (insufficient samples)", fontweight="bold", pad=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cumulative Probability")
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
    else:
        vals_sorted = np.sort(vals)
        cdf = np.arange(1, vals_sorted.size + 1, dtype=np.float64) / float(vals_sorted.size)

        c_main = "#1f77b4"
        method_label = method_display_name(method)
        ax.plot(vals_sorted, cdf, color=c_main, linewidth=2.8, label=f"{method_label} CDF")

        for th in thresholds:
            hit_rate = float(np.mean(vals <= th))
            ax.axvline(th, color="#444444", linestyle="--", linewidth=1.2, alpha=0.6)
            ax.text(
                th,
                min(0.98, max(0.05, hit_rate + 0.03)),
                f"R@{th:.1f}m={hit_rate*100.0:.1f}%",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=11,
                color="#333333",
            )

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cumulative Probability")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
        leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.95)
        leg.get_frame().set_linewidth(1.2)
        leg.get_frame().set_edgecolor("#222222")

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _paired_metric_values(rows: List[Dict[str, float]], metric_key: str, use_abs: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y_baseline, y_gru = paired_metric_arrays(rows, metric_key)
    n = min(y_baseline.size, y_gru.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    y_baseline = y_baseline[:n]
    y_gru = y_gru[:n]
    if use_abs:
        y_baseline = np.abs(y_baseline)
        y_gru = np.abs(y_gru)
    finite_mask = np.isfinite(y_baseline) & np.isfinite(y_gru)
    return y_baseline[finite_mask], y_gru[finite_mask]


def _plot_two_method_cdf(ax, vals_baseline: np.ndarray, vals_gru: np.ndarray, title: str, x_label: str):
    if vals_baseline.size < 5 or vals_gru.size < 5:
        ax.set_title(f"{title} (insufficient samples)", fontweight="bold", pad=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cumulative Probability")
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
        return

    vb = np.sort(vals_baseline)
    vg = np.sort(vals_gru)
    cb = np.arange(1, vb.size + 1, dtype=np.float64) / float(vb.size)
    cg = np.arange(1, vg.size + 1, dtype=np.float64) / float(vg.size)

    ax.plot(vb, cb, color=method_color("ppo") or "#1f77b4", linewidth=2.8, label=method_display_name("ppo"))
    ax.plot(vg, cg, color=method_color("ppo_gru") or "#d62728", linewidth=2.8, label=method_display_name("ppo_gru"))

    x_max = float(max(np.percentile(vb, 99.5), np.percentile(vg, 99.5)))
    if np.isfinite(x_max) and x_max > 0:
        ax.set_xlim(left=0.0, right=x_max * 1.05)

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.95)
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_edgecolor("#222222")


def plot_paired_cdf_figure(
    rows: List[Dict[str, float]],
    metrics: List[Tuple[str, str, str, bool]],
    suptitle: str,
    out_png: str,
    out_pdf: str,
):
    set_paper_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 6.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, title, xlab, use_abs) in zip(axes, metrics):
        vb, vg = _paired_metric_values(rows, key, use_abs=use_abs)
        _plot_two_method_cdf(ax=ax, vals_baseline=vb, vals_gru=vg, title=title, x_label=xlab)

    fig.suptitle(suptitle, fontsize=24, fontweight="bold")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _cdf_summary_row(values: np.ndarray, thresholds: Tuple[float, ...]) -> Dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        out = {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
        for th in thresholds:
            out[f"R@{th:.1f}m"] = float("nan")
        return out
    out = {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90.0)),
    }
    for th in thresholds:
        out[f"R@{th:.1f}m"] = float(np.mean(vals <= th))
    return out


def plot_landing_cdf_summary_table_figure(
    rows: List[Dict[str, float]],
    out_png: str,
    out_pdf: str,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0),
):
    set_paper_style()
    vb, vg = _paired_metric_values(rows, "landing_error_xy_m", use_abs=False)
    sb = _cdf_summary_row(vb, thresholds)
    sg = _cdf_summary_row(vg, thresholds)

    col_labels = ["Method", "N", "Mean(m)", "Median(m)", "P90(m)"] + [f"R@{th:.1f}m" for th in thresholds]

    def _row(method_name: str, stats: Dict[str, float]):
        row = [
            method_name,
            f"{int(stats['n'])}",
            f"{stats['mean']:.3f}" if np.isfinite(stats['mean']) else "nan",
            f"{stats['median']:.3f}" if np.isfinite(stats['median']) else "nan",
            f"{stats['p90']:.3f}" if np.isfinite(stats['p90']) else "nan",
        ]
        for th in thresholds:
            k = f"R@{th:.1f}m"
            v = stats[k]
            row.append(f"{100.0*v:.1f}%" if np.isfinite(v) else "nan")
        return row

    cell_text = [
        _row(method_display_name("ppo"), sb),
        _row(method_display_name("ppo_gru"), sg),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 3.8), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.7)
    ax.set_title("Landing Error XY CDF Summary (paired cases, both dropped)", fontweight="bold", pad=12)

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)



def _set_coarse_ylim_from_raw(ax, curves: List[np.ndarray], q_low: float = 1.0, q_high: float = 99.0, min_span: float = 20.0, tick_step: float = 5.0):
    valid = []
    for c in curves:
        if c is None:
            continue
        c = np.asarray(c, dtype=np.float64)
        c = c[np.isfinite(c)]
        if c.size > 0:
            valid.append(c)
    if not valid:
        return
    all_vals = np.concatenate(valid, axis=0)
    lo = float(np.percentile(all_vals, q_low))
    hi = float(np.percentile(all_vals, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi < lo:
        lo, hi = hi, lo
    span = max(hi - lo, min_span)
    center = 0.5 * (hi + lo)
    lo2 = center - 0.5 * span
    hi2 = center + 0.5 * span
    if tick_step > 0:
        lo2 = np.floor(lo2 / tick_step) * tick_step
        hi2 = np.ceil(hi2 / tick_step) * tick_step
    if np.isfinite(lo2) and np.isfinite(hi2) and hi2 > lo2:
        ax.set_ylim(lo2, hi2)


def plot_paired_attitude_shadow_figure(
    rows: List[Dict[str, float]],
    metrics: List[Tuple[str, str, str]],
    suptitle: str,
    out_png: str,
    out_pdf: str,
):
    set_paper_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 6.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    c_baseline = method_color("ppo") or "#1f77b4"
    c_gru = method_color("ppo_gru") or "#d62728"

    for ax, (key, title, ylab) in zip(axes, metrics):
        y_baseline, y_gru = paired_metric_arrays(rows, key)
        n = min(y_baseline.size, y_gru.size)
        if n < 5:
            logger.warning(f"Skip attitude shadow plot due to insufficient paired finite samples: {key}")
            ax.set_title(f"{title} (insufficient samples)", fontweight="bold", pad=10)
            ax.set_xlabel("DROP Sample Index")
            ax.set_ylabel(ylab)
            ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
            continue

        x = np.arange(1, n + 1)
        y_baseline = y_baseline[:n]
        y_gru = y_gru[:n]
        y_baseline_ema = ema(y_baseline, alpha=0.97)
        y_gru_ema = ema(y_gru, alpha=0.97)

        # first-version style: raw traces as light background + smoothed foreground
        ax.plot(x, y_baseline, color=c_baseline, linewidth=1.0, alpha=0.18)
        ax.plot(x, y_gru, color=c_gru, linewidth=1.0, alpha=0.18)
        ax.plot(x, y_baseline_ema, color=c_baseline, linewidth=2.8, label=method_display_name("ppo"))
        ax.plot(x, y_gru_ema, color=c_gru, linewidth=2.8, label=method_display_name("ppo_gru"))

        _set_coarse_ylim_from_raw(ax, [y_baseline, y_gru])

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("DROP Sample Index")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
        leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95)
        leg.get_frame().set_linewidth(1.2)
        leg.get_frame().set_edgecolor("#222222")

    fig.suptitle(suptitle, fontsize=24, fontweight="bold")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_attitude_summary_table_figure(
    rows: List[Dict[str, float]],
    out_png: str,
    out_pdf: str,
):
    set_paper_style()

    def _stats(vals: np.ndarray) -> Dict[str, float]:
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "p90": float("nan")}
        return {
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90.0)),
        }

    metrics = [
        ("roll_deg", "|Roll|"),
        ("pitch_deg", "|Pitch|"),
        ("yaw_deg", "|Yaw|"),
    ]

    col_labels = ["Method", "Metric", "N", "Mean(deg)", "Std(deg)", "Median(deg)", "P90(deg)"]
    cell_text = []

    for method in ("ppo_baseline", "ppo_gru"):
        method_name = method_display_name(method)
        for key, label in metrics:
            vals = single_method_metric_array(rows=rows, method=method, metric_key=key)
            vals = np.abs(vals)
            st = _stats(vals)
            cell_text.append([
                method_name,
                label,
                f"{int(st['n'])}",
                f"{st['mean']:.3f}" if np.isfinite(st['mean']) else "nan",
                f"{st['std']:.3f}" if np.isfinite(st['std']) else "nan",
                f"{st['median']:.3f}" if np.isfinite(st['median']) else "nan",
                f"{st['p90']:.3f}" if np.isfinite(st['p90']) else "nan",
            ])

    fig, ax = plt.subplots(1, 1, figsize=(13.8, 4.6), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.65)
    ax.set_title("Pre-DROP Attitude Summary (absolute angles, paired DROP samples)", fontweight="bold", pad=12)

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)



def plot_paired_scatter_figure(
    rows: List[Dict[str, float]],
    metrics: List[Tuple[str, str, bool]],
    suptitle: str,
    out_png: str,
    out_pdf: str,
):
    """
    Paired scatter: x=baseline, y=GRU for each matched DROP case.
    metrics entries: (metric_key, title, use_abs)
    """
    set_paper_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 6.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, title, use_abs) in zip(axes, metrics):
        vb, vg = _paired_metric_values(rows, key, use_abs=use_abs)
        if vb.size < 5 or vg.size < 5:
            logger.warning(f"Skip paired scatter due to insufficient samples: {key}")
            ax.set_title(f"{title} (insufficient samples)", fontweight="bold", pad=10)
            ax.set_xlabel(method_display_name("ppo"))
            ax.set_ylabel(method_display_name("ppo_gru"))
            ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
            continue

        delta = vg - vb
        gru_better_rate = float(np.mean(delta < 0.0))
        delta_median = float(np.median(delta))

        lo = float(min(np.percentile(vb, 0.5), np.percentile(vg, 0.5)))
        hi = float(max(np.percentile(vb, 99.5), np.percentile(vg, 99.5)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(min(np.min(vb), np.min(vg)))
            hi = float(max(np.max(vb), np.max(vg)))
        pad = 0.05 * max(hi - lo, 1e-6)
        xmin, xmax = lo - pad, hi + pad

        ax.scatter(vb, vg, s=18, alpha=0.45, color="#1f77b4", edgecolors="none")
        ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1.4, color="#333333", alpha=0.9)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(method_display_name("ppo"))
        ax.set_ylabel(method_display_name("ppo_gru"))
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")

        summary_text = (
            f"GRU better={gru_better_rate*100.0:.1f}%\n"
            f"median(Δ=GRU-base)={delta_median:.3f}"
        )
        ax.text(
            0.03,
            0.97,
            summary_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#888888"),
        )

    fig.suptitle(suptitle, fontsize=24, fontweight="bold")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_attitude_impulse_summary_tables_figure(
    rows: List[Dict[str, float]],
    out_png: str,
    out_pdf: str,
):
    set_paper_style()

    def _stats(vals: np.ndarray) -> Dict[str, float]:
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "p90": float("nan")}
        return {
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90.0)),
        }

    def _build_rows(metrics: List[Tuple[str, str, bool]]):
        rows_out = []
        for method in ("ppo_baseline", "ppo_gru"):
            method_name = method_display_name(method)
            for key, label, use_abs in metrics:
                vals = single_method_metric_array(rows=rows, method=method, metric_key=key)
                if use_abs:
                    vals = np.abs(vals)
                st = _stats(vals)
                rows_out.append([
                    method_name,
                    label,
                    f"{int(st['n'])}",
                    f"{st['mean']:.3f}" if np.isfinite(st['mean']) else "nan",
                    f"{st['std']:.3f}" if np.isfinite(st['std']) else "nan",
                    f"{st['median']:.3f}" if np.isfinite(st['median']) else "nan",
                    f"{st['p90']:.3f}" if np.isfinite(st['p90']) else "nan",
                ])
        return rows_out

    attitude_metrics = [
        ("roll_deg", "|Roll|", True),
        ("pitch_deg", "|Pitch|", True),
        ("yaw_deg", "|Yaw|", True),
    ]
    impulse_metrics = [
        ("impulse_metric", "Impulse Metric", False),
    ]

    col_labels = ["Method", "Metric", "N", "Mean", "Std", "Median", "P90"]
    attitude_rows = _build_rows(attitude_metrics)
    impulse_rows = _build_rows(impulse_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(17.2, 5.2), constrained_layout=True)

    axes[0].axis("off")
    t0 = axes[0].table(cellText=attitude_rows, colLabels=col_labels, cellLoc="center", loc="center")
    t0.auto_set_font_size(False)
    t0.set_fontsize(11)
    t0.scale(1.0, 1.45)
    axes[0].set_title("Attitude Summary (|roll|, |pitch|, |yaw|)", fontweight="bold", pad=10)

    axes[1].axis("off")
    t1 = axes[1].table(cellText=impulse_rows, colLabels=col_labels, cellLoc="center", loc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(11)
    t1.scale(1.0, 1.9)
    axes[1].set_title("Impact Summary (impulse_metric)", fontweight="bold", pad=10)

    fig.suptitle("Pre-DROP Attitude & Impact Summary Tables", fontsize=22, fontweight="bold")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)



def _methods_in_rows(rows: List[Dict[str, float]], preferred_order: Optional[List[str]] = None) -> List[str]:
    present = sorted(set(str(r["method"]) for r in rows))
    if preferred_order is None:
        return present
    ordered: List[str] = []
    seen = set()
    for preferred in preferred_order:
        preferred_key = _canonical_method_key(preferred)
        for method in present:
            if method in seen:
                continue
            if _canonical_method_key(method) == preferred_key:
                ordered.append(method)
                seen.add(method)
    tail = [m for m in present if m not in seen]
    return ordered + tail


def plot_multi_method_cdf_single_metric(
    rows: List[Dict[str, float]],
    methods: List[str],
    metric_key: str,
    title: str,
    x_label: str,
    out_pdf: str,
    use_abs: bool = False,
):
    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 6.2), constrained_layout=True)
    plotted = 0
    for method in methods:
        vals = single_method_metric_array(rows=rows, method=method, metric_key=metric_key)
        vals = vals[np.isfinite(vals)]
        if use_abs:
            vals = np.abs(vals)
        if vals.size < 5:
            continue
        v_sorted = np.sort(vals)
        cdf = np.arange(1, v_sorted.size + 1, dtype=np.float64) / float(v_sorted.size)
        ax.plot(
            v_sorted,
            cdf,
            linewidth=2.6,
            label=method_display_name(method),
            color=method_color(method),
        )
        plotted += 1

    if plotted == 0:
        ax.set_title(f"{title} (insufficient samples)", fontweight="bold", pad=10)
    else:
        ax.set_title(title, fontweight="bold", pad=10)
        leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.95)
        leg.get_frame().set_linewidth(1.2)
        leg.get_frame().set_edgecolor("#222222")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _moving_mean_std(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return vals.copy(), vals.copy()
    if vals.size < 3:
        return vals.copy(), np.zeros_like(vals, dtype=np.float64)
    w = int(max(3, min(window, vals.size)))
    if w % 2 == 0:
        w -= 1
    w = int(max(3, min(w, vals.size if vals.size % 2 == 1 else vals.size - 1)))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    mean = np.convolve(vals, kernel, mode="same")
    mean_sq = np.convolve(vals * vals, kernel, mode="same")
    var = np.clip(mean_sq - mean * mean, a_min=0.0, a_max=None)
    std = np.sqrt(var)
    return mean, std


def plot_multi_method_roll_pitch_case_series(
    rows: List[Dict[str, float]],
    methods: List[str],
    title: str,
    out_pdf: str,
):
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(17.8, 6.4), constrained_layout=True, sharex=False)
    metric_specs = [
        ("roll_deg", "Roll"),
        ("pitch_deg", "Pitch"),
    ]

    for ax, (metric_key, y_label) in zip(axes, metric_specs):
        ax.set_facecolor("#efefef")
        plotted = 0
        max_len = 0
        for method in methods:
            y = single_method_metric_array(rows=rows, method=method, metric_key=metric_key)
            y = y[np.isfinite(y)]
            if y.size < 5:
                continue
            x = np.arange(1, y.size + 1, dtype=np.int64)
            win = max(11, min(101, int(max(11, y.size // 18))))
            y_mean, y_std = _moving_mean_std(y, window=win)

            c = method_color(method)
            ax.plot(x, y, color=c, linewidth=0.8, alpha=0.14)
            ax.fill_between(
                x,
                y_mean - y_std,
                y_mean + y_std,
                color=c,
                alpha=0.18,
                linewidth=0.0,
            )
            ax.plot(
                x,
                y_mean,
                color=c,
                linewidth=2.6,
                label=method_display_name(method),
            )
            max_len = max(max_len, int(y.size))
            plotted += 1

        if plotted == 0:
            ax.set_title(f"{y_label} (insufficient samples)", fontweight="bold", pad=8)
        else:
            ax.set_title(y_label, fontweight="bold", pad=8)
        if max_len > 0:
            ax.set_xlim(1, max_len)
        ax.set_ylabel("Pre-DROP Angle (deg)")
        ax.set_xlabel("DROP Sample Index")
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
        if plotted > 0:
            leg = ax.legend(loc="upper right", frameon=True, fancybox=False, framealpha=0.95)
            leg.get_frame().set_linewidth(1.1)
            leg.get_frame().set_edgecolor("#222222")

    fig.suptitle(title, fontsize=22, fontweight="bold")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _safe_summary(values: np.ndarray) -> Dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "p50": float(np.percentile(vals, 50.0)),
        "p90": float(np.percentile(vals, 90.0)),
        "p95": float(np.percentile(vals, 95.0)),
    }


def _pct_text(num: int, den: int) -> str:
    if den <= 0:
        return "nan"
    return f"{100.0 * float(num) / float(den):.1f}%"


def _method_case_map(rows: List[Dict[str, float]]) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for r in rows:
        method = str(r.get("method", ""))
        case_id = int(r.get("case_id", -1))
        if not method or case_id < 0:
            continue
        out.setdefault(method, {})[case_id] = r
    return out


def _method_done_rows(rows: List[Dict[str, float]], method: str) -> List[Dict[str, float]]:
    return [
        r
        for r in rows
        if str(r.get("method", "")) == method and int(r.get("done_reached", 0)) == 1
    ]


def plot_four_pdf_comparison_table(
    rows: List[Dict[str, float]],
    methods: List[str],
    title: str,
    out_pdf: str,
    large_error_threshold_m: float = 3.0,
):
    set_paper_style()
    col_labels_main = [
        "Method",
        "N(drop)",
        "Precision Mean(m)",
        "Precision P50(m)",
        "Precision P90(m)",
        "R@1m",
        "R@2m",
        "R@3m",
        "|Roll| Mean",
        "|Roll| P90",
        "|Pitch| Mean",
        "|Pitch| P90",
        "Impact Mean",
        "Impact P90",
        "Impact P95",
    ]
    main_rows: List[List[str]] = []

    for method in methods:
        landing = single_method_metric_array(rows=rows, method=method, metric_key="landing_error_xy_m")
        roll = np.abs(single_method_metric_array(rows=rows, method=method, metric_key="roll_deg"))
        pitch = np.abs(single_method_metric_array(rows=rows, method=method, metric_key="pitch_deg"))
        impulse = single_method_metric_array(rows=rows, method=method, metric_key="impulse_metric")

        s_land = _safe_summary(landing)
        s_roll = _safe_summary(roll)
        s_pitch = _safe_summary(pitch)
        s_imp = _safe_summary(impulse)
        n_drop = max(int(s_land["n"]), int(s_roll["n"]), int(s_pitch["n"]), int(s_imp["n"]))
        hit1 = float(np.mean(landing <= 1.0)) if landing.size > 0 else float("nan")
        hit2 = float(np.mean(landing <= 2.0)) if landing.size > 0 else float("nan")
        hit3 = float(np.mean(landing <= 3.0)) if landing.size > 0 else float("nan")

        main_rows.append(
            [
                method_display_name(method),
                f"{n_drop}",
                f"{s_land['mean']:.3f}" if np.isfinite(s_land["mean"]) else "nan",
                f"{s_land['p50']:.3f}" if np.isfinite(s_land["p50"]) else "nan",
                f"{s_land['p90']:.3f}" if np.isfinite(s_land["p90"]) else "nan",
                f"{100.0*hit1:.1f}%" if np.isfinite(hit1) else "nan",
                f"{100.0*hit2:.1f}%" if np.isfinite(hit2) else "nan",
                f"{100.0*hit3:.1f}%" if np.isfinite(hit3) else "nan",
                f"{s_roll['mean']:.3f}" if np.isfinite(s_roll["mean"]) else "nan",
                f"{s_roll['p90']:.3f}" if np.isfinite(s_roll["p90"]) else "nan",
                f"{s_pitch['mean']:.3f}" if np.isfinite(s_pitch["mean"]) else "nan",
                f"{s_pitch['p90']:.3f}" if np.isfinite(s_pitch["p90"]) else "nan",
                f"{s_imp['mean']:.3f}" if np.isfinite(s_imp["mean"]) else "nan",
                f"{s_imp['p90']:.3f}" if np.isfinite(s_imp["p90"]) else "nan",
                f"{s_imp['p95']:.3f}" if np.isfinite(s_imp["p95"]) else "nan",
            ]
        )

    # No-drop breakdown by method.
    col_labels_nodrop = [
        "Method",
        "N(done)",
        "N(drop)",
        "Drop Rate",
        "N(no_drop)",
        "No-drop Rate",
        "Crash Rate",
        "Crash+NoDrop",
        "Crash in NoDrop",
        "Timeout+NoDrop",
        "Both Flags",
    ]
    nodrop_rows: List[List[str]] = []
    for method in methods:
        m_rows = _method_done_rows(rows=rows, method=method)
        n_done = len(m_rows)
        n_drop = sum(int(r.get("dropped", 0)) for r in m_rows)
        n_no_drop = n_done - n_drop
        n_crash = sum(1 for r in m_rows if int(r.get("crashed", 0)) == 1)
        n_crash_no_drop = sum(
            1
            for r in m_rows
            if int(r.get("dropped", 0)) == 0 and int(r.get("crashed", 0)) == 1
        )
        n_timeout_no_drop = sum(
            1
            for r in m_rows
            if int(r.get("dropped", 0)) == 0 and int(r.get("timeout", 0)) == 1
        )
        n_both = sum(
            1
            for r in m_rows
            if int(r.get("dropped", 0)) == 0
            and int(r.get("crashed", 0)) == 1
            and int(r.get("timeout", 0)) == 1
        )

        nodrop_rows.append(
            [
                method_display_name(method),
                f"{n_done}",
                f"{n_drop}",
                _pct_text(n_drop, n_done),
                f"{n_no_drop}",
                _pct_text(n_no_drop, n_done),
                _pct_text(n_crash, n_done),
                f"{n_crash_no_drop}",
                _pct_text(n_crash_no_drop, n_no_drop),
                f"{n_timeout_no_drop}",
                f"{n_both}",
            ]
        )

    # Cross-case analysis: among MPC no-drop cases, how learned policies behave.
    col_labels_cross = [
        "Method",
        "In MPC-NoDrop",
        "Dropped",
        "Drop Rate",
        f"Err>{large_error_threshold_m:.1f}m",
        "LargeErr Rate (in dropped)",
        "Mean Err (in dropped)",
    ]
    cross_rows: List[List[str]] = []
    case_map = _method_case_map(rows)
    if "mpc" in case_map:
        mpc_ref = {
            cid
            for cid, r in case_map["mpc"].items()
            if int(r.get("done_reached", 0)) == 1 and int(r.get("dropped", 0)) == 0
        }
        for method in [m for m in methods if m != "mpc"]:
            if method not in case_map:
                continue
            in_ref_rows = [
                case_map[method][cid]
                for cid in sorted(mpc_ref)
                if cid in case_map[method] and int(case_map[method][cid].get("done_reached", 0)) == 1
            ]
            n_ref = len(in_ref_rows)
            dropped_rows = [r for r in in_ref_rows if int(r.get("dropped", 0)) == 1]
            n_drop = len(dropped_rows)
            err_vals = np.asarray(
                [float(r.get("landing_error_xy_m", float("nan"))) for r in dropped_rows], dtype=np.float64
            )
            err_vals = err_vals[np.isfinite(err_vals)]
            n_large = int(np.sum(err_vals > float(large_error_threshold_m))) if err_vals.size > 0 else 0
            mean_err = float(np.mean(err_vals)) if err_vals.size > 0 else float("nan")

            cross_rows.append(
                [
                    method_display_name(method),
                    f"{n_ref}",
                    f"{n_drop}",
                    _pct_text(n_drop, n_ref),
                    f"{n_large}",
                    _pct_text(n_large, n_drop),
                    f"{mean_err:.3f}" if np.isfinite(mean_err) else "nan",
                ]
            )

    fig, axes = plt.subplots(3, 1, figsize=(21.8, 11.6), constrained_layout=True)

    axes[0].axis("off")
    t0 = axes[0].table(cellText=main_rows, colLabels=col_labels_main, cellLoc="center", loc="center")
    t0.auto_set_font_size(False)
    t0.set_fontsize(10)
    t0.scale(1.0, 1.45)
    axes[0].set_title("Dropped-Only Quality", fontweight="bold", pad=8)

    axes[1].axis("off")
    t1 = axes[1].table(cellText=nodrop_rows, colLabels=col_labels_nodrop, cellLoc="center", loc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)
    t1.scale(1.0, 1.42)
    axes[1].set_title("No-Drop Breakdown (done but no release)", fontweight="bold", pad=8)

    axes[2].axis("off")
    if cross_rows:
        t2 = axes[2].table(cellText=cross_rows, colLabels=col_labels_cross, cellLoc="center", loc="center")
        t2.auto_set_font_size(False)
        t2.set_fontsize(10)
        t2.scale(1.0, 1.42)
    else:
        t2 = axes[2].table(
            cellText=[["No MPC rows available for cross-case analysis.", "", "", "", "", "", ""]],
            colLabels=col_labels_cross,
            cellLoc="center",
            loc="center",
        )
        t2.auto_set_font_size(False)
        t2.set_fontsize(10)
        t2.scale(1.0, 1.42)
    learned_labels = "/".join(method_display_name(m) for m in methods if m != "mpc")
    axes[2].set_title(
        f"Cross-Case Check: {learned_labels} on MPC No-Drop Cases "
        f"(large error > {large_error_threshold_m:.1f}m)",
        fontweight="bold",
        pad=8,
    )

    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _preferred_method_order() -> List[str]:
    return ["ppo", "ppo_gru", "ppo_rnn", "mpc"]


def save_condition_eval_artifacts(
    raw_dir: str,
    stats_dir: str,
    rows: List[Dict[str, float]],
    trace_records: List[Dict[str, Any]],
    summaries: List[MethodSummary],
    methods: List[str],
) -> Tuple[str, str]:
    raw_csv_path = os.path.join(raw_dir, "combined_drop_metrics.csv")
    trace_npz_path = os.path.join(raw_dir, "post_drop_trace.npz")
    write_rows_csv(raw_csv_path, rows)
    write_post_drop_traces_npz(trace_npz_path, trace_records)
    save_summary_tables(stats_dir=stats_dir, rows=rows, summaries=summaries)
    save_condition_key_metrics_csv(
        stats_dir=stats_dir,
        rows=rows,
        summaries=summaries,
        methods=methods,
    )
    return raw_csv_path, trace_npz_path


def render_condition_outputs(
    rows: List[Dict[str, Any]],
    methods: List[str],
    experiment_display_name: str,
    condition_display_name: str,
    fig_dir: str,
    large_error_threshold_m: float,
):
    title_prefix = _short_figure_title_prefix(
        experiment_display_name=experiment_display_name,
        condition_display_name=condition_display_name,
    )
    # 1) Precision CDF
    plot_multi_method_cdf_single_metric(
        rows=rows,
        methods=methods,
        metric_key="landing_error_xy_m",
        title=f"{title_prefix}: Landing Precision CDF",
        x_label="Landing Error XY (m)",
        out_pdf=os.path.join(fig_dir, "precision_cdf.pdf"),
        use_abs=False,
    )

    # 2) Case-level ROLL/PITCH series (with shadow)
    plot_multi_method_roll_pitch_case_series(
        rows=rows,
        methods=methods,
        title=f"{title_prefix}: Roll/Pitch",
        out_pdf=os.path.join(fig_dir, "roll_pitch_case.pdf"),
    )

    # 3) Impact CDF
    plot_multi_method_cdf_single_metric(
        rows=rows,
        methods=methods,
        metric_key="impulse_metric",
        title=f"{title_prefix}: Impact CDF",
        x_label="Impact Metric",
        out_pdf=os.path.join(fig_dir, "impact_cdf.pdf"),
        use_abs=False,
    )

    # 4) Summary table PDF
    plot_four_pdf_comparison_table(
        rows=rows,
        methods=methods,
        title=f"{title_prefix}: Summary Table",
        out_pdf=os.path.join(fig_dir, "summary_table.pdf"),
        large_error_threshold_m=float(large_error_threshold_m),
    )


def _wind_condition_style(condition_name: str) -> Tuple[str, str]:
    if condition_name == "weak_wind":
        return "Low Wind", "#2ca02c"
    if condition_name == "medium_wind":
        return "Medium Wind", "#ff7f0e"
    if condition_name == "strong_wind":
        return "High Wind", "#d62728"
    return condition_name, "#1f77b4"


def save_post_drop_wind_summary_csv(
    out_csv: str,
    condition_trace_bundles: Dict[str, Dict[str, np.ndarray]],
    methods: List[str],
):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "wind_condition",
                "wind_label",
                "n_valid_drop",
                "mean_final_error_m",
                "median_final_error_m",
                "p90_final_error_m",
                "mean_fall_time_s",
                "median_fall_time_s",
            ]
        )
        for condition_name, bundle in condition_trace_bundles.items():
            wind_label, _ = _wind_condition_style(condition_name)
            for method in methods:
                mask = (bundle["method"] == method) & (bundle["crashed"] == 0)
                n = int(np.sum(mask))
                final_err = bundle["error_xy_m"][mask, -1] if n > 0 else np.asarray([], dtype=np.float32)
                fall_time = (
                    bundle["trace_fall_time_s"][mask] if n > 0 else np.asarray([], dtype=np.float32)
                )
                mean_final = float(np.mean(final_err)) if final_err.size > 0 else float("nan")
                median_final = float(np.median(final_err)) if final_err.size > 0 else float("nan")
                p90_final = (
                    float(np.percentile(final_err, 90.0)) if final_err.size > 0 else float("nan")
                )
                mean_fall = float(np.mean(fall_time)) if fall_time.size > 0 else float("nan")
                median_fall = float(np.median(fall_time)) if fall_time.size > 0 else float("nan")
                w.writerow(
                    [
                        method_display_name(method),
                        condition_name,
                        wind_label,
                        n,
                        mean_final,
                        median_final,
                        p90_final,
                        mean_fall,
                        median_fall,
                    ]
                )


def plot_post_drop_wind_error_evolution(
    condition_trace_bundles: Dict[str, Dict[str, np.ndarray]],
    methods: List[str],
    title: str,
    out_pdf: str,
):
    set_paper_style()
    n_methods = len(methods)
    if n_methods <= 0:
        return
    if n_methods <= 3:
        ncols = n_methods
    else:
        ncols = 2
    nrows = int(math.ceil(float(n_methods) / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.6 * ncols, 5.6 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    ordered_conditions = [c for c in ["weak_wind", "medium_wind", "strong_wind"] if c in condition_trace_bundles]

    for ax_idx, method in enumerate(methods):
        ax = axes.flat[ax_idx]
        for condition_name in ordered_conditions:
            bundle = condition_trace_bundles[condition_name]
            mask = (bundle["method"] == method) & (bundle["crashed"] == 0)
            n = int(np.sum(mask))
            if n <= 0:
                continue
            x = np.asarray(bundle["norm_t"][mask][0], dtype=np.float64)
            curves = np.asarray(bundle["error_xy_m"][mask], dtype=np.float64)
            if curves.ndim != 2 or curves.shape[1] <= 0:
                continue
            median = np.nanmedian(curves, axis=0)
            q1 = np.nanpercentile(curves, 25.0, axis=0)
            q3 = np.nanpercentile(curves, 75.0, axis=0)
            p90 = np.nanpercentile(curves, 90.0, axis=0)
            wind_label, color = _wind_condition_style(condition_name)
            ax.plot(x, median, color=color, linewidth=2.6, label=f"{wind_label} (N={n})")
            ax.fill_between(x, q1, q3, color=color, alpha=0.18)
            ax.plot(x, p90, color=color, linewidth=1.2, linestyle="--", alpha=0.75)

        ax.set_title(method_display_name(method), fontweight="bold", pad=10)
        ax.set_xlabel("Normalized Post-DROP Time")
        ax.set_ylabel("Post-DROP XY Error (m)")
        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.5, color="#b0b0b0")
        leg = ax.legend(loc="best", frameon=True, fancybox=False, framealpha=0.95)
        leg.get_frame().set_linewidth(1.0)
        leg.get_frame().set_edgecolor("#222222")

    for ax in axes.flat[n_methods:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def render_experiment_level_outputs(exp_root: str, exp: ExperimentSpec):
    if exp.name not in {"exp4_fixedH_multiW_obs", "exp4_multiH_multiW_noObs"}:
        return
    _, exp_stats_dir, exp_fig_dir = ensure_dirs(exp_root)
    condition_trace_bundles: Dict[str, Dict[str, np.ndarray]] = {}
    for cond in exp.conditions:
        trace_path = os.path.join(exp_root, cond.name, "raw", "post_drop_trace.npz")
        if not os.path.exists(trace_path):
            logger.warning(f"Skip experiment-level wind figure, missing trace file: {trace_path}")
            return
        condition_trace_bundles[cond.name] = load_post_drop_traces_npz(trace_path)

    present_methods = set()
    for bundle in condition_trace_bundles.values():
        for method in bundle.get("method", np.asarray([], dtype="<U16")):
            present_methods.add(str(method))
    methods = _methods_in_rows(
        [{"method": m} for m in present_methods],
        preferred_order=_preferred_method_order(),
    )
    if not methods:
        logger.warning("Skip experiment-level wind figure due to empty post-drop traces.")
        return

    save_post_drop_wind_summary_csv(
        out_csv=os.path.join(exp_stats_dir, "post_drop_wind_summary.csv"),
        condition_trace_bundles=condition_trace_bundles,
        methods=methods,
    )
    plot_post_drop_wind_error_evolution(
        condition_trace_bundles=condition_trace_bundles,
        methods=methods,
        title=f"{_short_experiment_label(exp.display_name)}: Post-DROP Error Evolution",
        out_pdf=os.path.join(exp_fig_dir, "post_drop_error_evolution_wind.pdf"),
    )


def _compute_key_metric_stats(rows: List[Dict[str, float]], method: str) -> Dict[str, float]:
    landing = single_method_metric_array(rows=rows, method=method, metric_key="landing_error_xy_m")
    roll = np.abs(single_method_metric_array(rows=rows, method=method, metric_key="roll_deg"))
    pitch = np.abs(single_method_metric_array(rows=rows, method=method, metric_key="pitch_deg"))

    def _safe(arr: np.ndarray, fn: Callable[[np.ndarray], float]) -> float:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(fn(arr))

    return {
        "n_drop": int(np.isfinite(landing).sum()),
        "landing_mean_m": _safe(landing, np.mean),
        "landing_p90_m": _safe(landing, lambda x: np.percentile(x, 90.0)),
        "roll_abs_mean_deg": _safe(roll, np.mean),
        "pitch_abs_mean_deg": _safe(pitch, np.mean),
    }


def _summary_map_by_method(summaries: List[MethodSummary]) -> Dict[str, MethodSummary]:
    return {s.method: s for s in summaries}


def save_condition_key_metrics_csv(
    stats_dir: str,
    rows: List[Dict[str, float]],
    summaries: List[MethodSummary],
    methods: List[str],
) -> str:
    out_path = os.path.join(stats_dir, "key_metrics.csv")
    summary_map = _summary_map_by_method(summaries)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "n_drop",
                "landing_mean_m",
                "landing_p90_m",
                "roll_abs_mean_deg",
                "pitch_abs_mean_deg",
                "no_drop_rate",
            ]
        )
        for method in methods:
            st = _compute_key_metric_stats(rows, method)
            sm = summary_map.get(method, None)
            no_drop_rate = float("nan")
            if sm is not None:
                no_drop_rate = float(sm.no_drop_done) / float(max(sm.done_total, 1))
            w.writerow(
                [
                    method_display_name(method),
                    st["n_drop"],
                    st["landing_mean_m"],
                    st["landing_p90_m"],
                    st["roll_abs_mean_deg"],
                    st["pitch_abs_mean_deg"],
                    no_drop_rate,
                ]
            )
    return out_path


def plot_condition_key_metrics_table_pdf(
    rows: List[Dict[str, float]],
    summaries: List[MethodSummary],
    methods: List[str],
    out_pdf: str,
    title: str,
):
    set_paper_style()
    summary_map = _summary_map_by_method(summaries)
    col_labels = [
        "Method",
        "N(drop)",
        "Landing Mean (m)",
        "Landing P90 (m)",
        "|Roll| Mean (deg)",
        "|Pitch| Mean (deg)",
        "No-Drop Rate",
    ]
    cell_text = []
    for method in methods:
        st = _compute_key_metric_stats(rows, method)
        sm = summary_map.get(method, None)
        no_drop_rate = float("nan")
        if sm is not None:
            no_drop_rate = float(sm.no_drop_done) / float(max(sm.done_total, 1))
        cell_text.append(
            [
                method_display_name(method),
                f"{int(st['n_drop'])}",
                f"{st['landing_mean_m']:.3f}" if np.isfinite(st["landing_mean_m"]) else "nan",
                f"{st['landing_p90_m']:.3f}" if np.isfinite(st["landing_p90_m"]) else "nan",
                f"{st['roll_abs_mean_deg']:.3f}" if np.isfinite(st["roll_abs_mean_deg"]) else "nan",
                f"{st['pitch_abs_mean_deg']:.3f}" if np.isfinite(st["pitch_abs_mean_deg"]) else "nan",
                f"{100.0*no_drop_rate:.2f}%" if np.isfinite(no_drop_rate) else "nan",
            ]
        )

    fig, ax = plt.subplots(1, 1, figsize=(14.2, 4.4), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.65)
    ax.set_title(title, fontweight="bold", pad=12)
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_experiment_overview_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_experiment_overview_table_pdf(
    rows: List[Dict[str, Any]],
    out_pdf: str,
    title: str,
):
    set_paper_style()
    if not rows:
        return

    col_labels = [
        "Condition",
        "Method",
        "N(drop)",
        "Landing Mean (m)",
        "Landing P90 (m)",
        "|Roll| Mean (deg)",
        "|Pitch| Mean (deg)",
        "No-Drop Rate",
    ]
    cell_text = []
    for r in rows:
        cell_text.append(
            [
                r["condition"],
                r["method"],
                f"{int(r['n_drop'])}",
                f"{r['landing_mean_m']:.3f}" if np.isfinite(r["landing_mean_m"]) else "nan",
                f"{r['landing_p90_m']:.3f}" if np.isfinite(r["landing_p90_m"]) else "nan",
                f"{r['roll_abs_mean_deg']:.3f}" if np.isfinite(r["roll_abs_mean_deg"]) else "nan",
                f"{r['pitch_abs_mean_deg']:.3f}" if np.isfinite(r["pitch_abs_mean_deg"]) else "nan",
                f"{100.0*r['no_drop_rate']:.2f}%" if np.isfinite(r["no_drop_rate"]) else "nan",
            ]
        )

    fig_h = max(4.6, 0.42 * len(cell_text) + 2.6)
    fig, ax = plt.subplots(1, 1, figsize=(15.8, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    ax.set_title(title, fontweight="bold", pad=12)
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _experiment_specs() -> List[ExperimentSpec]:
    fixed_height = 10.0
    fixed_wind = 2.0
    weak_wind = 1.0
    med_wind = 2.0
    strong_wind = 3.0
    random_height_min = 4.0
    random_height_max = 13.0

    common_base = {
        "spawn_z_curriculum_enable": False,
        "dryden_config.enable_dryden": False,
        "gmm_force_config.local_wind_max_speed_min": 0.0,
        "gmm_force_config.local_wind_max_speed_max": 0.0,
        "drop_reward_config.drop_obstacle_radius_min": 0.1,
        "drop_reward_config.drop_obstacle_radius_max": 0.1,
    }

    exp1 = ExperimentSpec(
        name="exp1_fixedH_fixedW_noObs",
        display_name="Experiment 1: Fixed Height + Fixed Wind + No Obstacle",
        conditions=[
            EvalCondition(
                name="main",
                display_name="fixed_h_fixed_w_no_obs",
                overrides={
                    **common_base,
                    "spawn_use_random_z": False,
                    "spawn_fixed_z": fixed_height,
                    "gmm_force_config.main_wind_speed_min": fixed_wind,
                    "gmm_force_config.main_wind_speed_max": fixed_wind,
                    "drop_reward_config.drop_obstacle_enable": False,
                    "drop_reward_config.drop_obstacle_spawn_prob": 0.0,
                },
            )
        ],
    )

    exp2 = ExperimentSpec(
        name="exp2_fixedH_fixedW_obs_ablation",
        display_name="Experiment 2: Fixed Height + Fixed Wind + Obstacle Ablation",
        conditions=[
            EvalCondition(
                name="no_obstacle",
                display_name="no_obstacle",
                overrides={
                    **common_base,
                    "spawn_use_random_z": False,
                    "spawn_fixed_z": fixed_height,
                    "gmm_force_config.main_wind_speed_min": fixed_wind,
                    "gmm_force_config.main_wind_speed_max": fixed_wind,
                    "drop_reward_config.drop_obstacle_enable": False,
                    "drop_reward_config.drop_obstacle_spawn_prob": 0.0,
                },
            ),
            EvalCondition(
                name="random_obstacle",
                display_name="random_obstacle",
                overrides={
                    **common_base,
                    "spawn_use_random_z": False,
                    "spawn_fixed_z": fixed_height,
                    "gmm_force_config.main_wind_speed_min": fixed_wind,
                    "gmm_force_config.main_wind_speed_max": fixed_wind,
                    "drop_reward_config.drop_obstacle_enable": True,
                    "drop_reward_config.drop_obstacle_spawn_prob": 1.0,
                    "drop_reward_config.drop_obstacle_center_radius_min": 1.0,
                    "drop_reward_config.drop_obstacle_center_radius_max": 2.0,
                },
            ),
        ],
    )

    exp3 = ExperimentSpec(
        name="exp3_randomH_fixedW_fixedObs",
        display_name="Experiment 3: Random Height + Fixed Wind + Fixed Obstacle",
        conditions=[
            EvalCondition(
                name="main",
                display_name="random_h_fixed_w_fixed_obs",
                overrides={
                    **common_base,
                    "spawn_use_random_z": True,
                    "spawn_random_z_min": random_height_min,
                    "spawn_random_z_max": random_height_max,
                    "gmm_force_config.main_wind_speed_min": fixed_wind,
                    "gmm_force_config.main_wind_speed_max": fixed_wind,
                    "drop_reward_config.drop_obstacle_enable": True,
                    "drop_reward_config.drop_obstacle_spawn_prob": 1.0,
                },
                fixed_obstacle=True,
                fixed_obstacle_offset_xy=(1.5, 0.0),
                fixed_obstacle_radius=0.2,
            )
        ],
    )

    exp4 = ExperimentSpec(
        name="exp4_multiH_multiW_noObs",
        display_name="Experiment 4: Multi Height + No Obstacle + Multi Wind Strength",
        conditions=[
            EvalCondition(
                name="weak_wind",
                display_name="weak_wind",
                overrides={
                    **common_base,
                    "spawn_use_random_z": True,
                    "spawn_random_z_min": random_height_min,
                    "spawn_random_z_max": random_height_max,
                    "gmm_force_config.main_wind_speed_min": weak_wind,
                    "gmm_force_config.main_wind_speed_max": weak_wind,
                    "drop_reward_config.drop_obstacle_enable": False,
                    "drop_reward_config.drop_obstacle_spawn_prob": 0.0,
                },
            ),
            EvalCondition(
                name="medium_wind",
                display_name="medium_wind",
                overrides={
                    **common_base,
                    "spawn_use_random_z": True,
                    "spawn_random_z_min": random_height_min,
                    "spawn_random_z_max": random_height_max,
                    "gmm_force_config.main_wind_speed_min": med_wind,
                    "gmm_force_config.main_wind_speed_max": med_wind,
                    "drop_reward_config.drop_obstacle_enable": False,
                    "drop_reward_config.drop_obstacle_spawn_prob": 0.0,
                },
            ),
            EvalCondition(
                name="strong_wind",
                display_name="strong_wind",
                overrides={
                    **common_base,
                    "spawn_use_random_z": True,
                    "spawn_random_z_min": random_height_min,
                    "spawn_random_z_max": random_height_max,
                    "gmm_force_config.main_wind_speed_min": strong_wind,
                    "gmm_force_config.main_wind_speed_max": strong_wind,
                    "drop_reward_config.drop_obstacle_enable": False,
                    "drop_reward_config.drop_obstacle_spawn_prob": 0.0,
                },
            ),
        ],
    )

    exp5 = ExperimentSpec(
        name="exp5_complex_multiH_multiW_obs",
        display_name="Experiment 5: Complex Multi-Height + Multi-Wind + Obstacle",
        conditions=[
            EvalCondition(
                name="complex_mix",
                display_name="complex_mix",
                overrides={
                    "spawn_z_curriculum_enable": False,
                    "spawn_use_random_z": True,
                    "spawn_random_z_min": random_height_min,
                    "spawn_random_z_max": random_height_max,
                    "dryden_config.enable_dryden": True,
                    "gmm_force_config.main_wind_speed_min": weak_wind,
                    "gmm_force_config.main_wind_speed_max": strong_wind,
                    "gmm_force_config.local_wind_max_speed_min": 0.2,
                    "gmm_force_config.local_wind_max_speed_max": 0.3,
                    "drop_reward_config.drop_obstacle_enable": True,
                    "drop_reward_config.drop_obstacle_spawn_prob": 1.0,
                    "drop_reward_config.drop_obstacle_center_radius_min": 0.8,
                    "drop_reward_config.drop_obstacle_center_radius_max": 2.8,
                    "drop_reward_config.drop_obstacle_radius_min": 0.1,
                    "drop_reward_config.drop_obstacle_radius_max": 0.2,
                },
            )
        ],
    )

    return [exp1, exp2, exp3, exp4, exp5]


def _resolve_render_only_experiment(
    output_root: str, exp: ExperimentSpec
) -> Tuple[ExperimentSpec, str]:
    exp_root = os.path.join(output_root, exp.name)
    if os.path.exists(exp_root):
        return exp, exp_root

    # Older saved results used the previous experiment-4 directory name.
    # Keep this as render-only compatibility so figures can be regenerated
    # without rerunning simulation.
    if exp.name == "exp4_multiH_multiW_noObs":
        legacy_name = "exp4_fixedH_multiW_obs"
        legacy_root = os.path.join(output_root, legacy_name)
        if os.path.exists(legacy_root):
            legacy_exp = ExperimentSpec(
                name=legacy_name,
                display_name="Experiment 4: Fixed Height + Obstacle + Multi Wind Strength",
                conditions=exp.conditions,
            )
            logger.info(
                f"[render-only] Using legacy experiment-4 saved directory: {legacy_root}"
            )
            return legacy_exp, legacy_root

    return exp, exp_root


def _parse_experiment_ids(text: str) -> List[int]:
    out = []
    for token in text.split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper comparison: PPO vs Proposed vs optional PPO-GRU baseline vs MPC across five DROP experiments."
    )
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default=DEFAULT_BASELINE_CHECKPOINT,
        help="PPO baseline (MLP) checkpoint path.",
    )
    parser.add_argument(
        "--gru_checkpoint",
        type=str,
        default=DEFAULT_GRU_CHECKPOINT,
        help="Proposed MLP+GRU checkpoint path.",
    )
    parser.add_argument(
        "--rnn_checkpoint",
        type=str,
        default=DEFAULT_RNN_CHECKPOINT,
        help="Optional pure PPO-GRU baseline checkpoint path. If empty, PPO-GRU baseline is skipped.",
    )
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--render_only_from_saved",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help=(
            "Skip simulation and regenerate PDFs directly from saved "
            "raw/combined_drop_metrics.csv files under output_root."
        ),
    )
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episodes_per_seed", type=int, default=2000)
    parser.add_argument("--episode_len_steps", type=int, default=1000)
    parser.add_argument("--seeds", type=str, default="10")
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated experiment ids to run (1-5).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--headless",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="Headless mode (True/False).",
    )
    parser.add_argument(
        "--mpc_drop_error_threshold",
        type=float,
        default=1.8,
        help="MPC drop gate: predicted release error threshold (m).",
    )
    parser.add_argument(
        "--mpc_drop_attitude_deg_threshold",
        type=float,
        default=16.0,
        help="MPC drop gate: release attitude threshold (deg).",
    )
    parser.add_argument(
        "--mpc_drop_angular_vel_threshold",
        type=float,
        default=1.6,
        help="MPC drop gate: angular velocity norm threshold (rad/s).",
    )
    parser.add_argument(
        "--mpc_drop_dist_xy_threshold",
        type=float,
        default=10.0,
        help="MPC drop gate: target XY distance threshold (m).",
    )
    parser.add_argument(
        "--mpc_drop_min_z",
        type=float,
        default=0.8,
        help="MPC drop gate: minimum altitude for drop (m).",
    )
    parser.add_argument(
        "--mpc_force_drop_enable",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="Force MPC to drop near episode end for active non-crashed cases.",
    )
    parser.add_argument(
        "--mpc_force_drop_ratio",
        type=float,
        default=0.97,
        help="Force-drop starts at ratio*episode_len_steps (e.g., 0.90).",
    )
    parser.add_argument(
        "--large_error_threshold_m",
        type=float,
        default=3.0,
        help="Threshold (m) used in cross-case no-drop analysis for large landing error.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["AERIAL_GYM_EVAL_MODE"] = "1"
    seeds = parse_seed_list(args.seeds)
    os.makedirs(args.output_root, exist_ok=True)

    logger.info("Paper comparison: PPO vs Proposed vs optional PPO-GRU baseline vs MPC (five experiments)")
    logger.info(f"baseline_checkpoint={args.baseline_checkpoint}")
    logger.info(f"gru_checkpoint={args.gru_checkpoint}")
    logger.info(f"rnn_checkpoint={args.rnn_checkpoint or '<disabled>'}")
    logger.info(f"output_root={args.output_root}")

    all_experiments = _experiment_specs()
    selected_ids = _parse_experiment_ids(args.experiments)
    selected_experiments: List[ExperimentSpec] = []
    for exp_id in selected_ids:
        if exp_id < 1 or exp_id > len(all_experiments):
            raise ValueError(f"Invalid experiment id: {exp_id}, valid range is 1-{len(all_experiments)}")
        selected_experiments.append(all_experiments[exp_id - 1])

    if bool(args.render_only_from_saved):
        logger.info("Render-only mode: regenerate PDFs directly from saved raw CSV files.")
        for exp in selected_experiments:
            render_exp, exp_root = _resolve_render_only_experiment(args.output_root, exp)
            os.makedirs(exp_root, exist_ok=True)
            logger.info(f"[render-only] {render_exp.display_name}")

            for cond in render_exp.conditions:
                cond_root = os.path.join(exp_root, cond.name)
                raw_dir, stats_dir, fig_dir = ensure_dirs(cond_root)
                raw_csv_path = os.path.join(raw_dir, "combined_drop_metrics.csv")
                rows = load_rows_csv(raw_csv_path)
                methods = _methods_in_rows(rows, preferred_order=_preferred_method_order())
                summaries = build_summaries_from_rows(rows=rows, methods=methods)
                save_summary_tables(stats_dir=stats_dir, rows=rows, summaries=summaries)
                save_condition_key_metrics_csv(
                    stats_dir=stats_dir,
                    rows=rows,
                    summaries=summaries,
                    methods=methods,
                )
                render_condition_outputs(
                    rows=rows,
                    methods=methods,
                    experiment_display_name=render_exp.display_name,
                    condition_display_name=cond.display_name,
                    fig_dir=fig_dir,
                    large_error_threshold_m=float(args.large_error_threshold_m),
                )
                logger.info(
                    f"[render-only] Regenerated PDFs from {raw_csv_path} into {fig_dir}"
                )
            render_experiment_level_outputs(exp_root=exp_root, exp=render_exp)
        return

    if len(seeds) > 1:
        logger.warning(
            "Isaac Gym supports a single Foundation instance per process. "
            "This script will run only the first seed in one process. "
            "Run multiple times for multiple seeds."
        )
    seed = seeds[0]

    (
        ppo_actor,
        ppo_rms,
        ppo_obs_dim,
        ppo_act_dim,
        ppo_use_rnn,
        ppo_rnn_hidden_size,
        ppo_has_actor_mlp,
    ) = load_ppo_policy(
        checkpoint_path=args.baseline_checkpoint, device=args.device
    )
    (
        gru_actor,
        gru_rms,
        gru_obs_dim,
        gru_act_dim,
        gru_use_rnn,
        gru_rnn_hidden_size,
        gru_has_actor_mlp,
    ) = load_ppo_policy(
        checkpoint_path=args.gru_checkpoint, device=args.device
    )
    rnn_actor = None
    rnn_rms = None
    rnn_obs_dim = 0
    rnn_act_dim = None
    rnn_use_rnn = False
    rnn_hidden_size = 0
    rnn_has_actor_mlp = False
    rnn_checkpoint = str(args.rnn_checkpoint).strip()
    if rnn_checkpoint:
        (
            rnn_actor,
            rnn_rms,
            rnn_obs_dim,
            rnn_act_dim,
            rnn_use_rnn,
            rnn_hidden_size,
            rnn_has_actor_mlp,
        ) = load_ppo_policy(checkpoint_path=rnn_checkpoint, device=args.device)

    if ppo_use_rnn:
        raise RuntimeError(
            "baseline_checkpoint contains RNN weights. "
            "Please provide a pure PPO-MLP checkpoint."
        )
    if not gru_use_rnn:
        raise RuntimeError(
            "gru_checkpoint appears to be pure MLP. "
            "Please provide the Proposed MLP+GRU checkpoint."
        )
    if not gru_has_actor_mlp:
        raise RuntimeError(
            "gru_checkpoint appears to be RNN-only. "
            "Please provide a Proposed checkpoint with actor_mlp + GRU."
        )
    if rnn_checkpoint and (not rnn_use_rnn or rnn_has_actor_mlp):
        raise RuntimeError(
            "rnn_checkpoint must be a pure RNN checkpoint: GRU weights present and actor_mlp weights absent."
        )
    if int(ppo_act_dim) != int(gru_act_dim):
        raise RuntimeError(
            f"Action dim mismatch between PPO and Proposed checkpoints: {ppo_act_dim} vs {gru_act_dim}"
        )
    if rnn_checkpoint and int(ppo_act_dim) != int(rnn_act_dim):
        raise RuntimeError(
            f"Action dim mismatch between PPO and PPO-GRU baseline checkpoints: {ppo_act_dim} vs {rnn_act_dim}"
        )

    obs_dims = [int(ppo_obs_dim), int(gru_obs_dim)]
    if rnn_checkpoint:
        obs_dims.append(int(rnn_obs_dim))
    eval_obs_dim, use_augmented_obs, use_drop_decision_features = _infer_eval_obs_layout(obs_dims)
    logger.info(
        f"Policy dims: ppo(obs={ppo_obs_dim}, act={ppo_act_dim}, rnn={ppo_use_rnn}), "
        f"gru(obs={gru_obs_dim}, act={gru_act_dim}, rnn={gru_use_rnn}, hidden={gru_rnn_hidden_size}), "
        f"rnn(obs={rnn_obs_dim if rnn_checkpoint else 'disabled'}, "
        f"act={rnn_act_dim if rnn_checkpoint else 'disabled'}, "
        f"hidden={rnn_hidden_size if rnn_checkpoint else 'disabled'}), "
        f"eval_obs_dim={eval_obs_dim}, "
        f"use_augmented_obs={use_augmented_obs}, "
        f"use_drop_decision_features={use_drop_decision_features}"
    )

    first_overrides = (
        selected_experiments[0].conditions[0].overrides if selected_experiments and selected_experiments[0].conditions else {}
    )
    env = build_env(
        num_envs=args.num_envs,
        seed=seed,
        device=args.device,
        headless=args.headless,
        episode_len_steps=args.episode_len_steps,
        observation_space_dim=eval_obs_dim,
        use_wind_estimation_features=use_augmented_obs,
        use_drop_decision_features=use_drop_decision_features,
        overrides=first_overrides,
    )
    env_act_dim = int(env.task_config.action_space_dim)
    if (
        env_act_dim != int(ppo_act_dim)
        or env_act_dim != int(gru_act_dim)
        or (rnn_checkpoint and env_act_dim != int(rnn_act_dim))
    ):
        env.close()
        raise RuntimeError(
            f"Action dim mismatch: env={env_act_dim}, ppo={ppo_act_dim}, "
            f"gru={gru_act_dim}, rnn={rnn_act_dim if rnn_checkpoint else 'disabled'}"
        )
    if int(env.task_config.observation_space_dim) < eval_obs_dim:
        env.close()
        raise RuntimeError(
            f"Env observation dim {env.task_config.observation_space_dim} < required {eval_obs_dim}"
        )

    try:
        for exp in selected_experiments:
            exp_root = os.path.join(args.output_root, exp.name)
            os.makedirs(exp_root, exist_ok=True)
            logger.info(f"Running {exp.display_name}")

            for cond in exp.conditions:
                cond_root = os.path.join(exp_root, cond.name)
                raw_dir, stats_dir, fig_dir = ensure_dirs(cond_root)
                logger.info(f"Condition: {cond.display_name}")

                apply_runtime_overrides_to_env(env, cond.overrides)

                desired_alt = cond.overrides.get("spawn_fixed_z", 10.0)
                if bool(cond.overrides.get("spawn_use_random_z", False)):
                    zmin = float(cond.overrides.get("spawn_random_z_min", 4.0))
                    zmax = float(cond.overrides.get("spawn_random_z_max", 13.0))
                    desired_alt = 0.5 * (zmin + zmax)

                force_drop_step: Optional[int] = None
                if bool(args.mpc_force_drop_enable):
                    ratio = float(max(0.0, min(args.mpc_force_drop_ratio, 1.0)))
                    ep_steps = int(max(1, args.episode_len_steps))
                    force_drop_step = int(min(ep_steps - 1, max(0, round(ratio * ep_steps))))

                mpc_controller = MPCDropController(
                    obs_dim=eval_obs_dim,
                    desired_altitude=float(desired_alt),
                    drop_error_threshold=float(args.mpc_drop_error_threshold),
                    drop_attitude_deg_threshold=float(args.mpc_drop_attitude_deg_threshold),
                    drop_angular_vel_threshold=float(args.mpc_drop_angular_vel_threshold),
                    drop_dist_xy_threshold=float(args.mpc_drop_dist_xy_threshold),
                    drop_min_z=float(args.mpc_drop_min_z),
                )
                method_specs = [
                    MethodSpec(
                        name="ppo",
                        actor=ppo_actor,
                        rms=ppo_rms,
                        obs_dim=int(ppo_obs_dim),
                        use_rnn=bool(ppo_use_rnn),
                        rnn_hidden_size=int(ppo_rnn_hidden_size or 64),
                        kind="ppo",
                    ),
                    MethodSpec(
                        name="ppo_gru",
                        actor=gru_actor,
                        rms=gru_rms,
                        obs_dim=int(gru_obs_dim),
                        use_rnn=bool(gru_use_rnn),
                        rnn_hidden_size=int(gru_rnn_hidden_size or 64),
                        kind="ppo",
                    ),
                ]
                if rnn_checkpoint:
                    method_specs.append(
                        MethodSpec(
                            name="ppo_rnn",
                            actor=rnn_actor,
                            rms=rnn_rms,
                            obs_dim=int(rnn_obs_dim),
                            use_rnn=bool(rnn_use_rnn),
                            rnn_hidden_size=int(rnn_hidden_size or 64),
                            kind="ppo",
                        )
                    )
                method_specs.append(
                    MethodSpec(
                        name="mpc",
                        actor=None,
                        rms=None,
                        obs_dim=int(eval_obs_dim),
                        use_rnn=False,
                        kind="mpc",
                        controller=mpc_controller,
                        force_drop_step=force_drop_step,
                    )
                )

                post_reset_hook = _make_post_reset_hook(cond)
                rows, summaries, trace_records = evaluate_methods_single_seed(
                    env=env,
                    method_specs=method_specs,
                    seed=seed,
                    episodes_target=args.episodes_per_seed,
                    device=args.device,
                    post_reset_hook=post_reset_hook,
                )

                method_order = [m.name for m in method_specs]
                methods_present = _methods_in_rows(rows, preferred_order=method_order)
                methods = [m for m in method_order if m in methods_present]

                raw_csv_path, trace_npz_path = save_condition_eval_artifacts(
                    raw_dir=raw_dir,
                    stats_dir=stats_dir,
                    rows=rows,
                    trace_records=trace_records,
                    summaries=summaries,
                    methods=methods,
                )
                render_condition_outputs(
                    rows=rows,
                    methods=methods,
                    experiment_display_name=exp.display_name,
                    condition_display_name=cond.display_name,
                    fig_dir=fig_dir,
                    large_error_threshold_m=float(args.large_error_threshold_m),
                )

                logger.info(
                    f"Saved raw/stats/figures for {cond.display_name}: "
                    f"raw={raw_csv_path}, trace={trace_npz_path}, stats={stats_dir}, figures={fig_dir}"
                )
            render_experiment_level_outputs(exp_root=exp_root, exp=exp)
    finally:
        env.close()


if __name__ == "__main__":
    main()
