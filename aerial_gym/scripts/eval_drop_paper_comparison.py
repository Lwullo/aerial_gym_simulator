"""
Paper-style comparative evaluation for DROP task:
1) PPO(baseline) policy (loaded from checkpoint)
2) PPO-GRU policy (loaded from checkpoint)

Outputs:
- Raw CSVs, summary CSVs
- PNG + PDF figures (paper style)
"""

import argparse
import csv
import math
import os
import sys
import distutils
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root import path is available when script is launched directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.config.task_config.navigation_task_gmm_noise_config import task_config
from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import NavigationTaskGmmNoise

import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = CustomLogger("eval_drop_paper_comparison")

DEFAULT_CHECKPOINT = (
    "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/"
    "runs/PPO/nn/last_ppo_mlp_drop_v1_ep_2010_rew_19.82835.pth"
)
DEFAULT_BASELINE_CHECKPOINT = DEFAULT_CHECKPOINT
DEFAULT_GRU_CHECKPOINT = DEFAULT_CHECKPOINT
DEFAULT_OUTPUT_ROOT = (
    "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/result"
)
CASE_SEED_STRIDE = 10007


@dataclass
class MethodSummary:
    method: str
    seed: int
    done_total: int
    drop_done: int
    no_drop_done: int


class RunningMeanStd:
    def __init__(self, mean: torch.Tensor, var: torch.Tensor, count: torch.Tensor, device: str):
        self.mean = mean.to(device=device, dtype=torch.float32)
        self.var = var.to(device=device, dtype=torch.float32)
        self.count = count
        self.device = device

    def normalize(self, obs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        std = torch.sqrt(torch.clamp(self.var, min=epsilon))
        return (obs - self.mean) / (std + epsilon)


class RNNBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(x, h)


class PPOActorGRU(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.rnn = RNNBlock(input_size=64, hidden_size=64)
        self.layer_norm = nn.LayerNorm(64)
        self.mu = nn.Linear(64, act_dim)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.actor_mlp(obs)
        x = x.unsqueeze(0)  # [1, N, 64]
        x, h_next = self.rnn(x, h)
        x = x.squeeze(0)
        x = self.layer_norm(x)
        mu = self.mu(x)
        return mu, h_next


class PPOActorMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mu = nn.Linear(64, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu(self.actor_mlp(obs))


def _infer_ppo_dims(model_state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    obs_key = "a2c_network.actor_mlp.0.weight"
    act_key = "a2c_network.mu.weight"
    if obs_key not in model_state or act_key not in model_state:
        raise RuntimeError("Failed to infer PPO obs/action dimensions from checkpoint model state.")
    obs_dim = int(model_state[obs_key].shape[1])
    act_dim = int(model_state[act_key].shape[0])
    return obs_dim, act_dim


def _ppo_checkpoint_has_rnn(model_state: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("a2c_network.rnn.") for k in model_state.keys())


def load_ppo_policy(checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(model_state, dict):
        raise RuntimeError("Unsupported checkpoint format: model state dict not found.")
    ppo_obs_dim, ppo_act_dim = _infer_ppo_dims(model_state)
    use_rnn = _ppo_checkpoint_has_rnn(model_state)
    if use_rnn:
        actor = PPOActorGRU(obs_dim=ppo_obs_dim, act_dim=ppo_act_dim).to(device)
    else:
        actor = PPOActorMLP(obs_dim=ppo_obs_dim, act_dim=ppo_act_dim).to(device)

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

    return actor, rms, ppo_obs_dim, ppo_act_dim, use_rnn


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


def build_env(
    num_envs: int,
    seed: int,
    device: str,
    headless: bool,
    episode_len_steps: int,
    observation_space_dim: int = None,
    use_wind_estimation_features: bool = None,
):
    task_config.num_envs = num_envs
    task_config.seed = seed
    task_config.device = device
    task_config.headless = headless
    task_config.episode_len_steps = episode_len_steps
    if use_wind_estimation_features is not None:
        task_config.use_wind_estimation_features = bool(use_wind_estimation_features)
    if observation_space_dim is not None:
        task_config.observation_space_dim = int(observation_space_dim)
    env = NavigationTaskGmmNoise(task_config)
    env.reset()
    return env


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
            f"PPO(baseline)(drop/no_drop)={baseline_summary_batch.drop_done}/{baseline_summary_batch.no_drop_done}, "
            f"PPO-GRU(drop/no_drop)={gru_summary_batch.drop_done}/{gru_summary_batch.no_drop_done}"
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
                arr = np.array([float(r[metric]) for r in method_rows], dtype=np.float64)
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


def method_display_name(method: str) -> str:
    if method == "ppo_baseline":
        return "PPO(baseline)"
    if method == "ppo_gru":
        return "PPO-GRU"
    return method.upper()


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
    # Delta is defined as: PPO-GRU minus PPO(baseline).
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

    c_baseline = "#1f77b4"
    c_gru = "#d62728"

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

        ax.plot(x, y_baseline_ema, color=c_baseline, linewidth=2.8, label="PPO(baseline)")
        ax.plot(x, y_gru_ema, color=c_gru, linewidth=2.8, label="PPO-GRU")
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

    ax.plot(vb, cb, color="#1f77b4", linewidth=2.8, label="PPO(baseline)")
    ax.plot(vg, cg, color="#d62728", linewidth=2.8, label="PPO-GRU")

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
        _row("PPO(baseline)", sb),
        _row("PPO-GRU", sg),
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

    c_baseline = "#1f77b4"
    c_gru = "#d62728"

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
        ax.plot(x, y_baseline_ema, color=c_baseline, linewidth=2.8, label="PPO(baseline)")
        ax.plot(x, y_gru_ema, color=c_gru, linewidth=2.8, label="PPO-GRU")

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
            ax.set_xlabel("PPO(baseline)")
            ax.set_ylabel("PPO-GRU")
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
        ax.set_xlabel("PPO(baseline)")
        ax.set_ylabel("PPO-GRU")
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



def parse_args():
    parser = argparse.ArgumentParser(description="PPO(baseline) vs PPO-GRU paper comparison.")
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
        help="PPO-GRU checkpoint path.",
    )
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episodes_per_seed", type=int, default=2000)
    parser.add_argument("--episode_len_steps", type=int, default=1000)
    parser.add_argument("--seeds", type=str, default="10")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--headless",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="Headless mode (True/False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["AERIAL_GYM_EVAL_MODE"] = "1"
    seeds = parse_seed_list(args.seeds)
    raw_dir, stats_dir, fig_dir = ensure_dirs(args.output_root)

    logger.info("DROP paper comparison: PPO(baseline) vs PPO-GRU")
    logger.info(f"baseline_checkpoint={args.baseline_checkpoint}")
    logger.info(f"gru_checkpoint={args.gru_checkpoint}")
    logger.info(f"output_root={args.output_root}")
    logger.info(
        f"num_envs={args.num_envs}, episodes_per_seed={args.episodes_per_seed}, "
        f"episode_len_steps={args.episode_len_steps}, seeds={seeds}"
    )

    all_rows: List[Dict[str, float]] = []
    summaries: List[MethodSummary] = []

    if len(seeds) > 1:
        logger.warning(
            "Isaac Gym supports a single Foundation instance per process. "
            "This script will run only the first seed in one process. "
            "Run multiple times for multiple seeds."
        )
    seed = seeds[0]

    baseline_actor, baseline_rms, baseline_obs_dim, baseline_act_dim, baseline_use_rnn = load_ppo_policy(
        checkpoint_path=args.baseline_checkpoint, device=args.device
    )
    gru_actor, gru_rms, gru_obs_dim, gru_act_dim, gru_use_rnn = load_ppo_policy(
        checkpoint_path=args.gru_checkpoint, device=args.device
    )

    if baseline_use_rnn:
        logger.warning(
            "baseline_checkpoint appears to contain RNN weights. "
            "For PPO(baseline), use a pure-MLP checkpoint."
        )
    if not gru_use_rnn:
        logger.warning(
            "gru_checkpoint appears to be pure MLP. "
            "For PPO-GRU, use a checkpoint trained with RNN enabled."
        )

    eval_obs_dim = max(int(baseline_obs_dim), int(gru_obs_dim))
    use_augmented_obs = bool(eval_obs_dim > 12)
    logger.info(
        f"Policy dims: baseline(obs={baseline_obs_dim}, act={baseline_act_dim}, rnn={baseline_use_rnn}), "
        f"gru(obs={gru_obs_dim}, act={gru_act_dim}, rnn={gru_use_rnn}), eval_obs_dim={eval_obs_dim}"
    )
    env = build_env(
        num_envs=args.num_envs,
        seed=seed,
        device=args.device,
        headless=args.headless,
        episode_len_steps=args.episode_len_steps,
        observation_space_dim=eval_obs_dim,
        use_wind_estimation_features=use_augmented_obs,
    )
    env_act_dim = int(env.task_config.action_space_dim)
    if env_act_dim != int(baseline_act_dim) or env_act_dim != int(gru_act_dim):
        env.close()
        raise RuntimeError(
            f"Action dim mismatch: env={env_act_dim}, baseline={baseline_act_dim}, gru={gru_act_dim}"
        )
    if int(env.task_config.observation_space_dim) < eval_obs_dim:
        env.close()
        raise RuntimeError(
            f"Env observation dim {env.task_config.observation_space_dim} < required {eval_obs_dim}"
        )

    try:
        baseline_rows, gru_rows, baseline_summary, gru_summary = evaluate_paired_single_seed(
            env=env,
            seed=seed,
            episodes_target=args.episodes_per_seed,
            device=args.device,
            baseline_actor=baseline_actor,
            baseline_rms=baseline_rms,
            baseline_obs_dim=int(baseline_obs_dim),
            baseline_use_rnn=baseline_use_rnn,
            gru_actor=gru_actor,
            gru_rms=gru_rms,
            gru_obs_dim=int(gru_obs_dim),
            gru_use_rnn=gru_use_rnn,
        )

        all_rows.extend(baseline_rows)
        all_rows.extend(gru_rows)
        summaries.extend([baseline_summary, gru_summary])
        logger.info(
            f"[seed={seed}] paired cases={args.episodes_per_seed}, "
            f"PPO(baseline) done/drop/no_drop={baseline_summary.done_total}/{baseline_summary.drop_done}/{baseline_summary.no_drop_done}, "
            f"PPO-GRU done/drop/no_drop={gru_summary.done_total}/{gru_summary.drop_done}/{gru_summary.no_drop_done}"
        )
    finally:
        env.close()

    baseline_rows_all = [r for r in all_rows if r["method"] == "ppo_baseline"]
    gru_rows_all = [r for r in all_rows if r["method"] == "ppo_gru"]
    write_rows_csv(os.path.join(raw_dir, "ppo_baseline_drop_metrics.csv"), baseline_rows_all)
    write_rows_csv(os.path.join(raw_dir, "ppo_gru_drop_metrics.csv"), gru_rows_all)
    write_rows_csv(os.path.join(raw_dir, "combined_drop_metrics.csv"), all_rows)

    mean_std_path, median_iqr_path, no_drop_path = save_summary_tables(
        stats_dir=stats_dir, rows=all_rows, summaries=summaries
    )

    # Figure 1: landing_error_xy CDF comparison (PPO vs PPO-GRU)
    plot_paired_cdf_figure(
        rows=all_rows,
        metrics=[
            ("landing_error_xy_m", "Landing Error XY CDF", "Landing Error XY (m)", False),
        ],
        suptitle="DROP Precision CDF: PPO(baseline) vs PPO-GRU",
        out_png=os.path.join(fig_dir, "fig_landing_error_xy_cdf_compare.png"),
        out_pdf=os.path.join(fig_dir, "fig_landing_error_xy_cdf_compare.pdf"),
    )

    # Figure 2: summary table for Figure 1
    plot_landing_cdf_summary_table_figure(
        rows=all_rows,
        out_png=os.path.join(fig_dir, "fig_landing_error_xy_cdf_summary_table.png"),
        out_pdf=os.path.join(fig_dir, "fig_landing_error_xy_cdf_summary_table.pdf"),
        thresholds=(0.5, 1.0, 2.0),
    )

    # Figure 3: pre-DROP attitude trend comparison (first-style: shadow + smoothed)
    plot_paired_attitude_shadow_figure(
        rows=all_rows,
        metrics=[
            ("roll_deg", "Roll", "Pre-DROP Angle (deg)"),
            ("pitch_deg", "Pitch", "Pre-DROP Angle (deg)"),
            ("yaw_deg", "Yaw", "Pre-DROP Angle (deg)"),
        ],
        suptitle="Pre-DROP Attitude Comparison: PPO(baseline) vs PPO-GRU",
        out_png=os.path.join(fig_dir, "fig_pre_drop_attitude_compare.png"),
        out_pdf=os.path.join(fig_dir, "fig_pre_drop_attitude_compare.pdf"),
    )

    # Added scatter windows: attitude (3 subplots) + impact (1 subplot)
    plot_paired_scatter_figure(
        rows=all_rows,
        metrics=[
            ("roll_deg", "|Roll| Paired Scatter", True),
            ("pitch_deg", "|Pitch| Paired Scatter", True),
            ("yaw_deg", "|Yaw| Paired Scatter", True),
        ],
        suptitle="Paired Attitude Scatter: PPO(baseline) vs PPO-GRU",
        out_png=os.path.join(fig_dir, "fig_pre_drop_attitude_scatter_compare.png"),
        out_pdf=os.path.join(fig_dir, "fig_pre_drop_attitude_scatter_compare.pdf"),
    )

    plot_paired_scatter_figure(
        rows=all_rows,
        metrics=[
            ("impulse_metric", "Impulse Metric Paired Scatter", False),
        ],
        suptitle="Paired Impact Scatter: PPO(baseline) vs PPO-GRU",
        out_png=os.path.join(fig_dir, "fig_impulse_metric_scatter_compare.png"),
        out_pdf=os.path.join(fig_dir, "fig_impulse_metric_scatter_compare.pdf"),
    )

    # Combined table window (2 subplots): attitude table + impact table
    plot_attitude_impulse_summary_tables_figure(
        rows=all_rows,
        out_png=os.path.join(fig_dir, "fig_attitude_impact_summary_tables.png"),
        out_pdf=os.path.join(fig_dir, "fig_attitude_impact_summary_tables.pdf"),
    )

    # Figure 4: impulse metric CDF comparison
    plot_paired_cdf_figure(
        rows=all_rows,
        metrics=[
            ("impulse_metric", "Impulse Metric CDF", "Impulse Metric", False),
        ],
        suptitle="DROP Impact CDF: PPO(baseline) vs PPO-GRU",
        out_png=os.path.join(fig_dir, "fig_impulse_metric_cdf_compare.png"),
        out_pdf=os.path.join(fig_dir, "fig_impulse_metric_cdf_compare.pdf"),
    )

    logger.info(f"Saved raw metrics to: {raw_dir}")
    logger.info(f"Saved summaries to: {stats_dir}")
    logger.info(f"Saved figures to: {fig_dir}")
    logger.info(f"Summary files: {mean_std_path}, {median_iqr_path}, {no_drop_path}")


if __name__ == "__main__":
    main()
