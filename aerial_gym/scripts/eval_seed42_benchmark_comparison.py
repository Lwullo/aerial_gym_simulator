import argparse
import contextlib
import csv
import glob
import importlib
import logging
import math
import os
import re
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import isaacgym  # noqa: F401
import torch
import torch.nn as nn

from aerial_gym.config.task_config.navigation_task_gmm_noise_config import task_config
from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import NavigationTaskGmmNoise
from aerial_gym.utils.logging import CustomLogger


logger = CustomLogger("eval_seed42_benchmark_comparison")

BASE_OBS_DIM = 13
CASE_SEED_STRIDE = 10007
SUCCESS_THRESHOLD_M = 1.0
WIND_TIER_NAMES = {0: "weak", 1: "medium", 2: "strong"}
QUIET_LOGGER_MODULES = [
    "aerial_gym.task.base_task",
    "aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise",
    "aerial_gym.env_manager.IGE_env_manager",
    "aerial_gym.env_manager.env_manager",
    "aerial_gym.env_manager.asset_loader",
    "aerial_gym.env_manager.asset_manager",
    "aerial_gym.robots.robot_manager",
    "aerial_gym.robots.base_robot",
    "aerial_gym.robots.base_multirotor",
    "aerial_gym.sensors.warp.warp_sensor",
    "aerial_gym.control.control_allocation",
    "aerial_gym.control.controllers.base_lee_controller",
    "aerial_gym.control.controllers.position_control",
    "aerial_gym.control.controllers.velocity_control",
]


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
        x = x.unsqueeze(0)
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
    obs_dim, act_dim = _infer_ppo_dims(model_state)
    use_rnn = _ppo_checkpoint_has_rnn(model_state)
    if use_rnn:
        actor = PPOActorGRU(obs_dim=obs_dim, act_dim=act_dim).to(device)
    else:
        actor = PPOActorMLP(obs_dim=obs_dim, act_dim=act_dim).to(device)

    actor_state = {}
    for key, value in model_state.items():
        if key.startswith("a2c_network."):
            actor_state[key.replace("a2c_network.", "", 1)] = value
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        logger.warning("Actor missing keys: %s", missing)
    benign_unexpected = {"sigma", "value.weight", "value.bias"}
    unexpected = [key for key in unexpected if key not in benign_unexpected]
    if unexpected:
        logger.warning("Actor unexpected keys: %s", unexpected)
    actor.eval()

    rms_mean = model_state.get("running_mean_std.running_mean", None)
    rms_var = model_state.get("running_mean_std.running_var", None)
    rms_count = model_state.get("running_mean_std.count", None)
    if rms_mean is None or rms_var is None or rms_count is None:
        rms = None
    else:
        rms = RunningMeanStd(mean=rms_mean, var=rms_var, count=rms_count, device=device)
    return actor, rms, obs_dim, act_dim, use_rnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed-42 benchmark comparison: single-frame vs symmetric vs proposed"
    )
    parser.add_argument(
        "--singleframe-run-dir",
        type=str,
        default=(
            "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/"
            "aerial_gym/rl_training/rl_games/runs/acomparsion1-singleframe"
        ),
    )
    parser.add_argument(
        "--symmetric-run-dir",
        type=str,
        default=(
            "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/"
            "aerial_gym/rl_training/rl_games/runs/acomparsion2-proposed-symmetric"
        ),
    )
    parser.add_argument(
        "--proposed-run-dir",
        type=str,
        default=(
            "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/"
            "aerial_gym/rl_training/rl_games/runs/seed-42"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/"
            "aerial_gym/rl_training/rl_games/runs/benchmark_seed42_three_way_comparison"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-cases", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--episode-len-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--skip-eval", action="store_true", help="Only generate benchmark cases and report config.")
    return parser.parse_args()


def ensure_dirs(output_dir: str):
    raw_dir = os.path.join(output_dir, "raw")
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    return raw_dir, stats_dir


def _set_logger_level(logger_obj, level: int):
    if logger_obj is None:
        return
    logger_obj.disabled = True
    set_level_fn = getattr(logger_obj, "setLoggerLevel", None)
    if callable(set_level_fn):
        set_level_fn(level)
    else:
        logger_obj.setLevel(level)
        for handler in getattr(logger_obj, "handlers", []):
            handler.setLevel(level)
    logger_obj.propagate = False


def silence_project_loggers(level: int = logging.ERROR):
    logging.getLogger().setLevel(level)
    for module_name in QUIET_LOGGER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        _set_logger_level(getattr(module, "logger", None), level)


def silence_python_warnings():
    warnings.filterwarnings("ignore", message="distutils Version classes are deprecated.*")
    warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended.*")


@contextlib.contextmanager
def suppress_stdio(enabled: bool = True):
    if not enabled:
        yield
        return
    stdout_fd = None
    stderr_fd = None
    devnull_fd = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        if stdout_fd is not None:
            os.dup2(stdout_fd, 1)
            os.close(stdout_fd)
        if stderr_fd is not None:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


def checkpoint_epoch(path: str) -> int:
    match = re.search(r"_ep_(\d+)_", os.path.basename(path))
    if match is None:
        return -1
    return int(match.group(1))


def find_final_checkpoint(run_dir: str) -> str:
    nn_dir = os.path.join(run_dir, "nn")
    if not os.path.isdir(nn_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {nn_dir}")
    candidates = glob.glob(os.path.join(nn_dir, "last_gmm_noise_run_ep_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No last_gmm_noise_run_ep_*.pth found in {nn_dir}")
    candidates.sort(key=lambda p: (checkpoint_epoch(p), os.path.getmtime(p), p))
    return candidates[-1]


def configure_task_for_eval(obs_dim: int, seed: int, device: str, headless: bool, episode_len_steps: int):
    if obs_dim % BASE_OBS_DIM != 0:
        raise ValueError(f"Observation dimension {obs_dim} is not divisible by base dim {BASE_OBS_DIM}.")
    frame_stack = int(obs_dim // BASE_OBS_DIM)
    priv_dim = int(getattr(task_config, "privileged_observation_space_dim", 13))

    task_config.seed = seed
    task_config.device = device
    task_config.headless = headless
    task_config.use_warp = True
    task_config.num_envs = 1
    task_config.frame_stack = frame_stack
    task_config.observation_space_dim = obs_dim
    task_config.critic_observation_space_dim = obs_dim + priv_dim
    task_config.episode_len_steps = episode_len_steps
    task_config.spawn_z_curriculum_enable = False
    setattr(task_config.drop_model_config, "record_drop_trajectory_for_eval", False)
    setattr(task_config.drop_model_config, "confidence_gate_enable", False)


def build_env(obs_dim: int, seed: int, device: str, headless: bool, episode_len_steps: int):
    configure_task_for_eval(
        obs_dim=obs_dim,
        seed=seed,
        device=device,
        headless=headless,
        episode_len_steps=episode_len_steps,
    )
    with suppress_stdio():
        env = NavigationTaskGmmNoise(task_config)
        env.reset()
    return env


def close_env(env):
    if env is None:
        return
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def slice_actor_obs(full_obs: torch.Tensor, actor_obs_dim: int) -> torch.Tensor:
    full_obs_dim = int(full_obs.shape[1])
    if full_obs_dim == actor_obs_dim:
        return full_obs
    if full_obs_dim < actor_obs_dim:
        raise ValueError(f"Full observation dim {full_obs_dim} is smaller than actor obs dim {actor_obs_dim}.")
    if full_obs_dim % BASE_OBS_DIM != 0 or actor_obs_dim % BASE_OBS_DIM != 0:
        raise ValueError(
            f"Expected stacked observations divisible by {BASE_OBS_DIM}, got "
            f"full={full_obs_dim}, actor={actor_obs_dim}."
        )
    return full_obs[:, -actor_obs_dim:]


def wind_tier_thresholds() -> Tuple[float, float]:
    cfg = task_config.gmm_force_config
    speed_min = float(getattr(cfg, "main_wind_speed_min", 1.0))
    speed_max = float(getattr(cfg, "main_wind_speed_max", 3.0))
    if speed_max < speed_min:
        speed_min, speed_max = speed_max, speed_min
    delta = (speed_max - speed_min) / 3.0
    return speed_min + delta, speed_min + 2.0 * delta


def classify_wind_tier(main_wind_speed: float, thresholds: Tuple[float, float]) -> int:
    weak_hi, medium_hi = thresholds
    if main_wind_speed < weak_hi:
        return 0
    if main_wind_speed < medium_hi:
        return 1
    return 2


def probe_case(env, case_seed: int, thresholds: Tuple[float, float]) -> Dict[str, float]:
    with suppress_stdio():
        env.seed(case_seed)
        env.reset()

    main_wind_speed = float(env.main_wind_speed[0].item())
    tier_id = classify_wind_tier(main_wind_speed, thresholds)
    spawn = env.spawn_position[0].detach().cpu().numpy()
    target = env.target_position[0].detach().cpu().numpy()
    main_dir = env.main_wind_direction[0].detach().cpu().numpy()
    main_vec = env.main_wind_vector[0].detach().cpu().numpy()
    dry_sigma = env.dryden_sigma[0].detach().cpu().numpy()
    dry_tau = env.dryden_tau[0].detach().cpu().numpy()

    return {
        "case_seed": int(case_seed),
        "wind_tier_id": int(tier_id),
        "wind_tier": WIND_TIER_NAMES[int(tier_id)],
        "main_wind_speed": main_wind_speed,
        "main_wind_dir_x": float(main_dir[0]),
        "main_wind_dir_y": float(main_dir[1]),
        "main_wind_dir_z": float(main_dir[2]),
        "main_wind_vec_x": float(main_vec[0]),
        "main_wind_vec_y": float(main_vec[1]),
        "main_wind_vec_z": float(main_vec[2]),
        "spawn_x": float(spawn[0]),
        "spawn_y": float(spawn[1]),
        "spawn_z": float(spawn[2]),
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "target_z": float(target[2]),
        "dryden_sigma_x": float(dry_sigma[0]),
        "dryden_sigma_y": float(dry_sigma[1]),
        "dryden_sigma_z": float(dry_sigma[2]),
        "dryden_tau_x": float(dry_tau[0]),
        "dryden_tau_y": float(dry_tau[1]),
        "dryden_tau_z": float(dry_tau[2]),
    }


def generate_benchmark_cases(env, base_seed: int, num_cases: int) -> Tuple[List[Dict[str, float]], Dict[str, int], Tuple[float, float]]:
    thresholds = wind_tier_thresholds()
    base_quota = num_cases // 3
    quotas = {0: base_quota, 1: base_quota, 2: num_cases - 2 * base_quota}
    counts = {0: 0, 1: 0, 2: 0}
    rows: List[Dict[str, float]] = []

    candidate_idx = 0
    while len(rows) < num_cases:
        case_seed = int(base_seed + candidate_idx * CASE_SEED_STRIDE)
        row = probe_case(env, case_seed=case_seed, thresholds=thresholds)
        tier_id = int(row["wind_tier_id"])
        if counts[tier_id] < quotas[tier_id]:
            row["case_id"] = len(rows)
            rows.append(row)
            counts[tier_id] += 1
        candidate_idx += 1

    return rows, counts, thresholds


def write_csv(path: str, rows: List[Dict[str, float]]):
    if not rows:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["empty"])
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _info_tensor(infos: Dict, key: str, default: torch.Tensor) -> torch.Tensor:
    return infos[key] if key in infos else default


def collect_step_arrays(env, infos: Dict) -> Dict[str, np.ndarray]:
    num_envs_local = env.sim_env.num_envs
    zeros_bool = torch.zeros(num_envs_local, device=env.device, dtype=torch.bool)
    zeros_float = torch.zeros(num_envs_local, device=env.device, dtype=torch.float32)
    dones_tensor = (env.terminations > 0) | (env.truncations > 0)

    def _to_np_bool(t):
        return t.detach().cpu().numpy().astype(bool)

    def _to_np_float(t):
        return t.detach().cpu().numpy().astype(np.float64)

    return {
        "drops": _to_np_bool(_info_tensor(infos, "drops_snapshot", zeros_bool)),
        "dones": _to_np_bool(_info_tensor(infos, "dones_snapshot", dones_tensor)),
        "crashes": _to_np_bool(_info_tensor(infos, "crashes_snapshot", zeros_bool)),
        "timeouts": _to_np_bool(_info_tensor(infos, "timeouts_snapshot", zeros_bool)),
        "roll": _to_np_float(_info_tensor(infos, "drop_roll_snapshot", zeros_float)),
        "pitch": _to_np_float(_info_tensor(infos, "drop_pitch_snapshot", zeros_float)),
        "yaw": _to_np_float(_info_tensor(infos, "drop_yaw_snapshot", zeros_float)),
        "delta_theta": _to_np_float(_info_tensor(infos, "drop_delta_theta_snapshot", zeros_float)),
        "delta_v": _to_np_float(_info_tensor(infos, "drop_delta_v_snapshot", zeros_float)),
        "delta_omega": _to_np_float(_info_tensor(infos, "drop_delta_omega_snapshot", zeros_float)),
        "impulse": _to_np_float(_info_tensor(infos, "impulse_metric_snapshot", zeros_float)),
        "landing_error_xy": _to_np_float(_info_tensor(infos, "landing_error_xy_snapshot", zeros_float)),
        "release_to_target_xy": _to_np_float(_info_tensor(infos, "release_to_target_xy_snapshot", zeros_float)),
    }


def build_result_row(method: str, case_meta: Dict[str, float], sim_step: int, done_reached: bool, arrays: Dict[str, np.ndarray]):
    dropped = bool(arrays["drops"][0]) if done_reached else False
    crashed = bool(arrays["crashes"][0]) if done_reached else False
    timeout = bool(arrays["timeouts"][0]) if done_reached else False

    row = {
        "method": method,
        "case_id": int(case_meta["case_id"]),
        "case_seed": int(case_meta["case_seed"]),
        "wind_tier": case_meta["wind_tier"],
        "wind_tier_id": int(case_meta["wind_tier_id"]),
        "main_wind_speed": float(case_meta["main_wind_speed"]),
        "spawn_x": float(case_meta["spawn_x"]),
        "spawn_y": float(case_meta["spawn_y"]),
        "spawn_z": float(case_meta["spawn_z"]),
        "target_x": float(case_meta["target_x"]),
        "target_y": float(case_meta["target_y"]),
        "target_z": float(case_meta["target_z"]),
        "done_reached": int(done_reached),
        "dropped": int(dropped),
        "crashed": int(crashed),
        "timeout": int(timeout),
        "no_drop_done": int(done_reached and not dropped),
        "sim_step": int(sim_step),
        "roll_deg": float("nan"),
        "pitch_deg": float("nan"),
        "yaw_deg": float("nan"),
        "attitude_total_deg": float("nan"),
        "max_abs_roll_pitch_deg": float("nan"),
        "landing_error_xy_m": float("nan"),
        "release_to_target_xy_m": float("nan"),
        "delta_theta_rad": float("nan"),
        "delta_theta_deg": float("nan"),
        "delta_v_m_s": float("nan"),
        "delta_omega_rad_s": float("nan"),
        "delta_omega_deg_s": float("nan"),
        "impulse_metric": float("nan"),
        "success_1m": 0,
        "attitude_total_over_45deg": 0,
        "max_abs_roll_pitch_over_45deg": 0,
    }

    if dropped:
        roll_rad = float(arrays["roll"][0])
        pitch_rad = float(arrays["pitch"][0])
        yaw_rad = float(arrays["yaw"][0])
        roll_deg = float(np.degrees(roll_rad))
        pitch_deg = float(np.degrees(pitch_rad))
        yaw_deg = float(np.degrees(yaw_rad))
        attitude_total_deg = float(math.sqrt(roll_deg * roll_deg + pitch_deg * pitch_deg))
        max_abs_roll_pitch_deg = float(max(abs(roll_deg), abs(pitch_deg)))
        landing_error_xy = float(arrays["landing_error_xy"][0])
        success_1m = int(np.isfinite(landing_error_xy) and landing_error_xy < SUCCESS_THRESHOLD_M)
        row.update(
            {
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "yaw_deg": yaw_deg,
                "attitude_total_deg": attitude_total_deg,
                "max_abs_roll_pitch_deg": max_abs_roll_pitch_deg,
                "landing_error_xy_m": landing_error_xy,
                "release_to_target_xy_m": float(arrays["release_to_target_xy"][0]),
                "delta_theta_rad": float(arrays["delta_theta"][0]),
                "delta_theta_deg": float(np.degrees(arrays["delta_theta"][0])),
                "delta_v_m_s": float(arrays["delta_v"][0]),
                "delta_omega_rad_s": float(arrays["delta_omega"][0]),
                "delta_omega_deg_s": float(np.degrees(arrays["delta_omega"][0])),
                "impulse_metric": float(arrays["impulse"][0]),
                "success_1m": success_1m,
                "attitude_total_over_45deg": int(attitude_total_deg > 45.0),
                "max_abs_roll_pitch_over_45deg": int(max_abs_roll_pitch_deg > 45.0),
            }
        )
    return row


def verify_case_metadata(env, case_meta: Dict[str, float], atol: float = 1e-6):
    speed = float(env.main_wind_speed[0].item())
    if abs(speed - float(case_meta["main_wind_speed"])) > atol:
        raise RuntimeError(
            f"Benchmark case mismatch for seed {case_meta['case_seed']}: "
            f"main_wind_speed {speed} != {case_meta['main_wind_speed']}"
        )


def evaluate_case(
    env,
    actor,
    rms,
    obs_dim: int,
    use_rnn: bool,
    case_meta: Dict[str, float],
    max_steps: int,
    method: str,
):
    with suppress_stdio():
        env.seed(int(case_meta["case_seed"]))
        env.reset()
    verify_case_metadata(env, case_meta)

    hidden = None
    if use_rnn:
        hidden = torch.zeros((1, env.sim_env.num_envs, 64), device=env.device, dtype=torch.float32)

    for sim_step in range(1, max_steps + 1):
        obs = slice_actor_obs(env.task_obs["observations"], obs_dim)
        if rms is not None:
            obs = rms.normalize(obs)
        with torch.no_grad():
            if use_rnn:
                mu, hidden = actor(obs, hidden)
            else:
                mu = actor(obs)
            actions = torch.clamp(mu, -1.0, 1.0)

        _, _, _, _, infos = env.step(actions)
        arrays = collect_step_arrays(env, infos)
        if bool(arrays["dones"][0]):
            return build_result_row(
                method=method,
                case_meta=case_meta,
                sim_step=sim_step,
                done_reached=True,
                arrays=arrays,
            )

    return build_result_row(
        method=method,
        case_meta=case_meta,
        sim_step=max_steps,
        done_reached=False,
        arrays=collect_step_arrays(env, {}),
    )


def evaluate_method(
    env,
    method: str,
    actor,
    rms,
    obs_dim: int,
    act_dim: int,
    use_rnn: bool,
    checkpoint_path: str,
    cases: List[Dict[str, float]],
    max_steps: int,
) -> Dict[str, object]:
    rows = []
    for idx, case_meta in enumerate(cases, start=1):
        row = evaluate_case(
            env=env,
            actor=actor,
            rms=rms,
            obs_dim=obs_dim,
            use_rnn=use_rnn,
            case_meta=case_meta,
            max_steps=max_steps,
            method=method,
        )
        rows.append(row)
        if idx % 50 == 0 or idx == len(cases):
            logger.info("[%s] evaluated %d/%d cases", method, idx, len(cases))

    return {
        "method": method,
        "checkpoint_path": checkpoint_path,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "frame_stack": int(obs_dim // BASE_OBS_DIM),
        "rows": rows,
    }


def finite_values(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return vals[np.isfinite(vals)]


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size > 0 else float("nan")


def safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size > 0 else float("nan")


def safe_p90(values: np.ndarray) -> float:
    return float(np.percentile(values, 90.0)) if values.size > 0 else float("nan")


def summarize_rows(rows: List[Dict[str, float]], split_name: str, method: str) -> Dict[str, float]:
    n_cases = len(rows)
    dropped_rows = [r for r in rows if int(r["dropped"]) == 1]
    n_drop = len(dropped_rows)
    n_done = sum(int(r["done_reached"]) for r in rows)
    n_no_drop = sum(int(r["no_drop_done"]) for r in rows)
    n_crash = sum(int(r["crashed"]) for r in rows)
    n_success = sum(int(r["success_1m"]) for r in rows)

    landing = finite_values(dropped_rows, "landing_error_xy_m")
    attitude_total = finite_values(dropped_rows, "attitude_total_deg")
    delta_v = finite_values(dropped_rows, "delta_v_m_s")
    delta_theta_deg = finite_values(dropped_rows, "delta_theta_deg")
    delta_omega_deg_s = finite_values(dropped_rows, "delta_omega_deg_s")
    impulse = finite_values(dropped_rows, "impulse_metric")

    attitude_total_over_45 = np.asarray(
        [int(r["attitude_total_over_45deg"]) for r in dropped_rows],
        dtype=np.float64,
    )
    max_abs_roll_pitch_over_45 = np.asarray(
        [int(r["max_abs_roll_pitch_over_45deg"]) for r in dropped_rows],
        dtype=np.float64,
    )

    return {
        "split": split_name,
        "method": method,
        "n_cases": n_cases,
        "n_done": n_done,
        "n_drop": n_drop,
        "n_no_drop": n_no_drop,
        "done_rate": float(n_done / n_cases) if n_cases > 0 else float("nan"),
        "drop_rate": float(n_drop / n_cases) if n_cases > 0 else float("nan"),
        "no_drop_rate": float(n_no_drop / n_cases) if n_cases > 0 else float("nan"),
        "crash_rate": float(n_crash / n_cases) if n_cases > 0 else float("nan"),
        "success_rate_1m": float(n_success / n_cases) if n_cases > 0 else float("nan"),
        "success_rate_1m_given_drop": float(n_success / n_drop) if n_drop > 0 else float("nan"),
        "landing_error_mean_m": safe_mean(landing),
        "landing_error_median_m": safe_median(landing),
        "landing_error_p90_m": safe_p90(landing),
        "attitude_total_mean_deg": safe_mean(attitude_total),
        "attitude_total_p90_deg": safe_p90(attitude_total),
        "attitude_total_over_45deg_rate": safe_mean(attitude_total_over_45),
        "max_abs_roll_pitch_over_45deg_rate": safe_mean(max_abs_roll_pitch_over_45),
        "delta_v_mean_m_s": safe_mean(delta_v),
        "delta_v_p90_m_s": safe_p90(delta_v),
        "delta_theta_deg_mean": safe_mean(delta_theta_deg),
        "delta_theta_deg_p90": safe_p90(delta_theta_deg),
        "delta_omega_deg_s_mean": safe_mean(delta_omega_deg_s),
        "delta_omega_deg_s_p90": safe_p90(delta_omega_deg_s),
        "impulse_metric_mean": safe_mean(impulse),
        "impulse_metric_p90": safe_p90(impulse),
    }


def summarize_all(method_rows: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, float]]:
    summaries = []
    split_order = ["overall", "weak", "medium", "strong"]
    for method, rows in method_rows.items():
        split_rows = {
            "overall": rows,
            "weak": [r for r in rows if r["wind_tier"] == "weak"],
            "medium": [r for r in rows if r["wind_tier"] == "medium"],
            "strong": [r for r in rows if r["wind_tier"] == "strong"],
        }
        for split_name in split_order:
            summaries.append(summarize_rows(split_rows[split_name], split_name=split_name, method=method))
    return summaries


def write_summary_csv(path: str, rows: List[Dict[str, float]]):
    if not rows:
        write_csv(path, [])
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percent_str(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{100.0 * x:.2f}%"


def number_str(x: float, digits: int = 3) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:.{digits}f}"


def summary_lookup(summary_rows: List[Dict[str, float]], split_name: str, method: str) -> Dict[str, float]:
    for row in summary_rows:
        if row["split"] == split_name and row["method"] == method:
            return row
    raise KeyError(f"Missing summary for split={split_name}, method={method}")


def render_markdown_table(summary_rows: List[Dict[str, float]], split_name: str) -> str:
    singleframe = summary_lookup(summary_rows, split_name=split_name, method="singleframe")
    symmetric = summary_lookup(summary_rows, split_name=split_name, method="proposed_symmetric")
    proposed = summary_lookup(summary_rows, split_name=split_name, method="proposed")
    lines = [
        f"## {split_name.capitalize()}",
        "",
        "| Metric | singleframe | proposed_symmetric | proposed |",
        "| --- | ---: | ---: | ---: |",
        f"| Cases | {singleframe['n_cases']} | {symmetric['n_cases']} | {proposed['n_cases']} |",
        f"| Drop Rate | {percent_str(singleframe['drop_rate'])} | {percent_str(symmetric['drop_rate'])} | {percent_str(proposed['drop_rate'])} |",
        f"| No-Drop Rate | {percent_str(singleframe['no_drop_rate'])} | {percent_str(symmetric['no_drop_rate'])} | {percent_str(proposed['no_drop_rate'])} |",
        f"| Crash Rate | {percent_str(singleframe['crash_rate'])} | {percent_str(symmetric['crash_rate'])} | {percent_str(proposed['crash_rate'])} |",
        f"| Success Rate (<1m) | {percent_str(singleframe['success_rate_1m'])} | {percent_str(symmetric['success_rate_1m'])} | {percent_str(proposed['success_rate_1m'])} |",
        f"| Success Rate Given Drop | {percent_str(singleframe['success_rate_1m_given_drop'])} | {percent_str(symmetric['success_rate_1m_given_drop'])} | {percent_str(proposed['success_rate_1m_given_drop'])} |",
        f"| Landing Error Mean (m) | {number_str(singleframe['landing_error_mean_m'])} | {number_str(symmetric['landing_error_mean_m'])} | {number_str(proposed['landing_error_mean_m'])} |",
        f"| Landing Error Median (m) | {number_str(singleframe['landing_error_median_m'])} | {number_str(symmetric['landing_error_median_m'])} | {number_str(proposed['landing_error_median_m'])} |",
        f"| Landing Error P90 (m) | {number_str(singleframe['landing_error_p90_m'])} | {number_str(symmetric['landing_error_p90_m'])} | {number_str(proposed['landing_error_p90_m'])} |",
        f"| Attitude Total Mean (deg) | {number_str(singleframe['attitude_total_mean_deg'])} | {number_str(symmetric['attitude_total_mean_deg'])} | {number_str(proposed['attitude_total_mean_deg'])} |",
        f"| Attitude Total P90 (deg) | {number_str(singleframe['attitude_total_p90_deg'])} | {number_str(symmetric['attitude_total_p90_deg'])} | {number_str(proposed['attitude_total_p90_deg'])} |",
        f"| Attitude Total >45deg Rate | {percent_str(singleframe['attitude_total_over_45deg_rate'])} | {percent_str(symmetric['attitude_total_over_45deg_rate'])} | {percent_str(proposed['attitude_total_over_45deg_rate'])} |",
        f"| Max(|roll|,|pitch|) >45deg Rate | {percent_str(singleframe['max_abs_roll_pitch_over_45deg_rate'])} | {percent_str(symmetric['max_abs_roll_pitch_over_45deg_rate'])} | {percent_str(proposed['max_abs_roll_pitch_over_45deg_rate'])} |",
        f"| Delta-v Mean (m/s) | {number_str(singleframe['delta_v_mean_m_s'])} | {number_str(symmetric['delta_v_mean_m_s'])} | {number_str(proposed['delta_v_mean_m_s'])} |",
        f"| Delta-v P90 (m/s) | {number_str(singleframe['delta_v_p90_m_s'])} | {number_str(symmetric['delta_v_p90_m_s'])} | {number_str(proposed['delta_v_p90_m_s'])} |",
        f"| Delta-theta Mean (deg) | {number_str(singleframe['delta_theta_deg_mean'])} | {number_str(symmetric['delta_theta_deg_mean'])} | {number_str(proposed['delta_theta_deg_mean'])} |",
        f"| Delta-theta P90 (deg) | {number_str(singleframe['delta_theta_deg_p90'])} | {number_str(symmetric['delta_theta_deg_p90'])} | {number_str(proposed['delta_theta_deg_p90'])} |",
        f"| Delta-omega Mean (deg/s) | {number_str(singleframe['delta_omega_deg_s_mean'])} | {number_str(symmetric['delta_omega_deg_s_mean'])} | {number_str(proposed['delta_omega_deg_s_mean'])} |",
        f"| Delta-omega P90 (deg/s) | {number_str(singleframe['delta_omega_deg_s_p90'])} | {number_str(symmetric['delta_omega_deg_s_p90'])} | {number_str(proposed['delta_omega_deg_s_p90'])} |",
        f"| Impulse Metric Mean | {number_str(singleframe['impulse_metric_mean'])} | {number_str(symmetric['impulse_metric_mean'])} | {number_str(proposed['impulse_metric_mean'])} |",
        f"| Impulse Metric P90 | {number_str(singleframe['impulse_metric_p90'])} | {number_str(symmetric['impulse_metric_p90'])} | {number_str(proposed['impulse_metric_p90'])} |",
        "",
    ]
    return "\n".join(lines)


def write_report(
    path: str,
    args,
    thresholds: Tuple[float, float],
    wind_speed_range: Tuple[float, float],
    tier_counts: Dict[str, int],
    singleframe_info: Dict[str, object],
    symmetric_info: Dict[str, object],
    proposed_info: Dict[str, object],
    summary_rows: List[Dict[str, float]],
):
    weak_hi, medium_hi = thresholds
    speed_min, speed_max = wind_speed_range
    lines = [
        "# Benchmark Comparison Report",
        "",
        "## Configuration",
        "",
        f"- Seed: `{args.seed}`",
        f"- Number of benchmark cases: `{args.num_cases}`",
        f"- Success definition: `landing_error_xy < 1.0 m`",
        f"- Wind tier thresholds based on main wind speed:",
        f"  - weak: `[{speed_min:.6f}, {weak_hi:.6f}) m/s`",
        f"  - medium: `[{weak_hi:.6f}, {medium_hi:.6f}) m/s`",
        f"  - strong: `[{medium_hi:.6f}, {speed_max:.6f}] m/s`",
        f"- Tier counts: `weak={tier_counts['weak']}`, `medium={tier_counts['medium']}`, `strong={tier_counts['strong']}`",
        f"- singleframe checkpoint: `{singleframe_info['checkpoint_path']}`",
        f"- proposed_symmetric checkpoint: `{symmetric_info['checkpoint_path']}`",
        f"- proposed checkpoint: `{proposed_info['checkpoint_path']}`",
        f"- singleframe actor obs dim / frame stack: `{singleframe_info['obs_dim']}` / `{singleframe_info['frame_stack']}`",
        f"- proposed_symmetric actor obs dim / frame stack: `{symmetric_info['obs_dim']}` / `{symmetric_info['frame_stack']}`",
        f"- proposed actor obs dim / frame stack: `{proposed_info['obs_dim']}` / `{proposed_info['frame_stack']}`",
        f"- Evaluation mode: deterministic policy, single-env sequential paired benchmark",
        "",
        render_markdown_table(summary_rows, "overall"),
        render_markdown_table(summary_rows, "weak"),
        render_markdown_table(summary_rows, "medium"),
        render_markdown_table(summary_rows, "strong"),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def tier_count_dict(case_rows: List[Dict[str, float]]) -> Dict[str, int]:
    return {
        "weak": sum(1 for r in case_rows if r["wind_tier"] == "weak"),
        "medium": sum(1 for r in case_rows if r["wind_tier"] == "medium"),
        "strong": sum(1 for r in case_rows if r["wind_tier"] == "strong"),
    }


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]
    os.environ["AERIAL_GYM_EVAL_MODE"] = "0"
    silence_project_loggers()
    silence_python_warnings()

    raw_dir, stats_dir = ensure_dirs(args.output_dir)
    singleframe_ckpt = find_final_checkpoint(args.singleframe_run_dir)
    symmetric_ckpt = find_final_checkpoint(args.symmetric_run_dir)
    proposed_ckpt = find_final_checkpoint(args.proposed_run_dir)

    logger.info("Resolved singleframe checkpoint: %s", singleframe_ckpt)
    logger.info("Resolved proposed_symmetric checkpoint: %s", symmetric_ckpt)
    logger.info("Resolved proposed checkpoint: %s", proposed_ckpt)

    singleframe_actor, singleframe_rms, singleframe_obs_dim, singleframe_act_dim, singleframe_use_rnn = (
        load_ppo_policy(singleframe_ckpt, device=args.device)
    )
    symmetric_actor, symmetric_rms, symmetric_obs_dim, symmetric_act_dim, symmetric_use_rnn = load_ppo_policy(
        symmetric_ckpt, device=args.device
    )
    proposed_actor, proposed_rms, proposed_obs_dim, proposed_act_dim, proposed_use_rnn = load_ppo_policy(
        proposed_ckpt, device=args.device
    )
    shared_obs_dim = max(singleframe_obs_dim, symmetric_obs_dim, proposed_obs_dim)

    env = None
    try:
        env = build_env(
            obs_dim=shared_obs_dim,
            seed=args.seed,
            device=args.device,
            headless=args.headless,
            episode_len_steps=args.episode_len_steps,
        )
        case_rows, tier_counts_ids, thresholds = generate_benchmark_cases(
            env=env,
            base_seed=args.seed,
            num_cases=args.num_cases,
        )

        case_csv = os.path.join(raw_dir, "benchmark_cases.csv")
        write_csv(case_csv, case_rows)
        logger.info("Saved benchmark cases: %s", case_csv)

        if args.skip_eval:
            logger.info("Skip eval requested. Benchmark cases generated only.")
            return

        singleframe_info = evaluate_method(
            env=env,
            method="singleframe",
            actor=singleframe_actor,
            rms=singleframe_rms,
            obs_dim=singleframe_obs_dim,
            act_dim=singleframe_act_dim,
            use_rnn=singleframe_use_rnn,
            checkpoint_path=singleframe_ckpt,
            cases=case_rows,
            max_steps=args.max_steps,
        )
        symmetric_info = evaluate_method(
            env=env,
            method="proposed_symmetric",
            actor=symmetric_actor,
            rms=symmetric_rms,
            obs_dim=symmetric_obs_dim,
            act_dim=symmetric_act_dim,
            use_rnn=symmetric_use_rnn,
            checkpoint_path=symmetric_ckpt,
            cases=case_rows,
            max_steps=args.max_steps,
        )
        proposed_info = evaluate_method(
            env=env,
            method="proposed",
            actor=proposed_actor,
            rms=proposed_rms,
            obs_dim=proposed_obs_dim,
            act_dim=proposed_act_dim,
            use_rnn=proposed_use_rnn,
            checkpoint_path=proposed_ckpt,
            cases=case_rows,
            max_steps=args.max_steps,
        )

        singleframe_rows = singleframe_info["rows"]
        symmetric_rows = symmetric_info["rows"]
        proposed_rows = proposed_info["rows"]
        combined_rows = singleframe_rows + symmetric_rows + proposed_rows

        write_csv(os.path.join(raw_dir, "singleframe_results.csv"), singleframe_rows)
        write_csv(os.path.join(raw_dir, "proposed_symmetric_results.csv"), symmetric_rows)
        write_csv(os.path.join(raw_dir, "proposed_results.csv"), proposed_rows)
        write_csv(os.path.join(raw_dir, "combined_results.csv"), combined_rows)

        summary_rows = summarize_all(
            {
                "singleframe": singleframe_rows,
                "proposed_symmetric": symmetric_rows,
                "proposed": proposed_rows,
            }
        )
        summary_csv = os.path.join(stats_dir, "summary_by_split.csv")
        write_summary_csv(summary_csv, summary_rows)

        report_path = os.path.join(args.output_dir, "comparison_report.md")
        cfg = task_config.gmm_force_config
        speed_min = float(getattr(cfg, "main_wind_speed_min", 1.0))
        speed_max = float(getattr(cfg, "main_wind_speed_max", 3.0))
        if speed_max < speed_min:
            speed_min, speed_max = speed_max, speed_min
        write_report(
            path=report_path,
            args=args,
            thresholds=thresholds,
            wind_speed_range=(speed_min, speed_max),
            tier_counts=tier_count_dict(case_rows),
            singleframe_info=singleframe_info,
            symmetric_info=symmetric_info,
            proposed_info=proposed_info,
            summary_rows=summary_rows,
        )

        logger.info("Summary saved: %s", summary_csv)
        logger.info("Report saved: %s", report_path)
    finally:
        close_env(env)


if __name__ == "__main__":
    main()
