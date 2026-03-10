#!/usr/bin/env python3
"""
Oscillation-focused validation for lmf2_velocity_control on navigation_task_gmm_noise.

This script is intentionally separate from existing crash/stability tuning scripts.
It checks whether apparent "stable" behavior is achieved via persistent high-frequency
oscillations (which is undesirable for real motors/airframes).

Outputs:
- Per-case metrics (hover/step tracking) in isolated and task-disturbance modes
- Risk flags for persistent oscillation
- JSON report under aerial_gym/debug/reports/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

# Import aerial_gym modules BEFORE torch (Isaac Gym requirement)
from aerial_gym.task.navigation_task_gmm_noise import NavigationTaskGmmNoise
from aerial_gym.config.task_config import navigation_task_gmm_noise_config

import torch

FLIP_HEIGHT_M = 0.10
FLIP_TILT_DEG = 55.0


def infer_dt(task: NavigationTaskGmmNoise) -> float:
    try:
        return float(task.sim_env.sim_config.sim.dt)
    except Exception:
        return 0.01


def spectral_metrics(signal: np.ndarray, fs: float, hf_low_hz: float = 5.0, hf_high_hz: float = 30.0) -> Tuple[float, float]:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 64:
        return 0.0, 0.0
    x = x - np.mean(x)
    if not np.isfinite(x).all():
        return 0.0, 0.0

    window = np.hanning(x.size)
    if np.sum(window) <= 0:
        return 0.0, 0.0

    spectrum = np.fft.rfft(x * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

    valid = freqs >= 0.5
    total_power = float(power[valid].sum())
    if total_power <= 1e-12:
        return 0.0, 0.0

    hf_band = (freqs >= hf_low_hz) & (freqs <= min(hf_high_hz, fs * 0.5 - 1e-3))
    hf_ratio = float(power[hf_band].sum() / total_power) if np.any(hf_band) else 0.0

    peak_idx = np.argmax(power[valid])
    dominant_freq = float(freqs[valid][peak_idx])
    return hf_ratio, dominant_freq


def make_case_actions(case: str, step: int, num_envs: int, device: str) -> torch.Tensor:
    if case == "hover":
        action = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
    elif case == "step_vx":
        if step < 200:
            action = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
        else:
            action = torch.tensor([0.6, 0.0, 0.0, 0.0], device=device)
    else:
        raise ValueError(f"Unknown case: {case}")
    return action.view(1, 4).repeat(num_envs, 1)


def run_case(task: NavigationTaskGmmNoise, case: str, steps: int, steady_start: int) -> Dict[str, float]:
    dt = infer_dt(task)
    fs = 1.0 / dt
    num_envs = task.sim_env.num_envs
    device = task.device

    task.reset()

    vel_err_hist: List[float] = []
    angvel_mag_hist: List[float] = []
    roll_deg_hist: List[float] = []
    pitch_deg_hist: List[float] = []
    rp_mag_deg_hist: List[float] = []
    thrust_hist: List[np.ndarray] = []
    raw_action_hist: List[np.ndarray] = []
    first_crash_step = -1
    first_crash_env = -1
    first_crash_z_m = 0.0
    first_crash_tilt_deg = 0.0
    first_crash_contact_norm = 0.0
    first_crash_is_flip = False
    truncation_events = 0
    ever_crashed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    flip_crash_count = 0
    nonflip_crash_count = 0

    for step in range(steps):
        actions = make_case_actions(case, step, num_envs, device)
        obs, reward, terminated, truncated, info = task.step(actions)

        cmd_vel = task.action_transformation_function(actions)[:, :3]
        body_vel = task.obs_dict["robot_body_linvel"]
        body_angvel = task.obs_dict["robot_body_angvel"]
        euler = task.obs_dict["robot_euler_angles"]

        vel_err = torch.norm(body_vel - cmd_vel, dim=1)
        vel_err_hist.append(float(vel_err[0].item()))

        angvel_mag = torch.norm(body_angvel[0], dim=0).item()
        angvel_mag_hist.append(float(angvel_mag))

        roll_deg = float(torch.rad2deg(torch.abs(euler[0, 0])).item())
        pitch_deg = float(torch.rad2deg(torch.abs(euler[0, 1])).item())
        rp_mag_deg = float(np.sqrt(roll_deg ** 2 + pitch_deg ** 2))

        roll_deg_hist.append(roll_deg)
        pitch_deg_hist.append(pitch_deg)
        rp_mag_deg_hist.append(rp_mag_deg)

        raw_action_hist.append(actions[0].detach().cpu().numpy())

        motor_thrust = None
        try:
            motor_thrust = task.sim_env.robot_manager.robot.control_allocator.motor_model.current_motor_thrust
        except Exception:
            pass
        if motor_thrust is not None:
            thrust_hist.append(motor_thrust[0].detach().cpu().numpy().astype(np.float64))

        if bool(truncated[0].item()):
            truncation_events += 1

        crash_mask = terminated > 0
        new_crashes = crash_mask & (~ever_crashed)
        if torch.any(new_crashes):
            new_ids = torch.nonzero(new_crashes, as_tuple=False).squeeze(-1)
            for env_id_t in new_ids:
                env_id = int(env_id_t.item())
                z = float(task.obs_dict["robot_position"][env_id, 2].item())
                e = task.obs_dict["robot_euler_angles"][env_id]
                tilt_deg = float(torch.rad2deg(torch.sqrt(e[0] * e[0] + e[1] * e[1])).item())
                contact_norm = float(torch.norm(task.obs_dict["robot_contact_force_tensor"][env_id]).item())
                is_flip = (z < FLIP_HEIGHT_M) or (tilt_deg > FLIP_TILT_DEG)

                if is_flip:
                    flip_crash_count += 1
                else:
                    nonflip_crash_count += 1

                if first_crash_step < 0:
                    first_crash_step = step
                    first_crash_env = env_id
                    first_crash_z_m = z
                    first_crash_tilt_deg = tilt_deg
                    first_crash_contact_norm = contact_norm
                    first_crash_is_flip = bool(is_flip)

            ever_crashed |= new_crashes

    vel_err_arr = np.asarray(vel_err_hist, dtype=np.float64)
    angvel_arr = np.asarray(angvel_mag_hist, dtype=np.float64)
    roll_arr = np.asarray(roll_deg_hist, dtype=np.float64)
    pitch_arr = np.asarray(pitch_deg_hist, dtype=np.float64)
    rp_arr = np.asarray(rp_mag_deg_hist, dtype=np.float64)
    act_arr = np.asarray(raw_action_hist, dtype=np.float64)
    thrust_arr = np.asarray(thrust_hist, dtype=np.float64) if thrust_hist else None

    if vel_err_arr.size == 0:
        raise RuntimeError("No samples collected. Simulation likely failed immediately.")

    s0 = min(max(steady_start, 0), vel_err_arr.size - 1)
    steady_slice = slice(s0, None)

    vel_err_rms = float(np.sqrt(np.mean(vel_err_arr ** 2)))
    vel_err_steady_mean = float(np.mean(vel_err_arr[steady_slice]))
    vel_err_steady_pp = float(np.percentile(vel_err_arr[steady_slice], 95) - np.percentile(vel_err_arr[steady_slice], 5))

    angvel_rms = float(np.sqrt(np.mean(angvel_arr ** 2)))
    angvel_steady_rms = float(np.sqrt(np.mean(angvel_arr[steady_slice] ** 2)))

    rp_rms_deg = float(np.sqrt(np.mean(rp_arr ** 2)))
    rp_steady_rms_deg = float(np.sqrt(np.mean(rp_arr[steady_slice] ** 2)))
    rp_peak_deg = float(np.max(rp_arr))

    action_delta_rms = 0.0
    action_sat_rate = 0.0
    if act_arr.shape[0] >= 2:
        act_delta = np.diff(act_arr, axis=0)
        action_delta_rms = float(np.sqrt(np.mean(np.sum(act_delta ** 2, axis=1))))
        action_sat_rate = float(np.mean(np.abs(act_arr) >= 0.98))

    angvel_hf_ratio, angvel_dom_hz = spectral_metrics(angvel_arr[steady_slice], fs=fs)

    thrust_hf_ratio_mean = 0.0
    thrust_dom_hz_mean = 0.0
    thrust_cv_mean = 0.0
    if thrust_arr is not None and thrust_arr.ndim == 2 and thrust_arr.shape[0] >= 64:
        t_steady = thrust_arr[steady_slice]
        cv_list = []
        hf_list = []
        dom_list = []
        for i in range(t_steady.shape[1]):
            sig = t_steady[:, i]
            mean_abs = float(np.mean(np.abs(sig)))
            std = float(np.std(sig))
            cv_list.append(std / (mean_abs + 1e-6))
            hf_ratio, dom_hz = spectral_metrics(sig, fs=fs)
            hf_list.append(hf_ratio)
            dom_list.append(dom_hz)
        thrust_cv_mean = float(np.mean(cv_list))
        thrust_hf_ratio_mean = float(np.mean(hf_list))
        thrust_dom_hz_mean = float(np.mean(dom_list))

    return {
        "dt": dt,
        "requested_steps": int(steps),
        "samples": int(vel_err_arr.size),
        "first_crash_step": int(first_crash_step),
        "first_crash_env": int(first_crash_env),
        "first_crash_z_m": float(first_crash_z_m),
        "first_crash_tilt_deg": float(first_crash_tilt_deg),
        "first_crash_contact_norm": float(first_crash_contact_norm),
        "first_crash_is_flip": bool(first_crash_is_flip),
        "crashed_env_count": int(torch.count_nonzero(ever_crashed).item()),
        "flip_crash_count": int(flip_crash_count),
        "nonflip_crash_count": int(nonflip_crash_count),
        "truncation_events": int(truncation_events),
        "vel_err_rms_mps": vel_err_rms,
        "vel_err_steady_mean_mps": vel_err_steady_mean,
        "vel_err_steady_p95_p05_mps": vel_err_steady_pp,
        "angvel_rms_rps": angvel_rms,
        "angvel_steady_rms_rps": angvel_steady_rms,
        "roll_pitch_rms_deg": rp_rms_deg,
        "roll_pitch_steady_rms_deg": rp_steady_rms_deg,
        "roll_pitch_peak_deg": rp_peak_deg,
        "action_delta_rms": action_delta_rms,
        "action_saturation_rate": action_sat_rate,
        "angvel_hf_ratio": angvel_hf_ratio,
        "angvel_dom_hz": angvel_dom_hz,
        "thrust_hf_ratio_mean": thrust_hf_ratio_mean,
        "thrust_dom_hz_mean": thrust_dom_hz_mean,
        "thrust_cv_mean": thrust_cv_mean,
    }


def evaluate_risk(case: str, metrics: Dict[str, float]) -> Dict[str, object]:
    # Thresholds are intentionally conservative for sim-to-real safety.
    checks = []
    # If run is too short, steady-state metrics are not trustworthy.
    checks.append(("sufficient_samples", metrics["samples"] >= int(0.9 * metrics["requested_steps"])))

    if case == "hover":
        checks = [
            *checks,
            ("angvel_steady_rms_rps", metrics["angvel_steady_rms_rps"] <= 0.8),
            ("roll_pitch_steady_rms_deg", metrics["roll_pitch_steady_rms_deg"] <= 5.0),
            ("thrust_cv_mean", metrics["thrust_cv_mean"] <= 0.30),
        ]
        # Only evaluate HF ratios when signal magnitude is non-trivial.
        if metrics["angvel_steady_rms_rps"] > 0.05:
            checks.append(("angvel_hf_ratio", metrics["angvel_hf_ratio"] <= 0.30))
        if metrics["thrust_cv_mean"] > 0.02:
            checks.append(("thrust_hf_ratio_mean", metrics["thrust_hf_ratio_mean"] <= 0.35))
    elif case == "step_vx":
        checks = [
            *checks,
            ("vel_err_steady_mean_mps", metrics["vel_err_steady_mean_mps"] <= 0.12),
            ("vel_err_steady_p95_p05_mps", metrics["vel_err_steady_p95_p05_mps"] <= 0.20),
            ("angvel_steady_rms_rps", metrics["angvel_steady_rms_rps"] <= 1.0),
        ]
        if metrics["angvel_steady_rms_rps"] > 0.05:
            checks.append(("angvel_hf_ratio", metrics["angvel_hf_ratio"] <= 0.35))
        if metrics["thrust_cv_mean"] > 0.02:
            checks.append(("thrust_hf_ratio_mean", metrics["thrust_hf_ratio_mean"] <= 0.40))
    else:
        raise ValueError(f"Unknown case: {case}")

    failed = [name for name, ok in checks if not ok]
    if metrics["crashed_env_count"] > 0:
        failed.append("crash_detected")
    if metrics["flip_crash_count"] > 0:
        failed.append("flip_crash")
    if metrics["nonflip_crash_count"] > 0:
        failed.append("contact_crash_nonflip")

    fail_count = len(failed)
    if fail_count == 0:
        verdict = "PASS"
        risk_level = "low"
    elif fail_count <= 2:
        verdict = "WARN"
        risk_level = "medium"
    else:
        verdict = "FAIL"
        risk_level = "high"

    return {
        "verdict": verdict,
        "risk_level": risk_level,
        "failed_checks": failed,
    }


def configure_task(num_envs: int, device: str, headless: bool, enable_noise: bool, enable_force: bool, steps_hint: int) -> NavigationTaskGmmNoise:
    cfg = navigation_task_gmm_noise_config.task_config
    cfg.headless = headless
    cfg.num_envs = num_envs
    cfg.device = device
    cfg.env_name = "empty_env"
    cfg.num_obstacles_in_env = 0
    cfg.episode_len_steps = max(1_000_000, steps_hint + 50)
    # Controller oscillation analysis does not depend on image encoder quality.
    cfg.vae_config.use_vae = False
    cfg.noise_config.enable_noise = enable_noise
    cfg.gmm_force_config.enable_physical_force = enable_force
    task = NavigationTaskGmmNoise(cfg)

    # In empty_env there may be no depth sensor. Inject a safe fallback tensor so
    # reward code paths that read depth_range_pixels can still execute.
    if "depth_range_pixels" not in task.obs_dict:
        task.obs_dict["depth_range_pixels"] = torch.full(
            (task.sim_env.num_envs, 1, 1, 1),
            10.0,
            device=task.device,
            dtype=torch.float32,
        )
    return task


def main():
    parser = argparse.ArgumentParser(description="Oscillation-focused validation for lmf2 controller")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Enable rendering window")
    parser.add_argument("--hover_steps", type=int, default=700, help="Steps for hover test")
    parser.add_argument("--step_steps", type=int, default=900, help="Steps for step test")
    parser.add_argument(
        "--include_task_disturbance",
        action="store_true",
        help="Also run disturbance-enabled mode (slower, may be less stable on some setups)",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write JSON report")
    args = parser.parse_args()

    # Avoid Isaac Gym helper parsers from consuming this script's CLI args later.
    sys.argv = [sys.argv[0]]
    # Deterministic eval physics and clean crash-flag resets across episodes.
    os.environ["AERIAL_GYM_EVAL_MODE"] = "1"

    run_modes = [("isolated_controller", False, False)]
    if args.include_task_disturbance:
        run_modes.append(("task_disturbance", True, True))
    cases = [
        ("hover", args.hover_steps, int(0.60 * args.hover_steps)),
        ("step_vx", args.step_steps, max(300, int(0.60 * args.step_steps))),
    ]

    all_results: Dict[str, Dict[str, object]] = {}
    overall_failed_checks = []

    for mode_name, enable_noise, enable_force in run_modes:
        task = None
        try:
            task = configure_task(
                num_envs=args.num_envs,
                device=args.device,
                headless=args.headless,
                enable_noise=enable_noise,
                enable_force=enable_force,
                steps_hint=max(args.hover_steps, args.step_steps),
            )
            mode_results = {}
            for case_name, steps, steady_start in cases:
                metrics = run_case(task, case=case_name, steps=steps, steady_start=steady_start)
                risk = evaluate_risk(case_name, metrics)
                mode_results[case_name] = {
                    "metrics": metrics,
                    "risk": risk,
                }
                if risk["verdict"] != "PASS":
                    for ck in risk["failed_checks"]:
                        overall_failed_checks.append(f"{mode_name}.{case_name}.{ck}")
            all_results[mode_name] = mode_results
        finally:
            if task is not None:
                try:
                    task.close()
                except Exception:
                    pass

    isolated_verdicts = [
        all_results["isolated_controller"][k]["risk"]["verdict"] for k in ("hover", "step_vx")
    ]
    if all(v == "PASS" for v in isolated_verdicts):
        final_judgment = "controller_oscillation_status: acceptable"
    elif any(v == "FAIL" for v in isolated_verdicts):
        final_judgment = "controller_oscillation_status: unacceptable"
    else:
        final_judgment = "controller_oscillation_status: caution"

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "num_envs": args.num_envs,
            "device": args.device,
            "headless": args.headless,
            "hover_steps": args.hover_steps,
            "step_steps": args.step_steps,
        },
        "results": all_results,
        "final_judgment": final_judgment,
        "overall_failed_checks": overall_failed_checks,
    }

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "reports")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"lmf2_oscillation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("LMF2 Oscillation Validation Report")
    print("=" * 72)
    print(f"Output JSON: {out_path}")
    print(f"Final judgment: {final_judgment}")
    print("")

    for mode_name, _, _ in run_modes:
        print(f"[{mode_name}]")
        for case_name in ("hover", "step_vx"):
            case = all_results[mode_name][case_name]
            m = case["metrics"]
            r = case["risk"]
            print(
                f"  - {case_name:7s} | verdict={r['verdict']:4s} risk={r['risk_level']:6s} "
                f"| angvel_hf={m['angvel_hf_ratio']:.3f} thrust_hf={m['thrust_hf_ratio_mean']:.3f} "
                f"| angvel_steady={m['angvel_steady_rms_rps']:.3f} rad/s "
                f"| rp_steady={m['roll_pitch_steady_rms_deg']:.2f} deg "
                f"| vel_ss_mean={m['vel_err_steady_mean_mps']:.3f} m/s "
                f"| vel_ss_pp={m['vel_err_steady_p95_p05_mps']:.3f} m/s "
                f"| crashed_envs={m['crashed_env_count']} "
                f"| flip={m['flip_crash_count']} nonflip={m['nonflip_crash_count']}"
            )
            if r["failed_checks"]:
                print(f"    failed_checks: {', '.join(r['failed_checks'])}")
        print("")


if __name__ == "__main__":
    main()
