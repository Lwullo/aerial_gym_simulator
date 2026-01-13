#!/usr/bin/env python3
"""
Automatic tuning of Lee controller gains (K_vel, K_rot, K_angvel) using Optuna.
The script:
1. Creates a temporary copy of `lmf2_controller_config.py` with trial-specific gains.
2. Modifies the navigation task config to run in an empty environment (no obstacles, no presets).
3. Executes `test_lee_waypoint_convergence.py` and captures its output.
4. Parses velocity error, max attitude, and crash flag.
5. Computes a score = avg_error + 0.05*max_attitude + 10*crash_flag.
6. Optimises the gains to minimise the score.
7. Writes the best gains back to the original `lmf2_controller_config.py`.
8. Prints the best gains and a short report.

Requirements:
- `optuna` (`pip install optuna`)
- Standard library modules only (subprocess, re, json, shutil, tempfile, pathlib).
"""

import optuna
import subprocess
import re
import json
import shutil
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (adjust if your project layout changes)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../aerial_gym_simulator
CONFIG_DIR = PROJECT_ROOT / "aerial_gym" / "config" / "controller_config"
TASK_CONFIG = PROJECT_ROOT / "aerial_gym" / "config" / "task_config" / "navigation_task_gmm_noise_config.py"
TEST_SCRIPT = PROJECT_ROOT / "aerial_gym" / "debug" / "test_lee_waypoint_convergence.py"

LMF2_CONFIG_ORIG = CONFIG_DIR / "lmf2_controller_config.py"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def replace_gain_in_file(src_path: Path, dst_path: Path, k_vel: float, k_rot: float, k_angvel: float):
    """Create a copy of the controller config with the given gains.
    The original file defines K_*_tensor_max/min as torch.tensor([...]).
    We replace the numeric values (keeping the tensor shape).
    """
    text = src_path.read_text(encoding="utf-8")
    # Simple regex replacement for the three lines
    text = re.sub(r"K_vel_tensor_max\s*=\s*torch\.tensor\([^\]]+\)",
                  f"K_vel_tensor_max = torch.tensor([{k_vel:.4f}, {k_vel:.4f}, {k_vel:.4f}])",
                  text)
    text = re.sub(r"K_vel_tensor_min\s*=\s*torch\.tensor\([^\]]+\)",
                  f"K_vel_tensor_min = torch.tensor([{k_vel:.4f}, {k_vel:.4f}, {k_vel:.4f}])",
                  text)
    text = re.sub(r"K_rot_tensor_max\s*=\s*torch\.tensor\([^\]]+\)",
                  f"K_rot_tensor_max = torch.tensor([{k_rot:.4f}, {k_rot:.4f}, {k_rot:.4f}])",
                  text)
    text = re.sub(r"K_rot_tensor_min\s*=\s*torch\.tensor\([^\]]+\)",
                  f"K_rot_tensor_min = torch.tensor([{k_rot:.4f}, {k_rot:.4f}, {k_rot:.4f}])",
                  text)
    text = re.sub(r"K_angvel_tensor_max\s*=\s*torch\.tensor\([^\]]+\)",
                  f"K_angvel_tensor_max = torch.tensor([{k_angvel:.4f}, {k_angvel:.4f}, {k_angvel:.4f}])",
                  text)
    text = re.sub(r"K_angvel_tensor_min\s*=\s*torch\.tensor\([^\]]+\)",
                  f"K_angvel_tensor_min = torch.tensor([{k_angvel:.4f}, {k_angvel:.4f}, {k_angvel:.4f}])",
                  text)
    dst_path.write_text(text, encoding="utf-8")

def make_task_config_empty(task_cfg_path: Path):
    """Modify the task config to disable obstacles and presets.
    We edit the file in‑place (backup first) because the test script imports it.
    """
    backup = task_cfg_path.with_suffix('.py.bak')
    shutil.copy2(task_cfg_path, backup)
    txt = task_cfg_path.read_text(encoding="utf-8")
    txt = re.sub(r"num_obstacles_in_env\s*=\s*[^\n]+", "num_obstacles_in_env = 0", txt)
    txt = re.sub(r"preset_id\s*=\s*[^\n]+", "preset_id = -1", txt)
    task_cfg_path.write_text(txt, encoding="utf-8")
    return backup

def restore_task_config(task_cfg_path: Path, backup_path: Path):
    shutil.move(str(backup_path), str(task_cfg_path))

def run_test_and_collect():
    """Run the convergence test script and parse the output.
    Returns (avg_error, max_attitude_deg, crash_flag).
    """
    proc = subprocess.run(["python", str(TEST_SCRIPT)],
                          capture_output=True, text=True, timeout=300)
    out = proc.stdout
    # Extract all "Velocity Error:" lines (format: Velocity Error: X.XXX m/s)
    vel_errors = [float(m.group(1)) for m in re.finditer(r"Velocity Error:\s*([0-9\.]+)\s*m/s", out)]
    # Extract all "Max Attitude:" lines (format: Max Attitude: X.X°)
    attitudes = [float(m.group(1)) for m in re.finditer(r"Max Attitude:\s*([0-9\.]+)°", out)]
    avg_error = sum(vel_errors) / len(vel_errors) if vel_errors else 1e3
    max_att = max(attitudes) if attitudes else 0.0
    crash_flag = 1 if "⚠ Episode ended prematurely!" in out else 0
    return avg_error, max_att, crash_flag, out

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial):
    # Sample gains
    k_vel = trial.suggest_float("k_vel", 0.5, 3.5)
    k_rot = trial.suggest_float("k_rot", 2.0, 10.0)
    k_ang = trial.suggest_float("k_angvel", 1.0, 6.0)

    # Create a temporary controller config
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cfg = Path(tmpdir) / "lmf2_controller_config.py"
        replace_gain_in_file(LMF2_CONFIG_ORIG, tmp_cfg, k_vel, k_rot, k_ang)
        # Insert the temporary config directory at the front of PYTHONPATH for this run
        env = dict(**os.environ)
        env["PYTHONPATH"] = str(tmpdir) + os.pathsep + env.get("PYTHONPATH", "")

        # Ensure empty environment (obstacles disabled)
        backup_task = make_task_config_empty(TASK_CONFIG)
        try:
            # Run the test script with the temporary config on the path
            proc = subprocess.run(["python", str(TEST_SCRIPT)],
                                  env=env, capture_output=True, text=True, timeout=300)
            out = proc.stdout
            # Parse results
            vel_errors = [float(m.group(1)) for m in re.finditer(r"Velocity Error:\s*([0-9\.]+)\s*m/s", out)]
            attitudes = [float(m.group(1)) for m in re.finditer(r"Max Attitude:\s*([0-9\.]+)°", out)]
            avg_error = sum(vel_errors) / len(vel_errors) if vel_errors else 1e3
            max_att = max(attitudes) if attitudes else 0.0
            crash_flag = 1 if "⚠ Episode ended prematurely!" in out else 0
        finally:
            # Restore original task config regardless of success/failure
            restore_task_config(TASK_CONFIG, backup_task)

    # Compute score (lower is better)
    score = avg_error + 0.05 * max_att + 10.0 * crash_flag
    # Store the raw metrics for later analysis
    trial.set_user_attr("avg_error", avg_error)
    trial.set_user_attr("max_attitude", max_att)
    trial.set_user_attr("crash_flag", crash_flag)
    trial.set_user_attr("output", out)
    return score

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40, n_jobs=4)

    best = study.best_trial
    best_params = best.params
    print("\n=== Best Gains Found ===")
    print(f"K_vel    = {best_params['k_vel']:.4f}")
    print(f"K_rot    = {best_params['k_rot']:.4f}")
    print(f"K_angvel = {best_params['k_angvel']:.4f}")
    print(f"Score    = {best.value:.4f}")
    print(f"Avg Error = {best.user_attrs['avg_error']:.4f} m/s")
    print(f"Max Attitude = {best.user_attrs['max_attitude']:.2f}°")
    print(f"Crash Flag = {best.user_attrs['crash_flag']}")

    # Write the best gains back to the original controller config
    replace_gain_in_file(LMF2_CONFIG_ORIG, LMF2_CONFIG_ORIG,
                        best_params['k_vel'], best_params['k_rot'], best_params['k_angvel'])
    print("\nBest gains have been written back to:", LMF2_CONFIG_ORIG)

    # Optionally, save a JSON summary for reproducibility
    summary = {
        "best_gains": {
            "K_vel": best_params['k_vel'],
            "K_rot": best_params['k_rot'],
            "K_angvel": best_params['k_angvel']
        },
        "metrics": {
            "avg_error": best.user_attrs['avg_error'],
            "max_attitude": best.user_attrs['max_attitude'],
            "crash_flag": best.user_attrs['crash_flag']
        },
        "score": best.value
    }
    summary_path = Path(__file__).with_name("best_gains.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary written to", summary_path)
