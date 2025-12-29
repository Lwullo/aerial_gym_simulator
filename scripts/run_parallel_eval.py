#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import yaml


def str2bool(value):
    if isinstance(value, bool):
        return value
    val = value.strip().lower()
    if val in ("1", "true", "yes", "y", "t"):
        return True
    if val in ("0", "false", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run compute_score_map or policy eval for a single preset environment."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--run-runner",
        action="store_true",
        help="Run policy eval with runner.py (navigation_task_gmm_noise).",
    )
    mode_group.add_argument(
        "--run-score-map",
        action="store_true",
        help="Run compute_score_map.py only.",
    )
    parser.add_argument("--checkpoint", help="Path to trained policy checkpoint.")
    parser.add_argument("--ppo-config", dest="ppo_config", help="Path to PPO yaml.")
    parser.add_argument("--task", default="navigation_task_gmm_noise", help="Task name (runner only).")
    parser.add_argument(
        "--preset-id",
        type=int,
        default=0,
        help="Preset id to evaluate.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Num envs for policy eval.")
    parser.add_argument("--headless", type=str2bool, default=False, help="Headless for policy eval.")
    parser.add_argument("--use-warp", type=str2bool, default=True, help="Use warp for policy eval.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--experiment-name",
        default="gmm_noise_eval",
        help="Experiment name prefix for eval runs.",
    )
    parser.add_argument("--runner", default=None, help="Path to runner.py.")
    parser.add_argument(
        "--compute-score-map",
        default=None,
        dest="score_map",
        help="Path to compute_score_map.py.",
    )
    parser.add_argument("--log-dir", default=None, help="Directory for log files.")
    return parser.parse_args()


def _maybe_abs(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def _find_latest_run_dir(runs_dir, experiment_name):
    if not os.path.isdir(runs_dir):
        return None
    candidates = [
        os.path.join(runs_dir, name)
        for name in os.listdir(runs_dir)
        if name.startswith(experiment_name)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _copy_best_point(runs_dir, experiment_name, compare_dir, preset_id):
    run_dir = _find_latest_run_dir(runs_dir, experiment_name)
    if run_dir is None:
        raise FileNotFoundError(
            f"No run dir found under {runs_dir} for experiment '{experiment_name}'."
        )
    src = os.path.join(run_dir, "best_point.txt")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"best_point.txt not found in {run_dir}")
    dst = os.path.join(compare_dir, f"best_point_runner_p{preset_id}.txt")
    shutil.copy2(src, dst)
    return dst


def _copy_score_map_outputs(score_dir, compare_dir, preset_id):
    suffix = f"_preset_{preset_id}"
    filenames = [
        f"score_components_boxplot{suffix}.txt",
        f"score_components_boxplot{suffix}.png",
        f"score_3d_heatmap{suffix}.png",
    ]
    copied = []
    for name in filenames:
        src = os.path.join(score_dir, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing score map output: {src}")
        dst = os.path.join(compare_dir, name)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _write_single_episode_config(config_path, output_dir, tag):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format: {config_path}")
    params_cfg = config.setdefault("params", {})
    config_cfg = params_cfg.setdefault("config", {})
    player_cfg = config_cfg.setdefault("player", {})
    player_cfg["games_num"] = 1
    if "games_to_play" in player_cfg:
        player_cfg["games_to_play"] = 1
    out_path = os.path.join(output_dir, f"ppo_single_episode_{tag}.yaml")
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return out_path


def main():
    args = parse_args()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner_path = args.runner or os.path.join(
        root_dir, "aerial_gym", "rl_training", "rl_games", "runner.py"
    )
    score_map_path = args.score_map or os.path.join(
        root_dir, "aerial_gym", "debug", "bruteforce_score", "compute_score_map.py"
    )

    log_dir = args.log_dir or os.path.join(root_dir, "runs", "single_eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    compare_dir = os.path.join(root_dir, "aerial_gym", "debug", "compare")
    os.makedirs(compare_dir, exist_ok=True)

    python = sys.executable
    preset_id = int(args.preset_id)
    out_tag = f"preset_{preset_id}"

    if args.run_score_map:
        score_cmd = [
            python,
            score_map_path,
            "--preset_id",
            str(preset_id),
            "--out_tag",
            out_tag,
        ]
        if args.seed is not None:
            score_cmd += ["--seed", str(args.seed)]
        score_log = os.path.join(log_dir, f"compute_score_map_{out_tag}.log")
        with open(score_log, "w", encoding="utf-8") as log_file:
            score_code = subprocess.call(score_cmd, stdout=log_file, stderr=subprocess.STDOUT)
        if score_code != 0:
            print(f"score_map preset {preset_id} failed with code {score_code} (log: {score_log})")
            sys.exit(score_code)
        try:
            _copy_score_map_outputs(os.path.dirname(score_map_path), compare_dir, preset_id)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print("done")
        return

    if not args.checkpoint or not args.ppo_config:
        print("runner mode requires --checkpoint and --ppo-config", file=sys.stderr)
        sys.exit(2)

    ppo_config = _maybe_abs(args.ppo_config)
    checkpoint = _maybe_abs(args.checkpoint)
    exp_name = f"{args.experiment_name}_p{preset_id}"
    runner_task = "navigation_task_gmm_noise"
    if args.task != runner_task:
        print(f"warning: overriding --task to {runner_task} for runner mode")
    if args.num_envs != 1:
        print("warning: overriding --num-envs to 1 for single-episode runner mode")
    ppo_config = _write_single_episode_config(ppo_config, log_dir, out_tag)
    runner_num_envs = 1
    runner_cmd = [
        python,
        runner_path,
        "--play",
        "--file",
        ppo_config,
        "--task",
        runner_task,
        "--experiment_name",
        exp_name,
        "--checkpoint",
        checkpoint,
        "--preset_id",
        str(preset_id),
        "--num_envs",
        str(runner_num_envs),
        "--headless",
        str(args.headless),
        "--use_warp",
        str(args.use_warp),
    ]
    if args.seed is not None:
        runner_cmd += ["--seed", str(args.seed)]
    runner_log = os.path.join(log_dir, f"runner_{out_tag}.log")
    with open(runner_log, "w", encoding="utf-8") as log_file:
        runner_env = os.environ.copy()
        runner_env["AERIAL_GYM_LOG_STEP_SCORES"] = "1"
        runner_code = subprocess.call(
            runner_cmd, stdout=log_file, stderr=subprocess.STDOUT, env=runner_env
        )
    if runner_code != 0:
        print(f"runner preset {preset_id} failed with code {runner_code} (log: {runner_log})")
        sys.exit(runner_code)

    runner_dir = os.path.dirname(os.path.abspath(runner_path))
    runs_dir = os.path.join(runner_dir, "runs")
    try:
        _copy_best_point(runs_dir, exp_name, compare_dir, preset_id)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
