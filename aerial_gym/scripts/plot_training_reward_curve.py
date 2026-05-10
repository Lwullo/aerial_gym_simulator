#!/usr/bin/env python3
"""Plot a smoothed training reward curve from TensorBoard event files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TAG_PRIORITY = (
    "rewards/iter",
    "reward/iter",
    "episode_reward/iter",
    "rewards/step",
    "reward",
)


def _load_tensorboard_helpers():
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        from tensorboard.util import tensor_util
    except Exception as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "TensorBoard is required to read event files. Install tensorboard or "
            "run this script in the same environment used for training."
        ) from exc
    return EventFileLoader, tensor_util


def _scalar_value(summary_value, tensor_util) -> Optional[float]:
    if summary_value.HasField("tensor"):
        arr = tensor_util.make_ndarray(summary_value.tensor)
        try:
            return float(np.asarray(arr).reshape(-1)[0])
        except Exception:
            return None
    try:
        return float(summary_value.simple_value)
    except Exception:
        return None


def find_event_files(run_dir: Path) -> List[Path]:
    summary_dir = run_dir / "summaries"
    roots = [summary_dir, run_dir] if summary_dir.exists() else [run_dir]
    event_files: List[Path] = []
    for root in roots:
        event_files.extend(sorted(root.glob("events.out.tfevents*")))
    # Preserve order while removing duplicates.
    seen = set()
    unique: List[Path] = []
    for path in event_files:
        key = path.resolve()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def list_scalar_tags(event_files: Sequence[Path]) -> Dict[str, int]:
    EventFileLoader, tensor_util = _load_tensorboard_helpers()
    counts: Dict[str, int] = {}
    for event_path in event_files:
        for event in EventFileLoader(str(event_path)).Load():
            for value in event.summary.value:
                if _scalar_value(value, tensor_util) is None:
                    continue
                counts[value.tag] = counts.get(value.tag, 0) + 1
    return counts


def choose_reward_tag(tags: Iterable[str], requested_tag: Optional[str]) -> str:
    tag_set = set(tags)
    if requested_tag:
        if requested_tag not in tag_set:
            available = ", ".join(sorted(tag_set))
            raise ValueError(f"Requested tag '{requested_tag}' not found. Available tags: {available}")
        return requested_tag

    for tag in DEFAULT_TAG_PRIORITY:
        if tag in tag_set:
            return tag

    reward_like = [
        tag
        for tag in sorted(tag_set)
        if ("reward" in tag.lower() or "rew" in tag.lower()) and "shaped" not in tag.lower()
    ]
    if reward_like:
        return reward_like[0]

    available = ", ".join(sorted(tag_set))
    raise ValueError(f"No reward-like scalar tag found. Available tags: {available}")


def read_scalar_series(event_files: Sequence[Path], tag: str) -> Tuple[np.ndarray, np.ndarray]:
    EventFileLoader, tensor_util = _load_tensorboard_helpers()
    by_step: Dict[int, Tuple[float, float]] = {}
    for event_path in event_files:
        for event in EventFileLoader(str(event_path)).Load():
            for value in event.summary.value:
                if value.tag != tag:
                    continue
                scalar = _scalar_value(value, tensor_util)
                if scalar is None or not np.isfinite(scalar):
                    continue
                step = int(event.step)
                wall_time = float(getattr(event, "wall_time", 0.0))
                old = by_step.get(step)
                if old is None or wall_time >= old[0]:
                    by_step[step] = (wall_time, scalar)

    if not by_step:
        raise ValueError(f"No scalar data found for tag '{tag}'.")

    steps = np.asarray(sorted(by_step.keys()), dtype=np.float64)
    values = np.asarray([by_step[int(step)][1] for step in steps], dtype=np.float64)
    return steps, values


def read_checkpoint_reward_series(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    nn_dir = run_dir / "nn"
    if not nn_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {nn_dir}")

    pattern = re.compile(r"_ep_(?P<epoch>\d+)_rew_(?P<reward>[-+]?\d+(?:\.\d+)?)(?:_|\.pth)")
    by_epoch: Dict[int, float] = {}
    for path in sorted(nn_dir.glob("*.pth")):
        match = pattern.search(path.name)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        reward = float(match.group("reward"))
        by_epoch[epoch] = reward

    if not by_epoch:
        raise ValueError(f"No checkpoint filenames with '_ep_*_rew_*' found under: {nn_dir}")

    epochs = np.asarray(sorted(by_epoch.keys()), dtype=np.float64)
    rewards = np.asarray([by_epoch[int(epoch)] for epoch in epochs], dtype=np.float64)
    return epochs, rewards


def ema(values: np.ndarray, smooth: float) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    smooth = float(smooth)
    if not 0.0 <= smooth < 1.0:
        raise ValueError("--smooth must be in [0, 1).")
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for idx in range(1, values.size):
        out[idx] = smooth * out[idx - 1] + (1.0 - smooth) * values[idx]
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    mean = np.convolve(padded, kernel, mode="valid")
    mean_sq = np.convolve(padded * padded, kernel, mode="valid")
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def save_curve_csv(path: Path, epochs: np.ndarray, raw: np.ndarray, smooth_values: np.ndarray, band: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "reward_raw", "reward_smooth", "residual_std"])
        for e, r, s, b in zip(epochs, raw, smooth_values, band):
            writer.writerow([f"{e:.0f}", f"{r:.8f}", f"{s:.8f}", f"{b:.8f}"])


def plot_reward_curve(
    epochs: np.ndarray,
    rewards: np.ndarray,
    smooth_values: np.ndarray,
    band: np.ndarray,
    out_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.1,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)

    lower = smooth_values - band
    upper = smooth_values + band
    ax.fill_between(epochs, lower, upper, color="#3b6ea8", alpha=0.18, linewidth=0.0, label="Residual band")
    ax.plot(epochs, rewards, color="#9fb7d7", alpha=0.28, linewidth=0.75, label="Raw reward")
    ax.plot(epochs, smooth_values, color="#12355b", linewidth=2.4, label="Smoothed reward")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.20)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#c9c9c9")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Training run directory.")
    parser.add_argument(
        "--source",
        choices=("events", "checkpoints"),
        default="events",
        help="Read rewards from TensorBoard events or checkpoint filenames.",
    )
    parser.add_argument("--tag", default=None, help="TensorBoard scalar tag. Defaults to rewards/iter if present.")
    parser.add_argument("--smooth", default=0.99, type=float, help="EMA smoothing coefficient.")
    parser.add_argument("--residual-window", default=101, type=int, help="Rolling window for residual shadow.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--output-name", default=None, help="PDF basename without extension.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if args.source == "checkpoints":
        tag = "checkpoint_filename_reward"
        epochs, rewards = read_checkpoint_reward_series(run_dir)
    else:
        event_files = find_event_files(run_dir)
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard event files found under: {run_dir}")

        if args.tag:
            tag = args.tag
        else:
            tag_counts = list_scalar_tags(event_files)
            tag = choose_reward_tag(tag_counts.keys(), args.tag)
        epochs, rewards = read_scalar_series(event_files, tag)
    smooth_values = ema(rewards, args.smooth)
    residual = rewards - smooth_values
    band = rolling_std(residual, args.residual_window)

    basename = args.output_name or f"{run_dir.name}_reward_convergence"
    out_pdf = args.out_dir / f"{basename}.pdf"
    out_csv = args.out_dir / f"{basename}.csv"

    plot_reward_curve(epochs, rewards, smooth_values, band, out_pdf)
    save_curve_csv(out_csv, epochs, rewards, smooth_values, band)

    print(f"tag={tag}")
    print(f"points={len(epochs)}")
    print(f"pdf={out_pdf}")
    print(f"csv={out_csv}")


if __name__ == "__main__":
    main()
