import os

import isaacgym  # must be imported before torch
import torch

import aerial_gym.task  # registers tasks
from aerial_gym.registry.task_registry import task_registry


def _fmt_vec(vec):
    vals = [float(v) for v in vec]
    return "(" + ", ".join(f"{v:.4f}" for v in vals) + ")"


def _clamp_point(point, min_bounds, max_bounds):
    return torch.max(torch.min(point, max_bounds), min_bounds)


def _compute_mixture(point, centers, sigmas, weights):
    deltas = point.unsqueeze(0) - centers
    scaled = (deltas / sigmas).pow(2).sum(dim=-1)
    mixture = torch.exp(-0.5 * scaled)
    return (weights * mixture).sum()


def _sample_noise(point, centers, sigmas, weights, noise_scale):
    mixture = _compute_mixture(point, centers, sigmas, weights)
    noise_vec = torch.randn_like(point) * mixture * noise_scale
    return mixture, noise_vec


def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    task = task_registry.make_task(
        "navigation_task_gmm_noise",
        headless=True,
        num_envs=1,
    )
    task.reset()

    output_path = os.path.join(os.path.dirname(__file__), "gmm_noise_dump.txt")
    lines = []

    env_bounds_min = task.env_bounds_min.detach().cpu()
    env_bounds_max = task.env_bounds_max.detach().cpu()

    lines.append(f"env_bounds_min: {_fmt_vec(env_bounds_min)}")
    lines.append(f"env_bounds_max: {_fmt_vec(env_bounds_max)}")

    centers = task.noise_centers[0]
    sigmas = task.noise_sigmas[0]
    weights = task.noise_weights[0]

    lines.append(f"num_sources: {centers.shape[0]}")
    for idx in range(centers.shape[0]):
        lines.append(f"source[{idx}] center: {_fmt_vec(centers[idx].detach().cpu())}")

    source_idx = 0
    center = centers[source_idx]
    lines.append(f"sample_source_index: {source_idx}")
    lines.append(f"sample_source_center: {_fmt_vec(center.detach().cpu())}")

    distances = [0.0, 1.0, 5.0, 10.0]
    axes = ["x", "y", "z"]
    axis_vecs = [
        torch.tensor([1.0, 0.0, 0.0], device=center.device),
        torch.tensor([0.0, 1.0, 0.0], device=center.device),
        torch.tensor([0.0, 0.0, 1.0], device=center.device),
    ]

    for dist in distances:
        for axis, axis_vec in zip(axes, axis_vecs):
            raw_point = center + axis_vec * dist
            sample_point = _clamp_point(raw_point, task.env_bounds_min, task.env_bounds_max)
            mixture, noise_vec = _sample_noise(
                sample_point, centers, sigmas, weights, task.noise_config.noise_scale
            )
            clamped = bool((sample_point != raw_point).any().item())
            lines.append(
                "dist={:.1f} axis={} raw={} sample={} clamped={} mixture={:.6f} noise={}".format(
                    dist,
                    axis,
                    _fmt_vec(raw_point.detach().cpu()),
                    _fmt_vec(sample_point.detach().cpu()),
                    clamped,
                    float(mixture.detach().cpu()),
                    _fmt_vec(noise_vec.detach().cpu()),
                )
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"Saved to: {output_path}")

    task.close()


if __name__ == "__main__":
    main()
