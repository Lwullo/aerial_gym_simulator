import os

import isaacgym  # must be imported before torch
import torch
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Registers 3D projection.

import aerial_gym.task  # registers tasks
from aerial_gym.registry.task_registry import task_registry


def main():
    task = task_registry.make_task(
        "navigation_task_gmm_noise",
        headless=True,
        num_envs=1,
    )
    task.reset()

    device = task.device
    score_cfg = task.task_config.score_config

    bounds_min = task.env_bounds_min
    bounds_max = task.env_bounds_max

    grid_step = float(score_cfg.grid_step)
    xs = torch.arange(
        float(bounds_min[0].item()),
        float(bounds_max[0].item()) + 1e-6,
        grid_step,
        device=device,
    )
    ys = torch.arange(
        float(bounds_min[1].item()),
        float(bounds_max[1].item()) + 1e-6,
        grid_step,
        device=device,
    )
    zs = torch.arange(
        float(bounds_min[2].item()),
        float(bounds_max[2].item()) + 1e-6,
        grid_step,
        device=device,
    )

    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    goal = task.target_position[0]

    centers = task.noise_centers[0]
    sigmas = task.noise_sigmas[0]
    weights = task.noise_weights[0]

    deltas = points.unsqueeze(1) - centers.unsqueeze(0)
    scaled = (deltas / sigmas.unsqueeze(0)).pow(2).sum(dim=-1)
    mixture = torch.exp(-0.5 * scaled)
    mixture = (mixture * weights.unsqueeze(0)).sum(dim=1)

    # 计算全局噪声，将输入的mixture进行平方运算
    g_noise = mixture.pow(2)
    # 计算语音信号噪声，通过1减去全局噪声得到
    s_noise = 1.0 - g_noise
    # 对语音信号噪声进行裁剪，确保其值在[0.0, 1.0]范围内
    s_noise = torch.clamp(s_noise, 0.0, 1.0)

    obs_pos = task.obs_dict["obstacle_position"][0]
    in_bounds = ((obs_pos >= bounds_min) & (obs_pos <= bounds_max)).all(dim=1)
    obs_pos = obs_pos[in_bounds]
    if obs_pos.numel() == 0:
        raise ValueError("No obstacles found within bounds.")

    obs_deltas = points.unsqueeze(1) - obs_pos.unsqueeze(0)
    obs_dist = torch.norm(obs_deltas, dim=-1)
    d_obs_surface = torch.clamp(obs_dist - float(score_cfg.r_obs), min=0.0)
    d_obs = d_obs_surface.min(dim=1).values

    valid_mask = d_obs > 0.0

# 计算观测值的显著性分数，使用高斯函数公式进行计算
# score_cfg.c 是高斯函数的系数，d_obs 是观测值的距离
    s_obs = 1.0 - torch.exp(-float(score_cfg.c) * d_obs.pow(2))
# 将计算得到的显著性分数限制在 [0.0, 1.0] 范围内
# torch.clamp 函数确保值不会超过指定范围
    s_obs = torch.clamp(s_obs, 0.0, 1.0)

    d_max = torch.norm(bounds_max - bounds_min) #  计算边界框的最大距离
    s_dist = 1.0 - (torch.norm(points - goal, dim=1) / d_max) #  计算每个点到目标点的距离，并归一化到[0,1]范围 使用L2范数计算距离，并在dim=1维度上进行计算
    s_dist = torch.clamp(s_dist, 0.0, 1.0) #  将距离值限制在[0,1]范围内，确保值不会超出此区间

    total_score = (
        float(score_cfg.w1) * s_dist
        + float(score_cfg.w2) * s_noise
        + float(score_cfg.w3) * s_obs
    )
    neg_inf = torch.tensor(float("-inf"), device=device)
    total_score = torch.where(valid_mask, total_score, neg_inf)

    best_idx = torch.argmax(total_score)
    best_point = points[best_idx]
    best_score = total_score[best_idx]

    output_dir = os.path.dirname(__file__)
    txt_path = os.path.join(output_dir, "score_components_boxplot.txt")
    png_path = os.path.join(output_dir, "score_components_boxplot.png")
    heatmap_path = os.path.join(output_dir, "score_3d_heatmap.png")

    s_safe = s_obs
    w1 = float(score_cfg.w1)
    w2 = float(score_cfg.w2)
    w3 = float(score_cfg.w3)

    w1_dist = torch.where(valid_mask, w1 * s_dist, torch.tensor(float("nan"), device=device))
    w2_noise = torch.where(valid_mask, w2 * s_noise, torch.tensor(float("nan"), device=device))
    w3_safe = torch.where(valid_mask, w3 * s_safe, torch.tensor(float("nan"), device=device))

    data = torch.cat(
        [points, w1_dist.unsqueeze(1), w2_noise.unsqueeze(1), w3_safe.unsqueeze(1)], dim=1
    ).detach().cpu().numpy()
    header = (
        "x y z w1S_dist w2S_noise w3S_safe\n"
        f"bounds_min={bounds_min.detach().cpu().tolist()}\n"
        f"bounds_max={bounds_max.detach().cpu().tolist()}\n"
        f"grid_step={grid_step}\n"
        f"w1={score_cfg.w1} w2={score_cfg.w2} w3={score_cfg.w3} c={score_cfg.c} r_obs={score_cfg.r_obs}\n"
        f"d_max={float(d_max.detach().cpu())} G_max=1.0\n"
        f"best_point={best_point.detach().cpu().tolist()} best_score={float(best_score.detach().cpu())}"
    )
    np.savetxt(txt_path, data, fmt="%.6f", header=header)

    dist_vals = w1_dist[valid_mask].detach().cpu().numpy()
    noise_vals = w2_noise[valid_mask].detach().cpu().numpy()
    safe_vals = w3_safe[valid_mask].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [dist_vals, noise_vals, safe_vals],
        labels=["w1*S_dist", "w2*S_noise", "w3*S_safe"],
        showfliers=False,
    )
    ax.set_ylabel("Score")
    ax.set_title("Weighted Score Components (Box Plot)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    points_cpu = points[valid_mask].detach().cpu().numpy()
    scores_cpu = total_score[valid_mask].detach().cpu().numpy()
    goal_cpu = goal.detach().cpu().numpy()
    best_cpu = best_point.detach().cpu().numpy()
    obs_pos_cpu = obs_pos.detach().cpu().numpy()
    bounds_min_cpu = bounds_min.detach().cpu().numpy()
    bounds_max_cpu = bounds_max.detach().cpu().numpy()
    r_obs = float(score_cfg.r_obs)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        points_cpu[:, 0],
        points_cpu[:, 1],
        points_cpu[:, 2],
        c=scores_cpu,
        s=4,
        cmap="viridis",
        alpha=0.6,
        linewidth=0,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Total Score")

    sphere_u = np.linspace(0.0, 2.0 * np.pi, 18)
    sphere_v = np.linspace(0.0, np.pi, 12)
    sphere_x = np.outer(np.cos(sphere_u), np.sin(sphere_v))
    sphere_y = np.outer(np.sin(sphere_u), np.sin(sphere_v))
    sphere_z = np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    for center in obs_pos_cpu:
        ax.plot_surface(
            r_obs * sphere_x + center[0],
            r_obs * sphere_y + center[1],
            r_obs * sphere_z + center[2],
            color="gray",
            alpha=0.25,
            linewidth=0,
            antialiased=False,
        )

    ax.scatter(
        [goal_cpu[0]],
        [goal_cpu[1]],
        [goal_cpu[2]],
        c="red",
        s=120,
        marker="*",
        label="Goal",
    )
    ax.scatter(
        [best_cpu[0]],
        [best_cpu[1]],
        [best_cpu[2]],
        c="black",
        s=80,
        marker="X",
        label="Best",
    )

    ax.set_xlim(bounds_min_cpu[0], bounds_max_cpu[0])
    ax.set_ylim(bounds_min_cpu[1], bounds_max_cpu[1])
    ax.set_zlim(bounds_min_cpu[2], bounds_max_cpu[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Score Heatmap")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    print(f"Best point: {best_point.detach().cpu().tolist()} score={float(best_score.detach().cpu())}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved 3D heatmap PNG: {heatmap_path}")
    print(f"Saved TXT: {txt_path}")


if __name__ == "__main__":
    main()
