import argparse
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

URDF_BOX_SIZES = {
    "panel.urdf": (0.1, 1.2, 3.0),
    "cuboidal_rod.urdf": (0.1, 0.1, 2.0),
    "small_cube.urdf": (0.4, 0.4, 0.4),
}

URDF_BOX_COLORS = {
    "panel.urdf": "#2a6fdb",
    "cuboidal_rod.urdf": "#2aa84a",
    "small_cube.urdf": "#8a4be6",
}


def _draw_wireframe_box(ax, center, size, color="gray", linewidth=0.6, alpha=0.6):
    cx, cy, cz = center
    sx, sy, sz = (s * 0.5 for s in size)
    corners = np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start, end in edges:
        xs = (corners[start, 0], corners[end, 0])
        ys = (corners[start, 1], corners[end, 1])
        zs = (corners[start, 2], corners[end, 2])
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=alpha)


def _distance_to_axis_aligned_boxes(points, centers, size):
    half = torch.tensor(size, device=points.device, dtype=points.dtype) * 0.5
    deltas = torch.abs(points.unsqueeze(1) - centers.unsqueeze(0)) - half.view(1, 1, 3)
    deltas = torch.clamp(deltas, min=0.0)
    return torch.norm(deltas, dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute score map for navigation_task_gmm_noise.")
    parser.add_argument("--preset_id", type=int, default=None, help="Fixed env preset id (0/1/2).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--out_tag",
        type=str,
        default="",
        help="Optional suffix for output files (e.g., preset_0).",
    )
    parser.add_argument(
        "--use_adaptive",
        action="store_true",
        help="Use adaptive grid refinement instead of fixed grid.",
    )
    return parser.parse_args()


def compute_scores(
    points,
    goal,
    centers,
    sigmas,
    weights,
    obs_pos,
    r_obs_val,
    c_val,
    d_max,
    w1,
    w2,
    w3,
    device,
    obstacle_box_size=None,
):
    """Compute total scores for a batch of points."""
    # s_noise
    deltas = points.unsqueeze(1) - centers.unsqueeze(0)
    scaled = (deltas / sigmas.unsqueeze(0)).pow(2).sum(dim=-1)
    mixture = torch.exp(-0.5 * scaled)
    mixture = (mixture * weights.unsqueeze(0)).sum(dim=1)
    g_noise = mixture.pow(2)
    s_noise = torch.clamp(1.0 - g_noise, 0.0, 1.0)

    # s_obs
    if obstacle_box_size is not None:
        obs_dist = _distance_to_axis_aligned_boxes(points, obs_pos, obstacle_box_size)
        d_obs = obs_dist.min(dim=1).values
    else:
        obs_deltas = points.unsqueeze(1) - obs_pos.unsqueeze(0)
        obs_dist = torch.norm(obs_deltas, dim=-1)
        d_obs_surface = torch.clamp(obs_dist - r_obs_val, min=0.0)
        d_obs = d_obs_surface.min(dim=1).values
    valid_mask = d_obs > 0.0
    
    # Minimum deployment altitude filter - points below this height are invalid
    MIN_DEPLOYMENT_ALTITUDE = 2.0
    altitude_valid = points[:, 2] >= MIN_DEPLOYMENT_ALTITUDE
    valid_mask = valid_mask & altitude_valid
    
    s_obs = torch.clamp(1.0 - torch.exp(-c_val * d_obs.pow(2)), 0.0, 1.0)

    # s_dist
    s_dist = torch.clamp(1.0 - (torch.norm(points - goal, dim=1) / d_max), 0.0, 1.0)

    # total score
    total_score = w1 * s_dist + w2 * s_noise + w3 * s_obs
    neg_inf = torch.tensor(float("-inf"), device=device)
    total_score = torch.where(valid_mask, total_score, neg_inf)

    return total_score, s_dist, s_noise, s_obs, valid_mask


def generate_grid_points(bounds_min, bounds_max, step, device):
    """Generate a 3D grid of points."""
    xs = torch.arange(float(bounds_min[0]), float(bounds_max[0]) + 1e-6, step, device=device)
    ys = torch.arange(float(bounds_min[1]), float(bounds_max[1]) + 1e-6, step, device=device)
    zs = torch.arange(float(bounds_min[2]), float(bounds_max[2]) + 1e-6, step, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    return points


def generate_local_points(center, radius, step, bounds_min, bounds_max, device):
    """Generate points in a local region around a center point."""
    local_min = torch.clamp(center - radius, min=bounds_min)
    local_max = torch.clamp(center + radius, max=bounds_max)
    return generate_grid_points(local_min, local_max, step, device)


def adaptive_grid_search(
    goal,
    centers,
    sigmas,
    weights,
    obs_pos,
    r_obs_val,
    c_val,
    d_max,
    w1,
    w2,
    w3,
    bounds_min,
    bounds_max,
    adaptive_cfg,
    device,
    obstacle_box_size=None,
):
    """
    Three-stage adaptive grid refinement:
    1. Coarse grid search
    2. Local refinement around high-gradient regions
    3. Fine polish around top candidates
    """
    all_points = []
    all_scores = []

    # ========== Stage 1: Coarse Grid ==========
    print(f"[Stage 1] Coarse grid search (step={adaptive_cfg.coarse_step}m)...")
    coarse_points = generate_grid_points(bounds_min, bounds_max, adaptive_cfg.coarse_step, device)
    coarse_scores, _, _, _, coarse_valid = compute_scores(
        coarse_points,
        goal,
        centers,
        sigmas,
        weights,
        obs_pos,
        r_obs_val,
        c_val,
        d_max,
        w1,
        w2,
        w3,
        device,
        obstacle_box_size=obstacle_box_size,
    )
    all_points.append(coarse_points)
    all_scores.append(coarse_scores)
    print(f"  -> Sampled {coarse_points.shape[0]} points")

    # ========== Stage 2: Local Refinement ==========
    print(f"[Stage 2] Local refinement (step={adaptive_cfg.refine_step}m)...")
    valid_scores = coarse_scores[coarse_valid]
    valid_points = coarse_points[coarse_valid]

    if valid_scores.numel() > 0:
        # Find high-gradient regions (score differences between neighbors)
        # For simplicity, we use top percentile as refinement candidates
        threshold_score = torch.quantile(valid_scores, 1.0 - adaptive_cfg.top_percent_refine)
        refine_mask = valid_scores >= threshold_score
        refine_centers = valid_points[refine_mask]

        # Also add points with high local gradient (score varies significantly)
        # We approximate gradient by checking if point is near high-score and low-score regions
        score_range = valid_scores.max() - valid_scores.min()
        if score_range > 0:
            normalized_scores = (valid_scores - valid_scores.min()) / score_range
            # High gradient = points in middle range (not too high, not too low)
            gradient_mask = (normalized_scores > 0.3) & (normalized_scores < 0.8)
            gradient_centers = valid_points[gradient_mask]
            refine_centers = torch.cat([refine_centers, gradient_centers], dim=0)
            refine_centers = torch.unique(refine_centers, dim=0)

        print(f"  -> Found {refine_centers.shape[0]} refinement centers")

        # Generate refined points around each center
        refined_points_list = []
        for center in refine_centers:
            local_pts = generate_local_points(
                center, adaptive_cfg.refine_radius, adaptive_cfg.refine_step,
                bounds_min, bounds_max, device
            )
            refined_points_list.append(local_pts)

        if refined_points_list:
            refined_points = torch.cat(refined_points_list, dim=0)
            refined_points = torch.unique(refined_points, dim=0)
            refined_scores, _, _, _, _ = compute_scores(
                refined_points,
                goal,
                centers,
                sigmas,
                weights,
                obs_pos,
                r_obs_val,
                c_val,
                d_max,
                w1,
                w2,
                w3,
                device,
                obstacle_box_size=obstacle_box_size,
            )
            all_points.append(refined_points)
            all_scores.append(refined_scores)
            print(f"  -> Sampled {refined_points.shape[0]} refined points")

    # ========== Stage 3: Final Polish ==========
    print(f"[Stage 3] Final polish (step={adaptive_cfg.polish_step}m)...")
    combined_points = torch.cat(all_points, dim=0)
    combined_scores = torch.cat(all_scores, dim=0)

    # Find top N candidates
    valid_combined = combined_scores > float("-inf")
    valid_scores_combined = combined_scores[valid_combined]
    valid_points_combined = combined_points[valid_combined]

    if valid_scores_combined.numel() > 0:
        top_n = min(adaptive_cfg.polish_top_n, valid_scores_combined.shape[0])
        top_scores, top_indices = torch.topk(valid_scores_combined, top_n)
        top_centers = valid_points_combined[top_indices]

        print(f"  -> Polishing around top {top_n} candidates")

        polish_points_list = []
        for center in top_centers:
            local_pts = generate_local_points(
                center, adaptive_cfg.polish_radius, adaptive_cfg.polish_step,
                bounds_min, bounds_max, device
            )
            polish_points_list.append(local_pts)

        if polish_points_list:
            polish_points = torch.cat(polish_points_list, dim=0)
            polish_points = torch.unique(polish_points, dim=0)
            polish_scores, s_dist, s_noise, s_obs, valid_mask = compute_scores(
                polish_points,
                goal,
                centers,
                sigmas,
                weights,
                obs_pos,
                r_obs_val,
                c_val,
                d_max,
                w1,
                w2,
                w3,
                device,
                obstacle_box_size=obstacle_box_size,
            )
            all_points.append(polish_points)
            all_scores.append(polish_scores)
            print(f"  -> Sampled {polish_points.shape[0]} polished points")

    # Combine all results
    final_points = torch.cat(all_points, dim=0)
    final_scores = torch.cat(all_scores, dim=0)

    # Remove duplicates and keep best score for each unique point
    # For simplicity, just use unique points (last occurrence wins)
    final_points, inverse_indices = torch.unique(final_points, dim=0, return_inverse=True)

    # Recompute scores for final unique points
    final_scores, s_dist, s_noise, s_obs, valid_mask = compute_scores(
        final_points,
        goal,
        centers,
        sigmas,
        weights,
        obs_pos,
        r_obs_val,
        c_val,
        d_max,
        w1,
        w2,
        w3,
        device,
        obstacle_box_size=obstacle_box_size,
    )

    print(f"[Done] Total unique points: {final_points.shape[0]}")

    return final_points, final_scores, s_dist, s_noise, s_obs, valid_mask


def main(args=None):
    if args is None:
        args = parse_args()

    task_name = "navigation_task_gmm_noise"
    task_config = task_registry.get_task_config(task_name)
    if args.preset_id is not None:
        task_config.preset_id = int(args.preset_id)
    if args.seed is not None:
        task_config.seed = int(args.seed)

    task = task_registry.make_task(
        task_name,
        headless=True,
        num_envs=1,
        seed=args.seed,
    )
    task.reset()

    device = task.device
    score_cfg = task.task_config.score_config
    adaptive_cfg = getattr(task.task_config, "adaptive_grid_config", None)

    bounds_min = task.env_bounds_min
    bounds_max = task.env_bounds_max
    
    # Add wall margin to avoid selecting points at boundaries
    WALL_MARGIN = 0.5  # meters
    bounds_min = bounds_min + WALL_MARGIN
    bounds_max = bounds_max - WALL_MARGIN
    
    d_max = torch.norm(bounds_max - bounds_min)

    goal = task.target_position[0]
    centers = task.noise_centers[0]
    sigmas = task.noise_sigmas[0]
    weights = task.noise_weights[0]

    current_preset = None
    if args.preset_id is not None and hasattr(task_config, "fixed_env_presets"):
        current_preset = task_config.fixed_env_presets[int(args.preset_id)]

    if current_preset is not None and "obstacle_positions" in current_preset:
        obs_pos = torch.tensor(
            current_preset["obstacle_positions"], device=device, dtype=torch.float32
        )
        expected_obstacles = int(task.task_config.num_obstacles_in_env)
        if obs_pos.shape[0] != expected_obstacles:
            obs_pos = obs_pos[:expected_obstacles]
    else:
        obs_pos = task.obs_dict["obstacle_position"][0]

    in_bounds = ((obs_pos >= bounds_min) & (obs_pos <= bounds_max)).all(dim=1)
    obs_pos = obs_pos[in_bounds]
    if obs_pos.numel() == 0:
        raise ValueError("No obstacles found within bounds.")

    r_obs_val = float(current_preset.get("r_obs", score_cfg.r_obs)) if current_preset else float(score_cfg.r_obs)
    c_val = float(score_cfg.c)
    w1 = float(score_cfg.w1)
    w2 = float(score_cfg.w2)
    w3 = float(score_cfg.w3)
    obstacle_box_size = None
    if current_preset is not None:
        obstacle_urdf = current_preset.get("obstacle_urdf")
        if obstacle_urdf in URDF_BOX_SIZES:
            obstacle_box_size = URDF_BOX_SIZES[obstacle_urdf]

    # Choose search method
    if args.use_adaptive and adaptive_cfg is not None:
        print("=== Using Adaptive Grid Refinement ===")
        points, total_score, s_dist, s_noise, s_obs, valid_mask = adaptive_grid_search(
            goal,
            centers,
            sigmas,
            weights,
            obs_pos,
            r_obs_val,
            c_val,
            d_max,
            w1,
            w2,
            w3,
            bounds_min,
            bounds_max,
            adaptive_cfg,
            device,
            obstacle_box_size=obstacle_box_size,
        )
    else:
        print("=== Using Fixed Grid Search ===")
        grid_step = float(score_cfg.grid_step)
        points = generate_grid_points(bounds_min, bounds_max, grid_step, device)
        total_score, s_dist, s_noise, s_obs, valid_mask = compute_scores(
            points,
            goal,
            centers,
            sigmas,
            weights,
            obs_pos,
            r_obs_val,
            c_val,
            d_max,
            w1,
            w2,
            w3,
            device,
            obstacle_box_size=obstacle_box_size,
        )
        print(f"Sampled {points.shape[0]} points with step={grid_step}m")

    best_idx = torch.argmax(total_score)
    best_point = points[best_idx]
    best_score = total_score[best_idx]

    # Output paths
    output_dir = os.path.dirname(__file__)
    suffix = f"_{args.out_tag}" if args.out_tag else ""
    txt_path = os.path.join(output_dir, f"score_components_boxplot{suffix}.txt")
    png_path = os.path.join(output_dir, f"score_components_boxplot{suffix}.png")
    heatmap_path = os.path.join(output_dir, f"score_3d_heatmap{suffix}.png")
    best_point_path = os.path.join(output_dir, f"best_point{suffix}.txt")

    s_safe = s_obs
    w1_dist = torch.where(valid_mask, w1 * s_dist, torch.tensor(float("nan"), device=device))
    w2_noise = torch.where(valid_mask, w2 * s_noise, torch.tensor(float("nan"), device=device))
    w3_safe = torch.where(valid_mask, w3 * s_safe, torch.tensor(float("nan"), device=device))

    # Save all points with scores
    data = torch.cat(
        [points, total_score.unsqueeze(1), w1_dist.unsqueeze(1), w2_noise.unsqueeze(1), w3_safe.unsqueeze(1)], dim=1
    ).detach().cpu().numpy()
    header = (
        "x y z total_score w1S_dist w2S_noise w3S_safe\n"
        f"bounds_min={bounds_min.detach().cpu().tolist()}\n"
        f"bounds_max={bounds_max.detach().cpu().tolist()}\n"
        f"adaptive_search={args.use_adaptive}\n"
        f"w1={score_cfg.w1} w2={score_cfg.w2} w3={score_cfg.w3} c={score_cfg.c} r_obs={r_obs_val}\n"
        f"d_max={float(d_max.detach().cpu())} total_points={points.shape[0]}\n"
        f"preset_id={getattr(task.task_config, 'preset_id', None)} seed={args.seed}\n"
        f"best_point={best_point.detach().cpu().tolist()} best_score={float(best_score.detach().cpu())}"
    )
    np.savetxt(txt_path, data, fmt="%.6f", header=header)

    # Save best point separately
    with open(best_point_path, "w", encoding="utf-8") as f:
        f.write(f"preset_id={args.preset_id}\n")
        f.write(f"best_score={float(best_score.detach().cpu()):.6f}\n")
        f.write(f"best_position={best_point.detach().cpu().tolist()}\n")
        f.write(f"total_points_searched={points.shape[0]}\n")
        f.write(f"adaptive_search={args.use_adaptive}\n")
        f.write(f"r_obs={r_obs_val}\n")
        f.write(f"w1={w1} w2={w2} w3={w3} c={c_val}\n")

    # Box plot
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

    # 3D heatmap
    points_cpu = points[valid_mask].detach().cpu().numpy()
    scores_cpu = total_score[valid_mask].detach().cpu().numpy()
    goal_cpu = goal.detach().cpu().numpy()
    best_cpu = best_point.detach().cpu().numpy()
    obs_pos_cpu = obs_pos.detach().cpu().numpy()
    noise_centers_cpu = centers.detach().cpu().numpy()
    bounds_min_cpu = bounds_min.detach().cpu().numpy()
    bounds_max_cpu = bounds_max.detach().cpu().numpy()

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

    obstacle_urdf = current_preset.get("obstacle_urdf") if current_preset else None
    box_size = URDF_BOX_SIZES.get(obstacle_urdf)
    if box_size is not None:
        wire_color = URDF_BOX_COLORS.get(obstacle_urdf, "gray")
        for center in obs_pos_cpu:
            _draw_wireframe_box(ax, center, box_size, color=wire_color, linewidth=1.2, alpha=0.85)
    else:
        sphere_u = np.linspace(0.0, 2.0 * np.pi, 18)
        sphere_v = np.linspace(0.0, np.pi, 12)
        sphere_x = np.outer(np.cos(sphere_u), np.sin(sphere_v))
        sphere_y = np.outer(np.sin(sphere_u), np.sin(sphere_v))
        sphere_z = np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
        for center in obs_pos_cpu:
            ax.plot_surface(
                r_obs_val * sphere_x + center[0],
                r_obs_val * sphere_y + center[1],
                r_obs_val * sphere_z + center[2],
                color="gray",
                alpha=0.25,
                linewidth=0,
                antialiased=False,
            )

    ax.scatter([goal_cpu[0]], [goal_cpu[1]], [goal_cpu[2]], c="red", s=120, marker="*", label="Goal")
    ax.scatter([best_cpu[0]], [best_cpu[1]], [best_cpu[2]], c="black", s=80, marker="X", label="Best")
    ax.scatter(noise_centers_cpu[:, 0], noise_centers_cpu[:, 1], noise_centers_cpu[:, 2],
               c="orange", s=60, marker="^", label="Noise Source")

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

    print(f"\n=== Results ===")
    print(f"Best point: {best_point.detach().cpu().tolist()}")
    print(f"Best score: {float(best_score.detach().cpu()):.6f}")
    print(f"Total points searched: {points.shape[0]}")
    print(f"Saved best point: {best_point_path}")
    print(f"Saved all scores: {txt_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved 3D heatmap PNG: {heatmap_path}")


if __name__ == "__main__":
    main()
