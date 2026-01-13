#!/usr/bin/env python3
"""
Score Sensitivity Analysis: Visualize score components along X/Y/Z axes through optimal point.
Compares theoretical optimal point (compute_score_map) with trained model's best point (runner).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Preset configurations (same as in navigation_task_gmm_noise_config.py)
FIXED_ENV_PRESETS = {
    0: {
        "target_position": [8.50, 8.50, 4.00],
        "noise_centers": [[6.00, 5.50, 2.50], [4.00, 8.00, 3.00]],
        "noise_sigmas": [[1.80, 1.80, 1.50], [1.50, 1.50, 1.20]],
        "noise_weights": [0.70, 0.55],
        "obstacle_positions": [[5.5, 5.5, 2.25], [3.0, 7.5, 2.25]],
        "obstacle_sizes": [[0.6, 0.6, 4.5], [0.6, 0.6, 4.5]],  # Box half-sizes
    },
    1: {
        "target_position": [9.00, 5.00, 4.00],
        "noise_centers": [[2.00, 5.00, 2.50], [5.50, 7.50, 3.00], [7.00, 3.00, 2.00]],
        "noise_sigmas": [[2.00, 2.00, 1.50], [1.50, 1.50, 1.20], [1.80, 1.80, 1.50]],
        "noise_weights": [0.65, 0.50, 0.60],
        "obstacle_positions": [[2.5, 5.0, 2.25], [5.5, 7.0, 2.25], [7.0, 3.0, 2.25]],
        "obstacle_sizes": [[0.6, 0.6, 4.5], [0.6, 0.6, 4.5], [0.6, 0.6, 4.5]],
    },
    2: {
        "target_position": [8.00, 2.00, 3.50],
        "noise_centers": [[3.00, 3.00, 2.50], [6.00, 6.00, 3.00], [4.00, 8.00, 2.00], [8.00, 4.00, 3.50]],
        "noise_sigmas": [[1.50, 1.50, 1.20], [2.00, 2.00, 1.50], [1.80, 1.80, 1.50], [1.50, 1.50, 1.20]],
        "noise_weights": [0.55, 0.65, 0.50, 0.60],
        "obstacle_positions": [[3.0, 3.5, 2.25], [6.0, 6.5, 2.25], [4.5, 8.0, 2.25], [8.0, 4.5, 2.25]],
        "obstacle_sizes": [[0.6, 0.6, 4.5], [0.6, 0.6, 4.5], [0.6, 0.6, 4.5], [0.6, 0.6, 4.5]],
    },
}

# Score configuration (from navigation_task_gmm_noise_config.py)
SCORE_CONFIG = {
    "w1": 0.6,  # s_dist weight
    "w2": 0.7,  # s_noise weight
    "w3": 1.0,  # s_obs weight
    "c": 1.5,   # obstacle safety parameter
}

# Environment bounds
ENV_BOUNDS_MIN = np.array([0.0, 0.0, 0.0])
ENV_BOUNDS_MAX = np.array([10.0, 10.0, 4.5])
WALL_MARGIN = 0.5


def compute_s_dist(point, goal, d_max):
    """Compute distance score s_dist."""
    dist = np.linalg.norm(point - goal)
    s_dist = 1.0 - dist / d_max
    return np.clip(s_dist, 0.0, 1.0)


def compute_s_noise(point, noise_centers, noise_sigmas, noise_weights):
    """Compute noise score s_noise."""
    mixture = 0.0
    for center, sigma, weight in zip(noise_centers, noise_sigmas, noise_weights):
        delta = point - np.array(center)
        sigma = np.array(sigma)
        scaled = np.sum((delta / sigma) ** 2)
        mixture += weight * np.exp(-0.5 * scaled)
    g_noise = mixture ** 2
    s_noise = 1.0 - g_noise
    return np.clip(s_noise, 0.0, 1.0)


def compute_box_distance(point, center, half_size):
    """Compute distance from point to box surface."""
    delta = np.abs(point - center) - half_size
    delta = np.maximum(delta, 0)
    return np.linalg.norm(delta)


def compute_s_obs(point, obstacle_positions, obstacle_sizes, c):
    """Compute obstacle safety score s_obs."""
    min_dist = float('inf')
    for pos, size in zip(obstacle_positions, obstacle_sizes):
        d = compute_box_distance(point, np.array(pos), np.array(size))
        min_dist = min(min_dist, d)
    s_obs = 1.0 - np.exp(-c * min_dist ** 2)
    return np.clip(s_obs, 0.0, 1.0), min_dist


def compute_total_score(point, preset, score_cfg):
    """Compute total score and all components."""
    goal = np.array(preset["target_position"])
    d_max = np.linalg.norm(ENV_BOUNDS_MAX - ENV_BOUNDS_MIN)
    
    s_dist = compute_s_dist(point, goal, d_max)
    s_noise = compute_s_noise(
        point, 
        preset["noise_centers"], 
        preset["noise_sigmas"], 
        preset["noise_weights"]
    )
    s_obs, _ = compute_s_obs(
        point, 
        preset["obstacle_positions"], 
        preset["obstacle_sizes"],
        score_cfg["c"]
    )
    
    total = score_cfg["w1"] * s_dist + score_cfg["w2"] * s_noise + score_cfg["w3"] * s_obs
    
    return {
        "total": total,
        "w1_s_dist": score_cfg["w1"] * s_dist,
        "w2_s_noise": score_cfg["w2"] * s_noise,
        "w3_s_obs": score_cfg["w3"] * s_obs,
    }


def scan_along_axis(axis_idx, fixed_coords, preset, score_cfg, num_points=100):
    """Scan along one axis and compute scores."""
    axis_min = ENV_BOUNDS_MIN[axis_idx] + WALL_MARGIN
    axis_max = ENV_BOUNDS_MAX[axis_idx] - WALL_MARGIN
    
    axis_values = np.linspace(axis_min, axis_max, num_points)
    results = {"axis": axis_values, "total": [], "w1_s_dist": [], "w2_s_noise": [], "w3_s_obs": []}
    
    for val in axis_values:
        point = fixed_coords.copy()
        point[axis_idx] = val
        scores = compute_total_score(point, preset, score_cfg)
        for key in ["total", "w1_s_dist", "w2_s_noise", "w3_s_obs"]:
            results[key].append(scores[key])
    
    return results


def get_obstacle_ranges(obstacle_positions, obstacle_sizes, axis_idx, fixed_coords):
    """Get obstacle ranges along an axis that intersect with fixed coordinates."""
    ranges = []
    for pos, size in zip(obstacle_positions, obstacle_sizes):
        pos = np.array(pos)
        size = np.array(size)
        
        # Check if fixed coordinates are within obstacle bounds on other axes
        other_axes = [i for i in range(3) if i != axis_idx]
        in_range = True
        for ax in other_axes:
            if not (pos[ax] - size[ax] <= fixed_coords[ax] <= pos[ax] + size[ax]):
                in_range = False
                break
        
        if in_range:
            ranges.append((pos[axis_idx] - size[axis_idx], pos[axis_idx] + size[axis_idx]))
    
    return ranges


def plot_sensitivity(preset_id, optimal_point, runner_point=None, output_path=None):
    """Generate sensitivity analysis plot."""
    preset = FIXED_ENV_PRESETS[preset_id]
    optimal_point = np.array(optimal_point)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axis_names = ['x', 'y', 'z']
    
    for ax_idx, ax in enumerate(axes):
        # Fixed coordinates (use optimal point for other axes)
        fixed_coords = optimal_point.copy()
        
        # Scan along this axis
        results = scan_along_axis(ax_idx, fixed_coords, preset, SCORE_CONFIG)
        
        # Plot score components
        ax.plot(results["axis"], results["total"], 'k-', linewidth=2, label='Total Score')
        ax.plot(results["axis"], results["w3_s_obs"], 'b--', linewidth=1.5, label='w3*s_obs (Safety)')
        ax.plot(results["axis"], results["w1_s_dist"], color='orange', linestyle='--', linewidth=1.5, label='w1*s_dist (Distance)')
        ax.plot(results["axis"], results["w2_s_noise"], 'g--', linewidth=1.5, label='w2*s_noise (Noise)')
        
        # Mark optimal point
        ax.axvline(x=optimal_point[ax_idx], color='red', linestyle=':', linewidth=2, label=f'Optimal {axis_names[ax_idx]}')
        
        # Mark runner point if provided
        if runner_point is not None:
            runner_point = np.array(runner_point)
            ax.axvline(x=runner_point[ax_idx], color='blue', linestyle='-.', linewidth=2, label=f'Runner {axis_names[ax_idx]}')
            # Also mark the runner point score
            runner_scores = compute_total_score(runner_point, preset, SCORE_CONFIG)
            ax.scatter([runner_point[ax_idx]], [runner_scores["total"]], 
                      color='blue', s=100, marker='*', zorder=5, label='Runner Best')
        
        # Add obstacle shading
        obs_ranges = get_obstacle_ranges(
            preset["obstacle_positions"], 
            preset["obstacle_sizes"], 
            ax_idx, 
            fixed_coords
        )
        for r_min, r_max in obs_ranges:
            ax.axvspan(r_min, r_max, alpha=0.2, color='gray', label='Obstacle' if obs_ranges.index((r_min, r_max)) == 0 else None)
        
        # Labels
        other_axes = [i for i in range(3) if i != ax_idx]
        title_suffix = f"({axis_names[other_axes[0]]}={fixed_coords[other_axes[0]]:.1f}, {axis_names[other_axes[1]]}={fixed_coords[other_axes[1]]:.1f})"
        ax.set_title(f'Scan along {axis_names[ax_idx]}-axis\n{title_suffix}')
        ax.set_xlabel(f'{axis_names[ax_idx]} coordinate')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 3.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    
    # Main title
    opt_str = f"x={optimal_point[0]:.1f}, y={optimal_point[1]:.1f}, z={optimal_point[2]:.1f}"
    fig.suptitle(f'Score Scan Through Optimal Point ({opt_str}) - Preset {preset_id}', fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Score Sensitivity Analysis")
    parser.add_argument("--preset_id", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--optimal_point", type=str, default="6.5,7.5,3.5", 
                        help="Optimal point as 'x,y,z'")
    parser.add_argument("--runner_point", type=str, default=None,
                        help="Runner's best point as 'x,y,z' (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: show plot)")
    args = parser.parse_args()
    
    optimal = [float(x) for x in args.optimal_point.split(",")]
    runner = None
    if args.runner_point:
        runner = [float(x) for x in args.runner_point.split(",")]
    
    plot_sensitivity(args.preset_id, optimal, runner, args.output)


if __name__ == "__main__":
    main()
