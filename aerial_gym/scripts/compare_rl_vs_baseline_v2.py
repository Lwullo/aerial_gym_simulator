"""
Comparative Experiment: RL (PPO) vs PID + Greedy Gradient Descent Baseline

This script conducts a rigorous comparative evaluation using 100 parallel environments:
1. Phase 1: Evaluates trained PPO model for 600 steps
2. Phase 2: Evaluates PID+Greedy baseline from identical initial states
3. Aggregates J(p) curves and generates academic-quality plots

Author: Antigravity AI
Date: 2026-01-19
"""

import os
import sys
import numpy as np
import csv

# Isaac Gym must be imported before torch (already handled)
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.config.task_config.navigation_task_gmm_noise_config import task_config
from aerial_gym.config.asset_config.env_object_config import object_asset_params, panel_asset_params
from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import NavigationTaskGmmNoise

import torch
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib import rcParams

logger = CustomLogger("compare_rl_vs_baseline")

# ============================================================================
# CRITICAL: Enable Evaluation Mode
# This activates deterministic physics and crash flag clearing in env_manager.py
# WITHOUT affecting training behavior (training uses original random physics)
# ============================================================================
import os
os.environ["AERIAL_GYM_EVAL_MODE"] = "1"

# ============================================================================
# Configuration
# ============================================================================

# Model path
# Model path
PPO_CHECKPOINT = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gmm_noise_run_30-01-27-14/nn/last_gmm_noise_run_ep_9000_rew__5.073401_.pth"

# Experiment parameters
NUM_ENVS = 100
EPISODE_LENGTH = 500
RANDOM_SEED = 10

# J(p) parameters (from config)
# J(p) parameters (from config)
W_D = 1.0
W_N = 1.0
D_0 = 10.0 # task_config.reward_parameters["potential_d0"]

# EMA smoothing
EMA_ALPHA = 0.99
RAD2DEG = 180.0 / np.pi

# Output paths
OUTPUT_DIR = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Running Mean/Std Normalization
# ============================================================================

class RunningMeanStd:
    """Tracks running mean and std for observation normalization"""
    def __init__(self, mean, std, count, device='cuda:0'):
        self.mean = torch.tensor(mean, device=device, dtype=torch.float32)
        self.std = torch.tensor(std, device=device, dtype=torch.float32)
        self.count = count
        self.device = device
    
    def normalize(self, obs, epsilon=1e-8):
        """Normalize observations using running mean/std"""
        return (obs - self.mean) / (self.std + epsilon)


def compute_normalization_stats(env, num_samples=2000):
    """
    Compute normalization statistics by sampling random actions in the environment.
    
    Args:
        env: Environment instance
        num_samples: Number of steps to collect observations
    
    Returns:
        RunningMeanStd: Normalization statistics
    """
    logger.info(f"Computing normalization statistics from {num_samples} environment steps...")
    
    obs_list = []
    
    env.reset()
    for i in range(num_samples):
        # Random actions
        random_actions = torch.rand(env.sim_env.num_envs, env.task_config.action_space_dim, device=env.device) * 2 - 1
        
        # Step environment
        env.step(random_actions)
        
        # Collect observation
        obs = env.task_obs["observations"].clone()
        obs_list.append(obs)
    
    # Compute statistics
    all_obs = torch.cat(obs_list, dim=0)
    mean = all_obs.mean(dim=0)
    std = all_obs.std(dim=0)
    count = all_obs.shape[0]
    
    logger.info(f"Computed normalization stats: mean={mean[:3].cpu().numpy()}, std={std[:3].cpu().numpy()}")
    
    return RunningMeanStd(
        mean=mean.cpu().numpy(),
        std=std.cpu().numpy(),
        count=count,
        device=env.device
    )


# ============================================================================
# Utility Functions
# ============================================================================

def compute_J_potential(env):
    """
    Compute unified potential J(p) = w_d * (d/d0)^2 + w_n * n_hat
    
    Returns:
        J_values: (num_envs,) tensor of potential values
    """
    position = env.obs_dict["robot_position"]
    
    # Distance term
    dist_to_target = torch.norm(position - env.target_position, dim=1)
    J_dist = W_D * (dist_to_target / D_0)
    
    # Noise term (normalized)
    current_noise = env._compute_gmm_mixture(position)
    if hasattr(env, "estimated_n_min") and hasattr(env, "estimated_n_max"):
        # [EVALUATION ONLY] Normalize noise to 0-1 for standard metric reporting
        # Training uses RAW noise (0-5.0) for gradients, but verification uses normalized J.
        n_hat = (current_noise - env.estimated_n_min) / (env.estimated_n_max - env.estimated_n_min + 1e-6)
        n_hat = torch.clamp(n_hat, 0.0, 1.0)
    else:
        n_hat = current_noise

    
    J_noise = W_N * n_hat
    
    # Total potential
    J_total = J_dist + J_noise
    
    return J_total


def save_state_snapshot(env):
    """Save complete environment state for restoration"""
    snapshot = {
        'robot_position': env.obs_dict['robot_position'].clone(),
        'robot_orientation': env.obs_dict['robot_orientation'].clone(),
        'robot_linvel': env.obs_dict['robot_linvel'].clone(),
        'robot_angvel': env.obs_dict['robot_angvel'].clone(),
        'target_position': env.target_position.clone(),
        'noise_centers': env.noise_centers.clone(),
        'noise_sigmas': env.noise_sigmas.clone(),
        'noise_weights': env.noise_weights.clone(),
        'obstacle_position': env.obs_dict['obstacle_position'].clone(),
        'obstacle_orientation': env.obs_dict['obstacle_orientation'].clone(),
    }
    
    # Save estimated noise range if available
    if hasattr(env, 'estimated_n_min'):
        snapshot['estimated_n_min'] = env.estimated_n_min.clone()
        snapshot['estimated_n_max'] = env.estimated_n_max.clone()
    
    return snapshot


def restore_state_snapshot(env, snapshot):
    """Restore environment to saved state"""
    env.obs_dict['robot_position'].copy_(snapshot['robot_position'])
    env.obs_dict['robot_orientation'].copy_(snapshot['robot_orientation'])
    env.obs_dict['robot_linvel'].copy_(snapshot['robot_linvel'])
    env.obs_dict['robot_angvel'].copy_(snapshot['robot_angvel'])
    env.target_position.copy_(snapshot['target_position'])
    env.noise_centers.copy_(snapshot['noise_centers'])
    env.noise_sigmas.copy_(snapshot['noise_sigmas'])
    env.noise_weights.copy_(snapshot['noise_weights'])
    env.obs_dict['obstacle_position'].copy_(snapshot['obstacle_position'])
    env.obs_dict['obstacle_orientation'].copy_(snapshot['obstacle_orientation'])
    
    if 'estimated_n_min' in snapshot:
        env.estimated_n_min.copy_(snapshot['estimated_n_min'])
        env.estimated_n_max.copy_(snapshot['estimated_n_max'])
    
    # Write state to Isaac Gym simulation
    env.sim_env.IGE_env.write_to_sim()


def spawn_target_area_obstacles(env):
    """
    Spawn one random cube obstacle (0.5m × 0.5m × 0.5m) within 3m of the target
    for each environment. This creates additional challenge near the goal.
    
    VERIFICATION SCRIPT ONLY - Not used during training.
    """
    logger.info("Spawning random obstacles near target positions...")
    
    num_envs = env.sim_env.num_envs
    num_obstacles = env.obs_dict['obstacle_position'].shape[1]
    
    if num_obstacles == 0:
        logger.warning("No obstacles in environment - cannot add target area obstacle")
        return
    
    # Use the last obstacle slot for the target-area obstacle
    target_obstacle_idx = num_obstacles - 1
    
    for env_id in range(num_envs):
        # Get target position for this environment
        target_pos = env.target_position[env_id].cpu().numpy()
        
        # Generate random position within 3m sphere around target
        # Using spherical coordinates for uniform distribution
        radius = torch.rand(1).item() * 1.0  # 0 to 1.0m (UPDATED: 3.0 -> 1.0)
        theta = torch.rand(1).item() * 2 * torch.pi  # azimuth angle
        phi = torch.acos(2 * torch.rand(1) - 1).item()  # polar angle (uniform on sphere)
        
        # Convert to Cartesian offset
        offset_x = radius * torch.sin(torch.tensor(phi)).item() * torch.cos(torch.tensor(theta)).item()
        offset_y = radius * torch.sin(torch.tensor(phi)).item() * torch.sin(torch.tensor(theta)).item()
        offset_z = radius * torch.cos(torch.tensor(phi)).item()
        
        # Calculate obstacle position
        obs_x = target_pos[0] + offset_x
        obs_y = target_pos[1] + offset_y
        obs_z = target_pos[2] + offset_z
        
        # Clamp to environment bounds
        obs_x = max(env.env_bounds_min[0].item(), min(obs_x, env.env_bounds_max[0].item()))
        obs_y = max(env.env_bounds_min[1].item(), min(obs_y, env.env_bounds_max[1].item()))
        obs_z = max(env.env_bounds_min[2].item(), min(obs_z, env.env_bounds_max[2].item()))
        
        # Set obstacle position (x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz)
        env.obs_dict['obstacle_position'][env_id, target_obstacle_idx, 0] = obs_x
        env.obs_dict['obstacle_position'][env_id, target_obstacle_idx, 1] = obs_y
        env.obs_dict['obstacle_position'][env_id, target_obstacle_idx, 2] = obs_z
        
        # Set orientation to identity quaternion (no rotation)
        env.obs_dict['obstacle_orientation'][env_id, target_obstacle_idx, 0] = 1.0  # qw
        env.obs_dict['obstacle_orientation'][env_id, target_obstacle_idx, 1] = 0.0  # qx
        env.obs_dict['obstacle_orientation'][env_id, target_obstacle_idx, 2] = 0.0  # qy
        env.obs_dict['obstacle_orientation'][env_id, target_obstacle_idx, 3] = 0.0  # qz
        
        # Zero velocities
        env.obs_dict['obstacle_position'][env_id, target_obstacle_idx, 7:13] = 0.0
    
    # Write updated obstacle positions to simulation
    env.sim_env.IGE_env.write_to_sim()
    
    logger.info(f"✅ Spawned {num_envs} random obstacles near targets (using obstacle slot {target_obstacle_idx})")



# ============================================================================
# APF Baseline Agent (from test_baseline_pid.py)
# ============================================================================

# Depth Camera Configuration (matching training)
DEPTH_HEIGHT = 135
DEPTH_WIDTH = 240
DEPTH_HFOV_DEG = 87.0
DEPTH_MAX_RANGE = 10.0
DEPTH_MIN_RANGE = 0.2

def depth_image_to_obstacle_points(depth_pixels, max_points=500):
    """
    Convert depth image to 3D obstacle points in robot frame
    
    Args:
        depth_pixels: (num_envs, H, W) normalized depth values [0-1]
        max_points: maximum number of obstacle points to extract per env
    
    Returns:
        obstacle_points: (num_envs, max_points, 3) 3D points in robot frame
        valid_mask: (num_envs, max_points) boolean mask of valid points
    """
    num_envs, H, W = depth_pixels.shape
    device = depth_pixels.device
    
    # Denormalize depth to meters
    depth_m = depth_pixels * DEPTH_MAX_RANGE
    
    # Replace invalid/too far values (0 or > max_range) with max_range
    depth_m = torch.where(depth_m <= 0.0, DEPTH_MAX_RANGE, depth_m)
    
    # Camera intrinsics
    hfov_rad = np.deg2rad(DEPTH_HFOV_DEG)
    fx = (W / 2.0) / np.tan(hfov_rad / 2.0)  # focal length in pixels
    fy = fx  # assume square pixels
    cx = W / 2.0  # principal point
    cy = H / 2.0
    
    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    #  Camera coordinate system (OpenCV convention):
    # X: right, Y: down, Z: forward
    # Robot coordinate system:
    # X: forward, Y: left, Z: up
    
    # Back-project to 3D in camera frame
    Z_cam = depth_m  # (num_envs, H, W)
    X_cam = (u.unsqueeze(0) - cx) / fx * Z_cam
    Y_cam = (v.unsqueeze(0) - cy) / fy * Z_cam
    
    # Transform from camera frame to robot frame
    # Camera mounted looking forward, rotated -90° around X, then -90° around Z
    # Simplified: X_robot = Z_cam, Y_robot = -X_cam, Z_robot = -Y_cam
    X_robot = Z_cam
    Y_robot = -X_cam
    Z_robot = -Y_cam
    
    # Stack to (num_envs, H, W, 3)
    points_3d = torch.stack([X_robot, Y_robot, Z_robot], dim=-1)
    
    # Flatten to (num_envs, H*W, 3)
    points_flat = points_3d.view(num_envs, H * W, 3)
    
    # Filter points: only keep points within reasonable range and not too close
    distances = torch.norm(points_flat, dim=-1)  # (num_envs, H*W)
    valid = (distances > DEPTH_MIN_RANGE) & (distances < DEPTH_MAX_RANGE)
    
    # Sample max_points for efficiency
    # Strategy: For each env, randomly sample from valid points
    obstacle_points = torch.zeros((num_envs, max_points, 3), device=device)
    valid_mask = torch.zeros((num_envs, max_points), dtype=torch.bool, device=device)
    
    for env_idx in range(num_envs):
        valid_indices = torch.where(valid[env_idx])[0]
        num_valid = valid_indices.shape[0]
        
        if num_valid > 0:
            # Sample uniformly or take closest points
            if num_valid > max_points:
                # Take closest max_points obstacles
                dists_valid = distances[env_idx, valid_indices]
                _, closest_idx = torch.topk(dists_valid, max_points, largest=False)
                sample_indices = valid_indices[closest_idx]
            else:
                sample_indices = valid_indices
            
            num_samples = sample_indices.shape[0]
            obstacle_points[env_idx, :num_samples] = points_flat[env_idx, sample_indices]
            valid_mask[env_idx, :num_samples] = True
    
    return obstacle_points, valid_mask


def compute_apf_obstacle_repulsion_from_points(robot_pos, obstacle_points, valid_mask, 
                                                 w_obs=2.0, d_obs=1.5):
    """
    Compute APF obstacle repulsion gradient from 3D obstacle points
    
    Args:
        robot_pos: (num_envs, 3) robot positions (only used for relative positioning)
        obstacle_points: (num_envs, max_points, 3) obstacle points in robot frame
        valid_mask: (num_envs, max_points) validity mask
        w_obs: obstacle weight
        d_obs: obstacle influence radius
    
    Returns:
        grad_obs: (num_envs, 3) repulsion gradient
    """
    num_envs = obstacle_points.shape[0]
    device = obstacle_points.device
    
    # Obstacle points are already in robot frame (relative to robot)
    # Distance from robot (at origin of robot frame) to each obstacle point
    diff = obstacle_points  # (num_envs, max_points, 3)
    dist = torch.norm(diff, dim=2)  # (num_envs, max_points)
    
    # Mask out invalid points by setting their distance to large value
    dist = torch.where(valid_mask, dist, torch.tensor(1000.0, device=device))
    
    # APF repulsion: U = 0.5 * (1/d - 1/d0)^2 if d < d0, else 0
    # grad U = (1/d - 1/d0) * (-1/d^2) * (diff / d) = (1/d - 1/d0) * (-1/d^3) * diff
    
    mask = (dist < d_obs) & (dist > 0.1)  # within influence radius and not too close
    
    # Repulsion magnitude
    repulsion_mag = (1.0 / dist - 1.0 / d_obs) * (1.0 / (dist ** 2))  # (num_envs, max_points)
    repulsion_mag = torch.where(mask, repulsion_mag, torch.tensor(0.0, device=device))
    
    # Unit vectors away from obstacles (gradient direction)
    obs_dirs = diff / (dist.unsqueeze(2) + 1e-6)  # (num_envs, max_points, 3)
    
    # Sum contributions from all obstacles
    grad_obs = torch.sum(repulsion_mag.unsqueeze(2) * obs_dirs, dim=1)  # (num_envs, 3)
    
    return grad_obs



class LocalAPFAgent:
    """
    Local (Blind) Artificial Potential Field Agent 
    - Uses Finite Difference for Noise Gradient (No internal formula access)
    - Uses Lidar/Depth for Obstacle Repulsion (No ground truth positions)
    """
    
    def __init__(self, env, w_dist=1.0, w_noise=5.0, w_obs=2.0, d_obs=1.5, max_speed_xy=0.8):
        self.env = env
        self.device = env.device
        self.w_dist = w_dist
        self.w_noise = w_noise
        self.w_obs = w_obs
        self.obs_influence_radius = d_obs
        self.max_speed_xy = max_speed_xy
        self.max_speed_z = 0.5
        self.grad_to_vel_gain = 2.0
        
        # Perturbations for finite difference gradient estimation
        self.epsilon = 0.1
        # Create perturbations: [[e,0,0], [-e,0,0], [0,e,0], [0,-e,0], [0,0,e], [0,0,-e]]
        self.perturbations = torch.tensor([
            [self.epsilon, 0, 0], [-self.epsilon, 0, 0],
            [0, self.epsilon, 0], [0, -self.epsilon, 0],
            [0, 0, self.epsilon], [0, 0, -self.epsilon]
        ], device=self.device)
    
    def compute_action(self, obs_dict):
        """Compute velocity command based on estimated local gradients"""
        num_envs = self.env.sim_env.num_envs
        position = obs_dict["robot_position"]
        
        # --- 1. Target Attraction (Known info) ---
        target_pos = self.env.target_position
        vec_to_target = position - target_pos
        # Gradient of 0.5 * dist^2 is just the vector (pos - target)
        # FORCE CLAMPING / VELOCITY SATURATION (Fix for saturation issue)
        # If distance is large, we shouldn't output infinite force. 
        # Normalize to max_magnitude = 1.0 (approx max cruise velocity/force)
        dist_to_target = torch.norm(vec_to_target, dim=1, keepdim=True)
        max_magnitude = 1.0
        
        # If dist > 1.0, scale vector to length 1.0. If dist < 1.0, keep as is (linear approach)
        scale_factor = torch.clamp(max_magnitude / (dist_to_target + 1e-6), max=1.0)
        grad_target = vec_to_target * scale_factor
        
        # --- 2. Noise Repulsion (Finite Difference Estimation) ---
        # We need to query noise at perturbed positions. 
        # Since we can't move the robot, we use the env's helper function strictly as a "sensor query"
        # provided we only pass hypothetical positions.
        
        # --- 2. Noise Repulsion (Finite Difference Estimation) ---
        # We need to query noise at perturbed positions. 
        # Since _compute_gmm_mixture expects (num_envs, 3) input to match internal GMM param shapes,
        # we cannot batch all perturbations 6x together. We must query them one by one or reshape internal params (too invasive).
        # We will loop over the 6 perturbation points.
        
        noise_readings = []
        
        # Iterate over the 6 perturbations
        for i in range(6):
             # p is (3,) -> expand to (num_envs, 3)
             perturbation = self.perturbations[i].unsqueeze(0).expand(num_envs, -1)
             query_pos = position + perturbation
             
             with torch.no_grad():
                 # Input: (num_envs, 3) -> Output: (num_envs,)
                 # This matches the shape expected by env's internal GMM parameters
                 val = self.env._compute_gmm_mixture(query_pos)
                 noise_readings.append(val)
        
        # Stack results: (num_envs, 6)
        noise_readings = torch.stack(noise_readings, dim=1)
        
        # Central difference approximation
        # dx = (f(x+e) - f(x-e)) / 2e
        # perturbations are ordered: +x, -x, +y, -y, +z, -z
        grad_noise_x = (noise_readings[:, 0] - noise_readings[:, 1]) / (2 * self.epsilon)
        grad_noise_y = (noise_readings[:, 2] - noise_readings[:, 3]) / (2 * self.epsilon)
        grad_noise_z = (noise_readings[:, 4] - noise_readings[:, 5]) / (2 * self.epsilon)
        
        grad_noise = torch.stack([grad_noise_x, grad_noise_y, grad_noise_z], dim=1)
        
        # --- 3. Obstacle Repulsion (Depth Camera Based) ---
        # Use depth image to detect obstacles (FAIR - same as RL training)
        depth_pixels = self.env.obs_dict["depth_range_pixels"].squeeze(1)  # (num_envs, H, W)
        
        # Convert depth image to 3D obstacle points in robot frame
        obstacle_points, valid_mask = depth_image_to_obstacle_points(depth_pixels, max_points=500)
        
        # Compute APF repulsion from obstacle points
        grad_obs = compute_apf_obstacle_repulsion_from_points(
            position, obstacle_points, valid_mask,
            w_obs=1.0,  # weight will be applied later
            d_obs=self.obs_influence_radius
        )
        
        
        # --- 4. Total Gradient ---
        grad_total = (self.w_dist * grad_target) + (self.w_noise * grad_noise) + (self.w_obs * grad_obs)
        
        # Descent direction
        vel_cmd = -self.grad_to_vel_gain * grad_total
        vel_cmd[:, 0:2] = torch.clamp(vel_cmd[:, 0:2], -self.max_speed_xy, self.max_speed_xy)
        vel_cmd[:, 2] = torch.clamp(vel_cmd[:, 2], -self.max_speed_z, self.max_speed_z)
        
        # Normalize to action space
        action = torch.zeros((num_envs, 4), device=self.device)
        action[:, 0] = vel_cmd[:, 0] / self.max_speed_xy
        action[:, 1] = vel_cmd[:, 1] / self.max_speed_xy
        action[:, 2] = vel_cmd[:, 2] / self.max_speed_z
        action[:, 3] = 0.0
        
        return torch.clamp(action, -1.0, 1.0)


class SimplePIDNavigator:
    """
    Simple PID Navigator - Pure target seeking with lidar-based obstacle avoidance
    - No noise optimization (completely ignores GMM noise field)
    - Only target attraction + obstacle repulsion
    - Baseline to show performance without noise awareness
    """
    
    def __init__(self, env, w_dist=1.0, w_obs=2.0, d_obs=1.5):
        self.env = env
        self.device = env.device
        self.w_dist = w_dist
        self.w_obs = w_obs
        self.obs_influence_radius = d_obs
        self.max_speed_xy = 0.8
        self.max_speed_z = 0.5
        self.grad_to_vel_gain = 2.0
    
    def compute_action(self, obs_dict):
        """Compute velocity command based on target attraction and obstacle avoidance only"""
        num_envs = self.env.sim_env.num_envs
        position = obs_dict["robot_position"]
        
        # --- 1. Target Attraction (Known info) ---
        target_pos = self.env.target_position
        vec_to_target = position - target_pos
        # Gradient of 0.5 * dist^2 is just the vector (pos - target)
        
        # FORCE CLAMPING / VELOCITY SATURATION (Fix for saturation issue)
        # Normalize to max_magnitude = 1.0
        dist_to_target = torch.norm(vec_to_target, dim=1, keepdim=True)
        max_magnitude = 1.0
        
        # If dist > 1.0, scale vector to length 1.0. If dist < 1.0, keep as is (linear approach)
        scale_factor = torch.clamp(max_magnitude / (dist_to_target + 1e-6), max=1.0)
        grad_target = vec_to_target * scale_factor
        
        # --- 2. Obstacle Repulsion (Depth Camera Based) ---
        # Use depth image to detect obstacles (FAIR - same as RL training)
        depth_pixels = self.env.obs_dict["depth_range_pixels"].squeeze(1)  # (num_envs, H, W)
        
        # Convert depth image to 3D obstacle points in robot frame
        obstacle_points, valid_mask = depth_image_to_obstacle_points(depth_pixels, max_points=500)
        
        # Compute APF repulsion from obstacle points
        grad_obs = compute_apf_obstacle_repulsion_from_points(
            position, obstacle_points, valid_mask,
            w_obs=1.0,  # weight will be applied later
            d_obs=self.obs_influence_radius
        )
        
        # --- 3. Total Gradient (NO NOISE OPTIMIZATION) ---
        grad_total = (self.w_dist * grad_target) + (self.w_obs * grad_obs)
        
        # Descent direction
        vel_cmd = -self.grad_to_vel_gain * grad_total
        vel_cmd[:, 0:2] = torch.clamp(vel_cmd[:, 0:2], -self.max_speed_xy, self.max_speed_xy)
        vel_cmd[:, 2] = torch.clamp(vel_cmd[:, 2], -self.max_speed_z, self.max_speed_z)
        
        # Normalize to action space
        action = torch.zeros((num_envs, 4), device=self.device)
        action[:, 0] = vel_cmd[:, 0] / self.max_speed_xy
        action[:, 1] = vel_cmd[:, 1] / self.max_speed_xy
        action[:, 2] = vel_cmd[:, 2] / self.max_speed_z
        action[:, 3] = 0.0
        
        return torch.clamp(action, -1.0, 1.0)



# ============================================================================
# PPO Model Loading
# ============================================================================

def load_ppo_model(checkpoint_path, env):
    """Load trained PPO model from checkpoint"""
    logger.info(f"Loading PPO model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    
    # Extract model state
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        model_state = checkpoint
    
    # Create simple inference wrapper network
    import torch.nn as nn
    
    obs_dim = env.task_config.observation_space_dim
    act_dim = env.task_config.action_space_dim
    
    # Infer network architecture from state dict
    # Actual rl_games architecture from checkpoint: 256 -> 128 -> 64 -> output
    class PPOActor(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.actor_mlp = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, act_dim),
            )
        
        def forward(self, obs):
            # Training config specifies 'mu_activation: None', so no Tanh here.
            # Output is raw mean, which might be clipped later or used directly.
            return self.actor_mlp(obs)
    
    model = PPOActor(obs_dim, act_dim).to(env.device)
    
    # Load only actor weights from checkpoint
    actor_state = {}
    for key, value in model_state.items():
        if 'a2c_network.actor_mlp' in key:
            # Remove 'a2c_network.' prefix
            new_key = key.replace('a2c_network.', '')
            actor_state[new_key] = value
    
    model.load_state_dict(actor_state, strict=False)
    model.eval()
    
    # Load running_mean_std for input normalization
    running_mean_std = None
    if 'running_mean_std' in checkpoint:
        rms_data = checkpoint['running_mean_std']
        running_mean_std = RunningMeanStd(
            mean=rms_data['running_mean'],
            std=rms_data['running_var'] ** 0.5,
            count=rms_data['count'],
            device=env.device
        )
        logger.info("✅ Loaded running_mean_std from checkpoint")
    else:
        logger.warning("⚠️  No running_mean_std in checkpoint - will compute from environment")
        # We'll compute it later after environment is fully set up
        # For now, return None and handle in main()
    
    logger.info("PPO model loaded successfully")
    return model, running_mean_std


# ============================================================================
# Main Experiment Loop
# ============================================================================

def run_phase(env, agent, episode_length, phase_name, running_mean_std=None):
    """
    Run one evaluation phase (RL or Baseline)
    
    Args:
        running_mean_std: For RL phase only, normalization parameters
    
    Returns:
        j_curve: (episode_length, num_envs) numpy array of J values
        attitude_data: dict with mean_abs/std arrays in degrees (shape: [episode_length, 3])
    """
    logger.info(f"Starting {phase_name} phase for {episode_length} steps...")
    # Track J(p) curve
    j_curve = np.zeros((episode_length, NUM_ENVS))
    
    # Track distance to target
    distance_curve = np.zeros((episode_length, NUM_ENVS))
    
    # Track attitude stability
    attitude_mean_abs = np.zeros((episode_length, 3))
    attitude_std = np.zeros((episode_length, 3))
    
    # Track control actions for smoothness analysis
    action_history = np.zeros((episode_length, NUM_ENVS, 4))
    
    # Track crashes (only first crash per environment)
    has_crashed = np.zeros(NUM_ENVS, dtype=bool)
    crash_count_per_step = np.zeros(episode_length)
    
    # Track survival times (all crash events)
    survival_times = []
    env_last_reset_step = np.zeros(NUM_ENVS, dtype=int)
    
    # Track 'Fair' Min J and Min Dist (valid only until first crash)
    # Initialize with infinity so we can find minimums
    fair_min_j_per_env = np.full(NUM_ENVS, np.inf)
    fair_min_dist_per_env = np.full(NUM_ENVS, np.inf)
    
    # Mask to track if environment is still valid (hasn't crashed yet)
    # Starts all True
    env_valid_for_metrics = np.ones(NUM_ENVS, dtype=bool)

    for step in range(episode_length):
        # Compute action
        if phase_name == "RL":
            obs = env.task_obs["observations"]
            
            # Apply normalization for RL
            if running_mean_std is not None:
                normalized_obs = running_mean_std.normalize(obs)
            else:
                normalized_obs = obs
            
            with torch.no_grad():
                action = agent(normalized_obs)
                if isinstance(action, tuple):
                    action = action[0]
        else:  # Baseline
            action = agent.compute_action(env.obs_dict)
        
        # Store action for smoothness analysis
        # We store the raw action sent to the controller (-1 to 1 or scaled)
        # Assuming agent.compute_action returns a tensor on device
        if isinstance(action, torch.Tensor):
            action_history[step] = action.detach().cpu().numpy()
        else:
            action_history[step] = action
        
        # Step environment
        _, _, _, _, _ = env.step(action)
        
        # CRITICAL FIX: Trigger "Safe Spawn" Logic
        # Training calls this to retry bad spawns. Verification must too.
        env.logging_sanity_check(env.infos)
        
        # Compute J(p)
        J_values = compute_J_potential(env)
        j_values_np = J_values.cpu().numpy()
        j_curve[step, :] = j_values_np
        
        # Compute distance to target
        robot_pos = env.obs_dict["robot_position"]
        target_pos = env.target_position
        distances = torch.norm(robot_pos - target_pos, dim=1)
        dist_values_np = distances.cpu().numpy()
        distance_curve[step, :] = dist_values_np

        # Attitude stability (roll/pitch/yaw) in degrees
        euler = env.obs_dict["robot_euler_angles"]
        euler_deg = euler * RAD2DEG
        attitude_mean_abs[step] = torch.abs(euler_deg).mean(dim=0).cpu().numpy()
        attitude_std[step] = euler_deg.std(dim=0).cpu().numpy()
        
        # Detect crashes
        crashed = env.obs_dict["crashes"].cpu().numpy()
        # Environments that crashed just now
        new_crashes = crashed & (~has_crashed)
        
        # UPDATE FAIR METRICS
        # Logic: 
        # 1. Update min_j/min_dist for environments that were valid at start of step AND didn't crash this step
        #    (We exclude 'new_crashes' because their J/Dist is from a fresh reset spawn)
        # 2. Mark environments that crashed (either before or now) as invalid for future
        
        # Envs that are valid for this step's metric update:
        # Currently valid (env_valid_for_metrics) AND NOT crashed in this step (new_crashes)
        # Note: env_valid_for_metrics is True if env hasn't crashed in PREVIOUS steps.
        # We combine this with ~crashed (which covers both prev and current crashes actually, since crashed comes from obs buffer)
        # But 'crashed' buffer might be fleeting if reset happens? 
        # In IsaacGym, 'crashed' is usually 1 during the reset step.
        
        currently_valid_mask = env_valid_for_metrics & (~crashed)
        
        # Update minimums just for valid envs
        if np.any(currently_valid_mask):
            fair_min_j_per_env[currently_valid_mask] = np.minimum(
                fair_min_j_per_env[currently_valid_mask], 
                j_values_np[currently_valid_mask]
            )
            fair_min_dist_per_env[currently_valid_mask] = np.minimum(
                fair_min_dist_per_env[currently_valid_mask], 
                dist_values_np[currently_valid_mask]
            )

        # Update global crash tracker
        has_crashed |= crashed
        crash_count_per_step[step] = new_crashes.sum()
        
        # Update metric validity mask (Once crashed, never valid again for min_j/dist)
        # We mark 'new_crashes' as invalid for NEXT step onwards
        # AND they were excluded from THIS step's update above by (~crashed)
        env_valid_for_metrics &= (~crashed)

        # Record survival times for newly crashed envs
        for env_id in np.where(new_crashes)[0]:
            survival_time = step - env_last_reset_step[env_id]
            survival_times.append(survival_time)
            env_last_reset_step[env_id] = step
        
        if step % 100 == 0:
            logger.info(f"  Step {step}/{episode_length} | Mean J = {J_values.mean().item():.4f} | Crashes: {int(crash_count_per_step[step])}/100")
    
    # Cumulative crash rate
    cumulative_crashes = np.cumsum(crash_count_per_step)
    cumulative_rate = (cumulative_crashes / NUM_ENVS) * 100
    
    crash_data = {
        'cumulative_count': cumulative_crashes,
        'cumulative_rate': cumulative_rate,
    }
    
    # Arrival statistics
    final_distances = torch.norm(
        env.obs_dict["robot_position"] - env.target_position, dim=1
    ).cpu().numpy()
    arrival_count = (final_distances < 2.0).sum()
    arrival_rate = (arrival_count / NUM_ENVS) * 100
    
    # Survival time statistics
    survival_data = {
        'mean_per_step': np.zeros(episode_length),
        'std_per_step': np.zeros(episode_length),
    }
    
    for step in range(episode_length):
        relevant_times = [st for st in survival_times if st <= step]
        if len(relevant_times) > 0:
            survival_data['mean_per_step'][step] = np.mean(relevant_times)
            survival_data['std_per_step'][step] = np.std(relevant_times)
        else:
            survival_data['mean_per_step'][step] = step
            survival_data['std_per_step'][step] = 0.0
    
    arrival_data = {
        'count': arrival_count,
        'rate': arrival_rate,
        'distances': final_distances,
    }
    
    attitude_data = {
        'mean_abs': attitude_mean_abs,
        'std': attitude_std,
    }
    
    # Use the FAIR metrics calculated incrementally
    # Replace any remaining infs (envs that crashed immediately at step 0) with initial values or clamp?
    # If fair_min_j is still inf, it means it crashed at step 0 and never had a valid step.
    # We can perform a fallback to the first step value (which might be the crash value) or just keep it high.
    # To avoid plotting errors, let's use the standard min for those corner cases, 
    # OR better: if it crashed immediately, its "performance" is indeed terrible (essentially infinite cost).
    # But for plotting, let's clamp Infinity to the max observed valid J to keep plots readable.
    
    valid_j_values = fair_min_j_per_env[np.isfinite(fair_min_j_per_env)]
    if len(valid_j_values) > 0:
        max_valid_j = np.max(valid_j_values)
        fair_min_j_per_env[np.isinf(fair_min_j_per_env)] = max_valid_j * 1.5 # Penalize immediate crashers
    else:
        # Fallback if everything crashed instantly (unlikely)
        fair_min_j_per_env = np.min(j_curve, axis=0)

    # Similar handling for distance
    valid_dist_values = fair_min_dist_per_env[np.isfinite(fair_min_dist_per_env)]
    if len(valid_dist_values) > 0:
        max_valid_dist = np.max(valid_dist_values)
        fair_min_dist_per_env[np.isinf(fair_min_dist_per_env)] = max_valid_dist
    else:
        fair_min_dist_per_env = np.min(distance_curve, axis=0)

    return j_curve, crash_data, survival_data, arrival_data, attitude_data, distance_curve, fair_min_j_per_env, fair_min_dist_per_env, action_history


def plot_attitude_stability(rl_attitude, baseline_attitude, pid_attitude, metric, filename_prefix, y_label):
    """Plot attitude stability comparison for roll/pitch/yaw."""
    logger.info(f"Generating attitude {metric} comparison plot...")
    
    rl_data = rl_attitude[metric]
    baseline_data = baseline_attitude[metric]
    pid_data = pid_attitude[metric]
    
    def compute_ema(data, alpha=EMA_ALPHA):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for t in range(1, len(data)):
            ema[t] = alpha * ema[t-1] + (1 - alpha) * data[t]
        return ema
    
    rl_ema = np.stack([compute_ema(rl_data[:, i]) for i in range(3)], axis=1)
    baseline_ema = np.stack([compute_ema(baseline_data[:, i]) for i in range(3)], axis=1)
    pid_ema = np.stack([compute_ema(pid_data[:, i]) for i in range(3)], axis=1)
    
    # Configure matplotlib
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 10
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 9
    
    fig_width_cm = 18.0
    fig_height_cm = 6.5
    fig, axes = plt.subplots(
        1, 3, figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54), sharex=True
    )
    
    timesteps = np.arange(EPISODE_LENGTH)
    rl_color = '#1f77b4'
    baseline_color = '#d62728'
    pid_color = '#2ca02c'
    angle_names = ["Roll", "Pitch", "Yaw"]
    
    for i, ax in enumerate(axes):
        ax.plot(
            timesteps,
            rl_data[:, i],
            color=rl_color,
            linewidth=0.8,
            alpha=0.25,
            label="RL raw" if i == 0 else None,
        )
        ax.plot(
            timesteps,
            baseline_data[:, i],
            color=baseline_color,
            linewidth=0.8,
            alpha=0.25,
            label="Baseline raw" if i == 0 else None,
        )
        ax.plot(
            timesteps,
            pid_data[:, i],
            color=pid_color,
            linewidth=0.8,
            alpha=0.25,
            label="PID raw" if i == 0 else None,
        )
        ax.plot(
            timesteps,
            rl_ema[:, i],
            color=rl_color,
            linewidth=1.6,
            linestyle='-',
            label="RL EMA" if i == 0 else None,
        )
        ax.plot(
            timesteps,
            baseline_ema[:, i],
            color=baseline_color,
            linewidth=1.6,
            linestyle='--',
            label="Baseline EMA" if i == 0 else None,
        )
        ax.plot(
            timesteps,
            pid_ema[:, i],
            color=pid_color,
            linewidth=1.6,
            linestyle=':',
            label="PID EMA" if i == 0 else None,
        )
        ax.set_title(angle_names[i])
        ax.set_xlabel('Time Steps')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_xlim([0, EPISODE_LENGTH])
        if i == 0:
            ax.set_ylabel(y_label)
            ax.legend(loc='best', frameon=True, fancybox=False)
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_600dpi.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved attitude PNG to: {png_path}")
    
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    logger.info(f"Saved attitude PDF to: {pdf_path}")
    
    plt.close()
    
    csv_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step',
            'rl_roll_raw', 'rl_pitch_raw', 'rl_yaw_raw',
            'baseline_roll_raw', 'baseline_pitch_raw', 'baseline_yaw_raw',
            'pid_roll_raw', 'pid_pitch_raw', 'pid_yaw_raw',
            'rl_roll_ema', 'rl_pitch_ema', 'rl_yaw_ema',
            'baseline_roll_ema', 'baseline_pitch_ema', 'baseline_yaw_ema',
            'pid_roll_ema', 'pid_pitch_ema', 'pid_yaw_ema',
        ])
        for t in range(EPISODE_LENGTH):
            writer.writerow([
                t,
                rl_data[t, 0], rl_data[t, 1], rl_data[t, 2],
                baseline_data[t, 0], baseline_data[t, 1], baseline_data[t, 2],
                pid_data[t, 0], pid_data[t, 1], pid_data[t, 2],
                rl_ema[t, 0], rl_ema[t, 1], rl_ema[t, 2],
                baseline_ema[t, 0], baseline_ema[t, 1], baseline_ema[t, 2],
                pid_ema[t, 0], pid_ema[t, 1], pid_ema[t, 2],
            ])
    logger.info(f"Saved attitude stats to: {csv_path}")


def aggregate_and_plot(rl_curve, baseline_curve, pid_curve):
    """
    Aggregate data and generate academic-quality plot
    
    Args:
        rl_curve: (600, 100) array
        baseline_curve: (600, 100) array
        pid_curve: (600, 100) array
    """
    logger.info("Aggregating data and generating plots...")
    
    # Compute statistics
    rl_mean = rl_curve.mean(axis=1)
    rl_std = rl_curve.std(axis=1)
    baseline_mean = baseline_curve.mean(axis=1)
    baseline_std = baseline_curve.std(axis=1)
    pid_mean = pid_curve.mean(axis=1)
    pid_std = pid_curve.std(axis=1)
    
    # Compute EMA
    def compute_ema(data, alpha=EMA_ALPHA):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for t in range(1, len(data)):
            ema[t] = alpha * ema[t-1] + (1 - alpha) * data[t]
        return ema
    
    rl_ema = compute_ema(rl_mean)
    baseline_ema = compute_ema(baseline_mean)
    pid_ema = compute_ema(pid_mean)
    
    # Save aggregated stats
    stats_csv = os.path.join(OUTPUT_DIR, "aggregated_stats.csv")
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'rl_mean', 'rl_std', 'baseline_mean', 'baseline_std', 'pid_mean', 'pid_std', 'rl_ema', 'baseline_ema', 'pid_ema'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, rl_mean[t], rl_std[t], baseline_mean[t], baseline_std[t], pid_mean[t], pid_std[t], rl_ema[t], baseline_ema[t], pid_ema[t]])
    logger.info(f"Saved aggregated statistics to: {stats_csv}")
    
    # Configure matplotlib for academic publication
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 10
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 9
    
    # Create figure (8.5 cm x 6.5 cm)
    fig_width_cm = 8.5
    fig_height_cm = 6.5
    fig, ax = plt.subplots(figsize=(fig_width_cm/2.54, fig_height_cm/2.54))
    
    # Time steps
    timesteps = np.arange(EPISODE_LENGTH)
    
    # Colors
    rl_color = '#1f77b4'  # Deep blue
    baseline_color = '#d62728'  # Deep red
    pid_color = '#2ca02c'  # Green
    
    # Plot RL curve
    ax.plot(timesteps, rl_ema, color=rl_color, linewidth=1.8, linestyle='-', label='RL (Proposed)')
    ax.fill_between(timesteps, rl_mean - rl_std, rl_mean + rl_std, color=rl_color, alpha=0.25)
    
    # Plot Baseline curve
    ax.plot(timesteps, baseline_ema, color=baseline_color, linewidth=1.8, linestyle='--', label='PID + Greedy')
    ax.fill_between(timesteps, baseline_mean - baseline_std, baseline_mean + baseline_std, color=baseline_color, alpha=0.25)
    
    # Plot Pure PID curve
    ax.plot(timesteps, pid_ema, color=pid_color, linewidth=1.8, linestyle=':', label='Pure PID')
    ax.fill_between(timesteps, pid_mean - pid_std, pid_mean + pid_std, color=pid_color, alpha=0.25)
    
    # Formatting
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Unified Potential $J(p)$')
    ax.legend(loc='best', frameon=True, fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim([0, EPISODE_LENGTH])
    
    # Tight layout
    plt.tight_layout()
    
    # Save PNG (600 DPI)
    png_path = os.path.join(OUTPUT_DIR, "j_convergence_comparison_600dpi.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved PNG plot to: {png_path}")
    
    # Save PDF (vector)
    pdf_path = os.path.join(OUTPUT_DIR, "j_convergence_comparison.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    logger.info(f"Saved PDF plot to: {pdf_path}")
    
    plt.close()
    
    # Print summary statistics
    logger.info("=== Summary Statistics ===")
    logger.info(f"RL Final EMA: {rl_ema[-1]:.4f}")
    logger.info(f"Baseline Final EMA: {baseline_ema[-1]:.4f}")
    logger.info(f"Pure PID Final EMA: {pid_ema[-1]:.4f}")
    logger.info(f"RL vs Baseline Improvement: {((baseline_ema[-1] - rl_ema[-1]) / baseline_ema[-1] * 100):.2f}%")
    logger.info(f"RL vs Pure PID Improvement: {((pid_ema[-1] - rl_ema[-1]) / pid_ema[-1] * 100):.2f}%")


def plot_fair_min_metrics_distribution(rl_min_j, baseline_min_j, pid_min_j, rl_min_dist, baseline_min_dist, pid_min_dist):
    """Plot distribution of FAIR minimum J and minimum distance to target using Sorted Performance Curves (Cactus Plots)."""
    
    # Define colors
    colors = ['blue', 'orange', 'green']
    labels = ['RL', 'PID+Greedy', 'Pure PID']
    
    def plot_sorted_performance_curve(ax, data_list, colors, labels, title, ylabel):
        """Plots sorted performance curves (Cactus Plots)"""
        for data, color, label in zip(data_list, colors, labels):
            # Sort data from best (lowest) to worst (highest)
            sorted_data = np.sort(data)
            x = np.arange(len(sorted_data))
            
            # Plot line
            ax.plot(x, sorted_data, color=color, linewidth=2, label=label)
            
            # Optional: Add small dots on the line
            ax.scatter(x, sorted_data, color=color, s=10, alpha=0.5)

        ax.set_xlabel('Episodes (Sorted by Performance)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle= '--', alpha=0.3)
        ax.set_xlim(0, len(data_list[0]))

    # 1. Minimum J Distribution (Sorted)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_sorted_performance_curve(
        ax, 
        [rl_min_j, baseline_min_j, pid_min_j], 
        colors, 
        labels,
        'Sorted Min J-Cost Performance (Lower is Better)',
        'Minimum J Value (Cost)'
    )
    
    png_path = os.path.join(OUTPUT_DIR, "min_j_sorted_curve.png")
    plt.savefig(png_path, dpi=300)
    logger.info(f"Saved min J sorted curve to: {png_path}")
    plt.close()
    
    # 2. Minimum Distance Distribution (Sorted)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_sorted_performance_curve(
        ax, 
        [rl_min_dist, baseline_min_dist, pid_min_dist], 
        colors, 
        labels,
        'Sorted Min Distance to Target (Lower is Better)',
        'Minimum Distance to Target (m)'
    )
    
    # Add a horizontal line at 2.0m (Success Threshold)
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Success Threshold (2.0m)')
    ax.legend()
    
    png_path = os.path.join(OUTPUT_DIR, "min_dist_sorted_curve.png")
    plt.savefig(png_path, dpi=300)
    logger.info(f"Saved min distance sorted curve to: {png_path}")
    plt.close()


def plot_raw_motor_commands(rl_actions, baseline_actions, pid_actions, env_idx=0):
    """
    Plot Raw Motor Commands for a single representative environment.
    Visualizes if actions are smooth or oscillating (sawtooth).
    """
    # Actions shape: (steps, num_envs, 4)
    # Extract actions for the specified environment: (steps, 4)
    rl_env_actions = rl_actions[:, env_idx, :]
    baseline_env_actions = baseline_actions[:, env_idx, :]
    pid_env_actions = pid_actions[:, env_idx, :]
    
    steps = np.arange(len(rl_env_actions))
    
    # Create 3 subplots (one for each method)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=False)  # Share X only, different Y scale for raw vs clipped
    
    # Plot RL Actions
    # We plot two versions: 
    # 1. Dashed line = Raw Output (What the network suggests)
    # 2. Solid line = Clipped Output (What the robot actually does, same range as PID)
    for i in range(4):
        # Raw (Original)
        # axes[0].plot(steps, rl_env_actions[:, i], linestyle=':', alpha=0.4, linewidth=1.0)
        
        # Clipped (Effective)
        rl_clipped = np.clip(rl_env_actions[:, i], -1.0, 1.0)
        axes[0].plot(steps, rl_clipped, label=f'Motor {i}', alpha=0.9, linewidth=1.5)
        
    # axes[0].set_ylim([-13, 13]) # Limit Y axis to show outliers if needed, but auto is fine for raw
    axes[0].set_title(f'RL (PPO) Effective Motor Commands (Clipped [-1, 1])')
    axes[0].set_ylabel('Effective Action [-1, 1]')
    axes[0].set_ylim([-1.1, 1.1]) # Focus on the effective range
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper right', ncol=4, fontsize='small')
    
    # Plot PID+Greedy Actions
    for i in range(4):
        axes[1].plot(steps, baseline_env_actions[:, i], label=f'Motor {i}', alpha=0.8, linewidth=1.5)
    axes[1].set_title(f'PID+Greedy Raw Motor Commands (Env {env_idx})')
    axes[1].set_ylabel('Raw Action [-1, 1]')
    axes[1].set_ylim([-1.1, 1.1])
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # Plot Pure PID Actions
    for i in range(4):
        axes[2].plot(steps, pid_env_actions[:, i], label=f'Motor {i}', alpha=0.8, linewidth=1.5)
    axes[2].set_title(f'Pure PID Raw Motor Commands (Env {env_idx})')
    axes[2].set_ylabel('Raw Action [-1, 1]')
    axes[2].set_ylim([-1.1, 1.1])
    axes[2].set_xlabel('Time Steps')
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "raw_motor_commands_env0.png")
    plt.savefig(png_path, dpi=300)
    logger.info(f"Saved raw motor commands plot to: {png_path}")
    plt.close()


def plot_crash_rate(rl_crash_data, baseline_crash_data, pid_crash_data):
    """Plot cumulative crash rate comparison"""
    logger.info("Generating crash rate comparison plot...")
    
    # Configure matplotlib
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 9
    
    fig_width_cm = 8.5
    fig_height_cm = 6.5
    fig, ax = plt.subplots(figsize=(fig_width_cm/2.54, fig_height_cm/2.54))
    
    timesteps = np.arange(EPISODE_LENGTH)
    rl_color = '#1f77b4'
    baseline_color = '#d62728'
    pid_color = '#2ca02c'
    
    # Plot cumulative crash rate (no EMA, already monotonic)
    ax.plot(timesteps, rl_crash_data['cumulative_rate'], color=rl_color, linewidth=1.8, linestyle='-', label='RL (Proposed)')
    ax.plot(timesteps, baseline_crash_data['cumulative_rate'], color=baseline_color, linewidth=1.8, linestyle='--', label='PID + Greedy')
    ax.plot(timesteps, pid_crash_data['cumulative_rate'], color=pid_color, linewidth=1.8, linestyle=':', label='Pure PID')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Crash Rate (%)')
    ax.legend(loc='best', frameon=True, fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim([0, EPISODE_LENGTH])
    ax.set_ylim([0, min(105, max(rl_crash_data['cumulative_rate'][-1], baseline_crash_data['cumulative_rate'][-1], pid_crash_data['cumulative_rate'][-1]) + 5)])
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "crash_rate_comparison_600dpi.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved crash rate PNG to: {png_path}")
    
    pdf_path = os.path.join(OUTPUT_DIR, "crash_rate_comparison.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    logger.info(f"Saved crash rate PDF to: {pdf_path}")
    
    plt.close()
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "crash_rate_stats.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'rl_crash_rate', 'baseline_crash_rate', 'pid_crash_rate'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, rl_crash_data['cumulative_rate'][t], baseline_crash_data['cumulative_rate'][t], pid_crash_data['cumulative_rate'][t]])
    logger.info(f"Saved crash rate stats to: {csv_path}")


def plot_distance_to_target(rl_distance, baseline_distance, pid_distance):
    """Plot distance to target over time comparison"""
    logger.info("Generating distance to target comparison plot...")
    
    # Compute statistics
    rl_mean = rl_distance.mean(axis=1)
    rl_std = rl_distance.std(axis=1)
    baseline_mean = baseline_distance.mean(axis=1)
    baseline_std = baseline_distance.std(axis=1)
    pid_mean = pid_distance.mean(axis=1)
    pid_std = pid_distance.std(axis=1)
    
    # Compute EMA
    def compute_ema(data, alpha=EMA_ALPHA):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for t in range(1, len(data)):
            ema[t] = alpha * ema[t-1] + (1 - alpha) * data[t]
        return ema
    
    rl_ema = compute_ema(rl_mean)
    baseline_ema = compute_ema(baseline_mean)
    pid_ema = compute_ema(pid_mean)
    
    # Configure matplotlib
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 9
    
    fig_width_cm = 8.5
    fig_height_cm = 6.5
    fig, ax = plt.subplots(figsize=(fig_width_cm/2.54, fig_height_cm/2.54))
    
    timesteps = np.arange(EPISODE_LENGTH)
    rl_color = '#1f77b4'
    baseline_color = '#d62728'
    pid_color = '#2ca02c'
    
    # Plot EMA curves with std shading
    ax.plot(timesteps, rl_ema, color=rl_color, linewidth=1.8, linestyle='-', label='RL (Proposed)')
    ax.fill_between(timesteps, rl_mean - rl_std, rl_mean + rl_std, color=rl_color, alpha=0.25)
    
    ax.plot(timesteps, baseline_ema, color=baseline_color, linewidth=1.8, linestyle='--', label='PID + Greedy')
    ax.fill_between(timesteps, baseline_mean - baseline_std, baseline_mean + baseline_std, color=baseline_color, alpha=0.25)
    
    ax.plot(timesteps, pid_ema, color=pid_color, linewidth=1.8, linestyle=':', label='Pure PID')
    ax.fill_between(timesteps, pid_mean - pid_std, pid_mean + pid_std, color=pid_color, alpha=0.25)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Distance to Target (m)')
    ax.set_title('Distance to Target Over Time')
    ax.legend(loc='best', frameon=True, fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim([0, EPISODE_LENGTH])
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "distance_to_target_comparison_600dpi.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved distance PNG to: {png_path}")
    
    pdf_path = os.path.join(OUTPUT_DIR, "distance_to_target_comparison.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    logger.info(f"Saved distance PDF to: {pdf_path}")
    
    plt.close()
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "distance_to_target_stats.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'rl_mean', 'rl_std', 'rl_ema', 'baseline_mean', 'baseline_std', 'baseline_ema', 'pid_mean', 'pid_std', 'pid_ema'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, rl_mean[t], rl_std[t], rl_ema[t], baseline_mean[t], baseline_std[t], baseline_ema[t], pid_mean[t], pid_std[t], pid_ema[t]])
    logger.info(f"Saved distance stats to: {csv_path}")
    
    # Print summary statistics
    logger.info("=== Distance Summary ===")
    logger.info(f"RL Final Distance (mean): {rl_mean[-1]:.3f}m")
    logger.info(f"Baseline Final Distance (mean): {baseline_mean[-1]:.3f}m")
    logger.info(f"Pure PID Final Distance (mean): {pid_mean[-1]:.3f}m")



def plot_survival_time(rl_survival_data, baseline_survival_data, pid_survival_data, rl_arrival_data, baseline_arrival_data, pid_arrival_data, rl_crash_data, baseline_crash_data, pid_crash_data, rl_j_curve, baseline_j_curve, pid_j_curve):
    """Plot average survival time comparison"""
    logger.info("Generating survival time comparison plot...")
    
    # Apply EMA smoothing
    def compute_ema(data, alpha=EMA_ALPHA):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for t in range(1, len(data)):
            ema[t] = alpha * ema[t-1] + (1 - alpha) * data[t]
        return ema
    
    rl_mean_ema = compute_ema(rl_survival_data['mean_per_step'])
    baseline_mean_ema = compute_ema(baseline_survival_data['mean_per_step'])
    pid_mean_ema = compute_ema(pid_survival_data['mean_per_step'])
    
    # Configure matplotlib
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 9
    
    fig_width_cm = 8.5
    fig_height_cm = 6.5
    fig, ax = plt.subplots(figsize=(fig_width_cm/2.54, fig_height_cm/2.54))
    
    timesteps = np.arange(EPISODE_LENGTH)
    rl_color = '#1f77b4'
    baseline_color = '#d62728'
    pid_color = '#2ca02c'
    
    # Plot EMA curves with std shading
    ax.plot(timesteps, rl_mean_ema, color=rl_color, linewidth=1.8, linestyle='-', label='RL (Proposed)')
    ax.fill_between(timesteps, 
                    rl_survival_data['mean_per_step'] - rl_survival_data['std_per_step'], 
                    rl_survival_data['mean_per_step'] + rl_survival_data['std_per_step'], 
                    color=rl_color, alpha=0.25)
    
    ax.plot(timesteps, baseline_mean_ema, color=baseline_color, linewidth=1.8, linestyle='--', label='PID + Greedy')
    ax.fill_between(timesteps, 
                    baseline_survival_data['mean_per_step'] - baseline_survival_data['std_per_step'], 
                    baseline_survival_data['mean_per_step'] + baseline_survival_data['std_per_step'], 
                    color=baseline_color, alpha=0.25)
    
    ax.plot(timesteps, pid_mean_ema, color=pid_color, linewidth=1.8, linestyle=':', label='Pure PID')
    ax.fill_between(timesteps, 
                    pid_survival_data['mean_per_step'] - pid_survival_data['std_per_step'], 
                    pid_survival_data['mean_per_step'] + pid_survival_data['std_per_step'], 
                    color=pid_color, alpha=0.25)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Average Survival Time (steps)')
    ax.legend(loc='best', frameon=True, fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim([0, EPISODE_LENGTH])
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "survival_time_comparison_600dpi.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved survival time PNG to: {png_path}")
    
    pdf_path = os.path.join(OUTPUT_DIR, "survival_time_comparison.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    logger.info(f"Saved survival time PDF to: {pdf_path}")
    
    plt.close()
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "survival_time_stats.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'rl_mean', 'rl_std', 'rl_ema', 'baseline_mean', 'baseline_std', 'baseline_ema', 'pid_mean', 'pid_std', 'pid_ema'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, 
                           rl_survival_data['mean_per_step'][t], rl_survival_data['std_per_step'][t], rl_mean_ema[t],
                           baseline_survival_data['mean_per_step'][t], baseline_survival_data['std_per_step'][t], baseline_mean_ema[t],
                           pid_survival_data['mean_per_step'][t], pid_survival_data['std_per_step'][t], pid_mean_ema[t]])
    logger.info(f"Saved survival time stats to: {csv_path}")
    
    # === Save Arrival Rate Comparison ===
    arrival_csv = os.path.join(OUTPUT_DIR, "arrival_rate_comparison.csv")
    with open(arrival_csv, 'w') as f:
        f.write("Method,Arrival Count,Arrival Rate (%),Mean Final Distance (m)\n")
        f.write(f"RL,{rl_arrival_data['count']},{rl_arrival_data['rate']:.2f},{rl_arrival_data['distances'].mean():.3f}\n")
        f.write(f"Baseline,{baseline_arrival_data['count']},{baseline_arrival_data['rate']:.2f},{baseline_arrival_data['distances'].mean():.3f}\n")
        f.write(f"Pure PID,{pid_arrival_data['count']},{pid_arrival_data['rate']:.2f},{pid_arrival_data['distances'].mean():.3f}\n")
    logger.info(f"Saved arrival rate comparison to: {arrival_csv}")
    
    # === Print Comparison Summary Table ===
    logger.info("="* 90)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 90)
    logger.info(f"{'Metric':<35} {'RL':<20} {'PID+Greedy':<20} {'Pure PID':<20}")
    logger.info("-" * 90)
    logger.info(f"{'Final Crash Rate (%)':<35} {rl_crash_data['cumulative_rate'][-1]:>18.1f}% {baseline_crash_data['cumulative_rate'][-1]:>18.1f}% {pid_crash_data['cumulative_rate'][-1]:>18.1f}%")
    logger.info(f"{'Arrival Rate @ 2m (%)':<35} {rl_arrival_data['rate']:>18.1f}% {baseline_arrival_data['rate']:>18.1f}% {pid_arrival_data['rate']:>18.1f}%")
    logger.info(f"{'Mean Final J Value':<35} {rl_j_curve[-1].mean():>19.3f} {baseline_j_curve[-1].mean():>19.3f} {pid_j_curve[-1].mean():>19.3f}")
    logger.info(f"{'Mean Final Distance (m)':<35} {rl_arrival_data['distances'].mean():>19.3f} {baseline_arrival_data['distances'].mean():>19.3f} {pid_arrival_data['distances'].mean():>19.3f}")
    logger.info("=" * 90)



def main():
    logger.info("=" * 70)
    logger.info("Comparative Experiment: RL vs PID+Greedy Baseline")
    logger.info("=" * 70)
    
    # Configure environment
    task_config.num_envs = NUM_ENVS
    task_config.headless = True
    task_config.device = "cuda:0"
    task_config.seed = RANDOM_SEED
    task_config.episode_len_steps = EPISODE_LENGTH
    
    # Initialize environment
    logger.info(f"Initializing environment (num_envs={NUM_ENVS}, seed={RANDOM_SEED})...")
    
    # CRITICAL FIX: Match training obstacle density (Training=11 vs Default=19)
    # USER REQUEST: Increased obstacle density to 20
    print(f"Forcing num_obstacles_in_env to 20 (was {task_config.num_obstacles_in_env})")
    task_config.num_obstacles_in_env = 20

    # Override Asset Params directly to bypass env_manager defaults
    # Total needed = 20. We have 6 walls always.
    # So we need 20 - 6 = 14 panels.
    print(f"Overriding Asset Params: Panels=14, Objects=0 (Total 14 panels + 6 walls = 20)")
    panel_asset_params.num_assets = 14
    object_asset_params.num_assets = 0
    
    env = NavigationTaskGmmNoise(task_config)
    env.reset()
    
    # Spawn random obstacles near targets (VERIFICATION ONLY)
    spawn_target_area_obstacles(env)  # [ENABLED] Spawn obstacles within 1m of target
    
    # Save initial state snapshot
    logger.info("Saving initial state snapshot...")
    initial_snapshot = save_state_snapshot(env)
    
    # === PHASE 1: RL Evaluation ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Evaluating RL (PPO) Model")
    logger.info("=" * 70)
    
    ppo_model, running_mean_std = load_ppo_model(PPO_CHECKPOINT, env)
    
    # Compute normalization if not in checkpoint
    if running_mean_std is None:
        logger.info("Computing normalization statistics from environment...")
        # Save current state before computing stats
        temp_snapshot = save_state_snapshot(env)
        running_mean_std = compute_normalization_stats(env, num_samples=2000)
        # Restore state after computation
        env.reset()
        restore_state_snapshot(env, temp_snapshot)
        logger.info("✅ Normalization statistics computed and environment state restored")
    
    rl_j_curve, rl_crash_data, rl_survival_data, rl_arrival_data, rl_attitude_data, rl_distance_curve, rl_min_j, rl_min_dist, rl_actions = run_phase(
        env, ppo_model, EPISODE_LENGTH, "RL", running_mean_std
    )
    
    # Save RL raw data
    rl_csv = os.path.join(OUTPUT_DIR, "rl_j_curve_raw.csv")
    np.savetxt(rl_csv, rl_j_curve, delimiter=',')
    logger.info(f"Saved RL raw data to: {rl_csv}")
    
    rl_npy = os.path.join(OUTPUT_DIR, "rl_j_curve.npy")
    np.save(rl_npy, rl_j_curve)
    
    # === PHASE 2: Baseline Evaluation ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Evaluating PID + Greedy Baseline")
    logger.info("=" * 70)
    
    # Reset environment and restore initial state
    logger.info("Restoring initial state...")
    env.reset()
    restore_state_snapshot(env, initial_snapshot)
    baseline_agent = LocalAPFAgent(
        env, 
        w_dist=W_D, 
        w_noise=W_N, 
        w_obs=1.0, 
        d_obs=1.0,  # Decreased observation radius (Aggressive)
        max_speed_xy=0.8 # Increased speed (Aggressive)
    )
    baseline_j_curve, baseline_crash_data, baseline_survival_data, baseline_arrival_data, baseline_attitude_data, baseline_distance_curve, baseline_min_j, baseline_min_dist, baseline_actions = run_phase(
        env, baseline_agent, EPISODE_LENGTH, "PID + Greedy"
    )
    
    # Save Baseline raw data
    baseline_csv = os.path.join(OUTPUT_DIR, "baseline_j_curve_raw.csv")
    np.savetxt(baseline_csv, baseline_j_curve, delimiter=',')
    logger.info(f"Saved Baseline raw data to: {baseline_csv}")
    
    baseline_npy = os.path.join(OUTPUT_DIR, "baseline_j_curve.npy")
    np.save(baseline_npy, baseline_j_curve)
    
    # === PHASE 3: Pure PID Evaluation ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Evaluating Pure PID Navigation")
    logger.info("=" * 70)
    
    # Reset environment and restore initial state
    logger.info("Restoring initial state...")
    env.reset()
    restore_state_snapshot(env, initial_snapshot)
    
    pid_agent = SimplePIDNavigator(env, w_dist=W_D, w_obs=1.0, d_obs=1.0)
    pid_j_curve, pid_crash_data, pid_survival_data, pid_arrival_data, pid_attitude_data, pid_distance_curve, pid_min_j, pid_min_dist, pid_actions = run_phase(
        env, pid_agent, EPISODE_LENGTH, "Pure PID"
    )
    
    # Save Pure PID raw data
    pid_csv = os.path.join(OUTPUT_DIR, "pid_j_curve_raw.csv")
    np.savetxt(pid_csv, pid_j_curve, delimiter=',')
    logger.info(f"Saved Pure PID raw data to: {pid_csv}")
    
    pid_npy = os.path.join(OUTPUT_DIR, "pid_j_curve.npy")
    np.save(pid_npy, pid_j_curve)

    
    # === AGGREGATION AND PLOTTING ===
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATION AND VISUALIZATION")
    logger.info("=" * 70)
    
    aggregate_and_plot(rl_j_curve, baseline_j_curve, pid_j_curve)
    plot_crash_rate(rl_crash_data, baseline_crash_data, pid_crash_data)
    plot_distance_to_target(rl_distance_curve, baseline_distance_curve, pid_distance_curve)
    plot_fair_min_metrics_distribution(rl_min_j, baseline_min_j, pid_min_j, rl_min_dist, baseline_min_dist, pid_min_dist)
    
    # 7. Plot Raw Motor Commands (Env 0)
    plot_raw_motor_commands(rl_actions, baseline_actions, pid_actions, env_idx=0)
    
    # Save arrival rates to CSVime(rl_survival_data, baseline_survival_data, pid_survival_data, rl_arrival_data, baseline_arrival_data, pid_arrival_data, rl_crash_data, baseline_crash_data, pid_crash_data, rl_j_curve, baseline_j_curve, pid_j_curve)
    plot_survival_time(rl_survival_data, baseline_survival_data, pid_survival_data, rl_arrival_data, baseline_arrival_data, pid_arrival_data, rl_crash_data, baseline_crash_data, pid_crash_data, rl_j_curve, baseline_j_curve, pid_j_curve)
    plot_attitude_stability(
        rl_attitude_data,
        baseline_attitude_data,
        pid_attitude_data,
        metric="mean_abs",
        filename_prefix="attitude_mean_abs_comparison",
        y_label="Mean Abs Angle (deg)",
    )
    plot_attitude_stability(
        rl_attitude_data,
        baseline_attitude_data,
        pid_attitude_data,
        metric="std",
        filename_prefix="attitude_std_comparison",
        y_label="Angle Std (deg)",
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("Experiment completed successfully!")
    logger.info(f"All results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
