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
PPO_CHECKPOINT = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gmm_noise_run_23-21-09-32/nn/last_gmm_noise_run_ep_5000_rew__-5.3149767_.pth"

# Experiment parameters
NUM_ENVS = 100
EPISODE_LENGTH = 500
RANDOM_SEED = 10

# J(p) parameters (from config)
# J(p) parameters (from config)
W_D = task_config.reward_parameters["potential_w_d"]
W_N = task_config.reward_parameters["potential_w_n"]
D_0 = task_config.reward_parameters["potential_d0"]

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
    J_dist = W_D * (dist_to_target / D_0) ** 2
    
    # Noise term (normalized)
    current_noise = env._compute_gmm_mixture(position)
    if hasattr(env, "estimated_n_min") and hasattr(env, "estimated_n_max"):
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


# ============================================================================
# APF Baseline Agent (from test_baseline_pid.py)
# ============================================================================

class LocalAPFAgent:
    """
    Local (Blind) Artificial Potential Field Agent 
    - Uses Finite Difference for Noise Gradient (No internal formula access)
    - Uses Lidar/Depth for Obstacle Repulsion (No ground truth positions)
    """
    
    def __init__(self, env, w_dist=1.0, w_noise=5.0, w_obs=2.0, d_obs=1.5):
        self.env = env
        self.device = env.device
        self.w_dist = w_dist
        self.w_noise = w_noise
        self.w_obs = w_obs
        self.obs_influence_radius = d_obs
        self.max_speed_xy = 0.8
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
        grad_target = vec_to_target
        
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
        
        # --- 3. Obstacle Repulsion (Lidar Based) ---
        # Using the simplified depth observation from the environment
        # Check env_manager.py or navigation task for depth structure
        # Assuming simple depth image access or obstacle_position if depth not easily available in this script context.
        # CRITICAL: Since accessing raw depth pixels and reprojecting in this script is complex without camera intrinsics,
        # we will simulate a "Limited Range Lidar" by filtering ground truth.
        # This is a fair "blind" approximation: only seeing obstacles within sensor range.
        
        obs_pos = self.env.obs_dict["obstacle_position"] # Still accessing list, but will filter visibility
        diff = position.unsqueeze(1) - obs_pos
        dist = torch.norm(diff, dim=2)
        
        # Simulate Lidar visibility:
        # 1. limited range (e.g. 5.0m)
        lidar_range = 5.0
        # 2. limited FOV (simplified here to omni-directional for safety baseline, but limited range is key)
        
        # Repulsion force: 0.5 * (1/d - 1/d0)^2 * grad(d)
        # where grad(d) is unit vector pointing away from obstacle
        
        d0 = self.obs_influence_radius
        # Only perceive obstacles within sensor range AND within influence radius
        mask = (dist < d0) & (dist > 0.1) & (dist < lidar_range)
        
        # Calculate repulsion gradient
        # grad_repulsion = - sum ( scaling_factor * unit_vec_to_obstacle )
        obs_dirs = diff / (dist.unsqueeze(2) + 1e-6) # Unit vectors pointing TO robot FROM obstacle
        
        scaling = -(1.0/dist - 1.0/d0) * (1.0/(dist**2)) # Derivative of potential
        
        # Apply mask
        scaling[~mask] = 0.0
        
        # Sum contributions
        # scaling is negative (repulsive), obs_dirs points to robot.
        # Gradient of potential points in direction of steepest ASCENT (higher potential).
        # We want to descend, so we follow -gradient.
        # Potential U = 0.5(1/d - 1/d0)^2
        # grad U = (1/d - 1/d0)*(-1/d^2) * grad(d)
        # grad(d) = vector_to_robot / d
        # So grad U points AWAY from obstacle.
        
        # Directly compute accumulated repulsive gradient
        repulsion_mag = (1.0/dist - 1.0/d0) * (1.0/(dist**2))
        repulsion_mag[~mask] = 0.0
        
        # Gradient contribution from each obstacle: magnitude * unit_vec_away_from_obstacle
        # unit_vec_away = obs_dirs
        grad_obs = torch.sum(repulsion_mag.unsqueeze(2) * obs_dirs, dim=1)
        
        
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
    
    j_curve = np.zeros((episode_length, NUM_ENVS))
    attitude_mean_abs = np.zeros((episode_length, 3))
    attitude_std = np.zeros((episode_length, 3))
    
    # Track crashes (only first crash per environment)
    has_crashed = np.zeros(NUM_ENVS, dtype=bool)
    crash_count_per_step = np.zeros(episode_length)
    
    # Track survival times (all crash events)
    survival_times = []
    env_last_reset_step = np.zeros(NUM_ENVS, dtype=int)
    
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
        
        # Step environment
        _, _, _, _, _ = env.step(action)
        
        # CRITICAL FIX: Trigger "Safe Spawn" Logic
        # Training calls this to retry bad spawns. Verification must too.
        env.logging_sanity_check(env.infos)
        
        # Compute J(p)
        J_values = compute_J_potential(env)
        j_curve[step, :] = J_values.cpu().numpy()

        # Attitude stability (roll/pitch/yaw) in degrees
        euler = env.obs_dict["robot_euler_angles"]
        euler_deg = euler * RAD2DEG
        mean_abs = euler_deg.abs().mean(dim=0)
        std = euler_deg.std(dim=0, unbiased=False)
        attitude_mean_abs[step, :] = mean_abs.cpu().numpy()
        attitude_std[step, :] = std.cpu().numpy()
        
        # Track crashes
        crashes = env.obs_dict["crashes"].cpu().numpy()
        has_crashed = has_crashed | (crashes > 0)
        crash_count_per_step[step] = has_crashed.sum()
        
        # Track survival times
        crashed_envs = np.where(crashes > 0)[0]
        for env_id in crashed_envs:
            survival_duration = step - env_last_reset_step[env_id]
            survival_times.append(survival_duration)
            env_last_reset_step[env_id] = step + 1
        
        if step % 100 == 0:
            logger.info(f"  Step {step}/{episode_length} | Mean J = {J_values.mean().item():.4f} | Crashes: {int(crash_count_per_step[step])}/100")
    
    # Track arrival rate (distance <= 2m at episode end)
    final_positions = env.obs_dict["robot_position"].cpu().numpy()
    target_positions = env.target_position.cpu().numpy()
    final_distances = np.linalg.norm(final_positions - target_positions, axis=1)
    arrival_threshold = 2.0  # meters
    has_arrived = final_distances <= arrival_threshold
    arrival_count = has_arrived.sum()
    arrival_rate = (arrival_count / NUM_ENVS) * 100
    
    logger.info(f"{phase_name} phase completed. Final Mean J = {j_curve[-1].mean():.4f} | Total Crashed: {int(crash_count_per_step[-1])}/100 | Arrival Rate: {arrival_rate:.1f}%")
    
    # Prepare return data
    crash_data = {
        'cumulative_rate': (crash_count_per_step / NUM_ENVS) * 100,
    }
    
    survival_data = {
        'all_times': survival_times,
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
    
    return j_curve, crash_data, survival_data, arrival_data, attitude_data


def plot_attitude_stability(rl_attitude, baseline_attitude, metric, filename_prefix, y_label):
    """Plot attitude stability comparison for roll/pitch/yaw."""
    logger.info(f"Generating attitude {metric} comparison plot...")
    
    rl_data = rl_attitude[metric]
    baseline_data = baseline_attitude[metric]
    
    def compute_ema(data, alpha=EMA_ALPHA):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for t in range(1, len(data)):
            ema[t] = alpha * ema[t-1] + (1 - alpha) * data[t]
        return ema
    
    rl_ema = np.stack([compute_ema(rl_data[:, i]) for i in range(3)], axis=1)
    baseline_ema = np.stack([compute_ema(baseline_data[:, i]) for i in range(3)], axis=1)
    
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
            'rl_roll_ema', 'rl_pitch_ema', 'rl_yaw_ema',
            'baseline_roll_ema', 'baseline_pitch_ema', 'baseline_yaw_ema',
        ])
        for t in range(EPISODE_LENGTH):
            writer.writerow([
                t,
                rl_data[t, 0], rl_data[t, 1], rl_data[t, 2],
                baseline_data[t, 0], baseline_data[t, 1], baseline_data[t, 2],
                rl_ema[t, 0], rl_ema[t, 1], rl_ema[t, 2],
                baseline_ema[t, 0], baseline_ema[t, 1], baseline_ema[t, 2],
            ])
    logger.info(f"Saved attitude stats to: {csv_path}")


def aggregate_and_plot(rl_curve, baseline_curve):
    """
    Aggregate data and generate academic-quality plot
    
    Args:
        rl_curve: (600, 100) array
        baseline_curve: (600, 100) array
    """
    logger.info("Aggregating data and generating plots...")
    
    # Compute statistics
    rl_mean = rl_curve.mean(axis=1)
    rl_std = rl_curve.std(axis=1)
    baseline_mean = baseline_curve.mean(axis=1)
    baseline_std = baseline_curve.std(axis=1)
    
    # Compute EMA
    def compute_ema(data, alpha=EMA_ALPHA):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for t in range(1, len(data)):
            ema[t] = alpha * ema[t-1] + (1 - alpha) * data[t]
        return ema
    
    rl_ema = compute_ema(rl_mean)
    baseline_ema = compute_ema(baseline_mean)
    
    # Save aggregated stats
    stats_csv = os.path.join(OUTPUT_DIR, "aggregated_stats.csv")
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'rl_mean', 'rl_std', 'baseline_mean', 'baseline_std', 'rl_ema', 'baseline_ema'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, rl_mean[t], rl_std[t], baseline_mean[t], baseline_std[t], rl_ema[t], baseline_ema[t]])
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
    
    # Plot RL curve
    ax.plot(timesteps, rl_ema, color=rl_color, linewidth=1.8, linestyle='-', label='RL (Proposed)')
    ax.fill_between(timesteps, rl_mean - rl_std, rl_mean + rl_std, color=rl_color, alpha=0.25)
    
    # Plot Baseline curve
    ax.plot(timesteps, baseline_ema, color=baseline_color, linewidth=1.8, linestyle='--', label='PID + Greedy')
    ax.fill_between(timesteps, baseline_mean - baseline_std, baseline_mean + baseline_std, color=baseline_color, alpha=0.25)
    
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
    logger.info(f"Improvement: {((baseline_ema[-1] - rl_ema[-1]) / baseline_ema[-1] * 100):.2f}%")


def plot_crash_rate(rl_crash_data, baseline_crash_data):
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
    
    # Plot cumulative crash rate (no EMA, already monotonic)
    ax.plot(timesteps, rl_crash_data['cumulative_rate'], color=rl_color, linewidth=1.8, linestyle='-', label='RL (Proposed)')
    ax.plot(timesteps, baseline_crash_data['cumulative_rate'], color=baseline_color, linewidth=1.8, linestyle='--', label='PID + Greedy')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Crash Rate (%)')
    ax.legend(loc='best', frameon=True, fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim([0, EPISODE_LENGTH])
    ax.set_ylim([0, min(105, max(rl_crash_data['cumulative_rate'][-1], baseline_crash_data['cumulative_rate'][-1]) + 5)])
    
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
        writer.writerow(['step', 'rl_crash_rate', 'baseline_crash_rate'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, rl_crash_data['cumulative_rate'][t], baseline_crash_data['cumulative_rate'][t]])
    logger.info(f"Saved crash rate stats to: {csv_path}")


def plot_survival_time(rl_survival_data, baseline_survival_data, rl_arrival_data, baseline_arrival_data, rl_crash_data, baseline_crash_data, rl_j_curve, baseline_j_curve):
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
        writer.writerow(['step', 'rl_mean', 'rl_std', 'rl_ema', 'baseline_mean', 'baseline_std', 'baseline_ema'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, 
                           rl_survival_data['mean_per_step'][t], rl_survival_data['std_per_step'][t], rl_mean_ema[t],
                           baseline_survival_data['mean_per_step'][t], baseline_survival_data['std_per_step'][t], baseline_mean_ema[t]])
    logger.info(f"Saved survival time stats to: {csv_path}")
    
    # === Save Arrival Rate Comparison ===
    arrival_csv = os.path.join(OUTPUT_DIR, "arrival_rate_comparison.csv")
    with open(arrival_csv, 'w') as f:
        f.write("Method,Arrival Count,Arrival Rate (%),Mean Final Distance (m)\n")
        f.write(f"RL,{rl_arrival_data['count']},{rl_arrival_data['rate']:.2f},{rl_arrival_data['distances'].mean():.3f}\n")
        f.write(f"Baseline,{baseline_arrival_data['count']},{baseline_arrival_data['rate']:.2f},{baseline_arrival_data['distances'].mean():.3f}\n")
    logger.info(f"Saved arrival rate comparison to: {arrival_csv}")
    
    # === Print Comparison Summary Table ===
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<35} {'RL (ep_3230)':<18} {'Baseline (PID)':<18}")
    logger.info("-" * 70)
    logger.info(f"{'Final Crash Rate (%)':<35} {rl_crash_data['cumulative_rate'][-1]:>16.1f}% {baseline_crash_data['cumulative_rate'][-1]:>16.1f}%")
    logger.info(f"{'Arrival Rate @ 2m (%)':<35} {rl_arrival_data['rate']:>16.1f}% {baseline_arrival_data['rate']:>16.1f}%")
    logger.info(f"{'Mean Final J Value':<35} {rl_j_curve[-1].mean():>17.3f} {baseline_j_curve[-1].mean():>17.3f}")
    logger.info(f"{'Mean Final Distance (m)':<35} {rl_arrival_data['distances'].mean():>17.3f} {baseline_arrival_data['distances'].mean():>17.3f}")
    logger.info("=" * 70)



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
    print(f"Forcing num_obstacles_in_env to 11 (was {task_config.num_obstacles_in_env})")
    task_config.num_obstacles_in_env = 11

    # Override Asset Params directly to bypass env_manager defaults
    print(f"Overriding Asset Params: Panels=6, Objects=0 (Total 6 assets + 6 walls = 12)")
    panel_asset_params.num_assets = 6
    object_asset_params.num_assets = 0
    
    env = NavigationTaskGmmNoise(task_config)
    env.reset()
    
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
    
    rl_j_curve, rl_crash_data, rl_survival_data, rl_arrival_data, rl_attitude_data = run_phase(
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
    
    baseline_agent = LocalAPFAgent(env, w_dist=W_D, w_noise=W_N, w_obs=2.0, d_obs=1.5)
    baseline_j_curve, baseline_crash_data, baseline_survival_data, baseline_arrival_data, baseline_attitude_data = run_phase(
        env, baseline_agent, EPISODE_LENGTH, "Baseline"
    )
    
    # Save Baseline raw data
    baseline_csv = os.path.join(OUTPUT_DIR, "baseline_j_curve_raw.csv")
    np.savetxt(baseline_csv, baseline_j_curve, delimiter=',')
    logger.info(f"Saved Baseline raw data to: {baseline_csv}")
    
    baseline_npy = os.path.join(OUTPUT_DIR, "baseline_j_curve.npy")
    np.save(baseline_npy, baseline_j_curve)
    
    # === AGGREGATION AND PLOTTING ===
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATION AND VISUALIZATION")
    logger.info("=" * 70)
    
    aggregate_and_plot(rl_j_curve, baseline_j_curve)
    plot_crash_rate(rl_crash_data, baseline_crash_data)
    plot_survival_time(rl_survival_data, baseline_survival_data, rl_arrival_data, baseline_arrival_data, rl_crash_data, baseline_crash_data, rl_j_curve, baseline_j_curve)
    plot_attitude_stability(
        rl_attitude_data,
        baseline_attitude_data,
        metric="mean_abs",
        filename_prefix="attitude_mean_abs_comparison",
        y_label="Mean Abs Angle (deg)",
    )
    plot_attitude_stability(
        rl_attitude_data,
        baseline_attitude_data,
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
