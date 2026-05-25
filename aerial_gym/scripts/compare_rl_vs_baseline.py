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
from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import NavigationTaskGmmNoise

import torch
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib import rcParams

logger = CustomLogger("compare_rl_vs_baseline")

# ============================================================================
# Configuration
# ============================================================================

# Model path
PPO_CHECKPOINT = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gmm_noise_run_18-21-28-26/nn/last_gmm_noise_run_ep_4950_rew_69.69058.pth"

# Experiment parameters
NUM_ENVS = 100
EPISODE_LENGTH = 600
RANDOM_SEED = 42

# J(p) parameters (from config)
W_D = 1.0
W_N = 1.0
D_0 = 2.0

# EMA smoothing
EMA_ALPHA = 0.99

# Output paths
OUTPUT_DIR = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# APF Baseline Agent (legacy PID + greedy baseline)
# ============================================================================

class APFAgent:
    """Artificial Potential Field Agent (PID + Greedy Gradient Descent)"""
    
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
    
    def compute_action(self, obs_dict):
        """Compute velocity command based on -grad(Total_Potential)"""
        position = obs_dict["robot_position"].clone().detach().requires_grad_(True)
        
        # 1. Target attraction
        target_pos = self.env.target_position
        dist_vec = position - target_pos
        dist_sq = (dist_vec ** 2).sum(dim=1)
        J_target = 0.5 * dist_sq
        
        # 2. Noise repulsion
        J_noise = self.env._compute_gmm_mixture(position)
        
        # 3. Obstacle repulsion
        obs_pos = self.env.obs_dict["obstacle_position"]
        diff = position.unsqueeze(1) - obs_pos
        dist = torch.norm(diff, dim=2)
        d0 = self.obs_influence_radius
        mask = (dist < d0) & (dist > 0.05)
        repulsion_term = torch.zeros_like(dist)
        repulsion_term[mask] = 0.5 * (1.0/dist[mask] - 1.0/d0)**2
        J_obs = repulsion_term.sum(dim=1)
        
        # Total potential
        J_total = (self.w_dist * J_target) + (self.w_noise * J_noise) + (self.w_obs * J_obs)
        
        # Compute gradient
        grad_sum = torch.sum(J_total)
        grad_sum.backward()
        gradients = position.grad
        
        # Descent direction
        vel_cmd = -self.grad_to_vel_gain * gradients
        vel_cmd[:, 0:2] = torch.clamp(vel_cmd[:, 0:2], -self.max_speed_xy, self.max_speed_xy)
        vel_cmd[:, 2] = torch.clamp(vel_cmd[:, 2], -self.max_speed_z, self.max_speed_z)
        
        # Normalize to action space
        action = torch.zeros((self.env.sim_env.num_envs, 4), device=self.device)
        action[:, 0] = vel_cmd[:, 0] / self.max_speed_xy
        action[:, 1] = vel_cmd[:, 1] / self.max_speed_xy
        action[:, 2] = vel_cmd[:, 2] / self.max_speed_z
        action[:, 3] = 0.0
        
        action = torch.clamp(action, -1.0, 1.0)
        return action


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
            return torch.tanh(self.actor_mlp(obs))  # Tanh for [-1, 1] action space
    
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
    
    logger.info("PPO model loaded successfully")
    return model


# ============================================================================
# Main Experiment Loop
# ============================================================================

def run_phase(env, agent, episode_length, phase_name):
    """
    Run one evaluation phase (RL or Baseline)
    
    Returns:
        j_curve: (episode_length, num_envs) numpy array of J values
    """
    logger.info(f"Starting {phase_name} phase for {episode_length} steps...")
    
    j_curve = np.zeros((episode_length, NUM_ENVS))
    
    for step in range(episode_length):
        # Compute action
        if phase_name == "RL":
            obs = env.task_obs["observations"]
            with torch.no_grad():
                action = agent(obs)  # Model outputs actions directly
                # Handle different output formats
                if isinstance(action, tuple):
                    action = action[0]
        else:  # Baseline
            action = agent.compute_action(env.obs_dict)
        
        # Step environment
        _, _, _, _, _ = env.step(action)
        
        # Compute J(p)
        J_values = compute_J_potential(env)
        j_curve[step, :] = J_values.cpu().numpy()
        
        if step % 100 == 0:
            logger.info(f"  Step {step}/{episode_length} | Mean J = {J_values.mean().item():.4f}")
    
    logger.info(f"{phase_name} phase completed. Final Mean J = {j_curve[-1].mean():.4f}")
    return j_curve


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
    env = NavigationTaskGmmNoise(task_config)
    env.reset()
    
    # Save initial state snapshot
    logger.info("Saving initial state snapshot...")
    initial_snapshot = save_state_snapshot(env)
    
    # === PHASE 1: RL Evaluation ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Evaluating RL (PPO) Model")
    logger.info("=" * 70)
    
    ppo_model = load_ppo_model(PPO_CHECKPOINT, env)
    rl_j_curve = run_phase(env, ppo_model, EPISODE_LENGTH, "RL")
    
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
    
    baseline_agent = APFAgent(env, w_dist=1.0, w_noise=5.0, w_obs=2.0, d_obs=1.5)
    baseline_j_curve = run_phase(env, baseline_agent, EPISODE_LENGTH, "Baseline")
    
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
    
    logger.info("\n" + "=" * 70)
    logger.info("Experiment completed successfully!")
    logger.info(f"All results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
