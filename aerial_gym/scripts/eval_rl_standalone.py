"""
Standalone RL Model Evaluation Script

Evaluates the trained PPO model independently to verify performance
without comparison test complexities. Generates:
- J(p) convergence curve
- Cumulative crash rate 
- Average survival time

Author: Antigravity AI
Date: 2026-01-19
"""

import os
import sys
import numpy as np
import csv

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.config.task_config.navigation_task_gmm_noise_config import task_config
from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import NavigationTaskGmmNoise

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

logger = CustomLogger("eval_rl_standalone")

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

# ============================================================================
# Configuration
# ============================================================================

PPO_CHECKPOINT = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gmm_noise_run_18-21-28-26/nn/last_gmm_noise_run_ep_4950_rew_69.69058.pth"

NUM_ENVS = 100
EPISODE_LENGTH = 1200
RANDOM_SEED = 42

# J(p) parameters
W_D = 1.0
W_N = 1.0
D_0 = 2.0

# EMA smoothing
EMA_ALPHA = 0.99

OUTPUT_DIR = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/rl_standalone_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Utility Functions
# ============================================================================

def compute_J_potential(env):
    """Compute unified potential J(p) = w_d * (d/d0)^2 + w_n * n_hat"""
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


def load_ppo_model(checkpoint_path, env):
    """Load trained PPO model from checkpoint"""
    logger.info(f"Loading PPO model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        model_state = checkpoint
    
    import torch.nn as nn
    
    obs_dim = env.task_config.observation_space_dim
    act_dim = env.task_config.action_space_dim
    
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
            return torch.tanh(self.actor_mlp(obs))
    
    model = PPOActor(obs_dim, act_dim).to(env.device)
    
    # Load only actor weights
    actor_state = {}
    for key, value in model_state.items():
        if 'a2c_network.actor_mlp' in key:
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
            std=rms_data['running_var'] ** 0.5,  # Convert variance to std
            count=rms_data['count'],
            device=env.device
        )
        logger.info("Loaded running_mean_std for input normalization")
    else:
        logger.warning("No running_mean_std found in checkpoint - using raw observations")
    
    logger.info("PPO model loaded successfully")
    return model, running_mean_std


def run_evaluation(env, model, running_mean_std, episode_length):
    """
    Run RL model evaluation
    
    Args:
        running_mean_std: RunningMeanStd object for input normalization (can be None)
    
    Returns:
        j_curve: (episode_length, num_envs) numpy array
        crash_data: dict with cumulative_rate
        survival_data: dict with mean/std per step
    """
    logger.info(f"Starting RL evaluation for {episode_length} steps...")
    
    j_curve = np.zeros((episode_length, NUM_ENVS))
    
    # Track crashes
    has_crashed = np.zeros(NUM_ENVS, dtype=bool)
    crash_count_per_step = np.zeros(episode_length)
    
    # Track survival times
    survival_times = []
    env_last_reset_step = np.zeros(NUM_ENVS, dtype=int)
    
    for step in range(episode_length):
        # Compute action
        obs = env.task_obs["observations"]
        
        # Apply normalization if available
        if running_mean_std is not None:
            normalized_obs = running_mean_std.normalize(obs)
        else:
            normalized_obs = obs
        
        with torch.no_grad():
            action = model(normalized_obs)
            if isinstance(action, tuple):
                action = action[0]
        
        # Step environment
        _, _, _, _, _ = env.step(action)
        
        # Compute J(p)
        J_values = compute_J_potential(env)
        j_curve[step, :] = J_values.cpu().numpy()
        
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
            logger.info(f"  Step {step}/{episode_length} | Mean J = {J_values.mean().item():.4f} | Crashes: {int(crash_count_per_step[step])}/{NUM_ENVS}")
    
    logger.info(f"Evaluation completed. Final Mean J = {j_curve[-1].mean():.4f} | Total Crashed: {int(crash_count_per_step[-1])}/{NUM_ENVS}")
    
    # Prepare crash data
    crash_data = {
        'cumulative_rate': (crash_count_per_step / NUM_ENVS) * 100,
    }
    
    # Prepare survival data
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
    
    return j_curve, crash_data, survival_data


def compute_ema(data, alpha=EMA_ALPHA):
    """Compute exponential moving average"""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * ema[t-1] + (1 - alpha) * data[t]
    return ema


def plot_j_curve(j_curve):
    """Plot J(p) convergence curve"""
    logger.info("Generating J(p) curve plot...")
    
    mean = j_curve.mean(axis=1)
    std = j_curve.std(axis=1)
    ema = compute_ema(mean)
    
    # Configure matplotlib
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    
    fig_width_cm = 8.5
    fig_height_cm = 6.5
    fig, ax = plt.subplots(figsize=(fig_width_cm/2.54, fig_height_cm/2.54))
    
    timesteps = np.arange(EPISODE_LENGTH)
    color = '#1f77b4'
    
    ax.plot(timesteps, ema, color=color, linewidth=1.8, linestyle='-', label='RL Model')
    ax.fill_between(timesteps, mean - std, mean + std, color=color, alpha=0.25)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Unified Potential $J(p)$')
    ax.set_title('RL Model J-Curve (Standalone Evaluation)')
    ax.legend(loc='best', frameon=True, fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim([0, EPISODE_LENGTH])
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "rl_standalone_j_curve.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved J-curve plot to: {png_path}")
    
    plt.close()
    
    # Save stats
    csv_path = os.path.join(OUTPUT_DIR, "rl_standalone_j_stats.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'mean', 'std', 'ema'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, mean[t], std[t], ema[t]])
    logger.info(f"Saved J-curve stats to: {csv_path}")
    
    logger.info(f"Final EMA J-value: {ema[-1]:.4f}")


def plot_crash_rate(crash_data):
    """Plot cumulative crash rate"""
    logger.info("Generating crash rate plot...")
    
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    
    fig_width_cm = 8.5
    fig_height_cm = 6.5
    fig, ax = plt.subplots(figsize=(fig_width_cm/2.54, fig_height_cm/2.54))
    
    timesteps = np.arange(EPISODE_LENGTH)
    color = '#1f77b4'
    
    ax.plot(timesteps, crash_data['cumulative_rate'], color=color, linewidth=1.8, linestyle='-', label='RL Model')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Crash Rate (%)')
    ax.set_title('RL Model Crash Rate (Standalone Evaluation)')
    ax.legend(loc='best', frameon=True, fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim([0, EPISODE_LENGTH])
    ax.set_ylim([0, min(105, crash_data['cumulative_rate'][-1] + 5)])
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "rl_standalone_crash_rate.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved crash rate plot to: {png_path}")
    
    plt.close()
    
    # Save stats
    csv_path = os.path.join(OUTPUT_DIR, "rl_standalone_crash_stats.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'crash_rate'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, crash_data['cumulative_rate'][t]])
    logger.info(f"Saved crash rate stats to: {csv_path}")
    
    logger.info(f"Final crash rate: {crash_data['cumulative_rate'][-1]:.2f}%")


def plot_survival_time(survival_data):
    """Plot average survival time"""
    logger.info("Generating survival time plot...")
    
    mean_ema = compute_ema(survival_data['mean_per_step'])
    
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    
    fig_width_cm = 8.5
    fig_height_cm = 6.5
    fig, ax = plt.subplots(figsize=(fig_width_cm/2.54, fig_height_cm/2.54))
    
    timesteps = np.arange(EPISODE_LENGTH)
    color = '#1f77b4'
    
    ax.plot(timesteps, mean_ema, color=color, linewidth=1.8, linestyle='-', label='RL Model')
    ax.fill_between(timesteps, 
                    survival_data['mean_per_step'] - survival_data['std_per_step'], 
                    survival_data['mean_per_step'] + survival_data['std_per_step'], 
                    color=color, alpha=0.25)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Average Survival Time (steps)')
    ax.set_title('RL Model Survival Time (Standalone Evaluation)')
    ax.legend(loc='best', frameon=True, fancybox=False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim([0, EPISODE_LENGTH])
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "rl_standalone_survival_time.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    logger.info(f"Saved survival time plot to: {png_path}")
    
    plt.close()
    
    # Save stats
    csv_path = os.path.join(OUTPUT_DIR, "rl_standalone_survival_stats.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'mean', 'std', 'ema'])
        for t in range(EPISODE_LENGTH):
            writer.writerow([t, survival_data['mean_per_step'][t], survival_data['std_per_step'][t], mean_ema[t]])
    logger.info(f"Saved survival time stats to: {csv_path}")


def main():
    logger.info("=" * 70)
    logger.info("Standalone RL Model Evaluation")
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
    
    # Load model
    model, running_mean_std = load_ppo_model(PPO_CHECKPOINT, env)
    
    # Run evaluation
    logger.info("\n" + "=" * 70)
    logger.info("Running Evaluation")
    logger.info("=" * 70)
    
    j_curve, crash_data, survival_data = run_evaluation(env, model, running_mean_std, EPISODE_LENGTH)
    
    # Save raw data
    j_npy = os.path.join(OUTPUT_DIR, "rl_standalone_j_curve.npy")
    np.save(j_npy, j_curve)
    logger.info(f"Saved raw J-curve data to: {j_npy}")
    
    # Generate plots
    logger.info("\n" + "=" * 70)
    logger.info("Generating Plots")
    logger.info("=" * 70)
    
    plot_j_curve(j_curve)
    plot_crash_rate(crash_data)
    plot_survival_time(survival_data)
    
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation completed successfully!")
    logger.info(f"All results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
