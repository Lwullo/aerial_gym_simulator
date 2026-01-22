"""
Sample Size Effect Verification Script

Runs num_envs=1 for 100 independent trials to verify whether the perceived
good performance in inference mode is just a statistical artifact of small sample size.

This will show the distribution of crash times and compare with the 100-env parallel evaluation.

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

logger = CustomLogger("verify_sample_size")

# ============================================================================
# Configuration
# ============================================================================

PPO_CHECKPOINT = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gmm_noise_run_18-21-28-26/nn/last_gmm_noise_run_ep_4950_rew_69.69058.pth"

NUM_TRIALS = 100  # Run 100 independent single-env episodes
EPISODE_LENGTH = 1200
BASE_SEED = 42

# J(p) parameters
W_D = 1.0
W_N = 1.0
D_0 = 2.0

OUTPUT_DIR = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/sample_size_verification"
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

# ============================================================================
# Utility Functions
# ============================================================================

def compute_J_potential(env):
    """Compute unified potential J(p)"""
    position = env.obs_dict["robot_position"]
    dist_to_target = torch.norm(position - env.target_position, dim=1)
    J_dist = W_D * (dist_to_target / D_0) ** 2
    
    current_noise = env._compute_gmm_mixture(position)
    if hasattr(env, "estimated_n_min") and hasattr(env, "estimated_n_max"):
        n_hat = (current_noise - env.estimated_n_min) / (env.estimated_n_max - env.estimated_n_min + 1e-6)
        n_hat = torch.clamp(n_hat, 0.0, 1.0)
    else:
        n_hat = current_noise
    
    J_noise = W_N * n_hat
    J_total = J_dist + J_noise
    
    return J_total


def load_ppo_model(checkpoint_path, env):
    """Load trained PPO model"""
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
            std=rms_data['running_var'] ** 0.5,
            count=rms_data['count'],
            device=env.device
        )
        logger.info("Loaded running_mean_std for input normalization")
    else:
        logger.warning("No running_mean_std found in checkpoint - using raw observations")
    
    logger.info("PPO model loaded successfully")
    return model, running_mean_std


def run_single_episode(env, model, running_mean_std, episode_length, trial_idx):
    """
    Run one episode with num_envs=1
    
    Args:
        running_mean_std: Normalization parameters (can be None)
    
    Returns:
        crash_step: step at which crash occurred (or episode_length if no crash)
        final_j: final J value
        mean_j: mean J value across episode
    """
    j_values = []
    crashed = False
    crash_step = episode_length
    
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
        J_val = compute_J_potential(env).item()
        j_values.append(J_val)
        
        # Check crash
        if env.obs_dict["crashes"][0] > 0 and not crashed:
            crashed = True
            crash_step = step
        
        # Optional: early stop if crashed (to save time)
        # Commenting out to collect full episode data
        # if crashed:
        #     break
    
    final_j = j_values[-1] if j_values else 0.0
    mean_j = np.mean(j_values) if j_values else 0.0
    
    logger.info(f"  Trial {trial_idx+1}/{NUM_TRIALS} | Crashed: {crashed} | Crash Step: {crash_step} | Mean J: {mean_j:.4f}")
    
    return crash_step, final_j, mean_j


def main():
    logger.info("=" * 70)
    logger.info("Sample Size Effect Verification (num_envs=1, 100 trials)")
    logger.info("=" * 70)
    
    # Configure environment for single-env trials
    task_config.num_envs = 1
    task_config.headless = True
    task_config.device = "cuda:0"
    task_config.episode_len_steps = EPISODE_LENGTH
    
    # Initialize environment
    logger.info(f"Initializing environment (num_envs=1, will run {NUM_TRIALS} trials)...")
    env = NavigationTaskGmmNoise(task_config)
    
    # Load model
    model, running_mean_std = load_ppo_model(PPO_CHECKPOINT, env)
    
    # Run trials
    logger.info("\n" + "=" * 70)
    logger.info("Running Trials")
    logger.info("=" * 70)
    
    crash_steps = []
    final_js = []
    mean_js = []
    
    for trial_idx in range(NUM_TRIALS):
        # Reset with different seed
        torch.manual_seed(BASE_SEED + trial_idx)
        np.random.seed(BASE_SEED + trial_idx)
        env.reset()
        
        # Run episode
        crash_step, final_j, mean_j = run_single_episode(env, model, running_mean_std, EPISODE_LENGTH, trial_idx)
        
        crash_steps.append(crash_step)
        final_js.append(final_j)
        mean_js.append(mean_j)
    
    crash_steps = np.array(crash_steps)
    final_js = np.array(final_js)
    mean_js = np.array(mean_js)
    
    # Calculate statistics
    crashed_mask = crash_steps < EPISODE_LENGTH
    crash_rate = crashed_mask.sum() / NUM_TRIALS * 100
    mean_crash_step = crash_steps[crashed_mask].mean() if crashed_mask.any() else EPISODE_LENGTH
    
    logger.info("\n" + "=" * 70)
    logger.info("Results Summary")
    logger.info("=" * 70)
    logger.info(f"Total Trials: {NUM_TRIALS}")
    logger.info(f"Crashed: {crashed_mask.sum()}")
    logger.info(f"Survived (timeout): {(~crashed_mask).sum()}")
    logger.info(f"Crash Rate: {crash_rate:.2f}%")
    logger.info(f"Mean Crash Step (for crashed): {mean_crash_step:.0f}")
    logger.info(f"Mean Final J: {final_js.mean():.4f}")
    logger.info(f"Mean Episode J: {mean_js.mean():.4f}")
    
    # Save raw data
    csv_path = os.path.join(OUTPUT_DIR, "trial_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'crash_step', 'crashed', 'final_j', 'mean_j'])
        for i in range(NUM_TRIALS):
            writer.writerow([i, crash_steps[i], crashed_mask[i], final_js[i], mean_js[i]])
    logger.info(f"\nSaved trial results to: {csv_path}")
    
    # Generate plots
    logger.info("\n" + "=" * 70)
    logger.info("Generating Plots")
    logger.info("=" * 70)
    
    plot_crash_distribution(crash_steps, crashed_mask)
    plot_cumulative_crash_rate(crash_steps)
    plot_comparison_with_parallel()
    
    logger.info("\n" + "=" * 70)
    logger.info("Verification completed!")
    logger.info(f"All results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


def plot_crash_distribution(crash_steps, crashed_mask):
    """Plot histogram of crash steps"""
    logger.info("Generating crash distribution histogram...")
    
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of crash steps (only for crashed episodes)
    crashed_steps = crash_steps[crashed_mask]
    
    bins = np.arange(0, EPISODE_LENGTH + 100, 100)
    ax.hist(crashed_steps, bins=bins, edgecolor='black', alpha=0.7, color='#d62728')
    
    # Add vertical line for mean
    if len(crashed_steps) > 0:
        mean_crash = crashed_steps.mean()
        ax.axvline(mean_crash, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_crash:.0f} steps')
    
    ax.set_xlabel('Crash Step')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Crash Times (num_envs=1, {NUM_TRIALS} trials)\nCrash Rate: {crashed_mask.sum()}/{NUM_TRIALS} = {crashed_mask.sum()/NUM_TRIALS*100:.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "crash_distribution.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved crash distribution to: {png_path}")
    
    plt.close()


def plot_cumulative_crash_rate(crash_steps):
    """Plot cumulative crash rate over time"""
    logger.info("Generating cumulative crash rate plot...")
    
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate cumulative crash rate
    timesteps = np.arange(0, EPISODE_LENGTH + 1)
    cumulative_crashes = np.array([np.sum(crash_steps <= t) for t in timesteps])
    cumulative_rate = (cumulative_crashes / NUM_TRIALS) * 100
    
    ax.plot(timesteps, cumulative_rate, linewidth=2, color='#d62728', label='num_envs=1 (100 trials)')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Crash Rate (%)')
    ax.set_title('Cumulative Crash Rate: Single-Env Trials')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, EPISODE_LENGTH])
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "cumulative_crash_rate.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved cumulative crash rate to: {png_path}")
    
    plt.close()


def plot_comparison_with_parallel():
    """Compare single-env trials with 100-env parallel evaluation"""
    logger.info("Generating comparison plot...")
    
    # Load parallel evaluation data
    parallel_data_path = "/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/rl_standalone_results/rl_standalone_crash_stats.csv"
    
    if not os.path.exists(parallel_data_path):
        logger.warning(f"Parallel evaluation data not found: {parallel_data_path}")
        logger.warning("Skipping comparison plot")
        return
    
    parallel_data = np.loadtxt(parallel_data_path, delimiter=',', skiprows=1)
    parallel_steps = parallel_data[:, 0]
    parallel_crash_rate = parallel_data[:, 1]
    
    # Load single-env trial data
    trial_data_path = os.path.join(OUTPUT_DIR, "trial_results.csv")
    trial_data = np.loadtxt(trial_data_path, delimiter=',', skiprows=1)
    crash_steps = trial_data[:, 1]
    
    # Calculate cumulative crash rate for single-env trials
    timesteps = np.arange(0, EPISODE_LENGTH + 1)
    cumulative_crashes = np.array([np.sum(crash_steps <= t) for t in timesteps])
    single_crash_rate = (cumulative_crashes / NUM_TRIALS) * 100
    
    # Plot comparison
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(timesteps, single_crash_rate, linewidth=2.5, color='#ff7f0e', 
            label=f'num_envs=1 ({NUM_TRIALS} trials)', linestyle='-')
    ax.plot(parallel_steps, parallel_crash_rate, linewidth=2.5, color='#1f77b4', 
            label='num_envs=100 (parallel)', linestyle='--')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Crash Rate (%)')
    ax.set_title('Comparison: Single-Env Trials vs Parallel Evaluation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, EPISODE_LENGTH])
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "comparison_single_vs_parallel.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to: {png_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
