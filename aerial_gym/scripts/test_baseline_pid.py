import os
import numpy as np
import time
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.config.task_config.navigation_task_gmm_noise_config import task_config
from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import NavigationTaskGmmNoise
import torch

logger = CustomLogger("baseline_agent")

def get_args():
    custom_args = [
        {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments"},
        {"name": "--headless", "action": "store_true", "help": "Run headless"},
        {"name": "--max_steps", "type": int, "default": 2000, "help": "Max steps per episode"},
        {"name": "--w_dist", "type": float, "default": 1.0, "help": "Weight for distance attraction"},
        {"name": "--w_noise", "type": float, "default": 5.0, "help": "Weight for noise repulsion"},
        {"name": "--w_obs", "type": float, "default": 2.0, "help": "Weight for obstacle repulsion"},
        {"name": "--d_obs", "type": float, "default": 1.5, "help": "Influence radius for obstacles"},
    ]
    # Simple argument parsing wrapper for standalone script
    import argparse
    parser = argparse.ArgumentParser()
    for arg in custom_args:
        name = arg.pop("name")
        parser.add_argument(name, **arg)
    args = parser.parse_args()
    return args

class APFAgent:
    """
    Artificial Potential Field (APF) Agent
    Combines:
    1. Attractive Potential to Target (Quadratic)
    2. Repulsive Potential from Noise (GMM Intensity)
    3. Repulsive Potential from Obstacles (Inverse Quadratic)
    """
    def __init__(self, env, args):
        self.env = env
        self.device = env.device
        
        # Gains
        self.w_dist = args.w_dist
        self.w_noise = args.w_noise
        self.w_obs = args.w_obs
        
        # Parameters
        self.obs_influence_radius = args.d_obs # meters
        self.max_speed_xy = 0.8 # m/s (from config)
        self.max_speed_z = 0.5  # m/s (from config)
        
        # PID equivalent gain (Gradient descent step size -> Velocity)
        self.grad_to_vel_gain = 2.0 

    def compute_action(self, obs_dict):
        """
        Compute velocity command based on -grad(Total_Potential)
        """
        # We need gradients w.r.t position, so we clone and enable grad
        position = obs_dict["robot_position"].clone().detach().requires_grad_(True)
        
        # 1. Target Attraction Potential: J_target = 0.5 * ||p - p_target||^2
        # (This naturally produces a linear restoring force/velocity towards target)
        target_pos = self.env.target_position
        dist_vec = position - target_pos
        dist_sq = (dist_vec ** 2).sum(dim=1)
        J_target = 0.5 * dist_sq
        
        # 2. Noise Repulsion Potential: J_noise = GMM_Intensity(p)
        # Using the environment's internal GMM computation logic
        # Note: We access _compute_gmm_mixture acting on our tensor with grad
        J_noise = self.env._compute_gmm_mixture(position)
        
        # 3. Obstacle Repulsion Potential: J_obs
        # Standard APF: 0.5 * eta * (1/d - 1/d0)^2 if d < d0 else 0
        
        # Obstacle positions: (num_envs, num_obstacles, 3)
        # We assume standard spherical obstacles for repulsion roughly
        # The environment stores them in obs_dict["obstacle_position"]
        # But that might be Observation buffer, let's use the ground truth if available
        # or defaults. In this task, obstacles are in obs_dict["obstacle_position"]
        
        obs_pos = self.env.obs_dict["obstacle_position"] # Shape: (N, MaxObs, 3)
        # Remove "invalid" obstacles (coordinates at -1000) by masking distance
        
        # Distance from robot to all obstacles
        # position: (N, 3) -> (N, 1, 3)
        # obs_pos: (N, K, 3)
        diff = position.unsqueeze(1) - obs_pos
        dist = torch.norm(diff, dim=2) # (N, K)
        
        # Thresholding
        d0 = self.obs_influence_radius
        mask = (dist < d0) & (dist > 0.05) # Avoid division by zero
        
        # Calculate Repulsion Potential per obstacle
        # U = 0.5 * (1/d - 1/d0)^2
        repulsion_term = torch.zeros_like(dist)
        repulsion_term[mask] = 0.5 * (1.0/dist[mask] - 1.0/d0)**2
        
        J_obs = repulsion_term.sum(dim=1) # Sum over all obstacles
        
        # Total Potential
        J_total = (self.w_dist * J_target) + (self.w_noise * J_noise) + (self.w_obs * J_obs)
        
        # Compute Gradient
        # We want grad of Sum(J) to get per-batch gradients efficiently
        grad_sum = torch.sum(J_total)
        grad_sum.backward()
        
        gradients = position.grad # (N, 3)
        
        # Descent Direction: v = - Gain * Gradient
        vel_cmd = -self.grad_to_vel_gain * gradients
        
        # Clamp Velocity components separately for safety/compliance with env limits
        vel_cmd[:, 0:2] = torch.clamp(vel_cmd[:, 0:2], -self.max_speed_xy, self.max_speed_xy)
        vel_cmd[:, 2] = torch.clamp(vel_cmd[:, 2], -self.max_speed_z, self.max_speed_z)
        
        # Normalize for Action Space [-1, 1]
        # The environment controller expects inputs in [-1, 1] mapped to max speeds
        # Action[0,1] = vel_xy / 0.8
        # Action[2]   = vel_z  / 0.5
        # Action[3]   = yaw_rate (set to 0 for PID baseline usually, or point to target)
        
        action = torch.zeros((self.env.sim_env.num_envs, 4), device=self.device)
        action[:, 0] = vel_cmd[:, 0] / self.max_speed_xy
        action[:, 1] = vel_cmd[:, 1] / self.max_speed_xy
        action[:, 2] = vel_cmd[:, 2] / self.max_speed_z
        action[:, 3] = 0.0 # Maintain Yaw
        
        # Clamp final actions to strict [-1, 1] to be safe
        action = torch.clamp(action, -1.0, 1.0)
        
        return action

import sys

def main():
    args = get_args()
    
    # CRITICAL: Clear sys.argv to prevent Isaac Gym / SimBuilder from parsing custom arguments (like --max_steps)
    # and conflicting or erroring out.
    sys.argv = [sys.argv[0]]
    
    # Configure Environment
    task_config.num_envs = args.num_envs
    task_config.headless = args.headless
    task_config.device = "cuda:0"
    
    # Create Environment
    logger.info("Initializing NavigationTaskGmmNoise Environment...")
    env = NavigationTaskGmmNoise(task_config)
    
    # Create Agent
    logger.info("Initializing APF Baseline Agent...")
    agent = APFAgent(env, args)
    
    # Run Loop
    logger.info(f"Starting Simulation ({args.max_steps} steps)...")
    
    total_rewards = torch.zeros(env.sim_env.num_envs, device=env.device)
    total_steps = 0
    success_count = 0
    crash_count = 0
    
    observation = env.reset()
    
    try:
        for step in range(args.max_steps):
            # Compute Action
            actions = agent.compute_action(env.obs_dict)
            
            # Step Env
            # Env returns (obs, rewards, terminations, truncations, infos)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            dones = (terminations > 0) | (truncations > 0)
            
            total_rewards += rewards
            total_steps += 1
            
            # Check for resets
            resets = dones.nonzero().squeeze()
            if len(resets.shape) > 0 and resets.numel() > 0:
                # Log outcome for just the first env if it resets, for brevity
                if 0 in resets:
                    r = total_rewards[0].item()
                    s = "Success" if env.success_buf[0] else "Crash/Timeout"
                    current_dist = torch.norm(env.obs_dict["robot_position"][0] - env.target_position[0]).item()
                    logger.info(f"Env 0 Reset. Reward: {r:.2f}, Outcome: {s}, Final Dist: {current_dist:.2f}m")
                
                # Simple statistics
                success_count += env.success_buf[resets].sum().item()
                crash_count += env.obs_dict["crashes"][resets].sum().item()
                total_rewards[resets] = 0.0
                
            if step % 100 == 0:
                logger.info(f"Step {step}/{args.max_steps} | Avg Reward: {total_rewards.mean().item():.3f}")
                
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
        
    logger.info("=== Final Stats ===")
    logger.info(f"Total Steps: {total_steps}")
    logger.info(f"Total Resets: {success_count + crash_count}")
    logger.info(f"Successes: {success_count}")
    logger.info(f"Crashes: {crash_count}")
    
    # Avoid immediate closure if visualizing
    if not args.headless:
        pass # The script ends, gym usually closes window.

if __name__ == "__main__":
    main()
