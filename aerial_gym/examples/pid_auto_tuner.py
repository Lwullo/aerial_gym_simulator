#!/usr/bin/env python3
"""
PID Auto-Tuner for Navigation Task (GMM Noise)

Automatically finds optimal PID gains for the lmf2_velocity_control controller
using Bayesian optimization (Optuna).

Usage:
    cd /home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator
    python -m aerial_gym.examples.pid_auto_tuner --n_trials 50 --num_envs 16

Author: Auto-generated for aerial_gym
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("=" * 60)
    print("ERROR: optuna not installed. Please run:")
    print("    pip install optuna")
    print("=" * 60)
    raise

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("pid_auto_tuner")


class PIDAutoTuner:
    """
    Automatic PID tuner using Bayesian optimization.
    
    This class runs the navigation task with different PID parameters
    and uses Optuna to find the optimal configuration.
    """
    
    def __init__(
        self,
        num_envs: int = 16,
        episode_len: int = 300,
        num_eval_episodes: int = 3,
        device: str = "cuda:0",
        headless: bool = True,
        task_name: str = "navigation_task_gmm_noise",
    ):
        """
        Initialize the PID tuner.
        
        Args:
            num_envs: Number of parallel environments
            episode_len: Steps per episode for evaluation
            num_eval_episodes: Number of episodes to average over
            device: CUDA device
            headless: Run without visualization
            task_name: Name of the task to use
        """
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.num_eval_episodes = num_eval_episodes
        self.device = device
        self.headless = headless
        self.task_name = task_name
        
        # Create task environment
        logger.info(f"Creating task environment: {task_name}")
        logger.info(f"num_envs={num_envs}, episode_len={episode_len}, headless={headless}")
        
        self.env = task_registry.make_task(
            task_name,
            headless=headless,
            num_envs=num_envs,
        )
        self.env.reset()
        
        # Get controller reference
        self.controller = self.env.sim_env.robot_manager.robot.controller
        
        # Store original parameters for restoration
        self.original_params = self._get_current_params()
        logger.info(f"Original controller parameters saved")
        
        # Best result tracking
        self.best_cost = float('inf')
        self.best_params = None
        self.trial_history = []
        
    def _get_current_params(self) -> Dict[str, torch.Tensor]:
        """Get current controller parameters."""
        return {
            'K_vel_xy': self.controller.K_linvel_tensor_current[0, 0].item(),
            'K_vel_z': self.controller.K_linvel_tensor_current[0, 2].item(),
            'K_rot_xy': self.controller.K_rot_tensor_current[0, 0].item(),
            'K_rot_z': self.controller.K_rot_tensor_current[0, 2].item(),
            'K_angvel_xy': self.controller.K_angvel_tensor_current[0, 0].item(),
            'K_angvel_z': self.controller.K_angvel_tensor_current[0, 2].item(),
        }
    
    def _apply_params(self, params: Dict[str, float]) -> None:
        """Apply new PID parameters to the controller."""
        num_envs = self.controller.K_linvel_tensor_current.shape[0]
        device = self.controller.K_linvel_tensor_current.device
        
        # Velocity gains (K_vel)
        self.controller.K_linvel_tensor_current[:, 0] = params['K_vel_xy']
        self.controller.K_linvel_tensor_current[:, 1] = params['K_vel_xy']
        self.controller.K_linvel_tensor_current[:, 2] = params['K_vel_z']
        
        # Rotation gains (K_rot)
        self.controller.K_rot_tensor_current[:, 0] = params['K_rot_xy']
        self.controller.K_rot_tensor_current[:, 1] = params['K_rot_xy']
        self.controller.K_rot_tensor_current[:, 2] = params['K_rot_z']
        
        # Angular velocity gains (K_angvel)
        self.controller.K_angvel_tensor_current[:, 0] = params['K_angvel_xy']
        self.controller.K_angvel_tensor_current[:, 1] = params['K_angvel_xy']
        self.controller.K_angvel_tensor_current[:, 2] = params['K_angvel_z']
    
    def _evaluate_episode(self) -> Tuple[float, float, float, float]:
        """
        Run one evaluation episode and compute metrics.
        
        Returns:
            tracking_error: Average velocity tracking error
            oscillation: Average angular velocity magnitude (oscillation measure)
            crash_rate: Fraction of environments that crashed
            stability: Measure of attitude stability
        """
        self.env.reset()
        
        total_tracking_error = 0.0
        total_oscillation = 0.0
        total_stability = 0.0
        crash_count = torch.zeros(self.num_envs, device=self.device)
        step_count = 0
        
        # Generate smooth velocity commands for evaluation
        # Using sinusoidal patterns to test tracking capability
        t = torch.linspace(0, 2 * np.pi, self.episode_len, device=self.device)
        
        with torch.no_grad():
            for step in range(self.episode_len):
                # Generate test velocity commands (smooth trajectory)
                actions = torch.zeros(
                    (self.num_envs, self.env.task_config.action_space_dim),
                    device=self.device
                )
                # Forward velocity with sinusoidal variation
                actions[:, 0] = 0.5 * torch.sin(t[step])  # vx
                actions[:, 1] = 0.3 * torch.cos(t[step])  # vy
                actions[:, 2] = 0.2 * torch.sin(0.5 * t[step])  # vz
                actions[:, 3] = 0.1 * torch.sin(0.3 * t[step])  # yaw rate
                
                obs, reward, terminated, truncated, info = self.env.step(actions)
                
                # Track crashes
                crash_count += (terminated > 0).float()
                
                # Compute tracking error (difference between commanded and actual velocity)
                actual_vel = self.env.obs_dict["robot_body_linvel"]
                # Transform commanded velocity for comparison
                cmd_vel = self.env.action_transformation_function(actions)[:, :3]
                vel_error = torch.norm(actual_vel - cmd_vel, dim=1).mean().item()
                total_tracking_error += vel_error
                
                # Compute oscillation (angular velocity magnitude)
                angvel = self.env.obs_dict["robot_body_angvel"]
                oscillation = torch.norm(angvel, dim=1).mean().item()
                total_oscillation += oscillation
                
                # Compute stability (roll/pitch deviation from level)
                euler = self.env.obs_dict["robot_euler_angles"]
                attitude_error = torch.sqrt(euler[:, 0]**2 + euler[:, 1]**2).mean().item()
                total_stability += attitude_error
                
                step_count += 1
        
        avg_tracking_error = total_tracking_error / step_count
        avg_oscillation = total_oscillation / step_count
        crash_rate = (crash_count > 0).float().mean().item()
        avg_stability = total_stability / step_count
        
        return avg_tracking_error, avg_oscillation, crash_rate, avg_stability
    
    def _compute_cost(
        self,
        tracking_error: float,
        oscillation: float,
        crash_rate: float,
        stability: float
    ) -> float:
        """
        Compute overall cost from individual metrics.
        Lower cost = better controller.
        """
        # Weights for each component
        w_tracking = 1.0      # Tracking accuracy is important
        w_oscillation = 0.5   # Some oscillation penalty
        w_crash = 10.0        # Heavy penalty for crashes
        w_stability = 0.3     # Attitude stability
        
        cost = (
            w_tracking * tracking_error +
            w_oscillation * oscillation +
            w_crash * crash_rate +
            w_stability * stability
        )
        
        return cost
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Samples PID parameters, evaluates them, and returns the cost.
        """
        # Sample parameters
        params = {
            # Velocity gains (scaled values, original uses scale_vel=0.3)
            'K_vel_xy': trial.suggest_float('K_vel_xy', 0.3, 2.0, log=True),
            'K_vel_z': trial.suggest_float('K_vel_z', 0.2, 1.5, log=True),
            
            # Rotation gains (scaled values, original uses scale_rot=1.5)
            'K_rot_xy': trial.suggest_float('K_rot_xy', 1.0, 5.0, log=True),
            'K_rot_z': trial.suggest_float('K_rot_z', 0.2, 1.5, log=True),
            
            # Angular velocity gains (scaled values, original uses scale_angvel=2.7)
            'K_angvel_xy': trial.suggest_float('K_angvel_xy', 0.5, 3.0, log=True),
            'K_angvel_z': trial.suggest_float('K_angvel_z', 0.1, 0.8, log=True),
        }
        
        # Apply parameters
        self._apply_params(params)
        
        # Evaluate over multiple episodes
        total_tracking_error = 0.0
        total_oscillation = 0.0
        total_crash_rate = 0.0
        total_stability = 0.0
        
        for ep in range(self.num_eval_episodes):
            tracking_error, oscillation, crash_rate, stability = self._evaluate_episode()
            total_tracking_error += tracking_error
            total_oscillation += oscillation
            total_crash_rate += crash_rate
            total_stability += stability
        
        # Average over episodes
        avg_tracking_error = total_tracking_error / self.num_eval_episodes
        avg_oscillation = total_oscillation / self.num_eval_episodes
        avg_crash_rate = total_crash_rate / self.num_eval_episodes
        avg_stability = total_stability / self.num_eval_episodes
        
        # Compute cost
        cost = self._compute_cost(
            avg_tracking_error, avg_oscillation, avg_crash_rate, avg_stability
        )
        
        # Track best result
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()
            logger.info(f"New best cost: {cost:.4f}")
            logger.info(f"Best params: {params}")
        
        # Store history
        self.trial_history.append({
            'trial': trial.number,
            'cost': cost,
            'tracking_error': avg_tracking_error,
            'oscillation': avg_oscillation,
            'crash_rate': avg_crash_rate,
            'stability': avg_stability,
            'params': params.copy(),
        })
        
        return cost
    
    def run(self, n_trials: int = 50, timeout: int = None) -> Dict[str, Any]:
        """
        Run the optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting PID optimization with {n_trials} trials")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='pid_tuning',
        )
        
        # Run optimization
        start_time = time.time()
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        elapsed = time.time() - start_time
        
        # Get best parameters
        best_trial = study.best_trial
        best_params = best_trial.params
        
        logger.info("=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Time elapsed: {elapsed:.1f} seconds")
        logger.info(f"Best cost: {best_trial.value:.4f}")
        logger.info(f"Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value:.6f}")
        
        return {
            'best_params': best_params,
            'best_cost': best_trial.value,
            'n_trials': n_trials,
            'elapsed_time': elapsed,
            'history': self.trial_history,
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """
        Save optimization results to JSON file.
        
        Args:
            results: Results dictionary from run()
            output_dir: Output directory (default: aerial_gym/examples/pid_tuning_results)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(output_dir, "pid_tuning_results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pid_tuning_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert tensor types to native python for JSON serialization
        results_serializable = {
            'best_params': results['best_params'],
            'best_cost': float(results['best_cost']),
            'n_trials': results['n_trials'],
            'elapsed_time': results['elapsed_time'],
            'history': [
                {k: (float(v) if isinstance(v, (int, float)) else v) 
                 for k, v in h.items()}
                for h in results['history']
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def print_config_update(self, params: Dict[str, float]) -> None:
        """
        Print the code to update lmf2_controller_config.py with optimized params.
        """
        print("\n" + "=" * 70)
        print("COPY THE FOLLOWING TO UPDATE lmf2_controller_config.py:")
        print("=" * 70)
        print("""
# Optimized PID parameters (from auto-tuner)
# Replace the existing K_*_tensor_* definitions with:

K_vel_tensor_max = torch.tensor([{K_vel_xy:.4f}, {K_vel_xy:.4f}, {K_vel_z:.4f}])
K_vel_tensor_min = torch.tensor([{K_vel_xy:.4f}, {K_vel_xy:.4f}, {K_vel_z:.4f}])

K_rot_tensor_max = torch.tensor([{K_rot_xy:.4f}, {K_rot_xy:.4f}, {K_rot_z:.4f}])
K_rot_tensor_min = torch.tensor([{K_rot_xy:.4f}, {K_rot_xy:.4f}, {K_rot_z:.4f}])

K_angvel_tensor_max = torch.tensor([{K_angvel_xy:.4f}, {K_angvel_xy:.4f}, {K_angvel_z:.4f}])
K_angvel_tensor_min = torch.tensor([{K_angvel_xy:.4f}, {K_angvel_xy:.4f}, {K_angvel_z:.4f}])

# Note: These are final values (already scaled), so set scale_* = 1.0
# Or divide the values by original scale factors:
#   scale_vel = 0.3, scale_rot = 1.5, scale_angvel = 2.7
""".format(**params))
        print("=" * 70 + "\n")
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'env'):
            self.env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Automatic PID Tuner for Navigation Task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--n_trials', type=int, default=50,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--num_envs', type=int, default=16,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--episode_len', type=int, default=300,
        help='Steps per evaluation episode'
    )
    parser.add_argument(
        '--num_eval_episodes', type=int, default=3,
        help='Number of episodes to average for each trial'
    )
    parser.add_argument(
        '--headless', action='store_true', default=True,
        help='Run without visualization'
    )
    parser.add_argument(
        '--no-headless', action='store_false', dest='headless',
        help='Run with visualization'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='CUDA device'
    )
    parser.add_argument(
        '--timeout', type=int, default=None,
        help='Optional timeout in seconds'
    )
    parser.add_argument(
        '--task_name', type=str, default='navigation_task_gmm_noise',
        help='Task name to use'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PID AUTO-TUNER FOR NAVIGATION TASK")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  n_trials: {args.n_trials}")
    logger.info(f"  num_envs: {args.num_envs}")
    logger.info(f"  episode_len: {args.episode_len}")
    logger.info(f"  num_eval_episodes: {args.num_eval_episodes}")
    logger.info(f"  headless: {args.headless}")
    logger.info(f"  device: {args.device}")
    logger.info(f"  task_name: {args.task_name}")
    logger.info("=" * 60)
    
    tuner = None
    try:
        tuner = PIDAutoTuner(
            num_envs=args.num_envs,
            episode_len=args.episode_len,
            num_eval_episodes=args.num_eval_episodes,
            device=args.device,
            headless=args.headless,
            task_name=args.task_name,
        )
        
        results = tuner.run(n_trials=args.n_trials, timeout=args.timeout)
        
        # Save results
        tuner.save_results(results)
        
        # Print config update instructions
        tuner.print_config_update(results['best_params'])
        
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise
    finally:
        if tuner is not None:
            tuner.close()


if __name__ == "__main__":
    main()
