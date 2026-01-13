#!/usr/bin/env python3
"""
Test script 1: Hover with GMM Disturbance
Purpose: Verify that GMM physical forces produce visible disturbances (lateral drift, oscillation, attitude tilt)
"""

import time
# Import aerial_gym modules BEFORE torch (Isaac Gym requirement)
from aerial_gym.task.navigation_task_gmm_noise import NavigationTaskGmmNoise
from aerial_gym.config.task_config import navigation_task_gmm_noise_config
import torch  # Import torch AFTER isaacgym modules

def main():
    # Create task config with visualization enabled
    task_config = navigation_task_gmm_noise_config.task_config
    task_config.headless = False  # Enable visualization
    task_config.num_envs = 1  # Single environment for clear observation
    task_config.episode_len_steps = 1000  # Long episode for observation
    
    print("=" * 60)
    print("Test 1: Hover with GMM Disturbance")
    print("=" * 60)
    print(f"GMM Physical Force Enabled: {task_config.gmm_force_config.enable_physical_force}")
    print(f"Disturbance Coefficient (k): {task_config.gmm_force_config.disturbance_coefficient}")
    print(f"Drone Mass: {task_config.gmm_force_config.drone_mass} kg")
    print(f"Force Update Steps: {task_config.gmm_force_config.force_update_steps}")
    print(f"Noise Scale: {task_config.noise_config.noise_scale}")
    print("=" * 60)
    print("\nExpected Behavior:")
    print("  - Visible lateral drift (X/Y movement)")
    print("  - Oscillation/jitter in position")
    print("  - Attitude tilting")
    print("\nStarting simulation...")
    print("=" * 60)
    
    # Create environment
    env = NavigationTaskGmmNoise(task_config)
    
    # Reset environment (returns only obs, not (obs, info))
    obs = env.reset()
    
    # Hover action: zero velocity command (stay in place)
    hover_action = torch.zeros((1, 4), device=task_config.device)
    
    # Track position for analyzing drift
    initial_position = env.obs_dict["robot_position"][0].clone()
    print(f"\nInitial Position: {initial_position.cpu().numpy()}")
    
    max_drift = torch.zeros(3, device=task_config.device)
    
    # Run simulation
    for step in range(500):  # Run for 500 steps
        obs, reward, terminated, truncated, info = env.step(hover_action)
        
        # Calculate drift from initial position
        current_position = env.obs_dict["robot_position"][0]
        drift = torch.abs(current_position - initial_position)
        max_drift = torch.maximum(max_drift, drift)
        
        # Print status every 50 steps
        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(f"  Position: {current_position.cpu().numpy()}")
            print(f"  Drift (XYZ): {drift.cpu().numpy()}")
            print(f"  Max Drift: {max_drift.cpu().numpy()}")
        
        # Render
        env.render()
        time.sleep(0.01)  # Slow down for observation
        
        # Check if episode ended
        if terminated[0] or truncated[0]:
            print("\nEpisode ended!")
            if terminated[0]:
                print("  Reason: Collision/Crash")
            if truncated[0]:
                print("  Reason: Timeout")
            break
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"Final Position: {env.obs_dict['robot_position'][0].cpu().numpy()}")
    print(f"Maximum Drift (XYZ): {max_drift.cpu().numpy()}")
    print(f"  X-drift: {max_drift[0].item():.3f} m")
    print(f"  Y-drift: {max_drift[1].item():.3f} m")
    print(f"  Z-drift: {max_drift[2].item():.3f} m")
    
    # Verification
    drift_threshold = 0.1  # meters
    lateral_drift = max(max_drift[0].item(), max_drift[1].item())
    
    if lateral_drift > drift_threshold:
        print(f"\n✓ PASS: Visible lateral drift detected ({lateral_drift:.3f}m > {drift_threshold}m)")
    else:
        print(f"\n✗ FAIL: Insufficient lateral drift ({lateral_drift:.3f}m ≤ {drift_threshold}m)")
        print("  Suggestion: Increase disturbance_coefficient or check GMM force implementation")
    
    print("=" * 60)
    
    # Close environment
    env.close()
    print("\nTest complete!")


if __name__ == "__main__":
    main()
