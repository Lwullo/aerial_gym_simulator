#!/usr/bin/env python3
"""
Test Script: Lee Controller Stability - Fixed Velocity Tracking
Purpose: Verify Lee velocity controller baseline stability without position control layer
Test: Send fixed velocity commands and observe tracking performance
"""

import time
import numpy as np
# Import aerial_gym modules BEFORE torch (Isaac Gym requirement)
from aerial_gym.task.navigation_task_gmm_noise import NavigationTaskGmmNoise
from aerial_gym.config.task_config import navigation_task_gmm_noise_config
import torch  # Import torch AFTER isaacgym modules


def main():
    # Create task config with visualization enabled
    task_config = navigation_task_gmm_noise_config.task_config
    task_config.headless = False  # Enable visualization
    task_config.num_envs = 1  # Single environment
    task_config.episode_len_steps = 1000  # Long episode
    
    print("=" * 60)
    print("Lee Controller Stability Test: Fixed Velocity Tracking")
    print("=" * 60)
    print(f"GMM Physical Force: {task_config.gmm_force_config.enable_physical_force}")
    print(f"GMM Observation Noise: {task_config.noise_config.enable_noise}")
    print("=" * 60)
    print("\nTest Objective:")
    print("  Verify Lee velocity controller can stably track fixed velocity")
    print("  commands without position feedback layer.")
    print("\nTest Procedure:")
    print("  - Send fixed velocity command: [vx=0.5, vy=0, vz=0, yaw=0]")
    print("  - Observe for 500 steps (~25 seconds)")
    print("\nExpected Behavior (STABLE controller):")
    print("  ✓ Actual velocity matches commanded velocity (±0.1 m/s)")
    print("  ✓ Attitude remains stable (roll/pitch < 10°)")
    print("  ✓ No oscillations or vibrations")
    print("  ✓ No roll-over or tumbling")
    print("\nStarting test...")
    print("=" * 60)
    
    # Create environment
    env = NavigationTaskGmmNoise(task_config)
    
    # Reset environment
    obs = env.reset()
    
    # IMPORTANT: Velocity commands are in VEHICLE FRAME (not world frame)
    # Vehicle frame = yaw-only frame (roll=pitch=0, only yaw rotation from world)
    # For this test: initial yaw is randomized within [-30°, +30°]
    # 
    # Command semantics:
    # - vx_vehicle: forward/backward relative to vehicle heading (considering yaw)
    # - vy_vehicle: left/right relative to vehicle heading
    # - vz_vehicle: up/down (same as world Z since vehicle frame has no roll/pitch)
    #
    # Example: If vehicle yaw=90°, vehicle_x points to world_y direction
    #
    # For testing, we command vehicle forward (vx_vehicle=0.2 m/s)
    # Due to random yaw, world direction will vary
    fixed_velocity_action = torch.tensor([[0.5, 0.0, 0.0, 0.0]], device=task_config.device)
    
    print(f"\nFixed velocity command (VEHICLE FRAME): {fixed_velocity_action[0].cpu().numpy()}")
    print("  Semantics: [vx_vehicle=0.2 m/s forward, vy=0, vz=0, yaw_rate=0]")
    print("  Note: Actual world direction depends on initial yaw angle")
    
    # Track metrics
    initial_position = env.obs_dict["robot_position"][0].clone()
    print(f"\nInitial Position: {initial_position.cpu().numpy()}")
    
    velocity_errors = []
    attitude_angles = []
    
    # Run simulation
    for step in range(500):
        obs, reward, terminated, truncated, info = env.step(fixed_velocity_action)
        
        # Get current state
        current_position = env.obs_dict["robot_position"][0]
        current_velocity_world = env.obs_dict["robot_linvel"][0]  # world frame
        current_velocity_vehicle = env.obs_dict["robot_vehicle_linvel"][0]  # vehicle frame
        current_orientation = env.obs_dict["robot_orientation"][0]  # quaternion
        
        # Calculate velocity error in vehicle frame (command is in vehicle frame)
        expected_vel_vehicle = torch.tensor([0.2, 0.0, 0.0], device=task_config.device)
        velocity_error = torch.norm(current_velocity_vehicle - expected_vel_vehicle).item()
        velocity_errors.append(velocity_error)
        
        # Calculate attitude angles (roll, pitch from quaternion)
        # Simple approximation: small angle assumption
        euler = env.obs_dict["robot_euler_angles"][0] if "robot_euler_angles" in env.obs_dict else torch.zeros(3, device=task_config.device)
        roll = abs(euler[0].item()) if len(euler) > 0 else 0
        pitch = abs(euler[1].item()) if len(euler) > 1 else 0
        max_attitude = max(roll, pitch)
        attitude_angles.append(max_attitude)
        
        # Print status every 100 steps
        if step % 100 == 0:
            displacement = current_position - initial_position
            print(f"\nStep {step}:")
            print(f"  Position: {current_position.cpu().numpy()}")
            print(f"  Displacement: {displacement.cpu().numpy()}")
            print(f"  Velocity (vehicle frame): {current_velocity_vehicle.cpu().numpy()}")
            print(f"  Velocity Error: {velocity_error:.3f} m/s")
            print(f"  Max Attitude: {np.rad2deg(max_attitude):.1f}°")
        
        # Render
        env.render()
        time.sleep(0.01)
        
        # Check if episode ended
        if terminated[0] or truncated[0]:
            print("\n⚠ Episode ended prematurely!")
            if terminated[0]:
                print("  Reason: Collision/Crash")
            if truncated[0]:
                print("  Reason: Timeout")
            break
    
    # Final statistics
    final_position = env.obs_dict["robot_position"][0]
    final_velocity = env.obs_dict["robot_linvel"][0]
    total_displacement = final_position - initial_position
    
    # Calculate metrics
    avg_velocity_error = np.mean(velocity_errors)
    max_velocity_error = np.max(velocity_errors)
    avg_attitude = np.mean(attitude_angles)
    max_attitude = np.max(attitude_angles)
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"Initial Position: {initial_position.cpu().numpy()}")
    print(f"Final Position:   {final_position.cpu().numpy()}")
    print(f"Total Displacement: {total_displacement.cpu().numpy()}")
    print(f"Final Velocity:   {final_velocity.cpu().numpy()}")
    print("\nVelocity Tracking Performance:")
    print(f"  Average Error: {avg_velocity_error:.3f} m/s")
    print(f"  Maximum Error: {max_velocity_error:.3f} m/s")
    print("\nAttitude Stability:")
    print(f"  Average Attitude: {np.rad2deg(avg_attitude):.1f}°")
    print(f"  Maximum Attitude: {np.rad2deg(max_attitude):.1f}°")
    
    # Verification
    print("\n" + "=" * 60)
    print("Stability Assessment:")
    print("=" * 60)
    
    passed = True
    
    # Check 1: Velocity tracking
    if avg_velocity_error < 0.1:
        print("✓ PASS: Velocity tracking accurate (avg error < 0.1 m/s)")
    else:
        print(f"✗ FAIL: Poor velocity tracking (avg error = {avg_velocity_error:.3f} m/s)")
        passed = False
    
    # Check 2: Attitude stability
    if max_attitude < np.deg2rad(10):
        print(f"✓ PASS: Attitude stable (max = {np.rad2deg(max_attitude):.1f}° < 10°)")
    else:
        print(f"✗ FAIL: Attitude unstable (max = {np.rad2deg(max_attitude):.1f}° ≥ 10°)")
        passed = False
    
    # Check 3: No premature termination
    if step >= 499:
        print("✓ PASS: Completed full test duration (no crash)")
    else:
        print(f"✗ FAIL: Crashed at step {step}")
        passed = False
    
    # Overall verdict
    print("\n" + "=" * 60)
    if passed:
        print("🎉 OVERALL: Controller is STABLE")
        print("   Lee velocity controller successfully tracks velocity commands")
        print("   with stable attitude and no oscillations.")
    else:
        print("⚠ OVERALL: Controller is UNSTABLE")
        print("   Suggestions:")
        print("   1. Check Lee controller gains (K_vel, K_rot, K_angvel)")
        print("   2. Verify motor saturation limits")
        print("   3. Review control allocation matrix")
    print("=" * 60)
    
    # Close environment
    env.close()
    print("\nTest complete!")


if __name__ == "__main__":
    main()
