
import isaacgym
import torch
import os
import sys
import numpy as np
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("DiagnosticRollout")

# Add the source directory to python path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import NavigationTaskGmmNoise
from aerial_gym.config.task_config.navigation_task_gmm_noise_config import task_config

class InstrumentedNavigationTask(NavigationTaskGmmNoise):
    """
    Subclass that captures reward components during step.
    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.latest_reward_components = {}

    def _compute_reward_and_scores(self):
        # Call original method
        ret = super()._compute_reward_and_scores()
        
        # Calculate components again or extract them if possible
        # Since the original method returns scalars/tensors but doesn't store them all in self,
        # we have to re-calculate specific components to be sure, OR rely on what is returned.
        # Original returns: (self.rewards, improvement_reward, direction_reward, noise_reduction_reward, noise_intensity)
        
        # We need Safety, Smoothness, Collision which are NOT returned by default.
        # So we MUST re-calculate them here to capture them.
        
        # --- Re-calculation Logic (Copied from source for diagnostics) ---
        position = self.obs_dict["robot_position"]
        dist_to_target = torch.norm(self.target_position - position, dim=1)
        
        # Safety (Penalty Log-Barrier)
        depth_pixels = self.obs_dict["depth_range_pixels"].squeeze(1)
        
        # 1. Handle Invalid/Zero -> Max Range
        max_range = 10.0
        depth_pixels = torch.where(depth_pixels <= 0.0, torch.tensor(max_range, device=self.device), depth_pixels)
        
        # Denormalize depth (pixels are 0-1, need meters)
        distances = depth_pixels.view(self.sim_env.num_envs, -1) * max_range
        safe_distances = torch.clamp(distances, min=self.reward_params["min_safe_distance_clamp"])
        
        # DEBUG: Check typical distances
        if self.sim_env.sim_steps[0] % 10 == 0:
           print(f"[DEBUG] Dist: min={distances.min().item():.3f}, mean={distances.mean().item():.3f}, max={distances.max().item():.3f}")
        
        # 2. Threshold
        threshold = self.reward_params.get("safety_dist_threshold", torch.tensor(1.0, device=self.device))
        
        log_dist = torch.log(safe_distances)
        log_threshold = torch.log(threshold)
        
        # Penalty only when closer than threshold
        pixel_penalties = torch.clamp(log_dist - log_threshold, max=0.0)
        safety_reward = self.reward_params["safety_reward_magnitude"] * pixel_penalties.mean(dim=1)
        
        # Action Smoothness
        # Note: self.actions is updated in step() BEFORE this is called.
        k_a = self.reward_params["action_magnitude_penalty_weight"]
        action_norm_sq = torch.sum(self.actions.pow(2), dim=1)
        action_magnitude_penalty = -k_a * action_norm_sq
        
        k_Delta_a = self.reward_params["action_change_penalty_weight"]
        action_diff_norm_sq = torch.sum((self.actions - self.previous_actions).pow(2), dim=1)
        action_change_penalty = -k_Delta_a * action_diff_norm_sq
        action_smoothness_penalty = action_magnitude_penalty + action_change_penalty
        
        # Collision
        collision_penalty = self.reward_params["collision_penalty"]
        collision_mask = (self.obs_dict["crashes"] > 0).float()
        collision_reward = collision_penalty * collision_mask
        
        # Unified Potential (Gradient)
        # Note: self.previous_potential is updated in the original method, so calculation 'J_prev - J_current' 
        # might be tricky if J_prev is already overwritten? 
        # Actually, original code updates previous_potential AT THE END of _compute_reward_and_scores?
        # Let's check: No, it updates it inside.
        # But we called super(), so self.previous_potential MIGHT be the NEW one now.
        # However, ret[1] is improvement_reward calculation result.
        
        self.latest_reward_components = {
            "total": ret[0].clone(),
            "potential_gradient": ret[1].clone(), # improvement_reward
            "safety": safety_reward,
            "action_smoothness": action_smoothness_penalty,
            "collision": collision_reward,
            "direction": ret[2].clone()
        }
        
        return ret

def run_diagnostic():
    # Setup config
    cfg = task_config
    cfg.seed = 42  # integer
    cfg.sim_device = "cuda:0"
    cfg.graphics_device_id = 0
    cfg.headless = True
    
    # 3 Policies
    policies = {
        "1. Zero Action": lambda shape: torch.zeros(shape, device="cuda:0"),
        "2. Max Backward": lambda shape: torch.tensor([-1.0, 0.0, 0.0, 0.0], device="cuda:0").repeat(shape[0], 1),
        "3. Max Forward": lambda shape: torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0").repeat(shape[0], 1)
    }
    
    results = {}
    
    # Create env once
    # Signature: task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    env = InstrumentedNavigationTask(cfg, seed=42, num_envs=128, headless=True, device="cuda:0", use_warp=True)
    
    steps_limit = 1200 # Rollout length (Increased to full episode length)
    
    for policy_name, policy_fn in policies.items():
        print(f"\nRunning Policy: {policy_name}")
        
        # Reset
        env.reset()
        # Ensure consistent conditions if possible?
        # Since we use same seed and reset, hopefully it resets to same initial state if sequential?
        # Isaac Gym reset might be stochastic. But we run 128 envs in parallel, so we get statistical average.
        
        curr_obs = env.obs_dict
        
        num_envs = env.sim_env.num_envs
        stats = {
            "ep_len": torch.zeros(num_envs, device="cuda:0"),
            "return": torch.zeros(num_envs, device="cuda:0"),
            "comp_safety": torch.zeros(num_envs, device="cuda:0"),
            "comp_smooth": torch.zeros(num_envs, device="cuda:0"),
            "comp_collision": torch.zeros(num_envs, device="cuda:0"),
            "comp_potential": torch.zeros(num_envs, device="cuda:0"),
            "active": torch.ones(num_envs, device="cuda:0", dtype=torch.bool)
        }
        
        for step in range(steps_limit):
            # Generate action
            action = policy_fn((num_envs, 4))
            
            # Step
            env.step(action)
            
            # Record
            dones = env.terminations > 0
            timeouts = env.truncations > 0
            rewards = env.latest_reward_components["total"]
            
            # Accumulate stats for active envs
            active = stats["active"]
            
            stats["return"][active] += rewards[active]
            stats["ep_len"][active] += 1
            
            # Accumulate components
            comps = env.latest_reward_components
            stats["comp_safety"][active] += comps["safety"][active]
            stats["comp_smooth"][active] += comps["action_smoothness"][active]
            stats["comp_collision"][active] += comps["collision"][active]
            stats["comp_potential"][active] += comps["potential_gradient"][active]
            
            # Update active status
            # If done, it stops accumulating for that env
            stats["active"] = stats["active"] & (~dones) & (~timeouts)
            
            if not stats["active"].any():
                break
                
        # Calculate means
        avg_return = stats["return"].mean().item()
        avg_len = stats["ep_len"].mean().item()
        avg_safety = stats["comp_safety"].mean().item()
        avg_smooth = stats["comp_smooth"].mean().item()
        avg_collision = stats["comp_collision"].mean().item()
        avg_potential = stats["comp_potential"].mean().item()
        
        results[policy_name] = {
            "Return": avg_return,
            "Length": avg_len,
            "Safety": avg_safety,
            "Smoothness": avg_smooth,
            "Collision": avg_collision,
            "Potential": avg_potential,
            # Per-Step Metrics
            "Step_Safety": avg_safety / avg_len if avg_len > 0 else 0,
            "Step_Smooth": avg_smooth / avg_len if avg_len > 0 else 0,
            "Step_Potential": avg_potential / avg_len if avg_len > 0 else 0,
            "Step_Total": avg_return / avg_len if avg_len > 0 else 0
        }
        
    print("\n\n" + "="*50)
    print("DIAGNOSTIC RESULTS (Averaged over 128 envs)")
    print("="*50)
    print(f"{'Policy':<20} | {'Return':<10} | {'Length':<8} | {'Safety':<10} | {'Smooth':<10} | {'Collision':<10} | {'Potential':<10}")
    print("-" * 90)
    for name, res in results.items():
        print(f"{name:<20} | {res['Return']:<10.2f} | {res['Length']:<8.1f} | {res['Safety']:<10.2f} | {res['Smoothness']:<10.2f} | {res['Collision']:<10.2f} | {res['Potential']:<10.2f}")
    print("-" * 90)

    print("\n" + "="*50)
    print("DIAGNOSTIC B: PER-STEP AVERAGES (Crucial for Weight Tuning)")
    print("="*50)
    print(f"{'Policy':<20} | {'Avg/Step':<10} | {'Safe/Step':<10} | {'Smooth/Step':<12} | {'Pot/Step':<10}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<20} | {res['Step_Total']:<10.4f} | {res['Step_Safety']:<10.4f} | {res['Step_Smooth']:<12.4f} | {res['Step_Potential']:<10.4f}")
    print("-" * 70)
    
    # Verdict
    ret_zero = results["1. Zero Action"]["Return"]
    ret_back = results["2. Max Backward"]["Return"]
    ret_fwd = results["3. Max Forward"]["Return"]
    
    if ret_back > ret_zero and ret_back > ret_fwd:
        print("\n[VERDICT]: ✅ SUICIDE IS OPTIMAL. 'Max Backward' yields highest return.")
        print("Reasoning: Negative safety/smoothness penalties outweigh potential rewards, encouraging quick termination.")
    else:
        print("\n[VERDICT]: ❌ Suicide is NOT optimal based on returns.")

if __name__ == "__main__":
    run_diagnostic()
