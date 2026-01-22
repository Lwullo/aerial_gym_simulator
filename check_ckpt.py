import torch
ckpt = torch.load('/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gmm_noise_run_18-21-28-26/nn/last_gmm_noise_run_ep_4950_rew_69.69058.pth')
print("Keys in checkpoint:")
for key in ckpt.keys():
    print(f"  {key}")

print("\nenv_state type:", type(ckpt['env_state']))
if isinstance(ckpt['env_state'], dict):
    print("env_state keys:", list(ckpt['env_state'].keys()))
else:
    print("env_state is not a dict, it's:", ckpt['env_state'])
