import numpy as np
import os
import yaml
import json


import isaacgym


from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.helpers import parse_arguments

import gym
from gym import spaces
from argparse import Namespace

from rl_games.common import env_configurations, vecenv

import torch
import distutils

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# import warnings
# warnings.filterwarnings("error")


def _patch_rl_games_scalar_logging():
    """
    Filter selected rl_games performance scalars from TensorBoard.
    """
    from rl_games.common import a2c_common

    if getattr(a2c_common.A2CBase, "_aerial_scalar_filter_patched", False):
        return

    original_write_stats = a2c_common.A2CBase.write_stats
    blocked_exact = {"performance/rl_update_time"}
    blocked_prefixes = ("performance/step_inference_rl_",)

    def filtered_write_stats(self, *args, **kwargs):
        writer = getattr(self, "writer", None)
        if writer is None or not hasattr(writer, "add_scalar"):
            return original_write_stats(self, *args, **kwargs)

        original_add_scalar = writer.add_scalar

        def filtered_add_scalar(tag, *a, **kw):
            if tag in blocked_exact:
                return
            for prefix in blocked_prefixes:
                if tag.startswith(prefix):
                    return
            return original_add_scalar(tag, *a, **kw)

        writer.add_scalar = filtered_add_scalar
        try:
            return original_write_stats(self, *args, **kwargs)
        finally:
            writer.add_scalar = original_add_scalar

    a2c_common.A2CBase.write_stats = filtered_write_stats
    a2c_common.A2CBase._aerial_scalar_filter_patched = True


def _patch_rl_games_sac_obs_handling():
    """
    Patch rl_games SACAgent.play_steps for tensor/dict observation handling.
    Some rl_games versions call `next_obs.clone()` even when `next_obs` is a dict,
    and may keep stale `obs` tensors across rollout steps.
    """
    from rl_games.algos_torch import sac_agent as sac_agent_mod
    from rl_games.algos_torch import torch_ext
    import time

    if getattr(sac_agent_mod.SACAgent, "_aerial_sac_obs_patch_applied", False):
        return

    def _as_obs_tensor(x):
        if isinstance(x, dict):
            return x["obs"]
        return x

    def patched_play_steps(self, random_exploration=False):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic1_losses = []
        critic2_losses = []

        obs_tensor = _as_obs_tensor(self.obs).clone()

        for _ in range(self.num_steps_per_episode):
            self.set_eval()
            if random_exploration:
                action = (
                    torch.rand(
                        (self.num_actors, *self.env_info["action_space"].shape), device=self._device
                    )
                    * 2.0
                    - 1.0
                )
            else:
                with torch.no_grad():
                    action = self.act(obs_tensor.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.time()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_end = time.time()

            next_obs_tensor = _as_obs_tensor(next_obs)

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += step_end - step_start
            step_time += step_end - step_start

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()
            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            # Keep full observation container (dict/tensor) for next step.
            self.obs = next_obs

            rewards = self.rewards_shaper(rewards)
            self.replay_buffer.add(
                obs_tensor,
                action,
                torch.unsqueeze(rewards, 1),
                next_obs_tensor,
                torch.unsqueeze(dones, 1),
            )

            # Advance rollout state.
            obs_tensor = next_obs_tensor

            if not random_exploration:
                self.set_train()
                update_time_start = time.time()
                actor_loss_info, critic1_loss, critic2_loss = self.update(self.epoch_num)
                update_time_end = time.time()
                update_time = update_time_end - update_time_start

                self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
            else:
                update_time = 0

            total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return (
            step_time,
            play_time,
            total_update_time,
            total_time,
            actor_losses,
            entropies,
            alphas,
            alpha_losses,
            critic1_losses,
            critic2_losses,
        )

    sac_agent_mod.SACAgent.play_steps = patched_play_steps
    sac_agent_mod.SACAgent._aerial_sac_obs_patch_applied = True


def _patch_rl_games_sac_epoch_scalar_logging():
    """
    Add SAC TensorBoard scalars with epoch as x-axis.
    Mirrors:
    - rewards/step -> rewards/epoch
    - episode_lengths/step -> episode_lengths/epoch
    """
    from rl_games.algos_torch import sac_agent as sac_agent_mod

    if getattr(sac_agent_mod.SACAgent, "_aerial_sac_epoch_scalar_patch_applied", False):
        return

    original_train = sac_agent_mod.SACAgent.train

    def patched_train(self, *args, **kwargs):
        writer = getattr(self, "writer", None)
        if writer is None or not hasattr(writer, "add_scalar"):
            return original_train(self, *args, **kwargs)

        original_add_scalar = writer.add_scalar

        def add_scalar_with_epoch(tag, scalar_value, *a, **kw):
            out = original_add_scalar(tag, scalar_value, *a, **kw)

            epoch_num = int(getattr(self, "epoch_num", 0))
            if tag == "rewards/step":
                original_add_scalar("rewards/epoch", scalar_value, epoch_num)
            elif tag == "episode_lengths/step":
                original_add_scalar("episode_lengths/epoch", scalar_value, epoch_num)
            return out

        writer.add_scalar = add_scalar_with_epoch
        try:
            return original_train(self, *args, **kwargs)
        finally:
            writer.add_scalar = original_add_scalar

    sac_agent_mod.SACAgent.train = patched_train
    sac_agent_mod.SACAgent._aerial_sac_epoch_scalar_patch_applied = True


class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations["observations"]

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)

        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )

        return (
            observations["observations"],
            rewards,
            dones,
            infos,
        )

    def set_train_info(self, *args, **kwargs):
        if hasattr(self.env, "set_train_info"):
            self.env.set_train_info(*args, **kwargs)


class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )
        info["observation_space"] = spaces.Box(
            np.ones(self.env.task_config.observation_space_dim) * -np.Inf,
            np.ones(self.env.task_config.observation_space_dim) * np.Inf,
        )
        print(info["action_space"], info["observation_space"])
        return info

    def set_train_info(self, *args, **kwargs):
        if hasattr(self.env, "set_train_info"):
            self.env.set_train_info(*args, **kwargs)


env_configurations.register(
    "position_setpoint_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_acceleration_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_acceleration_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task_gmm_noise",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "navigation_task_gmm_noise", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_reconfigurable",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_reconfigurable", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_morphy",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_morphy", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real_end_to_end",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real_end_to_end", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)


def get_args():
    from isaacgym import gymutil

    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "Random seed, if larger than 0 will overwrite the value in yaml config.",
        },
        {
            "name": "--tf",
            "required": False,
            "help": "run tensorflow runner",
            "action": "store_true",
        },
        {
            "name": "--train",
            "required": False,
            "help": "train network",
            "action": "store_true",
        },
        {
            "name": "--play",
            "required": False,
            "help": "play(test) network",
            "action": "store_true",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": False,
            "help": "path to checkpoint",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_aerial_quad.yaml",
            "required": False,
            "help": "path to config",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": "1024",
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--sigma",
            "type": float,
            "required": False,
            "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "rl_games",
            "help": "the wandb's project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "the entity (team) of wandb's project",
        },
        {
            "name": "--task",
            "type": str,
            "default": "navigation_task",
            "help": "Override task from config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--preset_id",
            "type": int,
            "default": -1,
            "help": "Fixed environment preset id for tasks that support it (-1 disables).",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "False",
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "Choose whether to use warp or Isaac Gym rendeing pipeline.",
        },
    ]

    # parse argume                                                
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def update_config(config, args):

    if args["task"] is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["params"]["config"]["name"] = args["experiment_name"]
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]
    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        # config['params']['config']['num_envs'] = args['num_envs']
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config


if __name__ == "__main__":
    runner_dir = os.path.dirname(os.path.abspath(__file__))
    original_cwd = os.getcwd()
    os.chdir(runner_dir)
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())
    # Guardrail: training should never inherit eval-only behavior from a stale shell env.
    if args.get("train", False):
        os.environ["AERIAL_GYM_EVAL_MODE"] = "0"

    config_name = args["file"]
    if not os.path.isabs(config_name):
        config_name = os.path.abspath(os.path.join(original_cwd, config_name))

    print("Loading config: ", config_name)
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)

        config = update_config(config, args)

        if args.get("preset_id", -1) is not None:
            task_cfg = task_registry.get_task_config(args["task"])
            task_cfg.preset_id = int(args["preset_id"])
        else:
            task_cfg = task_registry.get_task_config(args["task"])

        task_overrides = config.get("params", {}).get("config", {}).get("task_overrides", {})
        if isinstance(task_overrides, dict) and len(task_overrides) > 0:
            print(f"Applying task_overrides for {args['task']}: {task_overrides}")
            for key, value in task_overrides.items():
                setattr(task_cfg, key, value)

        experiment_name = config.get("params", {}).get("config", {}).get("name", "gen_ppo")
        runs_dir = os.path.join(runner_dir, "runs")
        os.environ["AERIAL_GYM_RUNS_DIR"] = runs_dir
        os.environ["AERIAL_GYM_EXPERIMENT_NAME"] = experiment_name
        os.environ["AERIAL_GYM_PPO_CONFIG_PATH"] = config_name
        os.environ["AERIAL_GYM_PPO_CONFIG_JSON"] = json.dumps(config, ensure_ascii=False)
        checkpoint_path = args.get("checkpoint")
        if checkpoint_path is not None and str(checkpoint_path).strip() != "":
            checkpoint_path = str(checkpoint_path).strip()
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.abspath(os.path.join(original_cwd, checkpoint_path))
            os.environ["AERIAL_GYM_IS_RESUME"] = "1"
            os.environ["AERIAL_GYM_RESUME_CHECKPOINT"] = checkpoint_path
        else:
            os.environ["AERIAL_GYM_IS_RESUME"] = "0"
            os.environ["AERIAL_GYM_RESUME_CHECKPOINT"] = ""
        learning_rate = config.get("params", {}).get("config", {}).get("learning_rate")
        if learning_rate is not None:
            os.environ["AERIAL_GYM_LR"] = str(learning_rate)

        from rl_games.torch_runner import Runner

        _patch_rl_games_scalar_logging()
        _patch_rl_games_sac_obs_handling()
        _patch_rl_games_sac_epoch_scalar_logging()
        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args["track"] and rank == 0:
        import wandb

        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
    runner.run(args)

    if args["track"] and rank == 0:
        wandb.finish()
