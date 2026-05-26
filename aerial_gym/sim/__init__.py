from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig


sim_config_registry.register("base_sim", BaseSimConfig)
