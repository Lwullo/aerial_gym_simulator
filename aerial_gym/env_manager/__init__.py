import isaacgym

from aerial_gym.config.env_config.env_with_obstacles import EnvWithObstaclesCfg
from aerial_gym.registry.env_registry import env_config_registry


env_config_registry.register("env_with_obstacles", EnvWithObstaclesCfg)

