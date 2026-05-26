from aerial_gym.config.robot_config.lmf2_config import LMF2Cfg
from aerial_gym.robots.base_multirotor import BaseMultirotor
from aerial_gym.registry.robot_registry import robot_registry


robot_registry.register("lmf2", BaseMultirotor, LMF2Cfg)

