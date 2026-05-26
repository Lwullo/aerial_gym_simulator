from aerial_gym.control.controllers.acceleration_control import (
    LeeAccelerationController,
)
from aerial_gym.control.controllers.attitude_control import LeeAttitudeController
from aerial_gym.control.controllers.position_control import LeePositionController
from aerial_gym.control.controllers.rates_control import LeeRatesController
from aerial_gym.control.controllers.velocity_control import LeeVelocityController
from aerial_gym.config.controller_config.lmf2_controller_config import (
    control as lmf2_controller_config,
)
from aerial_gym.registry.controller_registry import controller_registry


controller_registry.register_controller(
    "lmf2_position_control", LeePositionController, lmf2_controller_config
)
controller_registry.register_controller(
    "lmf2_velocity_control", LeeVelocityController, lmf2_controller_config
)
controller_registry.register_controller(
    "lmf2_attitude_control", LeeAttitudeController, lmf2_controller_config
)
controller_registry.register_controller(
    "lmf2_rates_control", LeeRatesController, lmf2_controller_config
)
controller_registry.register_controller(
    "lmf2_acceleration_control", LeeAccelerationController, lmf2_controller_config
)

