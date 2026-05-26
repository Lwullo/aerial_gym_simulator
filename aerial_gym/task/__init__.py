from aerial_gym.task.navigation_task_gmm_noise.navigation_task_gmm_noise import (
    NavigationTaskGmmNoise,
)
from aerial_gym.config.task_config.navigation_task_gmm_noise_config import (
    task_config as navigation_task_gmm_noise_config,
)
from aerial_gym.registry.task_registry import task_registry


task_registry.register_task(
    "navigation_task_gmm_noise",
    NavigationTaskGmmNoise,
    navigation_task_gmm_noise_config,
)

