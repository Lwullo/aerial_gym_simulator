# 激光雷达观测 → VAE压缩 → 64维编码 → 策略网络输入流程

本文档描述 `navigation_task_gmm_noise` 任务中，从激光雷达观测到 64 维视觉编码，再到拼接进入策略网络的完整数据链路。

---

## 1. 传感器输出：激光雷达 range image

使用机器人配置 `lmf2`，启用激光雷达：
- 位置：`aerial_gym/config/robot_config/lmf2_config.py`
- 配置：`OSDome_64_Config`

雷达分辨率与范围（range image）：
- 位置：`aerial_gym/config/sensor_config/lidar_config/osdome_64_config.py`
- 分辨率：`height=64, width=512`
- 距离范围：`min_range=0.5, max_range=20.0`

`WarpSensor` 使用 `WarpLidar` 做 raycast，把结果写入全局张量 `depth_range_pixels`：
- 位置：`aerial_gym/robots/robot_manager.py:189-229`

---

## 2. 裁剪与归一化

传感器更新时执行：
- `apply_range_limits()`：裁剪超出量程的值  
- `normalize_observation()`：除以 `max_range`

位置：`aerial_gym/sensors/warp/warp_sensor.py:198-226`

此时 `depth_range_pixels` 是**归一化后的范围图**，形状约为：
```
(num_envs, num_sensors, 64, 512)
```

---

## 3. VAE 编码（64 维）

如果配置中 `use_vae=True`：
- 位置：`aerial_gym/config/task_config/navigation_task_gmm_noise_config.py:124-134`

`process_image_observation()` 取出 `depth_range_pixels`，送入 VAE 编码器：
- 位置：`aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:1653-1657`

VAE 编码器 `VAEImageEncoder` 的处理逻辑：
1) 输入保证为 `(N, 1, H, W)`
2) 如果分辨率不是 `(270, 480)`，则插值到该尺寸
3) 经过 VAE 编码得到 64 维 latent

位置：`aerial_gym/utils/vae/vae_image_encoder.py:33-63`

注意：VAE 权重来自预训练文件，不会在任务中重新训练。

---

## 4. 拼接到观测向量

观测向量布局（总维度 = 13 + 4 + 64）：
- 位置与姿态等状态信息：`observations[0:13]`
- 上一时刻动作：`observations[13:17]`
- 视觉编码（VAE 输出）：`observations[17:]`

姿态相关维度的具体含义与处理：
- `S[4]=roll`, `S[5]=pitch`, `S[6]=yaw_placeholder`
- roll/pitch 先经过 `ssa()` 归一化到 [-π, π]
- 然后加入**加性均匀扰动**：`euler + 0.1 * (rand - 0.5)`，即范围为 **[-0.05, +0.05] rad**
- `yaw_placeholder` 固定为 0.0（因为 yaw 信息已在“目标方向向量”中隐含）

位置：`aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:1670-1675`

拼接位置：  
`aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:1678-1679`

---

## 5. 进入策略网络

拼接后的 `observations` 作为 RL 的输入，由训练/推理框架（PPO/策略网络）直接使用。  
