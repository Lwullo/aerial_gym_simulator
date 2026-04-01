# 风扰逻辑说明（当前：主风 + 简化 Dryden 湍流）

本文档对应任务 `navigation_task_gmm_noise` 的当前实现。

## 1. 当前风场结构

当前总风为：

`w_total = w_main + w_dryden`

- `w_main`：每个环境在 reset 采样一次，整个 episode 内保持常值
- `w_dryden`：每步更新的三轴一阶湍流项（简化 Dryden）

说明：

- `noise_config.num_sources = 0`，已关闭 GMM 空间风源。
- 代码里仍保留 `gmm_*` 的历史命名，但当前风场主逻辑不再使用 GMM 空间源。

---

## 2. 关键参数（当前默认）

来自 `navigation_task_gmm_noise_config.py`：

- 主风：
  - `main_wind_speed_min = 1.0 m/s`
  - `main_wind_speed_max = 3.0 m/s`
  - `main_wind_horizontal_only = False`（主风允许 z 分量）
- 湍流（简化 Dryden）：
  - `enable_dryden = True`
  - `sigma_min = [0.2, 0.2, 0.15] m/s`
  - `sigma_max = [0.3, 0.3, 0.25] m/s`
  - `tau_min = [0.8, 0.8, 0.8] s`
  - `tau_max = [2.0, 2.0, 2.0] s`
  - `horizontal_only = False`
  - `clip_sigma = 3.0`
- 母机风力模型：
  - `enable_physical_force = True`
  - `drag_coefficient = 4.0 N/(m/s)`

---

## 3. 简化 Dryden 模型（实现形式）

每个环境、每个轴独立一阶过程：

`w_{k+1} = a * w_k + b * xi`

其中：

- `xi ~ N(0, 1)`
- `a = exp(-dt / tau)`
- `b = sigma * sqrt(1 - a^2)`

这等价于离散 OU 过程，满足：

- 有时间相关性（不是白噪声抖动）
- 稳态标准差约为 `sigma`
- 每个环境在 reset 重新采样 `(sigma, tau)`，环境间湍流统计独立

---

## 4. 母机如何受风影响

任务层风力：

`F_wind = c_drag * (w_total - v_uav)`

- `v_uav`：母机世界系线速度
- 力写入 `task_external_force_tensor` 并加到母机 `base link`

---

## 5. 子机如何受风影响（DROP 后）

子机自由落体模型：

`a_child = g + (c_child/m_child) * (w_total - v_child)`

默认：

- `c_child = 1.0`
- `m_child = 1.0`

子机积分到 `z <= 0`，记录 `landing_position` 和 `landing_error_xy`。

---

## 6. 与随机扰动的关系

`LMF2Cfg` 中的机器人随机扰动（`enable_disturbance=True`）是独立通道，
不属于 `w_main` 或 `w_dryden`。

---

## 7. 快速结论

你当前是：

1. 无 GMM 空间风源
2. 有主风（每回合常值）
3. 有简化 Dryden 三轴湍流（每步时变）
4. 母机受力、子机受相对风加速度，均使用同一 `w_total`

