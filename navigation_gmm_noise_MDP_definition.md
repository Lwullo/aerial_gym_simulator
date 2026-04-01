# Navigation GMM Noise Task - MDP 定义（当前实现）

> 任务名称: `navigation_task_gmm_noise`
> 
> 文档版本: v2.0
> 
> 最后更新: 2026-03-17

---

## 1. MDP 概述

本任务定义为标准 MDP:

`MDP = (S, A, R, P)`

- `S`: 状态空间
- `A`: 动作空间
- `R`: 奖励函数
- `P`: 状态转移（仿真动力学）

任务目标是: 母机在风场中选择合适时机投放子机，使子机落点尽量接近目标，同时控制投放冲击和姿态风险。

---

## 2. 状态空间 `S`

### 2.1 维度

当前观测维度为 `12`。

### 2.2 定义

`S = [x_rel, y_rel, z_rel, vx, vy, vz, roll, pitch, wx, wy, wz, height]`

- `x_rel, y_rel, z_rel`: 目标相对位置（机体系）
- `vx, vy, vz`: 机体线速度（机体系）
- `roll, pitch`: 欧拉角（`ssa` 归一化）
- `wx, wy, wz`: 机体角速度
- `height`: 当前绝对高度 `z`（未归一化）

备注:
- 观测中不包含显式风向/风速。
- 不再包含 yaw 占位、动作历史、VAE latent。

---

## 3. 动作空间 `A`

### 3.1 维度

当前动作为 `5` 维连续量，输入范围 `[-1, 1]`:

`A = [vx_cmd, vy_cmd, vz_cmd, yawrate_cmd, drop_switch]`

### 3.2 物理映射

- `vx_cmd, vy_cmd, vz_cmd` 映射到 `[-1.2, 1.2] m/s`
- `yawrate_cmd` 映射到 `[-pi/6, pi/6] rad/s`
- `drop_switch` 保持网络原始输出，使用阈值触发

DROP 触发条件:

`drop_switch > drop_threshold`, 其中当前 `drop_threshold = 0.7`。

---

## 4. 环境与初始化

### 4.1 空间边界

- `env_bounds_min = [-25, -25, 0]`
- `env_bounds_max = [25, 25, 20]`

即环境大小约为 `50 x 50 x 20`。

### 4.2 目标与出生点

- 目标固定在各并行环境中心（局部偏移 `[0,0,0]`），`z=0`
- 母机出生点固定（相对环境中心）:
  - `spawn_xy = [-12, 0]`
  - `spawn_z = 10`

### 4.3 障碍物

- `num_obstacles_in_env = 0`
- 非墙体资产已关闭

---

## 5. 风场与扰动模型

采用两层风场:

`w_total = w_main + w_local`

- 主风 `w_main`: 每个环境在 reset 时独立采样，整个 episode 常量
  - `|w_main| ~ U(1.0, 3.0) m/s`
  - 水平风（`z=0`）
- 局部风 `w_local`: GMM 强度调制的小尺度扰动
  - 上限 `|w_local|max ~ U(0.2, 0.3) m/s`
  - 水平方向

当前配置: `enable_physical_force=False`

- 母机不受物理风力项直接作用
- 子机自由落体仿真仍使用上述风场

---

## 6. DROP 与子机模型

### 6.1 DROP 事件时序（单步内）

1. 根据 `drop_switch` 判定是否触发 DROP
2. 记录母机 `before` 姿态/角速度
3. 对母机施加释放冲击
4. 执行本步仿真
5. 读取 `after` 姿态/角速度，计算冲击指标
6. 用子机自由落体模型计算落点误差
7. 计算 DROP 奖励
8. 将该环境标记为 episode 结束（truncation）

### 6.2 子机自由落体动力学

子机初值:
- 位置: DROP 瞬间母机位置
- 速度: DROP 瞬间母机速度（继承）

积分方程:

`a = g + (c_child / m_child) * (w - v_child)`

当前参数:
- `g = 9.81`
- `c_child = 1.0`
- `m_child = 1.0`
- `max_child_sim_steps = 5000`

终止条件:
- 积分到 `z <= 0`
- 记录 `landing_position`
- 落点误差只用 `XY`:
  - `landing_error_xy = ||landing_xy - target_xy||`

---

## 7. 奖励函数 `R`

奖励分为 WAIT 阶段与 DROP 阶段。

### 7.1 WAIT（未投放）

1. 时间项:

`R_wait = -time_penalty`

当前 `time_penalty = 0.0`。

2. 距离进度引导:

`R_dir = w_dir * (d_{t-1}^{xy} - d_t^{xy})`

当前 `w_dir = 0.3`，仅在 `d_t^{xy} > 0.1` 且未 DROP 时生效。

3. 高度软约束惩罚（全程叠加）:

`R_alt = -w_alt * max((z_ref - tol) - z, 0)`

当前:
- `z_ref = 10.0`
- `tol = 0.5`
- `w_alt = 1.0`

### 7.2 DROP（投放当步）

#### A) 分段得分奖励

先计算 `d_xy = landing_error_xy`。

阈值:
- `r = 2.0`
- `T = [0.2r, 2.2r, 6.2r] = [0.4, 4.4, 12.4]`

分数:
- `raw_score = 20` if `d_xy <= 0.4`
- `raw_score = 6` if `0.4 < d_xy <= 4.4`
- `raw_score = 1` if `4.4 < d_xy <= 12.4`
- `raw_score = 0` if `d_xy > 12.4`

得分项:

`R_score = w_score * raw_score`, 当前 `w_score = 1.0`。

超外圈覆盖惩罚:
- 若 `d_xy > 12.4`，DROP 奖励直接覆盖为 `-outside_region_penalty`
- 当前 `outside_region_penalty = 20`

#### B) 冲击与姿态项

冲击指标:

`impulse_metric = alpha * Delta_theta + beta * Delta_omega`

其中:
- `Delta_theta = ||(roll,pitch)_after - (roll,pitch)_before||`
- `Delta_omega = ||omega_after - omega_before||`
- 当前 `alpha=1.0, beta=0.5`

冲击惩罚:

`R_impulse = -lambda_imp * impulse_metric`, 当前 `lambda_imp = 0.05`。

姿态奖励（仅 `raw_score > 0` 时给）:

`R_att = w_posture * exp(-(theta_drop/theta0)^2) + w_angle * exp(-(phi_drop/phi0)^2)`

当前:
- `w_posture = 0.05`, `theta0 = 0.12`
- `w_angle = 0.05`, `phi0 = 0.35`

最终 DROP 当步奖励:

`R_drop_step = R_score + R_impulse + R_att (+ R_alt)`

### 7.3 终止附加惩罚

- 若 episode 在未 DROP 情况下结束，额外:
  - `R_no_drop = -no_drop_penalty`
  - 当前 `no_drop_penalty = 20`

---

## 8. 终止条件

episode 结束条件包含:

1. `DROP` 触发后，当步奖励结算完成即结束（truncation）
2. 超时（`episode_len_steps = 1000`）
3. 其他仿真终止（如 crash）

---

## 9. 可观测性说明

当前观测不含显式风向/风速，风是部分不可观测变量。

这意味着:
- 策略主要通过运动结果间接推断风影响
- 在风随机较强时，20分档（<=0.4m）可能出现可达率上限

---

## 10. 当前关键参数快照

- 状态维度: `12`
- 动作维度: `5`
- episode 长度: `1000`
- 母机速度上限: `1.2 m/s`
- DROP 阈值: `0.7`
- 主风: `1.0~3.0 m/s`
- 局部风上限: `0.2~0.3 m/s`
- 奖励核心权重:
  - `direction_reward_weight = 0.3`
  - `impulse_penalty_weight = 0.05`
  - `attitude_reward_weight = 0.05`
  - `drop_angle_reward_weight = 0.05`

---

## 11. 代码入口

- 配置文件:
  - `aerial_gym/config/task_config/navigation_task_gmm_noise_config.py`
- 任务实现:
  - `aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py`
- 奖励细则文档:
  - `docs/drop_reward_function_spec.md`
- 风场逻辑文档:
  - `docs/gmm_wind_disturbance_logic.md`
