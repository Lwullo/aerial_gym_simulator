# Navigation Task GMM Noise: DROP 奖励函数数学公式说明（严格对齐当前代码）

本文档与当前实现同步：

- 任务代码：`aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py`
- 配置文件：`aerial_gym/config/task_config/navigation_task_gmm_noise_config.py`
- 同步日期：2026-03-20

---

## 1. 符号定义

对任一环境 `i` 在时刻 `t`：

- `R_i(t)`：总奖励
- `z_i(t)`：母机当前高度（m）
- `d_i^xy(t)`：母机到目标点的平面距离（m）
- `d_i^xy(t-1)`：上一时刻平面距离（m）
- `e_i^xy`：该次 DROP 的子机落点平面误差（m）
- `roll_i, pitch_i, yaw_i`：DROP 前姿态（rad）
- `omega_i`：机体系角速度向量（rad/s）
- `Delta_theta_i`：DROP 前后 roll/pitch 的变化量（rad）
- `Delta_omega_i`：DROP 前后角速度变化量（rad/s）
- `omega_{xy,i}^{drop}`：DROP 前瞬间角速度平面范数，`omega_{xy,i}^{drop} = ||[w_x, w_y]||_2`（rad/s）

指示函数：

- `I_wait = 1[child_has_dropped == False]`
- `I_drop = 1[step_drop_event_mask == True]`
- `I_done = 1[termination or truncation]`
- `I_no_drop_done = 1[I_done and child_has_dropped == False]`

---

## 2. 总奖励结构

代码中的每步奖励可写为：

`R_i(t) = R_progress,i(t) + R_drop,i(t) + R_alt,i(t) + R_no_drop,i(t)`

其中：

- `R_progress,i` 仅在未投放阶段生效
- `R_drop,i` 仅在 DROP 当步生效
- `R_alt,i` 每步生效
- `R_no_drop,i` 仅在 done 且从未 DROP 时生效

---

## 3. 各分项公式

### 3.1 DROP 前距离进展项（方向引导）

`R_progress,i(t) = w_dir * (d_i^xy(t-1) - d_i^xy(t)) * I_wait`

其中 `d_i^xy(t) = ||p_target,xy - p_mother,xy||_2`。

说明：

- 靠近目标：`d(t-1) - d(t) > 0`，该项为正
- 远离目标：该项为负

### 3.2 DROP 当步项

DROP 当步先计算连续精度得分、冲击惩罚、姿态奖励，再应用“区域外覆盖惩罚”。

#### 3.2.1 连续精度得分

`S_i = score_max * exp(- (e_i^xy / d0)^p )`

`R_score,i = w_score * S_i`

其中：

- `e_i^xy = ||p_landing,xy - p_target,xy||_2`
- `d0 = score_d0`
- `p = score_p`

#### 3.2.2 冲击指标与冲击惩罚

`Delta_theta_i = || [roll_after - roll_before, pitch_after - pitch_before] ||_2`

`Delta_omega_i = || omega_after - omega_before ||_2`

`impulse_i = alpha * Delta_theta_i + beta * Delta_omega_i`

`R_impulse,i = -lambda_imp * impulse_i`

其中：

- `alpha = impulse_alpha`
- `beta = impulse_beta`
- `lambda_imp = impulse_penalty_weight`

#### 3.2.3 稳定投放奖励（姿态 + 角速度两项）

`theta_drop,i = || [roll_before, pitch_before] ||_2`

`R_posture,i = w_att * exp( - (theta_drop,i / theta0)^2 )`

`R_angvel,i = w_att * exp( - (omega_{xy,i}^{drop} / theta0)^2 )`

`R_att,i = (R_posture,i + R_angvel,i) * 1[S_i > 0]`

其中：

- `theta0 = attitude_theta0`
- `w_att = attitude_reward_weight`

说明：

- 第二项（角速度稳定项）与第一项（姿态稳定项）使用**同一权重和同一尺度**；
- 代码里复用了历史缓冲名 `step_drop_heading_error`，其当前含义为 `omega_{xy}^{drop}`（不再是方向夹角）。

#### 3.2.4 DROP 当步合成与区域外覆盖

先定义外圈阈值：

`d_outer = piecewise_r * piecewise_thresholds[-1]`

DROP 当步奖励为分段形式：

- 若 `e_i^xy <= d_outer`：
  `R_drop,i = R_score,i + R_impulse,i + R_att,i`
- 若 `e_i^xy > d_outer`：
  `R_drop,i = -P_out`

其中 `P_out = outside_region_penalty`。

### 3.3 高度软惩罚（每步）

`z_low = z_target - z_tol`

`R_alt,i(t) = -w_alt * max(z_low - z_i(t), 0)`

其中：

- `z_target = altitude_target_z`
- `z_tol = altitude_tolerance`
- `w_alt = altitude_low_penalty_weight`

### 3.4 未投放终止惩罚

`R_no_drop,i(t) = -P_no_drop * I_no_drop_done`

其中 `P_no_drop = no_drop_penalty`。

---

## 4. 当前代码参数值（drop_reward_config）

- `direction_reward_weight = 0.3`
- `direction_min_target_dist = 0.1`（当前奖励逻辑未使用，保留为兼容参数）
- `score_reward_weight = 1.0`
- `score_max = 20.0`
- `score_d0 = 3.6`
- `score_p = 1.0`
- `impulse_alpha = 1.0`
- `impulse_beta = 0.5`
- `impulse_penalty_weight = 0.3`
- `attitude_reward_weight = 0.5`
- `attitude_theta0 = 0.12`
- `outside_region_penalty = 20.0`
- `no_drop_penalty = 20.0`
- `altitude_target_z = 10.0`
- `altitude_tolerance = 0.5`
- `altitude_low_penalty_weight = 1.0`

兼容性保留但当前 `R_att` 不使用：

- `drop_angle_reward_weight`
- `drop_angle_theta0`
- `drop_angle_min_speed`

由当前阈值列表可得：

- `piecewise_r = 2.0`
- `piecewise_thresholds[-1] = 12.2`
- `d_outer = 24.4 m`

---

## 5. 与奖励触发相关的开关（非奖励权重）

- `drop_threshold = 0.7`：`drop_switch > drop_threshold` 触发 DROP
- `allow_multiple_drops = False`：每个 episode 仅允许一次 DROP

---

## 6. 代码逻辑一致性说明

- 奖励在 DROP 后当步计算完成，然后该环境当步置 `truncation=1` 结束 episode。
- `R_no_drop` 在 done 判定后追加，因此是终止附加惩罚。
- 文档公式与当前代码一一对应，不包含额外假设或改写。
