# NavigationTaskGmmNoise DROP模型说明（无LaTex版）

本文档基于当前代码实现，整理 DROP 相关模型、公式和参数。
该版本使用纯文本公式，避免 Markdown 数学渲染不支持导致的乱码。

## 1. 动作与 DROP 触发

动作空间：
```text
a_t = [u_x, u_y, u_z, u_yaw, u_drop],  每个分量范围 [-1, 1]
```

触发条件：
```text
DROP 发生 <=> u_drop > drop_threshold
```

当前：
- `drop_threshold = 0.7`
- `allow_multiple_drops = False`（每个 episode 最多一次 DROP）

## 2. 固定挂点释放（偏心反冲）

为模拟偏心释放导致的不同反冲：
1. 任务初始化时，在 4 个挂点里随机选 1 个。
2. 该次训练运行期间固定使用这个挂点。
3. 该挂点用于计算反冲力矩 tau = r x F。

挂点坐标（机体系，单位 m）：
```text
r1 = [ 0.0919,  0.0919, -0.13]
r2 = [ 0.0919, -0.0919, -0.13]
r3 = [-0.0919, -0.0919, -0.13]
r4 = [-0.0919,  0.0919, -0.13]
```

## 3. DROP 瞬间母机反冲模型（力实现）

### 3.1 子机弹出速度

```text
v_eject_body = eject_speed * eject_direction_body
```

当前：
```text
eject_speed = 0.5 m/s
eject_direction_body = [0, 0, -1]
```

### 3.2 目标反冲速度增量（按质量比）

按你确认口径：
```text
m_child  = 1.0 kg
m_mother = 11.04 kg
```

目标反冲速度增量：
```text
delta_v_target_body = -(m_child / m_mother) * v_eject_body
```

### 3.3 由目标 delta_v 反推一步等效力

为保持“力实现”路径，同时达到目标反冲量：
```text
F_recoil_body = (m_dyn / dt) * delta_v_target_body
```

其中：
- `m_dyn`：仿真当前动力学质量（来自 `robot_mass`）
- `dt`：当前控制步长

转换到世界系后施加：
```text
F_recoil_world = R_WB * F_recoil_body
```

### 3.4 偏心释放力矩

设固定挂点向量为 `r_payload_body`：
```text
tau_recoil_body  = r_payload_body x F_recoil_body
tau_recoil_world = R_WB * tau_recoil_body
```

说明：
- `F_body` 与 `tau_body` 都在 DROP 事件步写入外力/外力矩张量。
- 这一步不使用“直接改速度冲量法”。

## 4. 子机最简自由落体模型

子机采用无控制器点质量模型，积分到落地。

初始条件：
```text
p_child(t0) = p_mother(t_drop)
v_child(t0) = v_mother(t_drop)
```

加速度模型：
```text
a_child = g + (c_child / m_child) * (w - v_child)
```

当前参数：
```text
g       = [0, 0, -9.81]
c_child = 1.0
m_child = 1.0
```

离散积分：
```text
v_{k+1} = v_k + a_k * dt
p_{k+1} = p_k + v_{k+1} * dt
```

落地判据：
```text
z_{k+1} <= 0
```

落点误差（只看 XY）：
```text
d_xy = || p_landing_xy - p_target_xy ||_2
```

## 5. DROP 奖励核心公式

### 5.1 WAIT 阶段

```text
R_wait     = -time_penalty
R_progress = w_dir * (d_prev_xy - d_curr_xy)
R_WAIT     = R_wait + R_progress
```

### 5.2 DROP 阶段

得分项（连续函数）：
```text
R_score = w_score * score_max * exp( - (d_xy / score_d0)^score_p )
```

冲击指标：
```text
Delta_theta = sqrt( (Delta_roll)^2 + (Delta_pitch)^2 )
Delta_omega = || omega_after - omega_before ||_2
impulse_metric = alpha * Delta_theta + beta * Delta_omega
R_impulse = -lambda * impulse_metric
```

稳定投放项（DROP 瞬间）：
```text
R_att = w_att * exp( - (theta_drop / theta0)^2 )
      + w_ang * exp( - (phi_drop   / phi0)^2 )
```

DROP 总奖励：
```text
R_DROP = R_score + R_impulse + R_att
```

区域外硬惩罚覆盖：
```text
if d_xy > d_out: R_DROP = -P_out
```

未投放惩罚：
```text
if episode_end and no_drop: R = -P_no_drop
```

## 6. 当前最终参数（与代码一致）

### 6.1 DROP 触发与子机模型
- `drop_threshold = 0.7`
- `allow_multiple_drops = False`
- `child_gravity = 9.81`
- `child_drag_coefficient = 1.0`
- `child_mass = 1.0`
- `max_child_sim_steps = 5000`

### 6.2 反冲模型
- `use_force_recoil = True`
- `child_mass = 1.0`
- `mother_mass = 11.04`
- `eject_speed = 0.5`
- `eject_direction_body = [0, 0, -1]`
- `random_fixed_payload_mount = True`
- `fixed_payload_mount_index = -1`（启动时随机选 1 个挂点并固定）
- `payload_mount_points_body`：4 个偏心挂点（见第 2 节）
- `enable_random_angular_kick = False`

### 6.3 奖励参数
- `time_penalty = 0.0`
- `direction_reward_weight = 0.3`
- `score_reward_weight = 1.0`
- `score_max = 20.0`
- `score_d0 = 3.6`
- `score_p = 1.0`
- `impulse_alpha = 1.0`
- `impulse_beta = 0.5`
- `impulse_penalty_weight = 0.5`
- `attitude_reward_weight = 0.5`
- `attitude_theta0 = 0.12`
- `drop_angle_reward_weight = 0.5`
- `drop_angle_theta0 = 0.35`
- `outside_region_penalty = 20.0`
- `no_drop_penalty = 20.0`
- `altitude_target_z = 10.0`
- `altitude_tolerance = 0.5`
- `altitude_low_penalty_weight = 1.0`

## 7. 备注

1. 当前反冲是“外力/外力矩注入”路径，不是直接改速度冲量路径。
2. 当前子机初速度继承母机速度，未额外叠加 `v_eject` 到子机初速度。
3. 当前每个 episode 只允许一次 DROP。
