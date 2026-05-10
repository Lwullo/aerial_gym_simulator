# NavigationTaskGmmNoise DROP 物理建模说明（当前代码版，无 LaTeX）

本文档按当前代码实现整理 DROP 相关的物理建模、事件顺序和关键参数。

对齐文件：

- 任务实现：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)
- 任务配置：[navigation_task_gmm_noise_config.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)
- 奖励逻辑详解另见：[drop_reward_function_spec.md](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/docs/drop_reward_function_spec.md)

更新时间：2026-04-10

## 1. 动作与 DROP 触发

动作空间：

```text
a_t = [u_x, u_y, u_z, u_yaw, u_drop]
```

每个分量范围：

```text
[-1, 1]
```

DROP 触发条件：

```text
DROP 发生 <=> u_drop > drop_threshold
```

当前：

- `drop_threshold = 0.7`
- `allow_multiple_drops = False`

因此默认每个 episode 最多发生一次 DROP。

## 2. 当前 DROP 事件顺序

一次 DROP 事件在当前实现里按下面顺序发生：

1. 从动作中取出 `drop_switch`
2. 判定 `drop_switch > 0.7`
3. 记录释放前状态：
   - 母机位置
   - 母机线速度
   - roll / pitch / yaw
   - 角速度
4. 如启用反冲模型，先更新母机刚体属性
5. 对母机施加释放反冲
6. 当前 RL step 内执行一次仿真步进
7. 记录释放后母机状态
8. 计算冲击指标
9. 用子机自由落体模型积分到落地，得到落点误差
10. 计算 DROP 奖励
11. 当前环境立即结束 episode（`truncation=1`）

对应代码主入口：
- [navigation_task_gmm_noise.py:3568](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3568)

## 3. 固定挂点释放模型

当前代码支持多挂点，但默认不是随机挂点，而是固定挂点。

配置：

```text
random_fixed_payload_mount = False
fixed_payload_mount_index = 0
```

挂点列表（机体系，单位 m）：

```text
r1 = [ 0.0919,  0.0919, -0.13]
r2 = [ 0.0919, -0.0919, -0.13]
r3 = [-0.0919, -0.0919, -0.13]
r4 = [-0.0919,  0.0919, -0.13]
```

因此当前默认实际使用的是：

```text
r_mount = r1 = [0.0919, 0.0919, -0.13]
```

若后续把 `random_fixed_payload_mount=True` 且 `fixed_payload_mount_index` 不合法，代码才会在多个挂点中随机选一个并在整次运行期间固定。

对应代码：
- 挂点初始化：[navigation_task_gmm_noise.py:317](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:317)

## 4. 母机释放后的质量、惯量与质心更新

这是当前实现里和旧版文档差异最大的部分之一。

当前不是简单地“给一个反冲力就结束”，而是先更新母机的聚合刚体属性。

### 4.1 更新内容

在 DROP 瞬间，代码会：

1. 找到当前挂点对应的 payload 刚体
2. 将该 payload 刚体质量改成接近 0 的残余值
3. 将该 payload 刚体惯量改成极小残余惯量
4. 基于剩余各刚体重新计算整机：
   - 总质量
   - 总惯量
   - 总质心

### 4.2 聚合动力学计算

对每个刚体 `j`：

```text
mass_j
com_local_j
inertia_body_j
```

代码先把每个刚体的局部质心变换到基座坐标系，再计算总质心：

```text
com_total = (sum_j mass_j * com_j) / (sum_j mass_j)
```

总惯量通过平行轴定理聚合：

```text
I_total = sum_j [ R_j * I_body_j * R_j^T + mass_j * shift(com_j - com_total) ]
```

因此当前实现里：

- 总质量会变
- 总惯量会变
- 总质心也会变

对应代码：
- 聚合质量/惯量/质心计算：[navigation_task_gmm_noise.py:1157](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:1157)
- DROP 时母机刚体更新：[navigation_task_gmm_noise.py:1294](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:1294)

### 4.3 物理含义

所以当前释放建模已经包含了：

- 释放后母机质量变化
- 释放后母机惯量变化
- 释放后母机质心变化

但它仍然是：

- 事件式瞬时更新
- 不是连续多体分离过程

也就是说：

- 包含“释放后动力学参数变化”
- 不包含“释放过程中连续质心迁移轨迹”

## 5. 母机反冲模型

当前默认使用：

```text
use_force_recoil = True
```

即优先走“外力/外力矩注入”路径，而不是直接改速度。

### 5.1 弹射速度

释放方向在机体系定义为：

```text
v_eject_body = eject_speed * eject_direction_body
```

当前参数：

```text
eject_speed = 0.5 m/s
eject_direction_body = [0, 0, -1]
```

转到世界系：

```text
v_eject_world = R_WB * v_eject_body
```

对应代码：
- [navigation_task_gmm_noise.py:2030](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:2030)

### 5.2 目标反冲速度增量

代码按质量比给母机构造目标反冲速度：

```text
delta_v_target_body = -(m_child / m_dyn) * v_eject_body
```

其中：

- `m_child`：载荷质量
- `m_dyn`：DROP 后当前母机的动力学质量

注意：

- 这里不是直接用配置里固定的 `mother_mass`
- 默认优先使用更新后的 `robot_mass`

所以反冲量会跟释放后的真实聚合质量一致。

### 5.3 一步等效反冲力

为了把目标 `delta_v` 转成仿真里的外力输入，当前实现使用：

```text
control_dt = sim_dt * num_physics_steps_per_env_step_mean
F_recoil_body = (m_dyn / control_dt) * delta_v_target_body
```

再转到世界系：

```text
F_recoil_world = R_WB * F_recoil_body
```

并写入：

```text
task_external_force_tensor
```

含义：

- 当前是“一控制步平均等效反冲力”
- 不是瞬时解析冲量

对应代码：
- [navigation_task_gmm_noise.py:2086](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:2086)

### 5.4 相对更新后质心的反冲力矩

当前扭矩不是简单的：

```text
tau = r_mount x F
```

而是：

```text
r_body = r_mount - com_new
tau_recoil_body = r_body x F_recoil_body
tau_recoil_world = R_WB * tau_recoil_body
```

也就是说，反冲力矩是**相对更新后的整机质心**计算的。

这是你当前实现里已经包含“质心变换影响”的关键点。

对应代码：
- [navigation_task_gmm_noise.py:2111](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:2111)

### 5.5 旧版兼容路径

如果：

```text
use_force_recoil = False
```

则走旧版兼容路径：

- 直接给母机线速度加 `delta_v`
- 可选再叠加随机角速度 kick

但这不是当前默认实现。

## 6. 子机释放初始条件

### 6.1 初始位置

当前子机释放位置：

```text
p_child(t0) = p_mother(t_drop)
```

### 6.2 初始速度

当前子机初速度不是只继承母机速度，而是：

```text
v_child(t0) = v_mother(t_drop) + v_eject_world
```

前提是：

```text
add_child_eject_velocity = True
```

当前默认该开关就是 `True`。

这点和旧版文档不同。旧版写的是“不叠加弹射速度”，当前代码已经不是这样。

对应代码：
- [navigation_task_gmm_noise.py:3580](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3580)

## 7. 子机自由落体模型

子机在 DROP 后采用点质量自由落体积分到地面。

### 7.1 加速度模型

```text
a_child = g + (c_child / m_child) * (w_total - v_child)
```

其中：

- `g = [0, 0, -child_gravity]`
- `c_child = child_drag_coefficient`
- `m_child = child_mass`
- `w_total`：当前位置风向量

### 7.2 当前风场来源

子机自由落体阶段调用的是任务当前风场模型：

```text
w_total = main_wind + local_wind
```

在当前默认配置下：

- `num_noise_sources = 0`
- `dryden_enabled = True`

所以实际主要是：

```text
w_total = w_main + w_dryden
```

如果关闭 Dryden，则会退回到主风 + 局部方向扰动的旧命名路径。

### 7.3 离散积分

当前积分形式：

```text
v_{k+1} = v_k + a_k * dt
p_{k+1} = p_k + v_{k+1} * dt
```

### 7.4 落地条件

```text
z_{k+1} <= 0
```

若达到最大积分步数仍未落地，则用当前最后位置做兜底落点。

对应代码：
- [navigation_task_gmm_noise.py:2216](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:2216)

## 8. 落点误差与后处理

当前落点评估只看平面误差：

```text
landing_error_xy = || p_landing_xy - p_target_xy ||_2
```

对应：

- `child_landing_position`
- `child_landing_xy_distance`

此外当前实现还会保存：

- `release_to_target_xy`
- 投放后重采样轨迹 `post_drop_trace`

用于后续论文图和 `post_drop_error_evolution_wind.pdf`。

## 9. 当前冲击指标定义

当前 `impulse_metric` 的定义是：

```text
delta_v     = || v_after - v_before ||
delta_omega = || omega_after - omega_before ||

impulse_metric = alpha * delta_v + beta * delta_omega
```

当前参数：

- `alpha = impulse_alpha = 1.0`
- `beta  = impulse_beta  = 0.8`

注意：

- 当前不是用 `delta_theta + delta_omega`
- 而是用 `delta_v + delta_omega`

这点和旧版部分描述不同，当前代码应以这里为准。

对应代码：
- [navigation_task_gmm_noise.py:3628](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3628)

## 10. 与 DROP 直接相关的当前参数快照

### 10.1 DROP 触发与子机模型

- `drop_threshold = 0.7`
- `allow_multiple_drops = False`
- `enable_drop_model = True`
- `child_gravity = 9.81`
- `child_drag_coefficient = 1.0`
- `child_mass = 1.0`
- `max_child_sim_steps = 5000`

### 10.2 反冲模型

- `enable_impact = True`
- `add_child_eject_velocity = True`
- `use_force_recoil = True`
- `child_mass = 1.0`
- `mother_mass = 11.04`
- `eject_speed = 0.5`
- `eject_direction_body = [0, 0, -1]`
- `random_fixed_payload_mount = False`
- `fixed_payload_mount_index = 0`
- `payload_mount_points_body = 4` 个固定偏心挂点
- `enable_random_angular_kick = False`

### 10.3 与 DROP 直接相关的 reward 快照

这里只列和 DROP 物理结果直接关联的项，完整奖励文档请看：
- [drop_reward_function_spec.md](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/docs/drop_reward_function_spec.md)

当前关键值：

- `score_max = 20.0`
- `score_d0 = 2.0`
- `score_p = 2.0`
- `piecewise_thresholds[-1] = 8.0`
- `outside_region_penalty = 20.0`
- `impulse_alpha = 1.0`
- `impulse_beta = 0.8`
- `impulse_penalty_weight = 0.5`
- `attitude_reward_weight = 0.5`
- `attitude_theta0 = 0.12`

## 11. 当前模型边界

当前 DROP 物理建模已经包含：

1. 释放后母机质量变化
2. 释放后母机惯量变化
3. 释放后母机质心变化
4. 相对更新后质心的反冲力矩
5. 子机带风自由落体积分

但它仍然不是完整连续分离动力学模型。当前没有：

1. 释放过程中的连续多体耦合
2. 载荷脱离过程中的连续质心迁移轨迹
3. 柔性连接、接触碰撞、滑轨分离等高保真释放机构建模
4. 子机与母机的持续耦合动力学

因此当前实现更准确的表述应是：

```text
事件式 DROP 物理模型：
释放瞬间更新母机聚合刚体属性，
通过相对新质心的反冲力/力矩近似母机响应，
并用带风点质量模型积分子机自由落体到落地。
```

## 12. 一句话总结

当前代码逻辑下，你的 DROP 物理建模不是“只有一个简单反冲力”，而是：

- 有固定挂点
- 有释放后母机质量/惯量/质心更新
- 反冲力矩相对新质心计算
- 子机初速度包含弹射速度
- 子机在风场中积分到落地

这已经比最简单的“瞬时丢点模型”更完整，但仍属于事件式近似，不是高保真连续多体分离模型。
