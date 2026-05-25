# NavigationTaskGmmNoise 投放瞬态物理建模说明（当前代码版）

本文档基于当前代码实现，整理 `navigation_task_gmm_noise` 中与投放瞬态过程相关的物理建模、事件时序、反冲建模、子机释放初始条件与自由落体预测逻辑。

对齐文件：

- 任务实现：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)
- 任务配置：[navigation_task_gmm_noise_config.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)
- 奖励逻辑详解另见：[drop_reward_function_spec.md](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/docs/drop_reward_function_spec.md)

更新时间：2026-05-18

---

## 1. 文档适用范围

本文档讨论的是：

- 母机何时被允许执行投放
- 投放瞬间母机刚体属性如何更新
- 投放反冲如何作用于母机
- 子机释放时的初始位置与初始速度如何确定
- 子机如何在当前风场中积分到落地

需要强调的是：

- 这是一套**事件式投放瞬态近似模型**
- 它已经显式考虑了质量、惯量、质心和偏心反冲力矩
- 但它仍然不是完整连续多体分离动力学模型

---

## 2. 动作与“请求投放”判定

当前动作空间为：

```text
a_t = [u_x, u_y, u_z, u_yaw, u_drop]
```

其中：

- 前四维是母机控制量
- 第五维 `u_drop` 是投放请求信号

策略输出范围为：

```text
[-1, 1]
```

首先定义“请求投放”条件：

```text
requested_drop <=> u_drop > drop_threshold
```

当前默认参数：

- `drop_threshold = 0.7`

但要注意：

> 在当前代码里，“请求投放”不等于“真正发生投放”。

---

## 3. 从“请求投放”到“真正投放”的门控逻辑

### 3.1 单次投放限制

当前默认：

```text
allow_multiple_drops = False
```

因此默认每个 episode 最多只允许投放一次。

若已经投放过，则后续即使再次输出 `u_drop > threshold`，也不会再执行投放。

### 3.2 硬姿态与硬角速度门限

当前真正执行投放前，还必须通过安全门限：

```text
attitude_deg <= max_release_attitude_deg
omega_xy <= max_release_omega_xy
```

其中：

- `attitude_deg = sqrt(roll^2 + pitch^2)` 再换算为角度制
- `omega_xy = ||[wx, wy]||`

当前默认参数：

- `max_release_attitude_deg = 45.0`
- `max_release_omega_xy = 0.3`

因此，即使策略已经请求投放，只要当前综合倾角或横向角速度超过硬门限，本步仍不会真正执行投放。

### 3.3 可选置信门控

当前代码还支持可选的 release confidence gate。

若：

```text
confidence_gate_enable = True
```

则在硬门限之外，还需要满足：

```text
release_confidence > confidence_threshold
sim_steps >= confidence_min_steps
```

当前默认配置：

- `confidence_gate_enable = False`
- `confidence_threshold = 0.45`
- `confidence_min_steps = 3`

因此在当前默认设置下：

- 实际主要依赖硬姿态/角速度门限
- 置信门控默认不开启

### 3.4 真正投放事件

综合起来，当前代码中的投放事件可写为：

```text
drop_event = candidate_drop_mask AND hard_release_ready AND optional_confidence_ready
```

其中：

- `candidate_drop_mask`：已经请求投放且满足多次投放限制
- `hard_release_ready`：满足硬姿态/硬角速度门限
- `optional_confidence_ready`：若开启 confidence gate，还需通过置信度与最少步数要求

因此，当前实现必须区分：

- **请求投放**
- **允许投放**
- **真正发生投放**

这和早期“`u_drop > threshold` 就直接投放”的版本不同。

---

## 4. 当前投放事件时序

在当前实现中，一次真实投放事件按如下顺序进行：

1. 从动作中提取 `drop_switch`
2. 判定是否满足“请求投放”条件
3. 再经过多次投放限制、硬姿态门限、硬角速度门限以及可选 confidence gate
4. 对于最终允许投放的环境：
   - 计算子机释放位置与释放速度
   - 记录母机释放前姿态、线速度、角速度
   - 更新母机刚体属性
   - 施加释放反冲
5. 执行当前 RL 控制步对应的仿真步进
6. 记录母机释放后状态
7. 计算冲击指标
8. 用子机自由落体模型积分到落地，得到落点误差
9. 结算 DROP 奖励
10. 当前环境在该步奖励计算后立即结束 episode（通过 `truncation=1`）

因此：

- 投放是一个**事件式瞬时更新 + 单控制步响应**过程
- 不是持续若干步的连续分离过程

---

## 5. 固定挂点释放模型

### 5.1 挂点配置

当前代码支持多个挂点，但默认不是随机挂点，而是固定挂点：

```text
random_fixed_payload_mount = False
fixed_payload_mount_index = 0
```

挂点列表（机体系，单位 m）为：

```text
r1 = [ 0.0919,  0.0919, -0.13]
r2 = [ 0.0919, -0.0919, -0.13]
r3 = [-0.0919, -0.0919, -0.13]
r4 = [-0.0919,  0.0919, -0.13]
```

因此当前默认实际使用的是：

```text
r_mount_body = r1 = [0.0919, 0.0919, -0.13]
```

### 5.2 挂点相对母机的物理含义

由于释放位置和反冲力矩都与挂点位置有关，因此当前模型不是“从母机质心直接释放”。

相反，当前模型认为：

- 子机从偏心挂点位置释放
- 偏心挂点会带来相对于更新后质心的反冲力矩

这也是当前投放瞬态建模比最简化“中心点丢载荷”模型更接近真实机构的原因之一。

---

## 6. 母机刚体属性的瞬时更新

这是当前实现的核心之一。

### 6.1 为什么要更新母机刚体属性

当前代码不是简单地“给母机施加一个外力”就结束，而是先显式模拟：

- 载荷脱离后母机质量减少
- 载荷脱离后母机惯量改变
- 载荷脱离后母机整机质心改变

只有先做这一步，后续反冲力和反冲力矩才会基于释放后的动力学属性计算。

### 6.2 更新方法

在投放瞬间，代码会：

1. 找到当前挂点对应的 payload 刚体
2. 将该 payload 的质量改为一个极小残余质量
3. 将该 payload 的惯量改为极小残余惯量
4. 用剩余刚体重新聚合整机：
   - 总质量
   - 总惯量
   - 总质心

当前残余参数默认采用：

```text
dropped_payload_residual_mass    = 1e-4
dropped_payload_residual_inertia = 1e-6
```

因此在数值上可以近似视为：

- 被投放载荷已从母机动力学系统中移除

### 6.3 聚合计算的物理形式

对各个刚体 `j`，代码会聚合：

```text
mass_j
com_j
I_j
```

总质心满足：

```text
com_total = (sum_j mass_j * com_j) / (sum_j mass_j)
```

总惯量满足平行轴定理形式的聚合：

```text
I_total = sum_j [ R_j I_j R_j^T + mass_j * parallel_axis_shift ]
```

因此当前代码里，DROP 后母机的：

- `robot_mass`
- `robot_inertia`
- `_robot_com_body`

都会更新。

### 6.4 物理意义

所以当前释放瞬态建模已经包含了：

1. 释放后母机质量变化
2. 释放后母机惯量变化
3. 释放后母机质心变化

但它仍然是：

- **事件式瞬时更新**
- 而不是连续多体分离过程

---

## 7. 母机反冲模型

### 7.1 默认模式

当前默认：

```text
use_force_recoil = True
```

即采用：

> 一控制步内的等效外力 / 外力矩注入

而不是直接对母机速度做瞬时人工改写。

### 7.2 等效分离速度

为与当前代码变量保持一致，代码中仍使用：

```text
eject_speed
eject_direction_body
v_eject_body
v_eject_world
```

但在本文档以及论文表述中，这些量更准确的物理含义应理解为：

> 沿释放轴方向定义的等效分离速度，用于等效表征释放机构脱扣、清离与瞬时冲量交换过程，而不表示存在主动弹射器。

当前模型中，等效分离速度在机体系下定义为：

```text
v_eject_body = eject_speed * eject_direction_body
```

当前默认参数：

```text
eject_speed = 0.5 m/s
eject_direction_body = [0, 0, -1]
```

再通过母机姿态旋转到世界系：

```text
v_eject_world = R_WB * v_eject_body
```

因此当前模型并不是假设“子机被主动发射出去”，而是采用一个沿释放轴方向的小相对分离速度，借此建立与动量守恒一致的释放瞬态反冲模型。

### 7.3 目标反冲速度增量

代码首先构造母机的目标反冲速度：

```text
delta_v_target_body = -(m_child / m_dyn) * v_eject_body
```

其中：

- `m_child`：子机/载荷质量
- `m_dyn`：DROP 后当前母机的动力学质量

这一步的物理意义是：

- 子机在释放瞬间相对母机具有一个沿释放轴的小等效分离速度
- 为满足动量守恒，母机获得反向速度增量

注意：

- 这里默认优先使用更新后的 `robot_mass`
- 因此反冲量与释放后的真实聚合质量保持一致

### 7.4 一控制步等效反冲力

为了把目标 `delta_v` 转成仿真中的外力输入，代码使用：

```text
control_dt = sim_dt * num_physics_steps_per_env_step_mean
F_recoil_body = (m_dyn / control_dt) * delta_v_target_body
F_recoil_world = R_WB * F_recoil_body
```

并写入：

```text
task_external_force_tensor
```

其物理含义是：

- 当前不是解析瞬时冲量模型
- 而是将该冲量等效为一个控制步平均外力

### 7.5 相对更新后质心的偏心反冲力矩

若挂点偏离更新后的母机质心，则还会产生反冲力矩：

```text
r_body = r_mount_body - com_new_body
tau_recoil_body = r_body x F_recoil_body
tau_recoil_world = R_WB * tau_recoil_body
```

并写入：

```text
task_external_torque_tensor
```

这意味着当前模型中的反冲力矩并不是简单的：

```text
tau = r_mount x F
```

而是：

> 相对释放后母机新质心计算的偏心反冲力矩

这是当前代码里已经显式考虑质心变化影响的关键点。

### 7.6 旧版兼容路径

若：

```text
use_force_recoil = False
```

则代码退回到旧版兼容路径：

- 直接给母机线速度注入 `delta_v`
- 若开启随机角速度 kick，还会再注入随机 `delta_omega`

但这不是当前默认实现。

---

## 8. 子机释放初始条件

当前子机的释放初始条件比旧版本更完整。

### 8.1 释放位置

当前释放位置不是单纯：

```text
p_child(t0) = p_mother(t0)
```

而是：

```text
p_child(t0) = p_mother(t0) + r_mount_world
```

其中：

```text
r_mount_world = R_WB * r_mount_body
```

因此当前子机是从母机偏心挂点位置释放，而不是从母机参考点直接释放。

### 8.2 释放速度

当前子机释放速度也不是只继承母机平动速度，而是：

```text
v_child(t0) = v_mother(t0) + omega_world(t0) x r_mount_world + v_eject_world
```

其中：

- 第一项：母机平动速度
- 第二项：挂点相对于母机角速度产生的切向速度
- 第三项：沿释放轴方向的等效分离速度

若：

```text
add_child_eject_velocity = False
```

则最后一项被去掉。

当前默认：

```text
add_child_eject_velocity = True
```

因此在当前默认设置下，子机释放速度已经显式考虑了：

1. 母机平动
2. 母机转动引起的挂点切向速度
3. 等效分离速度

这比旧版“只继承母机速度”或“只附加一个未区分物理含义的额外速度项”的描述更完整。

---

## 9. 子机自由落体模型

### 9.1 基本模型

当前子机在投放后采用点质量自由落体积分到地面，其加速度模型为：

```text
a_child = g + (c_child / m_child) * (w_total - v_child)
```

其中：

- `g = [0, 0, -child_gravity]`
- `c_child = child_drag_coefficient`
- `m_child = child_mass`
- `w_total`：当前子机所在位置的风速向量

当前默认参数：

- `child_gravity = 9.81`
- `child_drag_coefficient = 1.0`
- `child_mass = 1.0`

### 9.2 当前风场来源

子机自由落体阶段使用的是任务当前真实风场。

当前风场统一写成：

```text
w_total = w_main + w_local
```

但在当前默认配置下：

- 主风始终开启
- `Dryden turbulence` 默认开启

因此默认实际可以更准确地写为：

```text
w_total = w_main + w_dryden
```

若关闭 Dryden，则退回到：

```text
w_total = w_main + w_local_gust
```

### 9.3 积分形式

当前离散积分形式为：

```text
v_{k+1} = v_k + a_k * dt
p_{k+1} = p_k + v_{k+1} * dt
```

其中：

- `dt` 默认来自仿真步长 `0.01 s`

### 9.4 落地条件

当满足：

```text
z_{k+1} <= 0
```

时认为子机落地，并记录：

- 落地位置
- 平面落点误差

若达到最大积分步数仍未落地，则使用最后位置做兜底落点。

当前默认：

- `max_child_sim_steps = 5000`

### 9.5 可选轨迹记录

若评估模式下开启：

```text
record_drop_trajectory_for_eval = True
```

则当前代码还会保存子机投放后的重采样轨迹信息，包括：

- 归一化时间轴
- 轨迹位置序列
- 对目标平面误差序列
- 落地总步数与总时间

这部分主要用于后续评估与论文可视化，不改变主物理过程。

---

## 10. 落点误差定义

当前落点评估只使用平面误差：

```text
landing_error_xy = || p_landing_xy - p_target_xy ||_2
```

因此当前投放精度评价不直接使用三维欧氏误差，而是仅关注最终地面落点的 XY 偏差。

此外，当前代码还会额外记录：

- `release_to_target_xy`
- 若开启轨迹记录时的 `post-drop trace`

用于分析释放时机与投放后误差演化。

---

## 11. 当前冲击指标定义

当前 `impulse_metric` 用来表征投放瞬间母机的动态突变，其定义为：

```text
delta_v     = || v_after - v_before ||
delta_omega = || omega_after - omega_before ||

impulse_metric = alpha * delta_v + beta * delta_omega
```

其中：

- `v_before` / `v_after`：投放前后母机线速度
- `omega_before` / `omega_after`：投放前后母机角速度
- `alpha = impulse_alpha`
- `beta = impulse_beta`

当前默认参数：

- `impulse_alpha = 1.0`
- `impulse_beta = 0.8`

因此当前代码的冲击定义重点是：

- 平动速度突变
- 角速度突变

而不是旧版某些描述中的：

- `delta_theta + delta_omega`

这一点在论文里需要严格按当前代码来写。

---

## 12. 与投放物理模型直接相关的当前参数快照

### 12.1 投放触发与门控

- `drop_threshold = 0.7`
- `allow_multiple_drops = False`
- `confidence_gate_enable = False`
- `confidence_threshold = 0.45`
- `confidence_min_steps = 3`
- `max_release_attitude_deg = 45.0`
- `max_release_omega_xy = 0.3`

### 12.2 子机模型

- `enable_drop_model = True`
- `child_gravity = 9.81`
- `child_drag_coefficient = 1.0`
- `child_mass = 1.0`
- `max_child_sim_steps = 5000`

### 12.3 反冲模型

- `enable_impact = True`
- `add_child_eject_velocity = True`
- `use_force_recoil = True`
- `child_mass = 1.0`
- `mother_mass = 11.04`
- `eject_speed = 0.5`
- `eject_direction_body = [0, 0, -1]`
- `random_fixed_payload_mount = False`
- `fixed_payload_mount_index = 0`
- `payload_mount_points_body`：4 个固定偏心挂点
- `enable_random_angular_kick = False`
- `dropped_payload_residual_mass = 1e-4`（若未显式覆盖，则代码默认如此）
- `dropped_payload_residual_inertia = 1e-6`（若未显式覆盖，则代码默认如此）

说明：

- 这里代码变量名仍然沿用 `eject_*`
- 但论文中建议解释为“等效分离速度”及其方向
- 不建议直接表述为“主动弹射速度”，以免与真实机械释放结构不符

### 12.4 与投放物理结果直接相关的当前奖励快照

这里只列与投放物理结果直接相关的奖励项，完整奖励请看：

- [drop_reward_function_spec.md](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/docs/drop_reward_function_spec.md)

当前关键值：

- `score_max = 20.0`
- `score_d0 = 4.0`（注意当前实际还启用了 `score_d0` 课程学习）
- `score_p = 2.0`
- `piecewise_thresholds[-1] = 8.0`
- `outside_region_penalty = 20.0`
- `impulse_alpha = 1.0`
- `impulse_beta = 0.8`
- `impulse_penalty_weight = 0.5`
- `blocked_drop_penalty = 2.0`
- `crash_no_drop_penalty = 20.0`
- `missed_drop_no_drop_penalty = 8.0`
- `reasonable_no_drop_reward = 0.0`

---

## 13. 当前模型边界

当前投放瞬态物理建模已经包含：

1. 投放请求与真正投放之间的门控差异
2. 释放后母机质量变化
3. 释放后母机惯量变化
4. 释放后母机质心变化
5. 相对更新后质心的偏心反冲力矩
6. 挂点位置释放
7. 母机角速度引起的挂点切向释放速度
8. 子机带风点质量自由落体积分

另外需要强调：

- 当前代码中的 `eject_speed` 更适合解释为“等效分离速度”
- 它用于通过动量守恒建立释放瞬时反冲
- 不宜在论文中直接表述为“主动弹射速度”

但它仍然不是完整连续分离动力学模型。当前仍然没有：

1. 释放过程中的连续多体耦合
2. 子机与母机的连续接触/滑轨分离过程
3. 柔性连接、结构振动、机构摩擦等高保真机构建模
4. 释放过程中持续的双向耦合动力学

因此，当前实现最准确的表述应是：

```text
事件式投放瞬态等效动力学模型：
在投放事件触发时先更新母机聚合刚体属性，
再基于释放后质量与质心计算等效反冲力和偏心反冲力矩，
并以挂点位置和挂点速度作为子机初始条件，
最后在当前风场中积分子机自由落体直至落地。
```

---

## 14. 一句话总结

当前代码下的投放瞬态物理建模不是“简单反冲 + 直线下落”，而是：

- 有投放前门控逻辑
- 有固定偏心挂点
- 有释放后母机质量/惯量/质心更新
- 有相对新质心计算的偏心反冲力矩
- 有包含 `omega x r_mount` 与等效分离速度的子机释放速度
- 有带主风和 Dryden 扰动的子机自由落体积分

这已经比最简单的瞬时丢点模型更完整，但本质上仍属于事件式近似模型，而非高保真连续多体分离动力学模型。
