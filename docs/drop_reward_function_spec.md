# Navigation Task GMM Noise：当前奖励函数说明（中文）

本文档严格对齐当前代码实现：

- 任务配置：[navigation_task_gmm_noise_config.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)
- 任务实现：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)

当前文档更新时间：2026-05-17

---

## 1. 当前奖励结构总览

当前版本的奖励不再是早期那套 `J`、悬停、安全、碰撞等混合势函数逻辑，而是围绕“**什么时候投放**、**投得准不准**、**投放动作是否平稳**、**不投放是否合理**”来设计。

从代码实现上看，总奖励由两部分组成：

```text
R_total = R_step_main + R_terminal_adjust
```

其中：

1. `R_step_main`

- 在 `_compute_reward_and_scores()` 中计算；
- 包含等待阶段 shaping、投放当步精度奖励、冲击惩罚、高度软约束。

2. `R_terminal_adjust`

- 在 `step()` 末尾对 done 但未投放的环境再做附加处理；
- 包含：
  - 被门控拦下的请求投放惩罚
  - crash + no_drop 惩罚
  - timeout + no_drop 的“合理不投 / 错过投放窗口”分支处理

因此，更准确地写可以表示为：

```text
R_total = R_wait_or_drop
        + R_alt
        + R_blocked_drop
        + R_no_drop_terminal
```

---

## 2. 主奖励入口

当前主奖励入口在：

- [_compute_reward_and_scores()](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:2028)

当前终止附加处理在：

- [step() 中 no-drop / blocked-drop 逻辑](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3411)

---

## 3. WAIT 阶段奖励

当该环境本步尚未发生投放时，奖励主要由两类 shaping 项构成。

### 3.1 平面距离进展奖励

公式为：

```text
R_dir = w_dir * (d_prev_xy - d_curr_xy)
```

其中：

- `d_prev_xy`：上一时刻母机到目标点的平面距离
- `d_curr_xy`：当前时刻母机到目标点的平面距离
- `w_dir = direction_reward_weight`

当前默认参数：

- `direction_reward_weight = 0.4`

物理意义：

- 朝目标靠近：正奖励
- 远离目标：负奖励

这一项只在 **尚未投放** 时生效。

### 3.2 预测投放误差改进奖励

当前代码会在每一步估计一次：

> “如果现在立刻投放，预计落点误差会是多少”

然后奖励这个预测误差相对于上一时刻是否变好了。

公式为：

```text
delta_e_pred = clip(e_pred_prev - e_pred_curr, -c, c)
R_pred = w_pred * delta_e_pred
```

其中：

- `e_pred_prev`：上一时刻的预测投放误差
- `e_pred_curr`：当前时刻的预测投放误差
- `c = pred_error_improvement_clip`
- `w_pred = pred_error_shaping_weight`

当前默认参数：

- `pred_error_shaping_weight = 0.5`
- `pred_error_improvement_clip = 1.0`

物理意义：

- 如果当前状态比上一时刻更适合投放，给正奖励
- 如果当前状态比上一时刻更不适合投放，给负奖励

注意这里奖励的是“**改进量**”，不是每一步直接惩罚绝对预测误差。

### 3.3 WAIT 阶段小结

因此在未投放阶段，可以近似写成：

```text
R_wait = R_dir + R_pred
```

---

## 4. DROP 当步奖励

一旦本步真正发生投放，当前奖励主项切换为：

```text
R_drop = R_score + R_impulse
```

注意：

- 当前版本 **没有单独的姿态奖励项**；
- 姿态稳定性更多体现在：
  - 投放是否能通过硬门限 / confidence gate
  - 投放后的冲击指标是否过大

### 4.1 连续落点精度奖励

当前采用连续指数型精度分数：

```text
raw_score = score_max * exp(-(landing_error_xy / score_d0)^score_p)
R_score = score_reward_weight * raw_score
```

其中：

- `landing_error_xy`：子机最终落点到目标点的平面误差
- `score_max`：理论最大奖励上限
- `score_d0`：误差衰减尺度
- `score_p`：衰减幂次

当前默认参数：

- `score_max = 20.0`
- `score_reward_weight = 1.0`
- `score_d0 = 4.0`
- `score_p = 2.0`

这项奖励的性质是：

- 误差越小，得分越高
- 误差越大，得分按指数形式快速下降

### 4.2 冲击惩罚

当前投放冲击指标定义为：

```text
impulse_metric = alpha * delta_v + beta * delta_omega
R_impulse = -lambda_imp * impulse_metric
```

其中：

- `delta_v`：投放前后母机线速度变化范数
- `delta_omega`：投放前后母机角速度变化范数
- `alpha = impulse_alpha`
- `beta = impulse_beta`
- `lambda_imp = impulse_penalty_weight`

当前默认参数：

- `impulse_alpha = 1.0`
- `impulse_beta = 0.8`
- `impulse_penalty_weight = 0.5`

物理意义：

- 投放导致母机速度突变越大，惩罚越大
- 投放导致母机角速度突变越大，惩罚越大

### 4.3 DROP 当步的基础形式

因此，在未触发外圈惩罚时，DROP 当步奖励为：

```text
R_drop = R_score + R_impulse
```

---

## 5. 外圈超界惩罚

当前代码仍保留了一个“外圈阈值”逻辑，但它已经不是旧版那种“直接固定改写为 -20”，而是：

> 超过最外层阈值后，按超出的距离线性给负惩罚，并裁剪到下限 `-outside_region_penalty`

### 5.1 外圈阈值

当前阈值由：

```text
piecewise_r = 1.0
piecewise_thresholds = [0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 8.0]
```

得到最外层阈值：

```text
d_outer = 8.0 m
```

### 5.2 超界惩罚形式

当：

```text
landing_error_xy > d_outer
```

时，代码计算：

```text
outside_error = landing_error_xy - d_outer
R_outside = clamp(-outside_error, min=-outside_region_penalty, max=0.0)
```

并将 DROP 奖励改为：

```text
R_drop = R_outside + R_impulse
```

同时：

- 原本的 `R_score` 被清零

当前默认参数：

- `outside_region_penalty = 20.0`

因此现在的逻辑是：

- 外圈内：按连续精度得分
- 外圈外：精度正奖励清零，改为“超出越多，惩罚越大”，但最差不低于 `-20`

---

## 6. 高度软约束惩罚

当前全程每一步都会叠加一个高度下界软惩罚：

```text
R_alt = -w_alt * max(z_low - z, 0)
```

其中：

- `z`：当前母机高度
- `z_low`：高度下界参考
- `w_alt = altitude_low_penalty_weight`

当前代码优先采用：

```text
z_low = spawn_height_reference - altitude_tolerance
```

若没有该参考，则退化为：

```text
z_low = altitude_target_z - altitude_tolerance
```

当前默认参数：

- `altitude_target_z = 10.0`
- `altitude_tolerance = 0.5`
- `altitude_low_penalty_weight = 1.0`

因此，实际效果是：

- 高度高于下界：不惩罚
- 高度低于下界：按“低了多少”线性惩罚

---

## 7. 被拦截投放请求惩罚

当前代码里，策略即使输出了 `drop_switch > drop_threshold`，也不一定能真正投放。

如果请求投放时未通过以下条件之一：

- 硬姿态门限
- 硬角速度门限
- 若开启 `confidence_gate`，还要通过置信门限和最少步数要求

则本步会被记为“请求了投放，但没有被允许执行”。

这类情况会额外受到惩罚：

```text
R_blocked_drop = -blocked_drop_penalty
```

当前默认参数：

- `blocked_drop_penalty = 2.0`

说明：

- 当 `confidence_gate_enable = False` 时，这个惩罚仍然可能生效，因为还有硬姿态/角速度门限
- 这个惩罚不是单独替换整步奖励，而是在已有奖励基础上再减去这部分

---

## 8. 未投放终止奖励

这是当前奖励设计中很重要的一部分。

如果一个 episode 结束时仍然没有发生投放，代码不会统一给一个固定惩罚，而是按终止原因再细分。

### 8.1 crash + no_drop

若 episode 因 crash 结束，且全程未投放：

```text
R_no_drop_crash = -crash_no_drop_penalty
```

当前默认参数：

- `crash_no_drop_penalty = 20.0`

这表示：

- 撞毁且没投放，被视为明确失败

### 8.2 timeout + no_drop

若 episode 因超时结束，且全程未投放，代码会在终止时刻做一次“若现在投放”的预测评估。

评估时会：

1. 计算当前时刻若投放，子机的初始位置和初速度；
2. 调用落点预测模型；
3. 得到当前时刻若投放的预测落点误差 `pred_error_xy`。

然后再细分两类。

#### 8.2.1 合理不投

若同时满足：

- `pred_error_xy > reasonable_no_drop_pred_error_threshold`
- 当前母机到目标的平面距离不算特别远，即 `timeout_dist_xy <= no_drop_eval_max_xy_dist`

则认为：

> 当前时刻即使投放也很难投准，因此“不投放”是合理规避风险。

此时加上：

```text
R_reasonable_no_drop = +reasonable_no_drop_reward
```

当前默认参数：

- `reasonable_no_drop_reward = 0.0`
- `reasonable_no_drop_pred_error_threshold = 5.0`
- `no_drop_eval_max_xy_dist = 10.0`

注意你当前配置里：

- `reasonable_no_drop_reward` 已经设成了 `0.0`

也就是说现在“合理不投”并不会额外加分，只是不再被归入“错过投放窗口”的那一类。

#### 8.2.2 错过投放窗口

若没有被判为“合理不投”，则认为：

> 其实存在可接受投放机会，但策略最终没有做出投放决策。

此时加上：

```text
R_missed_no_drop = -missed_drop_no_drop_penalty
```

当前默认参数：

- `missed_drop_no_drop_penalty = 8.0`

---

## 9. 当前奖励的完整近似表达

把当前主逻辑合在一起，可以写成下面这个更接近实现的形式。

### 9.1 未投放且未终止

```text
R = R_dir + R_pred + R_alt
```

### 9.2 本步发生投放，且落点未超外圈

```text
R = R_score + R_impulse + R_alt
```

### 9.3 本步发生投放，且落点超出外圈

```text
R = R_outside + R_impulse + R_alt
```

### 9.4 本步请求投放但被门控拦下

```text
R = R_wait + R_alt - blocked_drop_penalty
```

### 9.5 crash + no_drop

```text
R = R_current - crash_no_drop_penalty
```

### 9.6 timeout + no_drop 且错过窗口

```text
R = R_current - missed_drop_no_drop_penalty
```

### 9.7 timeout + no_drop 且合理不投

```text
R = R_current + reasonable_no_drop_reward
```

---

## 10. 当前真正生效的奖励相关参数

下面只列出当前代码里确实参与主奖励或终止奖励逻辑的参数。

### 10.1 `drop_reward_config`

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `altitude_target_z` | `10.0` | 高度下界的后备参考值 |
| `altitude_tolerance` | `0.5` | 高度容忍带 |
| `altitude_low_penalty_weight` | `1.0` | 高度过低线性惩罚系数 |
| `direction_reward_weight` | `0.4` | 等待阶段平面进展奖励系数 |
| `pred_error_shaping_weight` | `0.5` | 预测投放误差改进奖励系数 |
| `pred_error_improvement_clip` | `1.0` | 预测误差改进量裁剪范围 |
| `score_max` | `20.0` | 连续精度得分上限 |
| `score_d0` | `4.0` | 连续精度得分尺度参数 |
| `score_p` | `2.0` | 连续精度得分幂次 |
| `score_reward_weight` | `1.0` | 精度分数总权重 |
| `score_d0_curriculum_enable` | `True` | 是否启用 `score_d0` 课程学习 |
| `score_d0_curriculum_values` | `[4.0, 3.0, 2.5, 2.0, 1.5]` | `score_d0` 分阶段取值 |
| `score_d0_curriculum_min_drop_rate` | `[0.15, 0.35, 0.55, 0.65]` | 升级到下一阶段所需最小投放率 |
| `score_d0_curriculum_max_error_ema` | `[8.0, 5.0, 3.0, 1.2]` | 升级到下一阶段所需最大落点误差 EMA |
| `piecewise_r` | `1.0` | 外圈阈值缩放因子 |
| `piecewise_thresholds` | `[0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 8.0]` | 外圈阈值序列，最外层用于超界判断 |
| `outside_region_penalty` | `20.0` | 外圈超界惩罚下限 |
| `blocked_drop_penalty` | `2.0` | 请求投放但被门控拦下时的惩罚 |
| `crash_no_drop_penalty` | `20.0` | crash + no_drop 惩罚 |
| `missed_drop_no_drop_penalty` | `8.0` | timeout + no_drop 且错过窗口时的惩罚 |
| `reasonable_no_drop_reward` | `0.0` | timeout + no_drop 且合理不投时的奖励 |
| `reasonable_no_drop_pred_error_threshold` | `5.0` | 合理不投的预测误差阈值 |
| `no_drop_eval_max_xy_dist` | `10.0` | no-drop 评估时的距离过滤阈值 |
| `impulse_alpha` | `1.0` | 冲击指标中 `delta_v` 权重 |
| `impulse_beta` | `0.8` | 冲击指标中 `delta_omega` 权重 |
| `impulse_penalty_weight` | `0.5` | 冲击惩罚总权重 |
| `landing_error_ema_alpha` | `0.9` | 落点误差 EMA 统计系数 |
| `attitude_total_ema_alpha` | `0.9` | 投放姿态总倾角 EMA 统计系数 |

### 10.2 `drop_model_config` 中与奖励联动的参数

这些参数本身不直接写进 reward 公式，但会决定“能不能投放”，从而间接影响奖励。

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `drop_threshold` | `0.7` | 动作维 `drop_switch` 的触发阈值 |
| `allow_multiple_drops` | `False` | 是否允许同一 episode 多次投放 |
| `confidence_gate_enable` | `False` | 是否启用置信门控 |
| `confidence_threshold` | `0.45` | 置信门控阈值 |
| `confidence_min_steps` | `3` | 启用置信门控时的最少步数要求 |
| `max_release_attitude_deg` | `45.0` | 硬姿态上限，超出则禁止投放 |
| `max_release_omega_xy` | `0.3` | 硬角速度上限，超出则禁止投放 |

### 10.3 `drop_impact_config` 中与奖励联动的参数

这些参数会影响投放后的动力学结果，因此会间接影响：

- `landing_error_xy`
- `delta_v`
- `delta_omega`
- `impulse_metric`

典型参数包括：

- `enable_impact`
- `add_child_eject_velocity`
- `use_force_recoil`
- `child_mass`
- `mother_mass`
- `eject_speed`
- `eject_direction_body`
- `payload_mount_points_body`

---

## 11. 当前不应再写成“主奖励项”的旧参数 / 旧概念

下面这些内容在当前实现里已经不再是主奖励结构的一部分，不建议在论文里继续按旧版写法描述：

- `J` / potential reward 主逻辑
- 悬停奖励 `R_hover`
- 安全奖励 `R_safe`
- 动作平滑奖励 `R_smooth`
- 方向对齐余弦奖励 `direction_alignment_reward_magnitude`
- 早期版本中的姿态指数奖励主项
- 固定 `no_drop = -20` 的一刀切逻辑
- “超外圈直接固定改写成 `-20` 且保留旧姿态项”的表述

这些要么已经删除，要么只是保留了遗留配置字段，但不再参与当前主奖励计算。

---

## 12. 当前奖励设计的工程意图

当前版本的奖励目标可以概括为：

1. 鼓励母机逐步飞到更适合投放的位置；
2. 鼓励“此刻投放误差更小”的状态演化趋势；
3. 真正投放时，优先奖励高精度落点；
4. 惩罚投放引起的速度和角速度突变；
5. 对明显不合格的大误差投放进行外圈惩罚；
6. 对“该投却没投”进行惩罚；
7. 对“撞毁且没投”给予更重惩罚；
8. 允许未来通过 `reasonable_no_drop_reward` 控制“合理不投”的偏好强度。

因此，当前奖励并不是单纯“逼着策略尽快投”，而是：

- 鼓励有质量的投放；
- 惩罚粗糙和不稳定的投放；
- 同时区分“没投放到底是失误，还是规避风险”。
