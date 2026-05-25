# 当前奖励函数与参数（NavigationTaskGmmNoise，中文摘要版）

本文档是当前奖励函数的简要摘要版，完整说明请优先参考：

- [docs/drop_reward_function_spec.md](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/docs/drop_reward_function_spec.md)

对应代码：

- [navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)
- [navigation_task_gmm_noise_config.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)

更新时间：2026-05-17

---

## 1. 当前奖励主逻辑

当前奖励已经不再采用旧版的 `J` 势函数 / 悬停 / 安全 / 平滑奖励主框架，而是围绕“投放决策”设计：

```text
R_total = R_wait_or_drop + R_alt + R_blocked_drop + R_no_drop_terminal
```

其中：

- `R_wait_or_drop`：等待阶段 shaping 或投放当步奖励
- `R_alt`：高度软约束惩罚
- `R_blocked_drop`：请求投放但被门限拦下时的惩罚
- `R_no_drop_terminal`：episode 结束时仍未投放的附加惩罚或奖励

---

## 2. WAIT 阶段

未投放时：

```text
R_wait = R_dir + R_pred
```

### 2.1 平面距离进展奖励

```text
R_dir = direction_reward_weight * (d_prev_xy - d_curr_xy)
```

当前参数：

- `direction_reward_weight = 0.4`

### 2.2 预测投放误差改进奖励

```text
delta_e_pred = clip(e_pred_prev - e_pred_curr, -pred_error_improvement_clip, pred_error_improvement_clip)
R_pred = pred_error_shaping_weight * delta_e_pred
```

当前参数：

- `pred_error_shaping_weight = 0.5`
- `pred_error_improvement_clip = 1.0`

---

## 3. DROP 当步

若本步真实发生投放：

```text
R_drop = R_score + R_impulse
```

### 3.1 连续精度奖励

```text
raw_score = score_max * exp(-(landing_error_xy / score_d0)^score_p)
R_score = score_reward_weight * raw_score
```

当前参数：

- `score_max = 20.0`
- `score_reward_weight = 1.0`
- `score_d0 = 4.0`
- `score_p = 2.0`

### 3.2 冲击惩罚

```text
impulse_metric = impulse_alpha * delta_v + impulse_beta * delta_omega
R_impulse = -impulse_penalty_weight * impulse_metric
```

当前参数：

- `impulse_alpha = 1.0`
- `impulse_beta = 0.8`
- `impulse_penalty_weight = 0.5`

注意：

- 当前主奖励里没有单独的姿态指数奖励项
- 姿态主要通过“是否允许投放”和“投放后冲击大小”间接体现

---

## 4. 外圈超界惩罚

最外层阈值由：

- `piecewise_r = 1.0`
- `piecewise_thresholds = [0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 8.0]`

得到：

```text
d_outer = 8.0 m
```

若：

```text
landing_error_xy > d_outer
```

则：

```text
R_drop = R_outside + R_impulse
R_outside = clamp(-(landing_error_xy - d_outer), min=-outside_region_penalty, max=0)
```

当前参数：

- `outside_region_penalty = 20.0`

所以当前不是“超外圈固定 -20”，而是“线性超界惩罚，最差裁剪到 -20”。

---

## 5. 高度软约束

每步都叠加：

```text
R_alt = -altitude_low_penalty_weight * max(z_low - z, 0)
```

当前参数：

- `altitude_target_z = 10.0`
- `altitude_tolerance = 0.5`
- `altitude_low_penalty_weight = 1.0`

优先使用每个环境的 `spawn_height_reference` 作为高度参考。

---

## 6. 被拦截投放请求惩罚

若策略请求投放，但由于硬姿态门限、硬角速度门限或置信门控未通过，导致本步没有真正投放，则额外施加：

```text
R_blocked_drop = -blocked_drop_penalty
```

当前参数：

- `blocked_drop_penalty = 2.0`

---

## 7. 未投放终止项

### 7.1 crash + no_drop

```text
R_no_drop = -crash_no_drop_penalty
```

当前参数：

- `crash_no_drop_penalty = 20.0`

### 7.2 timeout + no_drop

系统会在终止时刻预测“如果现在投放”的落点误差。

若被判定为“错过窗口”：

```text
R_no_drop = -missed_drop_no_drop_penalty
```

若被判定为“合理不投”：

```text
R_no_drop = +reasonable_no_drop_reward
```

当前参数：

- `missed_drop_no_drop_penalty = 8.0`
- `reasonable_no_drop_reward = 0.0`
- `reasonable_no_drop_pred_error_threshold = 5.0`
- `no_drop_eval_max_xy_dist = 10.0`

因为你当前配置里 `reasonable_no_drop_reward = 0.0`，所以“合理不投”现在不加分，只是避免被算作“错过窗口”。

---

## 8. score_d0 课程学习

当前精度奖励尺度 `score_d0` 启用了课程学习：

- `score_d0_curriculum_enable = True`
- `score_d0_curriculum_values = [4.0, 3.0, 2.5, 2.0, 1.5]`
- `score_d0_curriculum_min_drop_rate = [0.15, 0.35, 0.55, 0.65]`
- `score_d0_curriculum_max_error_ema = [8.0, 5.0, 3.0, 1.2]`

含义是：

- 训练初期用较大的 `score_d0`，让精度奖励更宽松；
- 随着投放率上升、落点误差下降，再逐步缩小 `score_d0`，提高精度要求。

---

## 9. 当前有效奖励参数清单

### `drop_reward_config`

- `altitude_target_z`
- `altitude_tolerance`
- `altitude_low_penalty_weight`
- `direction_reward_weight`
- `pred_error_shaping_weight`
- `pred_error_improvement_clip`
- `score_max`
- `score_d0`
- `score_p`
- `score_reward_weight`
- `score_d0_curriculum_enable`
- `score_d0_curriculum_values`
- `score_d0_curriculum_min_drop_rate`
- `score_d0_curriculum_max_error_ema`
- `piecewise_r`
- `piecewise_thresholds`
- `outside_region_penalty`
- `blocked_drop_penalty`
- `crash_no_drop_penalty`
- `missed_drop_no_drop_penalty`
- `reasonable_no_drop_reward`
- `reasonable_no_drop_pred_error_threshold`
- `no_drop_eval_max_xy_dist`
- `impulse_alpha`
- `impulse_beta`
- `impulse_penalty_weight`

### 与奖励联动但不直接写进公式的参数

来自 `drop_model_config`：

- `drop_threshold`
- `allow_multiple_drops`
- `confidence_gate_enable`
- `confidence_threshold`
- `confidence_min_steps`
- `max_release_attitude_deg`
- `max_release_omega_xy`

来自 `drop_impact_config`：

- `enable_impact`
- `add_child_eject_velocity`
- `use_force_recoil`
- `child_mass`
- `mother_mass`
- `eject_speed`
- `eject_direction_body`

---

## 10. 当前不应再按主奖励描述的旧项

下面这些不要再写成“当前奖励函数的一部分”：

- `J` 势函数主逻辑
- `R_hover`
- `R_safe`
- `R_smooth`
- `direction_alignment_reward_magnitude`
- 旧版姿态指数奖励主项
- 固定 `no_drop = -20` 的旧逻辑

它们要么已经删除，要么现在只是遗留字段，并不参与当前主奖励计算。
