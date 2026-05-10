# Navigation Task GMM Noise: 当前 DROP 奖励函数说明

本文档严格对齐当前代码快照：

- 任务配置：[navigation_task_gmm_noise_config.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)
- 任务实现：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)
- 文档更新时间：2026-04-10

## 1. 奖励结构

当前每步奖励由四部分组成：

```text
R = R_progress + R_drop + R_alt + R_no_drop_terminal
```

其中：

- `R_progress`：未投放阶段的平面距离进展奖励
- `R_drop`：投放当步的精度/冲击/姿态奖励
- `R_alt`：全程高度软惩罚
- `R_no_drop_terminal`：episode 结束但未投放时的终止附加项

对应实现入口：
- [_compute_reward_and_scores()](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:2355)
- `step()` 里的 no-drop 终止处理：[navigation_task_gmm_noise.py:3694](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3694)

## 2. WAIT 阶段奖励

### 2.1 距离进展项

在尚未投放时：

```text
R_progress = w_dir * (d_prev_xy - d_curr_xy)
```

其中：

- `d_prev_xy`：上一时刻母机到目标点的 XY 距离
- `d_curr_xy`：当前时刻母机到目标点的 XY 距离
- `w_dir = direction_reward_weight = 0.75`

含义：

- 更接近目标：正奖励
- 远离目标：负奖励

当前没有再对“离目标足够远才给进展奖励”加额外门控；只要未投放，该项就生效。

## 3. DROP 当步奖励

DROP 当步先计算连续精度分数、冲击惩罚、姿态奖励，然后对“超外圈/撞障碍物”样本做覆盖惩罚。

### 3.1 连续精度分数

```text
raw_score = score_max * exp(-(landing_error_xy / score_d0)^score_p)
R_score = score_reward_weight * raw_score
```

当前参数：

- `score_max = 20.0`
- `score_reward_weight = 1.0`
- `score_d0 = 2.0`
- `score_p = 2.0`

相比旧版：

- 旧版 `score_d0 = 3.6`
- 旧版 `score_p = 1.0`

当前版本对大误差下降更快，明显更偏向压大偏差。

### 3.2 冲击惩罚

投放冲击指标定义为：

```text
impulse_metric = alpha * delta_theta + beta * delta_omega
R_impulse = -impulse_penalty_weight * impulse_metric
```

当前参数：

- `alpha = impulse_alpha = 1.0`
- `beta = impulse_beta = 0.8`
- `impulse_penalty_weight = 0.5`

其中：

- `delta_theta`：投放前后 roll/pitch 变化量范数
- `delta_omega`：投放前后角速度变化量范数

### 3.3 姿态稳定奖励

姿态项由两个指数奖励组成：

```text
theta_drop = ||[roll_before, pitch_before]||
omega_drop_xy = ||[wx_before, wy_before]||

R_posture = attitude_reward_weight * exp(-(theta_drop / attitude_theta0)^2)
R_angvel = attitude_reward_weight * exp(-(omega_drop_xy / attitude_theta0)^2)
R_att = R_posture + R_angvel
```

当前参数：

- `attitude_reward_weight = 0.5`
- `attitude_theta0 = 0.12`

注意：

- 当前实现里，角速度稳定项和姿态稳定项使用同一权重、同一尺度
- `drop_angle_reward_weight` / `drop_angle_theta0` 仍保留在配置里，但当前 DROP 奖励主分支并没有实际使用它们

### 3.4 超外圈与障碍物命中覆盖惩罚

当前外圈阈值由：

```text
piecewise_r = 1.0
piecewise_thresholds = [0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 8.0]
```

得到最外层阈值：

```text
d_outer = 8.0 m
```

若满足任一条件：

- `landing_error_xy > 8.0m`
- 投放后落点击中目标附近采样障碍物

则 DROP 当步奖励被直接覆盖为：

```text
R_drop = -outside_region_penalty = -20.0
```

同时清零：

- `R_score`
- `R_impulse`
- `R_att`

这意味着当前版本已经不是“误差很大但仍给一点小正分”的宽松逻辑，而是：

- 大误差或撞障碍直接视为失败型 DROP

## 4. 高度软惩罚

全程每步都有高度下界惩罚：

```text
lower_bound_z = spawn_height_reference - altitude_tolerance
R_alt = -altitude_low_penalty_weight * max(lower_bound_z - z, 0)
```

当前参数：

- `altitude_tolerance = 0.5`
- `altitude_low_penalty_weight = 1.0`

说明：

- 当前实现优先使用每个环境的 `spawn_height_reference`
- 所以随机高度训练时，这个惩罚会跟每个 case 的出生高度基准一起变化

## 5. 未投放终止项：当前已经改为风险感知逻辑

这是当前版本和旧文档差异最大的部分。

旧逻辑是：

```text
done 且未投放 => 固定 -20
```

当前逻辑不是一刀切，而是按终止类型区分。

### 5.1 crash + no_drop

若 episode 因撞毁结束且从未投放：

```text
R_no_drop_terminal = -crash_no_drop_penalty
```

当前：

- `crash_no_drop_penalty = 20.0`

### 5.2 timeout + no_drop

若 episode 因超时结束且从未投放，系统会在终止时刻做一次“如果现在投放会怎样”的预测：

- 取当前母机位置/速度
- 若开启了子机初始弹射速度，则把弹射速度加进去
- 调用 `_predict_child_landing_xy(...)` 估计此刻投放的预测落点误差

然后按阈值 `reasonable_no_drop_pred_error_threshold = 5.0m` 分两类：

#### 合理不投

若：

```text
pred_error_xy > 5.0m
```

则认为“不投放是合理避险”，给：

```text
+ reasonable_no_drop_reward = +2.0
```

#### 错过投放窗口

若：

```text
pred_error_xy <= 5.0m
```

则认为“其实可以投，但你错过了窗口”，给：

```text
- missed_drop_no_drop_penalty = -8.0
```

### 5.3 当前 no-drop 相关参数

- `no_drop_penalty = 0.0`：旧的一刀切 no-drop 惩罚已经停用
- `crash_no_drop_penalty = 20.0`
- `missed_drop_no_drop_penalty = 8.0`
- `reasonable_no_drop_reward = 2.0`
- `reasonable_no_drop_pred_error_threshold = 5.0`

## 6. 当前 reward 参数总览

当前 `drop_reward_config` 的关键参数为：

- `direction_reward_weight = 0.75`
- `direction_min_speed = 0.05`
- `direction_min_target_dist = 0.1`
- `score_max = 20.0`
- `score_reward_weight = 1.0`
- `score_d0 = 2.0`
- `score_p = 2.0`
- `piecewise_r = 1.0`
- `piecewise_thresholds = [0.2, 0.4, 0.8, 1.2, 2.2, 4.0, 6.2, 8.0]`
- `outside_region_penalty = 20.0`
- `impulse_alpha = 1.0`
- `impulse_beta = 0.8`
- `impulse_penalty_weight = 0.5`
- `attitude_reward_weight = 0.5`
- `attitude_theta0 = 0.12`
- `altitude_tolerance = 0.5`
- `altitude_low_penalty_weight = 1.0`
- `drop_obstacle_enable = True`
- `drop_obstacle_spawn_prob = 0.5`
- `crash_no_drop_penalty = 20.0`
- `missed_drop_no_drop_penalty = 8.0`
- `reasonable_no_drop_reward = 2.0`
- `reasonable_no_drop_pred_error_threshold = 5.0`

## 7. 当前奖励逻辑的工程意图

当前版本的设计目标可以概括为：

1. 奖励高精度 DROP
2. 惩罚高冲击/高姿态风险 DROP
3. 对明显不可投的极端状态允许“不投放”
4. 对撞毁不投、以及本可投却错过窗口的不投放继续惩罚
5. 对大误差或撞障碍的 DROP 直接判为失败型投放

所以当前版本不是简单的“逼着策略一定投”，而是：

- 鼓励有质量的投放
- 允许合理拒投
- 避免无意义乱投

## 8. 与旧文档相比的关键变化

如果你看到旧版结论：

- `no_drop = -20`
- 外圈阈值约 `24.4m`
- `score_d0 = 3.6`

那已经不是当前代码。

当前版本已经更新为：

- `no_drop` 按 crash / timeout 分支处理
- 合理不投可得 `+2`
- 错过窗口不投为 `-8`
- 外圈硬失败阈值为 `8.0m`
- 精度连续项收紧为 `score_d0=2.0, score_p=2.0`
