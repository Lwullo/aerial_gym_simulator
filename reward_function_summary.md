# 当前奖励函数与参数（NavigationTaskGmmNoise）

本文整理当前代码中的奖励项、公式与权重（基于 `aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py`
与 `aerial_gym/config/task_config/navigation_task_gmm_noise_config.py`）。

## 总奖励公式（当前启用项）

```
R_total = R_pot
        + R_dir
        + R_smooth
        + R_safe
        + R_hover
        + R_collision
        + R_success = 0
```

- `R_success` 目前为 0（成功判定仅用于终止，不额外加分）。
- 终端奖励 `terminal_reward` 已禁用（为 0）。
- 运行代价 `R_J` 已删除。

## J(p) 定义

```
J_t = w_d * (d / d0)^2 + w_n * n_hat
n_hat = clamp((n - n_min) / (n_max - n_min + 1e-6), 0, 1)
```

## 奖励项定义

### 1) 潜势改进奖励 R_pot
```
R_pot = k_J * tanh((J_prev - J_t) / s_J)
```

### 2) 方向奖励 R_dir（当前禁用）
```
R_dir = k_dir * max(0, cos(v, dir_to_target)) * moving_mask
moving_mask = (speed > 0.05)
```

### 3) 平滑项 R_smooth
```
R_mag   = -k_a * ||u_t||^2
R_delta = -k_D * ||u_t - u_{t-1}||^2
R_smooth = R_mag + R_delta
```

### 4) 安全奖励 R_safe（最近障碍主导）
```
R_safe = k_safe * min(log(d_min) - log(d_thresh), 0)
```
- d_min 为当前深度图像素的最小距离
- 深度值先按 max_range=10.0 反归一化，再做最小值

### 5) 悬停奖励 R_hover
```
R_hover = k_h * g_J * g_v
g_J = exp(-alpha_J * (J_t / J_h))
g_v = exp(-(||v|| / v_h)^2)
```
- 距离门控 g_d 已移除（等价于 1）

### 6) 碰撞惩罚 R_collision
```
R_collision = collision_penalty * crash_mask
```

### 7) 成功奖励 R_success（Scheme B）
成功条件（需连续 min_success_steps 步）：
- 位置：d <= 2.0 m
- 速度：||v|| <= 0.35 m/s
- 势能：J_ema <= J_best + 0.15
- J_ema 采用 EMA（alpha=0.9）

当前 `success_reward = 0.0`，因此不加分，只用于终止。

---

## 当前参数配置

### reward_parameters

| 参数 | 值 | 说明 |
| --- | --- | --- |
| potential_kj | 1.0 | R_pot 系数 k_J |
| potential_sj | 0.12 | R_pot 归一化尺度 s_J |
| potential_w_d | 1.0 | J 距离权重 w_d |
| potential_w_n | 1.0 | J 噪声权重 w_n |
| potential_d0 | 2.0 | 距离归一化 d0 |
| n_min_max_sample_size | 1000 | 噪声范围估计采样数 |
| terminal_reward | 0.0 | 终端到达奖励（禁用） |
| direction_alignment_reward_magnitude | 0.0 | 方向奖励系数（禁用） |
| safety_reward_magnitude | 0.3 | 安全奖励系数 k_safe |
| safety_dist_threshold | 1.0 | 安全阈值 d_thresh |
| min_safe_distance_clamp | 0.1 | 最小距离 clamp |
| action_magnitude_penalty_weight | 0.05 | k_a |
| action_change_penalty_weight | 0.1 | k_D |
| hover_reward_kh | 0.3 | 悬停系数 k_h |
| hover_reward_dh | 2.0 | 悬停距离阈值（当前未使用） |
| hover_reward_vh | 0.35 | 悬停速度阈值 v_h |
| hover_reward_jh | 2.0 | 悬停势能阈值 J_h |
| hover_reward_alpha_j | 3.0 | 悬停势能门控 alpha_J |
| collision_penalty | -20.0 | 碰撞惩罚 |

### success_config（终止判定）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| success_reward | 0.0 | 成功奖励（禁用） |
| success_radius | 2.0 | 成功半径 |
| stability_velocity_threshold | 0.35 | 速度阈值 |
| stability_potential_delta | 0.15 | 势能容差 |
| stability_potential_ema_alpha | 0.9 | 势能 EMA alpha |
| min_success_steps | 15 | 成功保持步数 |

