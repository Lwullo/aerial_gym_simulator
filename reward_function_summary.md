# 当前奖励函数完整总结

## 总体设计理念

**统一奖励函数**（Unified Reward Function）：无显式阶段切换，通过各奖励分量的自然组合实现：
- **导航行为**：由距离改进奖励和方向对齐奖励驱动
- **优化行为**：由噪声降低奖励驱动
- **悬停行为**：由悬停奖励和累积悬停奖励驱动

## 奖励函数组成

### 总奖励公式

```
Total Reward = 
    improvement_reward              (距离改进奖励)
  + noise_reduction_reward          (噪声降低奖励)
  + direction_reward                (方向对齐奖励)
  + hover_bonus                     (即时悬停奖励)
  + cumulative_hover_bonus          (累积悬停奖励)
  + velocity_smooth_penalty         (速度平滑惩罚)
  + safety_reward                   (安全奖励)
  + x_diff_penalty                  (X轴动作平滑惩罚 - 已禁用)
  + y_diff_penalty                  (Y轴动作平滑惩罚 - 已禁用)
  + z_diff_penalty                  (Z轴动作平滑惩罚 - 已禁用)
  + yaw_rate_penalty                (偏航角速度惩罚 - 已禁用)
  + speed_penalty                   (速度惩罚 - 已禁用)
  + collision_reward                (碰撞惩罚)
```

---

## 各奖励分量详解

### 1. 距离改进奖励 (Distance Improvement Reward) ⭐
**作用**：主要导航驱动力，鼓励靠近目标

**公式**：
```
distance_improvement = previous_distance - current_distance
improvement_reward = magnitude × max(0, distance_improvement)
```

**权重系数**：
- `distance_improvement_reward_magnitude`: **8.0**

**特点**：
- 梯度型奖励，只奖励"变得更近"的行为
- 使用`clamp(min=0)`确保只有正向改进才获得奖励

---

### 2. 噪声降低奖励 (Noise Reduction Reward) ⭐
**作用**：鼓励移动到低噪声区域，驱动优化行为

**公式**：
```
noise_reduction = prev_noise - current_noise
noise_reduction_reward = magnitude × max(0, noise_reduction)
```

**权重系数**：
- `noise_reduction_reward_magnitude`: **8.0**

**特点**：
- 梯度型奖励，只奖励"噪声降低"的行为
- 与距离改进奖励权重相同，两者共同驱动导航

---

### 3. 方向对齐奖励 (Direction Alignment Reward)
**作用**：辅助导航，鼓励速度方向与目标方向一致

**公式**：
```
direction_to_target = (target_position - position) / distance
velocity_direction = velocity / speed
alignment = dot_product(direction_to_target, velocity_direction)
direction_reward = magnitude × max(0, alignment) × moving_mask
```

**权重系数**：
- `direction_alignment_reward_magnitude`: **2.0**

**特殊处理**：
- 只在速度 > 0.05 m/s 时生效（`moving_mask`）
- 静止时不给奖励，避免悬停状态下的噪声

---

### 4. 即时悬停奖励 (Hover Bonus) ⭐⭐⭐
**作用**：鼓励在优质位置（靠近目标且低噪声）稳定悬停

**公式**：
```
position_quality = -(distance × 8.0 + noise × 8.0)
is_good_position = position_quality > quality_threshold
is_hovering = speed < speed_threshold

hover_bonus = hover_bonus_magnitude  (if both conditions met)
             = 0                      (otherwise)
```

**权重系数**：
- `hover_bonus_magnitude`: **50.0**
- `hover_quality_threshold`: **-3.0**
- `hover_speed_threshold`: **0.2** m/s

**触发条件**：
- 位置质量足够好（靠近目标且噪声低）
- 速度足够低（接近悬停）

---

### 5. 累积悬停奖励 (Cumulative Hover Bonus) ⭐⭐
**作用**：鼓励持续停留在最优位置，防止无休止探索

**公式**：
```
hover_time_counter += 1  (if in optimal zone)
                    = 0  (otherwise)

cumulative_hover_bonus = min(hover_time_counter × rate, max_value)
```

**权重系数**：
- `cumulative_hover_rate`: **0.02**
- `cumulative_hover_max`: **10.0**

**特点**：
- 随时间累积，最大值为10.0
- 离开最优区域立即重置为0

---

### 6. 动态速度平滑惩罚 (Dynamic Velocity Smoothness Penalty) ⭐
**作用**：鼓励平滑运动，避免剧烈加速/减速

**公式**：
```
velocity_change_xy = current_velocity_xy - previous_velocity_xy
velocity_diff_norm = norm(velocity_change_xy)

smooth_weight = penalty_far   (if distance > threshold)
              = penalty_near   (otherwise)

velocity_smooth_penalty = -smooth_weight × velocity_diff_norm
```

**权重系数**：
- `velocity_smoothness_penalty_far`: **0.1** (远离目标时)
- `velocity_smoothness_penalty_near`: **0.2** (靠近目标时)
- `distance_threshold_for_smooth`: **2.0** m

**特点**：
- 动态权重：靠近目标时惩罚更重
- 只考虑XY平面速度变化

---

### 7. 安全奖励 (Safety Reward) ⭐⭐
**作用**：基于深度图鼓励远离障碍物

**公式**：
```
depth_pixels = depth_range_pixels.flatten()  # 所有像素
safe_distances = max(depth_pixels, min_clamp)
log_distances = log(safe_distances)
safety_reward = magnitude × mean(log_distances)
```

**权重系数**：
- `safety_reward_magnitude`: **2.0**
- `min_safe_distance_clamp`: **0.1** m

**特点**：
- 使用对数刻度，鼓励与障碍物保持距离
- 基于所有深度图像素的平均值

---

### 8. 动作平滑惩罚 (Action Smoothness Penalties) - **已禁用**
**作用**：惩罚相邻动作之间的剧烈变化

**公式**：
```
action_diff = current_action - previous_action
x_diff_penalty = -magnitude × |action_diff_x|^exponent
y_diff_penalty = -magnitude × |action_diff_y|^exponent
z_diff_penalty = -magnitude × |action_diff_z|^exponent
```

**权重系数** (当前全部为0，已禁用)：
- `x_action_diff_penalty_magnitude`: **0.0**
- `y_action_diff_penalty_magnitude`: **0.0**
- `z_action_diff_penalty_magnitude`: **0.0**
- `x_action_diff_penalty_exponent`: **2.0**
- `y_action_diff_penalty_exponent`: **2.0**
- `z_action_diff_penalty_exponent`: **2.0**

---

### 9. 偏航角速度惩罚 (Yaw Rate Penalty) - **已禁用**
**作用**：防止无人机快速旋转

**公式**：
```
yaw_rate = angular_velocity_z
yaw_rate_penalty = -magnitude × |yaw_rate|^exponent
```

**权重系数** (当前为0，已禁用)：
- `yaw_rate_penalty_magnitude`: **0.0**
- `yaw_rate_penalty_exponent`: **2.0**

---

### 10. 速度惩罚 (Speed Penalty) - **已禁用**
**作用**：限制最大飞行速度

**公式**：
```
speed_excess = max(0, speed - max_safe_speed)
speed_penalty = -magnitude × speed_excess²
```

**权重系数** (当前为0，已禁用)：
- `speed_penalty_magnitude`: **0.0**
- `max_safe_speed`: **5.0** m/s

---

### 11. 碰撞惩罚 (Collision Penalty)
**作用**：强烈惩罚碰撞行为

**公式**：
```
collision_reward = collision_penalty  (if crashed)
                 = 0                  (otherwise)
```

**权重系数**：
- `collision_penalty`: **-100.0**

**特点**：
- 固定大额惩罚
- 碰撞后episode终止

---

## 权重系数汇总表

| 奖励分量             | 参数名                                     | 权重值        | 状态   |
| ---------------- | --------------------------------------- | ---------- | ---- |
| **距离改进奖励**       | `distance_improvement_reward_magnitude` | **8.0**    | ✅ 启用 |
| **噪声降低奖励**       | `noise_reduction_reward_magnitude`      | **8.0**    | ✅ 启用 |
| **方向对齐奖励**       | `direction_alignment_reward_magnitude`  | **2.0**    | ✅ 启用 |
| **即时悬停奖励**       | `hover_bonus_magnitude`                 | **50.0**   | ✅ 启用 |
| **累积悬停奖励(rate)** | `cumulative_hover_rate`                 | **0.02**   | ✅ 启用 |
| **累积悬停奖励(max)**  | `cumulative_hover_max`                  | **10.0**   | ✅ 启用 |
| **速度平滑惩罚(远)**    | `velocity_smoothness_penalty_far`       | **0.1**    | ✅ 启用 |
| **速度平滑惩罚(近)**    | `velocity_smoothness_penalty_near`      | **0.2**    | ✅ 启用 |
| **安全奖励**         | `safety_reward_magnitude`               | **2.0**    | ✅ 启用 |
| **碰撞惩罚**         | `collision_penalty`                     | **-100.0** | ✅ 启用 |
| X轴动作平滑惩罚         | `x_action_diff_penalty_magnitude`       | **0.0**    | ❌ 禁用 |
| Y轴动作平滑惩罚         | `y_action_diff_penalty_magnitude`       | **0.0**    | ❌ 禁用 |
| Z轴动作平滑惩罚         | `z_action_diff_penalty_magnitude`       | **0.0**    | ❌ 禁用 |
| 偏航角速度惩罚          | `yaw_rate_penalty_magnitude`            | **0.0**    | ❌ 禁用 |
| 速度惩罚             | `speed_penalty_magnitude`               | **0.0**    | ❌ 禁用 |

---

## 阈值参数汇总

| 参数 | 参数名 | 值 | 说明 |
|-----|-------|---|------|
| 悬停质量阈值 | `hover_quality_threshold` | **-3.0** | position_quality > -3.0 才能获得悬停奖励 |
| 悬停速度阈值 | `hover_speed_threshold` | **0.2** m/s | 速度 < 0.2 m/s 才被认为是悬停 |
| 平滑距离阈值 | `distance_threshold_for_smooth` | **2.0** m | 根据距离切换速度平滑惩罚权重 |
| 最小安全距离 | `min_safe_distance_clamp` | **0.1** m | 安全奖励计算中深度值的最小阈值 |
| 最大安全速度 | `max_safe_speed` | **5.0** m/s | 速度惩罚阈值（当前禁用） |

---

## 奖励函数设计哲学

1. **梯度驱动**：主要奖励（距离改进、噪声降低）都是梯度型，鼓励持续改进而非绝对状态

2. **行为涌现**：不同行为通过奖励自然涌现，无需显式阶段切换：
   - 导航：距离改进 + 方向对齐
   - 优化：噪声降低
   - 悬停：即时悬停奖励 + 累积悬停奖励

3. **安全优先**：安全奖励和碰撞惩罚确保避障行为

4. **平滑运动**：动态速度平滑惩罚鼓励平稳飞行

5. **禁用冗余项**：动作平滑、偏航角速度、速度惩罚当前禁用，避免过度限制
