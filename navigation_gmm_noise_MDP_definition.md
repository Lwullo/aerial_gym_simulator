# Navigation GMM Noise Task - MDP定义文档

> **任务名称**: `navigation_gmm_noise_task`  
> **创建时间**: 2026-01-15  
> **目的**: 训练无人机在GMM噪声场中寻找最优投放位置

---

## 1. 马尔科夫决策过程 (MDP) 概述

本任务是一个标准的马尔科夫决策过程 (Markov Decision Process)，定义为四元组：

**MDP = (S, A, R, P)**

- **S**: 状态空间 (State Space)
- **A**: 动作空间 (Action Space)
- **R**: 奖励函数 (Reward Function)
- **P**: 状态转移概率 (Transition Probability)

---

## 2. 状态空间 S (State Space)

### 2.1 总维度
**81维** (`observation_space_dim = 13 + 4 + 64`)

### 2.2 详细组成

#### (1) 目标信息 - 4维 `[0:3]`
| 索引 | 名称 | 描述 |
|------|------|------|
| 0-2 | `vec_to_target` | 机器人到目标的单位方向向量 (机体坐标系) |
| 3 | `dist_to_target` | 到目标的欧氏距离 (m) |

**特征**:
- 方向向量在**机体坐标系**下表示 (body frame)
- 带有GMM位置噪声扰动 (`pos_noise`)
- 额外添加±10%的随机扰动增强鲁棒性

#### (2) 姿态信息 - 3维 `[4:6]`
| 索引 | 名称 | 描述 |
|------|------|------|
| 4 | `roll` | 横滚角 (rad, 归一化到[-π, π]) |
| 5 | `pitch` | 俯仰角 (rad, 归一化到[-π, π]) |
| 6 | `yaw_placeholder` | 固定为0 (yaw信息已在方向向量中隐含) |

**特征**:
- 使用 `ssa()` 函数进行角度归一化
- 带有±5%的随机扰动

#### (3) 速度信息 - 6维 `[7:12]`
| 索引 | 名称 | 描述 |
|------|------|------|
| 7-9 | `body_linvel` | 机体坐标系线速度 (m/s) |
| 10-12 | `body_angvel` | 机体坐标系角速度 (rad/s) |

**特征**:
- 均在**机体坐标系**下表示
- 无额外噪声扰动

#### (4) 动作历史 - 4维 `[13:16]`
| 索引 | 名称 | 描述 |
|------|------|------|
| 13-16 | `prev_actions` | 上一时刻的动作输出 (归一化 [-1, 1]) |

**特征**:
- 提供动作连续性信息，有助于学习平滑控制策略

#### (5) 视觉编码 - 64维 `[17:80]`
| 索引 | 名称 | 描述 |
|------|------|------|
| 17-80 | `vae_latent` | VAE编码的深度图像潜在特征 |

**特征**:
- 输入: 270×480 深度图像 (LiDAR范围像素)
- 编码器: 预训练VAE模型 (`LD_64_epoch_49.pth`)
- 输出: 64维潜在向量，包含障碍物和环境几何信息

---

## 3. 动作空间 A (Action Space)

### 3.1 总维度
**4维连续动作**，每维范围: **[-1, 1]**

### 3.2 动作映射

| 维度 | 物理含义 | 归一化范围 | 物理范围 |
|------|----------|-----------|----------|
| `A[0]` | X方向速度命令 | [-1, +1] | [-0.8, +0.8] m/s |
| `A[1]` | Y方向速度命令 | [-1, +1] | [-0.8, +0.8] m/s |
| `A[2]` | Z方向速度命令 | [-1, +1] | [-0.5, +0.5] m/s |
| `A[3]` | Yaw角速度命令 | [-1, +1] | [-π/6, +π/6] rad/s (±30°/s) |

### 3.3 动作转换函数

```python
def action_transformation_function(action):
    clamped_action = torch.clamp(action, -1.0, 1.0)
    max_speed = [0.8, 0.8, 0.5]  # X, Y, Z
    max_yawrate = π/6  # 30°/s
    
    processed_action[:, 0:3] = max_speed * clamped_action[:, 0:3]
    processed_action[:, 3] = max_yawrate * clamped_action[:, 3]
    return processed_action
```

**设计rationale**:
- XY速度限制较低(0.8 m/s)以提高安全性和稳定性
- Z速度更保守(0.5 m/s)防止快速上升/下降
- Yaw速度适中(30°/s)平衡响应性和稳定性

---

## 4. 奖励函数 R (Reward Function)

### 4.1 设计理念
**统一奖励函数** (无显式阶段切换):
- 导航行为: 由改进奖励 + 方向奖励驱动
- 优化行为: 由噪声降低奖励驱动
- 悬停行为: 由悬停奖励在好位置自然涌现

### 4.2 奖励组件

#### 核心奖励 (导航 + 优化)

| 组件 | 权重 | 公式 | 说明 |
|------|------|------|------|
| **距离改进奖励** | 8.0 | `max(0, prev_dist - curr_dist)` | 鼓励靠近目标 (梯度型) |
| **噪声降低奖励** | 8.0 | `max(0, prev_noise - curr_noise)` | 鼓励降低GMM噪声强度 |
| **方向对齐奖励** | 2.0 | `max(0, dot(vel_dir, target_dir))` | 速度方向与目标方向对齐 |

#### 悬停稳定奖励

| 组件 | 权重 | 触发条件 | 说明 |
|------|------|----------|------|
| **位置质量悬停奖励** | 0~25.0 | `position_quality > -3.0` | 渐进式奖励，速度越低奖励越高 |
| **累积悬停奖励** | 0~10.0 | 持续悬停 | 鼓励保持在最优位置 |

```python
# 位置质量评估
position_quality = -(dist_to_target * 8.0 + noise_intensity * 8.0)

# 速度因子 (越慢奖励越高)
speed_factor = clamp(1.0 - speed / 0.3, 0, 1)

# 悬停奖励
hover_bonus = 25.0 * speed_factor  # if position_quality > -3.0
```

#### 安全与约束惩罚

| 组件 | 权重 | 公式 | 说明 |
|------|------|------|------|
| **安全奖励** | 2.0 | `mean(log(depth_distances))` | 基于深度图避障 |
| **速度平滑惩罚** | -0.1~-0.3 | `-w * ||Δv||` | 动态权重(远: 0.1, 近: 0.3) |
| **倾角惩罚** | -1.0 | `-max(0, tilt - 15°)²` | 防止过度倾斜 |
| **碰撞惩罚** | -100.0 | 固定值 | 碰撞时立即终止 |

### 4.3 总奖励公式

```
R_total = improvement_reward           (8.0 × improvement)
        + noise_reduction_reward        (8.0 × noise_reduction)
        + direction_reward              (2.0 × alignment)
        + hover_bonus                   (0~25.0, progressive)
        + cumulative_hover_bonus        (0~10.0, cumulative)
        + velocity_smooth_penalty       (dynamic: -0.1 or -0.3)
        + safety_reward                 (2.0 × mean(log(distances)))
        + tilt_penalty                  (-1.0 × tilt²)
        + collision_penalty             (-100.0 if crash)
```

---

## 5. 状态转移概率 P (Transition Probability)

### 5.1 动力学模型
**Lee速度控制器 + 四旋翼物理仿真**

```
状态转移: s_{t+1} = f(s_t, a_t, ω_t)
```

其中:
- `f(·)`: Isaac Gym中的物理引擎
- `ω_t`: 随机扰动项

### 5.2 确定性部分

#### (1) Lee速度控制器
- 输入: 期望速度 `v_des` (由动作 `a_t` 确定)
- 输出: 电机推力命令
- 控制分配: 4个电机的推力分配矩阵

#### (2) 四旋翼动力学
- 质量: 12.04 kg
- 惯性矩阵: 根据URDF定义
- 推力模型: 比例于电机转速平方

### 5.3 随机扰动部分

#### (1) GMM物理力扰动
**启用条件**: `enable_physical_force = True`

```python
# 力的大小计算
mixture_intensity = Σ w_i * exp(-0.5 * ||p - μ_i||²/σ_i²)
F_magnitude = mixture_intensity × mass × g × k

# 参数
k = 0.01  # 扰动系数 (降低后更温和)
mass = 12.04 kg
g = 9.81 m/s²
```

**方向更新**:
- 每5步随机生成新目标方向
- 低通滤波平滑: `dir = 0.9 × dir_old + 0.1 × dir_target`
- 防止突然"踢"干扰

#### (2) 位置观测噪声
```python
noise = randn() × mixture_intensity × noise_scale
noisy_position = true_position + noise
```

### 5.4 终止条件

| 条件 | 类型 | 说明 |
|------|------|------|
| 碰撞 | Termination | 与障碍物/边界碰撞 |
| 超时 | Truncation | 超过1200步 (episode_len_steps) |
| 成功 | Termination | 满足成功条件(见下) |

---

## 6. 成功条件 (Success Criteria)

### 6.1 综合条件 (需同时满足)

| 条件 | 阈值 | 说明 |
|------|------|------|
| **距离** | `< 2.0 m` | 进入目标区域 |
| **速度** | `< 0.4 m/s` | 接近静止 (放宽至0.4) |
| **姿态** | `roll/pitch < 15°` | 姿态稳定 |
| **持续时间** | `≥ 50 steps` | 连续满足条件 |

### 6.2 成功判定逻辑

```python
# 单步判定
is_near_target = dist_to_target < 2.0
is_slow = linvel_magnitude < 0.4
is_level = sqrt(roll² + pitch²) < 15° (0.262 rad)

is_success_step = is_near_target AND is_slow AND is_level

# 连续判定
success_counter += 1 if is_success_step else reset to 0
episode_success = (success_counter >= 50)
```

---

## 7. 环境参数

### 7.1 空间边界
```python
env_bounds_min = [0.0, 0.0, 0.0]  # m
env_bounds_max = [10.0, 10.0, 10.0]  # m
```

### 7.2 障碍物配置
- 数量: 11个随机障碍物
- 类型: 面板 (panels) - URDF定义
- 碰撞检测: Isaac Gym物理引擎

### 7.3 GMM噪声场配置
```python
num_sources = 5  # 噪声源数量
sigma_min = [5.0, 5.0, 2.5]  # 各向异性协方差下限
sigma_max = [15.0, 15.0, 7.5]  # 协方差上限
weight_min = 0.1  # 权重下限
weight_max = 1.0  # 权重上限
noise_scale = 1.0  # 总噪声缩放系数
```

**重采样**: 每次episode重置时随机生成新的GMM参数

### 7.4 目标位置
- 采样范围: 环境空间的20%-80%区域 (距离边界≥2m)
- 重采样: 每次episode重置

---

## 8. 可观测性分析

### 8.1 完全可观测信息
✅ 相对目标位置和距离  
✅ 自身姿态(欧拉角)  
✅ 自身速度(线速度+角速度)  
✅ 深度图像(通过VAE编码)  
✅ 上一步动作  

### 8.2 部分可观测信息
⚠️ **GMM噪声场**: 只能通过当前位置的噪声强度间接感知，无法直接获取:
   - 噪声源中心 `μ_i`
   - 协方差矩阵 `Σ_i`
   - 权重 `w_i`

⚠️ **障碍物**: 只能通过深度图推断，无法获取:
   - 全局精确位置
   - 几何形状参数

### 8.3 不可观测信息
❌ 全局绝对位置 (只知道相对目标位置)  
❌ 全局坐标系下的yaw角 (只知道相对目标方向)  

**设计rationale**: 
- 鼓励策略学习**相对导航**而非记忆固定轨迹
- 提高泛化能力到不同起点和目标配置

---

## 9. 训练配置

### 9.1 时间步设置
```python
episode_len_steps = 1200  # 最大步数
sim_dt = 0.01  # 仿真时间步长 (推测)
max_episode_time ≈ 12 seconds  # 实际物理时间
```

### 9.2 并行环境
```python
num_envs = 256  # 并行环境数量
device = "cuda:0"  # GPU加速
use_warp = True  # Warp物理引擎加速
```

### 9.3 课程学习 (Curriculum)
```python
obstacle_count = 11  # 固定障碍物数量 (已禁用动态课程)
```

---

## 10. 任务目标总结

### 10.1 主要目标
在10×10×10m的环境中，训练无人机自主寻找**最优投放位置**:

1. **靠近目标区域** (距离 < 2.0m)
2. **低噪声强度** (GMM噪声场低值区域)
3. **稳定悬停** (速度 < 0.4 m/s, 倾角 < 15°, 持续50步)
4. **安全避障** (避免碰撞)

### 10.2 学习挑战

| 挑战 | 说明 |
|------|------|
| **探索-利用平衡** | 需要探索空间寻找低噪声区域，又要稳定悬停 |
| **多目标优化** | 同时优化距离、噪声、安全性 |
| **扰动鲁棒性** | 对抗GMM物理力和观测噪声 |
| **部分可观测** | GMM场不可直接观测，需推断 |
| **长时间稳定** | 需连续50步满足条件才算成功 |

---

## 11. 参考文件

- **配置文件**: [`navigation_task_gmm_noise_config.py`](file:///home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)
- **任务实现**: [`navigation_task_gmm_noise.py`](file:///home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)
- **控制器**: `lmf2_velocity_control` (Lee速度控制器)
- **VAE模型**: `ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth`

---

**文档版本**: v1.0  
**最后更新**: 2026-01-15  
**作者**: AI分析生成
