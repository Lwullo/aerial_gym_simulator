# navigation_task_gmm_noise 当前网络与观测说明

本文档基于当前 `navigation_task_gmm_noise` 实际代码整理，用于说明：

- 当前 MDP 输入结构
- 帧堆叠逻辑
- 非对称 Actor-Critic 输入通路
- 特权信息定义
- 当前主要训练 YAML 的网络配置

对齐文件：

- 任务配置：[navigation_task_gmm_noise_config.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)
- 任务实现：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)
- 主实验配置：[ppo_aerial_quad_navigation_seed42.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_seed42.yaml)
- 单帧基线：[ppo_acomparsion1_singleframe.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion1_singleframe.yaml)
- 对称基线：[ppo_acomparsion2_proposed_symmetric.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion2_proposed_symmetric.yaml)
- 去物理特权基线：[ppo_acomparsion4_proposed_wo_physics.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion4_proposed_wo_physics.yaml)

更新时间：2026-05-20

## 1. 当前总体结构

当前项目已经不再使用旧版的 `GRU / MLP+GRU` 主线结构，现行实现是：

- Actor：纯 MLP
- Critic：纯 MLP
- Actor 使用帧堆叠观测
- Critic 可选使用特权信息
- 通过 `rl_games` 的 `central value` 通路实现非对称 Critic

默认主实验为：

- 基础观测 `13D`
- 帧堆叠长度 `6`
- Actor 输入 `78D`
- Critic 输入 `91D = 78D + 13D`

对应代码见：

- 观测维度与输入通路初始化：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:120)
- Critic 状态拼接：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3186)

## 2. 基础观测定义

当前基础观测固定为 `13D`，定义如下：

```text
[0:3]   目标相对位置（机体系）target_relative_position
[3:6]   机体系线速度 body_linvel
[6]     roll
[7]     pitch
[8]     yaw
[9:12]  机体系角速度 body_angvel
[12]    当前高度 z
```

对应代码：

- [navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3062)

说明：

- 相对目标位置已经在机体系下表达，便于策略直接学习“朝目标修正”的控制行为。
- 欧拉角中显式包含 `yaw`，因此当前基础观测不是 12 维，而是 13 维。
- 当前主线代码中不再把旧版障碍物位置观测、旧的高层启发式特征作为默认网络输入。

## 3. 帧堆叠机制

当前时序建模不再通过 GRU 完成，而是通过固定长度帧堆叠实现。

默认配置：

- `frame_stack = 6`

对应代码：

- 帧堆叠长度初始化：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:129)
- 帧缓冲区定义：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:244)
- 每步刷新逻辑：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3181)
- reset 时整段填充：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:2726)

具体逻辑为：

1. 环境维护一个形状为 `num_envs × frame_stack × 13` 的观测缓冲区。
2. 每个控制步先计算当前 `13D` 基础观测。
3. 之后将缓冲区整体向前滚动一帧，并把最新观测写入最后一帧。
4. 最后将 `6 × 13` 展平为 `78D`，作为 Actor 输入。
5. 在环境 reset 时，用“当前初始观测”填满全部 6 帧，而不是只填最后一帧。

因此当前帧堆叠的作用是：

- 为 Actor 提供短时速度、角速度、姿态变化历史
- 让策略无需显式记忆网络，也能隐式推断风扰和瞬态变化趋势

## 4. Actor 输入

Actor 输入只使用堆叠观测，不使用特权信息。

默认维度：

- 基础观测：`13D`
- 帧堆叠：`6`
- Actor 输入：`13 × 6 = 78D`

对应代码：

- Actor 输入维度计算：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:161)
- Actor 观测写入：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3184)

这意味着部署时只需要环境可测观测，不依赖风速、质量、惯量等真实物理量。

## 5. Critic 特权信息

当前特权信息固定为 `13D`，定义如下：

```text
[0:3]   世界系真实风速 wind_world
[3:4]   当前母机总质量 robot_mass
[4:7]   当前惯量对角项 inertia_diag
[7:10]  预测反冲力 recoil_force_world
[10:13] 预测反冲力矩 recoil_tau_world
```

对应代码：

- 特权信息生成：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3112)

说明：

- 风速使用世界系真实值。
- 质量使用当前实时值，因此投放前后质量变化会直接反映到 Critic 输入。
- 惯量使用当前实时惯量张量的对角项。
- 预测反冲力/力矩表示“如果当前时刻立即投放”，系统按当前投放物理模型估算得到的瞬态项。

## 6. 非对称 Critic 输入通路

当：

- `use_central_value = True`
- `critic_use_privileged_obs = True`

时，Critic 输入由两部分拼接而成：

- 前半段：Actor 同款 `78D` 堆叠观测
- 后半段：`13D` 特权信息

即：

```text
Critic 输入维度 = 78 + 13 = 91
```

对应代码：

- Critic 输入维度计算：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:164)
- `states` 拼接逻辑：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3186)

这就是当前项目中的标准 asymmetric actor-critic 设计：

- Actor 只看可部署观测
- Critic 额外看训练期特权物理量

## 7. wind_only 模式

当前代码还支持：

- `critic_privileged_mode = "full"`
- `critic_privileged_mode = "wind_only"`

对应代码：

- 模式定义与合法性检查：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:138)
- `wind_only` 处理逻辑：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3125)

其中：

- `full`：保留全部 13 维特权信息
- `wind_only`：仅保留前 3 维真实风速，其余 `10` 维清零

因此在 `wind_only` 模式下，Critic 输入维度虽然仍是 `91D`，但额外物理信息实际上只剩下风速。

## 8. 特权信息域随机化

当前特权信息支持单独做域随机化，开关为：

- `privileged_randomization_enable`

对应代码：

- 开关读取：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:147)
- 扰动施加：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3131)

当前支持的扰动对象包括：

- 风速
- 质量
- 惯量
- 预测反冲力
- 预测反冲力矩

但要注意：

- 如果 `critic_privileged_mode = "wind_only"`，则随机化仅作用于风速维度。
- 其余质量、惯量、反冲项已经被清零，不再参与 Critic 输入。

## 9. 动作空间

当前动作维度为 `5D`：

```text
[vx_cmd, vy_cmd, vz_cmd, yawrate_cmd, drop_switch]
```

对应配置：

- [navigation_task_gmm_noise_config.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py:116)

含义为：

- 前四维是母机速度控制命令
- 第五维是投放决策开关

因此当前策略网络输出的是 `5` 维连续动作均值，以及对应的对数标准差参数。

## 10. 当前网络结构

当前主线训练 YAML 中，Actor 与 Critic 都采用纯 MLP 结构：

```text
输入 -> 256 -> 128 -> 64 -> 输出
```

对应配置：

- 主实验：[ppo_aerial_quad_navigation_seed42.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_seed42.yaml:11)
- 单帧基线：[ppo_acomparsion1_singleframe.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion1_singleframe.yaml:11)
- 对称基线：[ppo_acomparsion2_proposed_symmetric.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion2_proposed_symmetric.yaml:11)
- 去物理特权基线：[ppo_acomparsion4_proposed_wo_physics.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion4_proposed_wo_physics.yaml:11)

共同设置为：

- `mlp.units = [256, 128, 64]`
- `activation = elu`
- `fixed_sigma = True`
- `sigma_init = 0`

说明：

- 这里不再使用 GRU。
- 当前时序建模完全依赖帧堆叠。
- `fixed_sigma = True` 表示策略方差不由状态逐点输出，而是采用全局可学习的对数标准差参数。

## 11. 当前主要实验配置对比

### 11.1 Proposed 主实验

配置文件：

- [ppo_aerial_quad_navigation_seed42.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_seed42.yaml)

设置：

- `frame_stack = 6`
- Actor 输入 `78D`
- Critic 输入 `91D`
- 特权模式 `full`
- 特权随机化 `enable`

语义：

- 6 帧堆叠
- 非对称 Critic
- 使用完整物理特权信息

### 11.2 单帧基线

配置文件：

- [ppo_acomparsion1_singleframe.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion1_singleframe.yaml)

设置：

- `frame_stack = 1`
- Actor 输入 `13D`
- Critic 输入 `26D = 13 + 13`
- 特权模式 `full`

语义：

- 无帧堆叠
- 保留非对称 Critic
- 用于验证帧堆叠时序建模的贡献

### 11.3 Proposed Symmetric 基线

配置文件：

- [ppo_acomparsion2_proposed_symmetric.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion2_proposed_symmetric.yaml)

设置：

- `frame_stack = 6`
- Actor 输入 `78D`
- Critic 输入 `78D`
- `use_central_value = False`
- `critic_use_privileged_obs = False`

语义：

- 保留 6 帧堆叠
- 去掉特权信息
- 用于验证非对称 Critic 的贡献

### 11.4 Proposed without full physics privileged information

配置文件：

- [ppo_acomparsion4_proposed_wo_physics.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_acomparsion4_proposed_wo_physics.yaml)

设置：

- `frame_stack = 6`
- Actor 输入 `78D`
- Critic 输入 `91D`
- 特权模式 `wind_only`
- 特权随机化 `enable`

语义：

- 仍使用非对称 Critic
- 但只给 Critic 真实风速，不给质量、惯量和反冲项
- 用于验证完整物理特权信息的贡献

## 12. 归一化说明

当前环境本身返回的是原始物理量，环境内部没有手工把每一维观测压到固定量纲区间。

训练侧采用的是 `rl_games` 自带输入归一化：

- `normalize_input = True`

对应 YAML：

- [ppo_aerial_quad_navigation_seed42.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_seed42.yaml:68)

因此当前实际流程是：

1. 环境输出原始观测和原始特权信息
2. `rl_games` 在训练中对输入做在线统计归一化
3. Actor 和 Critic 使用归一化后的输入进行学习

这样做的好处是：

- 不需要为每个物理量手工设计归一化系数
- 能自动适应不同训练阶段的数据分布
- 对多输入量纲混合场景更稳定

## 13. 当前文档结论

截至当前代码版本，`navigation_task_gmm_noise` 的网络与输入结构可以概括为：

1. 基础观测固定为 `13D`
2. 通过 `6` 帧堆叠实现短时序建模
3. Actor 默认输入 `78D`
4. Critic 默认输入 `91D`
5. Critic 的 `13D` 特权信息由真实风速、质量、惯量、预测反冲力和预测反冲力矩组成
6. 当前主线网络是纯 `MLP`，不再依赖 `GRU`
7. 通过 single-frame、symmetric、wind-only 等基线可以分别验证帧堆叠、非对称 Critic 和完整物理特权信息的作用

如果后续继续修改以下内容，请同步更新本文档：

- 基础观测维度
- 帧堆叠长度
- 特权信息组成
- Critic 特权模式
- 主实验 YAML 的网络结构
