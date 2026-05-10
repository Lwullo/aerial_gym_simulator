# Navigation Task GMM Noise: 当前观测、网络与训练配置说明

本文档记录当前与 `navigation_task_gmm_noise` 相关的观测定义、网络结构以及最近引入的轻量 `MLP+RNN` 配置。

对齐文件：

- 任务配置：[navigation_task_gmm_noise_config.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)
- 任务实现：[navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)
- 训练配置：
  - [ppo_aerial_quad_navigation.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation.yaml)
  - [ppo_aerial_quad_navigation_rnn_only.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_rnn_only.yaml)
  - [ppo_aerial_quad_navigation_mlp_rnn_lite.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_mlp_rnn_lite.yaml)

更新时间：2026-04-10

## 1. 观测空间

### 1.1 基础观测

当前基础观测前缀为 15 维：

```text
[0:3]   target_relative_position (x_rel, y_rel, z_rel)
[3:6]   robot_body_linvel       (vx, vy, vz)
[6]     roll
[7]     pitch
[8:11]  robot_body_angvel       (wx, wy, wz)
[11]    robot_height_z
[12:15] drop_obstacle_position  (obs_x, obs_y, obs_z)
```

对应代码：
- [navigation_task_gmm_noise.py:3460](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3460)

说明：

- 当前障碍观测仍是“单个 DROP 风险障碍物位置”；
- 若障碍不存在，位置会写入 `drop_obstacle_absent_obs_value = -1e3`。

### 1.2 风估计增强观测

当 `use_wind_estimation_features=True` 时，在基础观测后追加：

```text
prev_cmd_body      : 4 dims
delta_v            : 3 dims
linvel_history     : 3 * obs_linvel_history_frames dims
```

默认 `obs_linvel_history_frames = 4`，因此增强维度总计：

```text
4 + 3 + 12 = 19
```

所以：

- 基础观测：`15D`
- 风增强观测：`15 + 19 = 34D`

### 1.3 新增的高层投放特征

本次更新新增可选开关：

```python
use_drop_decision_features = False
```

当开启时，在当前观测末尾再追加 3 个高层特征：

1. `predicted_drop_error_xy`
2. `predicted_fall_time`
3. `predicted_release_risk`

对应代码：
- 开关定义：[navigation_task_gmm_noise_config.py:107](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py:107)
- 特征计算：[navigation_task_gmm_noise.py:3425](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3425)
- 特征写入：[navigation_task_gmm_noise.py:3494](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py:3494)

这三个量当前定义为：

### `predicted_drop_error_xy`

```text
|| target_rel_xy - body_linvel_xy * predicted_fall_time ||
```

含义：
- 如果“现在立刻释放”，按当前高度和当前水平速度估计的 XY 偏差有多大。

### `predicted_fall_time`

```text
sqrt(2 * z / g)
```

含义：
- 现在投放后，自由落体到地面的估计时间。

### `predicted_release_risk`

```text
attitude_deg + 10 * omega_xy
```

含义：
- 当前释放姿态和角速度综合形成的启发式风险分数。

### 1.4 当前支持的观测维度

现在任务和评估脚本支持 4 种观测布局：

- `15D`：基础观测
- `18D`：基础观测 + 3 个高层投放特征
- `34D`：基础观测 + 风增强
- `37D`：基础观测 + 风增强 + 3 个高层投放特征

评估脚本已做显式识别：
- [eval_drop_paper_comparison.py:76](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/eval_drop_paper_comparison.py:76)
- [eval_drop_paper_comparison.py:367](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/eval_drop_paper_comparison.py:367)

并增加了保护：

- `18D` 基础观测 + 新特征版本
- 不能和 `34D/37D` 风增强版本在同一次评估里混跑

原因：
- 两者前缀布局不同，强行混跑会导致语义错位。

## 2. 当前主要训练配置

### 2.1 PPO-MLP-RNN（原配置）

配置文件：
- [ppo_aerial_quad_navigation.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation.yaml)

结构：

- MLP：`[256, 128, 64]`
- RNN：`GRU(64)`
- `seq_length = 8`
- `learning_rate = 5e-5`

语义：
- 这是“MLP 前端 + GRU”结构

### 2.2 PPO-RNN-only

配置文件：
- [ppo_aerial_quad_navigation_rnn_only.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_rnn_only.yaml)

结构：

- MLP：`[]`
- RNN：`GRU(64)`
- `seq_length = 8`
- `learning_rate = 5e-5`

语义：
- 这是“纯 GRU”，不是 vanilla 无门控 RNN

### 2.3 新增 PPO-MLP-RNN-lite

为了压尾部误差、并给 `MLP+RNN` 一个更合理的轻量前端，本次新增：

- [ppo_aerial_quad_navigation_mlp_rnn_lite.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_mlp_rnn_lite.yaml)

关键改动：

- `mlp.units: [128, 64]`
- `seq_length: 16`
- `learning_rate: 3e-5`
- `env_config.use_drop_decision_features: True`

设计目的：

1. 降低原版 `[256,128,64]` 前端过重带来的尾部风险
2. 用更长序列增强时序判断
3. 利用 3 个高层投放特征帮助 `MLP+RNN` 做更明确的“何时释放”决策

## 3. 训练入口兼容性更新

训练入口 `runner.py` 会把 YAML 中的 `env_config` 字段直接传给任务注册器。

本次为了支持：

```yaml
env_config:
  use_drop_decision_features: True
```

对注册器做了兼容扩展：

- [task_registry.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/registry/task_registry.py)

现在 `TaskRegistry.make_task(...)` 会：

- 接收额外 `env_config` 字段
- 若对应字段存在于 `task_config`，则写回配置后再创建任务

这样新配置训练时不会再报：

```text
make_task() got an unexpected keyword argument 'use_drop_decision_features'
```

## 4. 评估脚本的当前约束

评估脚本：
- [eval_drop_paper_comparison.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/eval_drop_paper_comparison.py)

当前评估脚本会根据 checkpoint 自动推断：

- 是否有 `actor_mlp`
- 是否有 `GRU`
- 输入观测维度是多少

然后自动设置环境：

- `observation_space_dim`
- `use_wind_estimation_features`
- `use_drop_decision_features`

对应逻辑：
- [eval_drop_paper_comparison.py:3579](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/eval_drop_paper_comparison.py:3579)

因此：

- 旧的 `PPO-MLP`
- 旧的 `PPO-MLP-RNN`
- 旧的 `PPO-RNN-only`
- 新的 `PPO-MLP-RNN-lite`

都可以通过同一评估脚本加载，只要观测布局不冲突。

## 5. 当前工程结论

截至这次更新，当前代码逻辑已经从“单一固定观测 + 单一 PPO-GRU 配置”演化为：

1. 支持基础观测和风增强观测
2. 支持额外高层投放特征
3. 支持纯 RNN、MLP+RNN、轻量 MLP+RNN 三类结构
4. 评估脚本支持按 checkpoint 自动适配观测维度

如果后续你继续改：

- 观测维度
- 高层投放特征定义
- 轻量 MLP-RNN 配置
- 风增强开关

请同步更新本文档，避免网络结构说明和实际训练代码再次脱节。
