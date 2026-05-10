# Navigation Task GMM Noise: 训练与评估代码更新说明

本文档记录 2026-04 前后相对于旧版流程的主要代码更新，重点覆盖：

- 训练侧新增的观测/配置逻辑
- 对比评估脚本新增的数据保存与重绘能力
- 实验 4 风强时间序列图的生成逻辑

对齐文件：

- [eval_drop_paper_comparison.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/scripts/eval_drop_paper_comparison.py)
- [navigation_task_gmm_noise.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)
- [task_registry.py](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/registry/task_registry.py)

更新时间：2026-04-10

## 1. 训练侧更新

### 1.1 新增高层投放特征

任务配置新增：

```python
use_drop_decision_features = False
```

开启后追加 3 个特征：

- `predicted_drop_error_xy`
- `predicted_fall_time`
- `predicted_release_risk`

主要用途：

- 给 `MLP+RNN` 一个更明确的单步释放决策输入
- 帮助压极端误投带来的长尾误差

### 1.2 新增轻量 MLP+RNN 训练配置

新增训练 YAML：

- [ppo_aerial_quad_navigation_mlp_rnn_lite.yaml](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation_mlp_rnn_lite.yaml)

和旧版 `ppo_aerial_quad_navigation.yaml` 的差异：

- MLP：`[256,128,64] -> [128,64]`
- `seq_length: 8 -> 16`
- `learning_rate: 5e-5 -> 3e-5`
- `env_config.use_drop_decision_features: True`

### 1.3 训练入口兼容扩展

为了支持 YAML 里新的：

```yaml
env_config:
  use_drop_decision_features: True
```

`task_registry.make_task(...)` 现在支持额外任务覆盖字段，并在创建任务前写回 `task_config`。

这样训练入口无需修改调用方式，只改 YAML 就能启用新特征。

## 2. 对比评估脚本更新

### 2.1 当前支持的方法

当前比较脚本支持：

- `PPO`：纯 MLP
- `PPO-GRU`：MLP + GRU
- `PPO-RNN`：纯 GRU
- `MPC`

其中 `PPO-RNN` 是可选方法，只有传 `--rnn_checkpoint` 时才加载。

### 2.2 观测布局自动识别

评估脚本会从 checkpoint 自动推断：

- 是否包含 `actor_mlp`
- 是否包含 `GRU`
- 输入维度是多少

然后自动设置环境的：

- `observation_space_dim`
- `use_wind_estimation_features`
- `use_drop_decision_features`

当前已支持：

- `15D`
- `18D`
- `34D`
- `37D`

并且显式禁止：

- `18D` 基础观测 + 高层特征
- 与 `34D/37D` 风增强模型在同一轮比较里混跑

### 2.3 当前每个条件保存的内容

对比实验现在不是只出 PDF，而是每个实验条件都会额外保存：

```text
raw/combined_drop_metrics.csv
raw/post_drop_trace.npz
stats/summary_mean_std.csv
stats/summary_median_iqr.csv
stats/no_drop_rate.csv
stats/key_metrics.csv
figures/precision_cdf.pdf
figures/roll_pitch_case.pdf
figures/impact_cdf.pdf
figures/summary_table.pdf
```

这样后续改标题、图例、配色，不需要重新跑仿真。

### 2.4 render-only 重绘模式

新增参数：

```text
--render_only_from_saved True
```

作用：

- 跳过 Isaac Gym 仿真
- 直接从保存下来的 `raw/combined_drop_metrics.csv` 重生成 PDF

注意：

- 如果该次实验没有保存 raw CSV，就不能只重绘

## 3. post-drop 时间序列数据

### 3.1 保存形式

现在对每个已投放 case，脚本会额外保存：

- `post_drop_trace.npz`

其中包含：

- `norm_t`
- `error_xy_m`
- `pos_x_m`
- `pos_y_m`
- `pos_z_m`
- `trace_fall_steps`
- `trace_fall_time_s`

### 3.2 这些轨迹的真实含义

这里保存的不是“投放后继续跑很多 RL step 的 episode 轨迹”，而是：

- 在 DROP 当步内部
- 子机自由落体积分过程
- 统一重采样到固定长度后的时间序列

因此它适合做：

- 投放后误差演化图
- 不同风强下的 post-drop 对比图

### 3.3 实验 4 风强时间序列图

对于实验 4，会额外在实验根目录下生成：

- `figures/post_drop_error_evolution_wind.pdf`
- `stats/post_drop_wind_summary.csv`

图的结构是：

- 每个算法一个子图
- 每个子图里三条曲线：弱风 / 中风 / 强风
- 实线：中位数
- 阴影：P25-P75
- 虚线：P90
- 图例里标注有效样本数 `N`

这样可以避免只看成功样本曲线而忽略样本量差异。

## 4. 当前建议的使用方式

### 训练新轻量 MLP+RNN

```bash
cd aerial_gym/rl_training/rl_games

/home/lwulo/miniconda3/envs/rlgpu/bin/python runner.py \
  --train \
  --file=./ppo_aerial_quad_navigation_mlp_rnn_lite.yaml \
  --task=navigation_task_gmm_noise \
  --experiment_name=PPO-MLP-RNN-lite \
  --num_envs=1024 \
  --headless=True \
  --use_warp=True
```

### 重跑完整对比实验

```bash
/home/lwulo/miniconda3/envs/rlgpu/bin/python aerial_gym/scripts/eval_drop_paper_comparison.py \
  --headless True \
  --experiments 1,2,3,4,5 \
  --output_root <your_result_dir> \
  --baseline_checkpoint <ppo_mlp_ckpt> \
  --gru_checkpoint <ppo_mlp_rnn_lite_ckpt> \
  --rnn_checkpoint <ppo_rnn_ckpt>
```

### 只重绘

```bash
/home/lwulo/miniconda3/envs/rlgpu/bin/python aerial_gym/scripts/eval_drop_paper_comparison.py \
  --render_only_from_saved True \
  --headless True \
  --experiments 1,2,3,4,5 \
  --output_root <your_result_dir>
```

## 5. 当前版本的文档入口建议

如果你后面只想快速对照当前逻辑，建议优先看：

1. [drop_reward_function_spec.md](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/docs/drop_reward_function_spec.md)
2. [navigation_task_gmm_noise_network_spec.md](/home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/docs/navigation_task_gmm_noise_network_spec.md)
3. 本文档

这三份文档分别负责：

- 奖励逻辑
- 观测/网络逻辑
- 训练/评估更新逻辑
