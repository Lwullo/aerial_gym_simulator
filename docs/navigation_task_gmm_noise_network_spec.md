# Navigation GMM Noise Task 神经网络与训练稳定性说明（当前代码快照）

> 任务名：`navigation_task_gmm_noise`  
> 更新时间：2026-03-19  
> 文档用途：记录当前任务所用策略网络架构、关键超参数，以及可复现实验中的稳定性论证

---

## 1. 配置来源（Source of Truth）

- 任务配置：`aerial_gym/config/task_config/navigation_task_gmm_noise_config.py`
- PPO 配置：`aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation.yaml`
- 训练入口：`aerial_gym/rl_training/rl_games/runner.py`

---

## 2. 网络输入输出定义

### 2.1 输入层（Observation）

当前输入维度为 `12`：

```text
s = [x_rel, y_rel, z_rel, vx, vy, vz, roll, pitch, wx, wy, wz, height]
```

- `x_rel,y_rel,z_rel`：目标相对位置
- `v_x,v_y,v_z`：机体线速度
- `roll,pitch`：roll/pitch
- `wx,wy,wz`：机体角速度
- `h`：当前高度（绝对 z）

说明：训练侧启用了 `normalize_input=True`，进入策略网络前会进行运行均值/方差归一化。

### 2.2 输出层（Action）

当前动作维度为 `5`：

```text
a = [a_vx, a_vy, a_vz, a_yawrate, a_drop] in [-1, 1]^5
```

动作映射（环境侧）：

- `v_cmd = 1.2 * [a_vx, a_vy, a_vz]` (m/s)
- `yawrate_cmd = (pi/6) * a_yawrate` (rad/s)
- `drop_switch = a_drop`，当 `drop_switch > 0.7` 触发 DROP

---

## 3. 网络架构（Actor-Critic + RNN）

当前使用 `rl_games` 的 `continuous_a2c_logstd`，并开启 PPO 训练模式。

### 3.1 主干结构

- `separate=False`：Actor/Critic 共享特征主干
- MLP：`12 -> 256 -> 128 -> 64`
- 激活函数：`ELU`
- RNN：`GRU(64)`，`layers=1`，`before_mlp=False`（即 MLP 后接 GRU）
- `layer_norm=True`（对 GRU 输出做 LayerNorm）

### 3.2 输出头

- Actor 均值头（`mu`）：`Linear(64, 5)`，`mu_activation=None`
- Actor 方差头（`logstd`）：`fixed_sigma=True`，5 维全局可训练参数（非状态相关）
- Critic 价值头：`Linear(64, 1)`

### 3.3 可训练参数量

按当前配置实例化后统计得到：

- **总可训练参数：`69,963`**

主要分布：

- MLP：44,480
- GRU：24,960
- LayerNorm：128
- Value head：65
- Mu head：325
- LogStd 参数：5

### 3.4 按层展开（你要的“输入/隐藏/激活/输出”）

```text
输入层:
  Linear in: 12-dim observation

隐藏层:
  MLP hidden-1: 256
  MLP hidden-2: 128
  MLP hidden-3: 64
  RNN hidden:   GRU(64), 1 layer

激活函数:
  MLP activation: ELU
  GRU internal gates: sigmoid/tanh (框架内部)
  mu head activation: None
  sigma head activation: None (logstd -> exp 转为 sigma)

输出层:
  Actor mu:    Linear(64 -> 5)
  Actor sigma: fixed trainable 5-dim logstd/sigma parameter
  Critic V:    Linear(64 -> 1)
```

---

## 4. 关键超参数（PPO）

以下为 `ppo_aerial_quad_navigation.yaml` 当前值：

- 算法：`a2c_continuous`（`ppo=True`）
- 学习率：`1e-4`
- 学习率策略：`adaptive`
- KL 阈值：`0.008`
- 折扣因子 `gamma`：`0.99`
- GAE `tau`：`0.95`
- PPO clip `e_clip`：`0.2`
- 熵系数 `entropy_coef`：`0.008`
- Critic 系数 `critic_coef`：`2`
- 梯度裁剪：`grad_norm=1.0`，`truncate_grads=True`
- rollout 长度：`horizon_length=32`
- minibatch：`2048`
- mini-epochs：`2`
- RNN 序列长度：`seq_length=8`
- 优势归一化：`normalize_advantage=True`
- 观测归一化：`normalize_input=True`
- 价值归一化：`normalize_value=True`
- reward 缩放：`reward_shaper.scale_value=0.1`
- 最大 epoch：`9000`

运行注意：

- `runner.py` 默认会把 `num_envs` 覆盖为命令行参数值（默认 `1024`），并同步覆盖 `num_actors`。
- 若你训练时显式传入 `--num_envs`，以命令行为准。

---

## 5. 稳定性证明（工程可验证版）

### 5.1 结论范围

这里给出的是**训练稳定性与数值有界性论证**（practical stability），不是“全局最优收敛”的严格数学证明。

### 5.2 命题 A：动作有界

策略输出先经 `[-1,1]` 截断，再映射到控制指令，因此：

```text
|vx|, |vy|, |vz| <= 1.2
|yawrate| <= pi/6
```

因此策略不会因动作无界导致控制量爆炸。

### 5.3 命题 B：回报有界（有限时域）

episode 长度有限（当前 `1000` 步），且主要奖励项幅值受限：

- 评分项：`R_score \in [0, 20]`
- 投放姿态奖励：`R_att \in [0, 1.0]`（两项各 `<=0.5`）
- 额外终止惩罚：`no_drop=-20`，外圈投放 `-20`
- 高度软惩罚由环境高度区间与权重限制在有限范围内

在“仿真状态有界”的条件下（Isaac Gym 常规安全边界与重置机制），回报为有界随机变量，方差可控。

### 5.4 命题 C：策略更新步长受控

PPO 目标：

```text
L_clip(theta) = E[min(r_t(theta) * A_hat_t,
                      clip(r_t(theta), 1-eps, 1+eps) * A_hat_t)]
```

其中 `eps=0.2`。配合：

- `kl_threshold=0.008` + `lr_schedule=adaptive`
- `grad_norm=1.0` 梯度裁剪

可限制单次更新过大，抑制策略突变。

### 5.5 命题 D：RNN 数值稳定性增强

当前 RNN 为 GRU + LayerNorm，并使用短序列截断反传（`seq_length=8`）：

- GRU 门控结构可抑制长期依赖下的数值漂移
- LayerNorm 降低 hidden state 分布漂移
- 短序列 BPTT + 梯度裁剪降低梯度爆炸风险

### 5.6 工程判据（建议持续监控）

建议同时观察：

- `rewards/*`：是否持续上升并趋于平稳
- `losses/entropy`：是否异常单调上升（可能探索过强）
- `losses/*`（policy/value）：是否出现震荡放大
- `performance/landing_error_xy_ema`：是否下降
- `performance/attitude_total_deg_ema` 与 `performance/attitude_total_deg_std`：投放姿态稳定性
- `performance/impulse_metric_mean/std`：投放冲击是否收敛

若出现“reward 上升但 entropy 持续上升且动作抖动”，通常需要联调：

- 熵系数
- 学习率/KL 阈值
- 奖励项权重比例（尤其 `R_score` 与冲击/姿态项）

---

## 6. 版本注记

本说明针对当前仓库配置快照。后续若修改了：

- `observation_space_dim / action_space_dim`
- `network.mlp / rnn`
- `learning_rate / entropy_coef / kl_threshold`
- DROP 奖励结构

请同步更新本文档，避免“文档参数”与“训练参数”不一致。
