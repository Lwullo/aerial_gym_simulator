# Navigation Task GMM Noise: Actor-Critic 训练流程总览（详细框图）

> 面向你当前工程的 PPO 训练链路（`rl_games`）。
> 目标：回答“观测先到哪里、Actor/Critic 怎么走、Loss 在哪里算、优化了什么参数”。

---

## 1. 先回答你的核心问题

是的，你现在的链路是：

1. 环境产生观测 `obs`。
2. `obs` 进入 Actor-Critic 网络（共享主干 + 两个输出头）。
3. Actor 输出动作分布参数，采样/确定动作并回传给环境。
4. Critic 输出状态价值 `V(s)`。
5. 收集一段 rollout 后，进入 PPO 的 Loss 计算与优化。

注意：当前 `ppo_aerial_quad_navigation.yaml` 里 `separate: False`，表示 **Actor 和 Critic 共享特征主干**，不是完全独立两套 backbone。

---

## 2. 端到端训练流程框图（环境交互到参数更新）

```mermaid
flowchart TD
    A[Env Reset<br/>NavigationTaskGmmNoise.reset()] --> B[返回观测字典<br/>obs_dict['observations']]
    B --> C[ExtractObsWrapper<br/>提取 observations 张量]
    C --> D[rl_games: play_steps / play_steps_rnn]

    D --> E[观测预处理<br/>normalize_input: RunningMeanStd]
    E --> F[Actor-Critic 前向]

    F --> F1[共享主干特征<br/>当前: MLP 12->256->128->64]
    F1 --> F2[Actor 头<br/>mu + logstd/fixed_sigma]
    F1 --> F3[Critic 头<br/>V(s)]

    F2 --> G[采样动作 a_t<br/>并裁剪到动作范围]
    G --> H[env.step(a_t)]
    H --> I[得到 r_t, done_t, next_obs]
    I --> J[写入 ExperienceBuffer<br/>obs, action, logp_old, value_old, reward, done]

    J --> K{达到 horizon_length?}
    K -- 否 --> D
    K -- 是 --> L[Bootstrap 最后一步 V(s_T)]

    L --> M[计算 GAE 优势 A_t 与回报 R_t]
    M --> N[prepare_dataset()<br/>构造 PPO minibatch 数据]

    N --> O[mini_epochs 循环]
    O --> P[minibatch 循环]
    P --> Q[calc_gradients()<br/>计算 a_loss/c_loss/entropy/b_loss]
    Q --> R[total_loss 反传<br/>梯度裁剪 + optimizer.step()]
    R --> S[计算 KL]
    S --> T[Adaptive LR Scheduler<br/>按 KL 调整 lr/entropy_coef]

    T --> U[写 TensorBoard 指标 + 存 checkpoint]
    U --> V[下一 epoch 继续]
```

---

## 3. 你这个任务里的“观测->动作->环境”具体内容

### 3.1 观测（当前默认）

`obs_dim = 12`

```text
[x_rel, y_rel, z_rel, vx, vy, vz, roll, pitch, wx, wy, wz, height]
```

### 3.2 动作

`act_dim = 5`

```text
[a_vx, a_vy, a_vz, a_yawrate, a_drop] in [-1, 1]^5
```

环境里 `a_drop > drop_threshold(0.7)` 会触发 DROP 事件。

---

## 4. Loss 在哪里算？怎么组合？

## 4.1 代码位置（你当前工程）

- 训练入口：
  - `aerial_gym/rl_training/rl_games/runner.py`
- PPO 主循环（rollout + update）：
  - `rl_games/common/a2c_common.py`
- 连续动作 PPO 梯度计算：
  - `rl_games/algos_torch/a2c_continuous.py`
- actor/critic loss 公式函数：
  - `rl_games/common/common_losses.py`

（`rl_games` 路径在你的 conda 环境中：
`/home/lwulo/miniconda3/envs/rlgpu/lib/python3.8/site-packages/rl_games/...`）

### 4.2 数学形式（对应 rl_games 实现）

1. PPO 比率：

\[
r_t(\theta)=\exp\big(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)\big)
\]

2. Actor（clip）损失：

\[
L_{actor}=\max\left(-A_t r_t(\theta),\,-A_t\,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\right)
\]

3. Critic 损失：

\[
L_{critic}=(R_t - V_\theta(s_t))^2
\]

4. 熵项（鼓励探索）：

\[
L_{entropy}=H\big(\pi_\theta(\cdot|s_t)\big)
\]

5. 动作边界正则（bounds loss，rl_games 内置）：

\[
L_{bound}=\sum_i \big[\max(\mu_i-1.1,0)^2 + \max(-1.1-\mu_i,0)^2\big]
\]

6. 总损失（对应 `a2c_continuous.py`）：

\[
L_{total}=L_{actor}+0.5\cdot \text{critic\_coef}\cdot L_{critic}
-\text{entropy\_coef}\cdot L_{entropy}
+\text{bounds\_loss\_coef}\cdot L_{bound}
\]

---

## 5. 优化的是哪些参数？

当前（你配置里 RNN 注释掉）主要优化：

1. 共享 MLP 主干参数（256/128/64）
2. Actor 的 `mu` 头参数
3. Critic 的 value 头参数
4. `fixed_sigma=True` 时的可训练 `logstd` 参数（全局参数，不依赖状态）

如果你打开 RNN（GRU），还会额外优化：

5. GRU 参数（以及对应 layer norm 参数）

---

## 6. “Loss/优化”细化框图（你问的关键中间环节）

```mermaid
flowchart LR
    A[Rollout Buffer<br/>obs, actions, old_logp, old_values, rewards, dones] --> B[GAE/Returns 计算]
    B --> C[Dataset 切分 minibatch]

    C --> D[前向: model(batch)]
    D --> D1[new_neglogp]
    D --> D2[new_values]
    D --> D3[entropy]
    D --> D4[mu/sigma]

    D1 --> E[Actor Clip Loss]
    D2 --> F[Critic Loss]
    D3 --> G[Entropy Term]
    D4 --> H[Bounds Loss + KL]

    E --> I[合成 total_loss]
    F --> I
    G --> I
    H --> I

    I --> J[backward()]
    J --> K[梯度裁剪]
    K --> L[Adam optimizer.step()]
    L --> M[按 KL 自适应更新 lr 与 entropy_coef]
```

---

## 7. 你当前配置下的重要训练超参数（PPO）

来自 `ppo_aerial_quad_navigation.yaml`：

- `algo`: `a2c_continuous` + `ppo=True`
- `learning_rate`: `1e-4`
- `lr_schedule`: `adaptive`
- `kl_threshold`: `0.007`
- `gamma`: `0.99`
- `tau(GAE)`: `0.95`
- `e_clip`: `0.2`
- `entropy_coef`: `0.008`
- `critic_coef`: `2`
- `horizon_length`: `32`
- `minibatch_size`: `2048`
- `mini_epochs`: `2`
- `normalize_input`: `True`
- `normalize_advantage`: `True`
- `normalize_value`: `True`

---

## 8. 对你问题的“一句话版本”

你现在就是“环境给观测 -> 同一个 Actor-Critic 网络前向得到动作和值函数 -> 采样收集 rollout -> 用 PPO 的 actor/critic/entropy/bounds 组成总损失 -> 反向传播更新网络参数 -> 下一个 epoch 重复”。

