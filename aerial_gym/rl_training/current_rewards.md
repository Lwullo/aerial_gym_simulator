# 当前奖励函数文档 (Current Rewards)

> **Updated: 2026-01-30**
> **Context**: Configured for **Raw Noise** (No Normalization) and **High Obstacle Density**.
> **Note**: Unused rewards (Terminal, Anchor, Success) have been removed from this document.

## 1. 总奖励公式 (Total Reward)

$$
R_t = R_{pot} + R_{dir} + R_{smooth} + R_{safe} + R_{hover} + R_{crash}
$$

---

## 2. 各项分量详解

### A. 统一势能奖励 (Unified Potential Reward) - $R_{pot}$

基于势能场的变化量计算奖励，鼓励由高势能向低势能移动。
**注意：噪声项不再归一化，直接使用原始物理强度 (0~5.0+)。**

$$
R_{pot} = k_J \cdot \tanh\left( \frac{J_{t-1} - J_t}{s_J} \right)
$$

其中势能函数 $J(\mathbf{p})$ 定义为：

$$
J(\mathbf{p}) = w_d \left(\frac{d}{d_0}\right)^2 + w_n \cdot n_{raw}(\mathbf{p})
$$

*   **参数配置**:
    *   `potential_kj` ($k_J$): **1.0**
    *   `potential_sj` ($s_J$): **0.1**
    *   `potential_w_d` ($w_d$): **1.0**
    *   `potential_w_n` ($w_n$): **1.0**
    *   `potential_d0` ($d_0$): **2.5 m**
    *   $n_{raw}(\mathbf{p})$: **原始噪音强度** (通常 0~5.0，不再归一化)。

---

### B. 悬停奖励 (Hover Reward) - $R_{hover}$

鼓励在“高质量区域 + 低噪声 + 低速度”的位置稳定悬停。**注意：当前实现中去除了显式的距离门控，$J$ 值低隐含了距离近。**

$$
R_{hover} = k_h \cdot g_J(J_t) \cdot g_v(||\mathbf{v}_t||)
$$

门控函数：
*   质量门控: $g_J(J) = \exp(-J/J_h)$
*   速度门控: $g_v(v) = \exp(-(v/v_h)^2)$

*   **参数配置**:
    *   `hover_reward_kh` ($k_h$): **1.0**
    *   `hover_reward_jh` ($J_h$): **1.0**
    *   `hover_reward_vh` ($v_h$): **0.2 m/s**

---

### C. 方向对齐奖励 (Direction Alignment Reward) - $R_{dir}$

鼓励机头朝向目标点 (Yaw Alignment)。

$$
R_{dir} = k_{dir} \cdot \cos(\theta_{error})
$$

*   **参数配置**:
    *   `direction_alignment_reward_magnitude` ($k_{dir}$): **0.5**

---

### D. 安全奖励 (Safety Reward) - $R_{safe}$

惩罚型 Log-Barrier，仅当距离障碍物小于阈值时生效。

$$
R_{safe} = k_{safe} \cdot \frac{1}{N} \sum_{i=1}^{N} \min\left( \log(d_i) - \log(d_{thresh}), \ 0 \right)
$$

*   **参数配置**:
    *   `safety_reward_magnitude` ($k_{safe}$): **0.5**
    *   `safety_dist_threshold`: **1.0 m**

---

### E. 动作平滑惩罚 (Action Smoothness) - $R_{smooth}$

$$
R_{smooth} = -k_a ||\mathbf{u}_t||^2 - k_{\Delta a} ||\mathbf{u}_t - \mathbf{u}_{t-1}||^2
$$

*   **参数配置**:
    *   `action_magnitude_penalty_weight` ($k_a$): **0.05**
    *   `action_change_penalty_weight` ($k_{\Delta a}$): **0.1**

---

### F. 碰撞惩罚 (Collision Penalty) - $R_{crash}$

$$
R_{crash} = \begin{cases} -C_{crash} & \text{if collision} \\ 0 & \text{otherwise} \end{cases}
$$

*   **参数配置**:
    *   `collision_penalty` ($C_{crash}$): **-50.0**

---

## 3. 环境与物理配置

*   **障碍物数量**: **30** (高密度)
*   **物理扰动**: GMM Force Enabled ($k=0.05$)
