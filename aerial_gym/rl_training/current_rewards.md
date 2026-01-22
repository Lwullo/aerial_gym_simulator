# 当前奖励函数文档 (Current Rewards)

> **Updated: 2026-01-20**
> **Context**: Added **Hover Reward** (continuous gating) and **Terminal Reward Scheme B** (stability-based). Updated parameters for conservative policy.

## 1. 总奖励公式 (Total Reward)

$$
R_t = R_{pot} + R_{dir} + R_{smooth} + R_{safe} + R_{hover} + R_{crash} + R_{term}
$$

---

## 2. 各项分量详解

### A. 统一势能奖励 (Unified Potential Reward) - $R_{pot}$

基于势能场的变化量计算奖励，鼓励由高势能向低势能移动 (即: 靠近目标且降低噪音)。

$$
R_{pot} = k_J \cdot \tanh\left( \frac{J_{t-1} - J_t}{s_J} \right)
$$

其中势能函数 $J(\mathbf{p})$ 定义为：

$$
J(\mathbf{p}) = w_d \left(\frac{d}{d_0}\right)^2 + w_n \cdot \hat{n}(\mathbf{p})
$$

*   **参数配置**:
    *   `potential_kj` ($k_J$): **1.5** (最大奖励幅度，已增大以鼓励探索)
    *   `potential_sj` ($s_J$): **0.08** (敏感度系数，已减小以增强对微小变化的感知)
    *   `potential_w_d` ($w_d$): **1.0**
    *   `potential_w_n` ($w_n$): **1.0**
    *   `potential_d0` ($d_0$): **2.0 m**
    *   $\hat{n}(\mathbf{p})$: 归一化噪音强度 (0~1)。

---

### B. 悬停奖励 (Hover Reward) - $R_{hover}$ **[NEW]**

鼓励在“目标附近 + 低噪声 + 低速度”的位置稳定悬停。由三个平滑门控函数组成。

$$
R_{hover} = k_h \cdot g_d(d_t) \cdot g_J(J_t) \cdot g_v(||\mathbf{v}_t||)
$$

门控函数：
*   距离门控: $g_d(d) = \exp(-(d/d_h)^2)$
*   质量门控: $g_J(J) = \exp(-J/J_h)$
*   速度门控: $g_v(v) = \exp(-(v/v_h)^2)$

*   **参数配置**:
    *   `hover_reward_kh` ($k_h$): **0.3** (已降低权重以防止过早陷入局部最优)
    *   `hover_reward_dh` ($d_h$): **2.0 m**
    *   `hover_reward_vh` ($v_h$): **0.35 m/s**
    *   `hover_reward_jh` ($J_h$): **1.0**

---

### C. 终端奖励 (Terminal Reward - Scheme B) - $R_{term}$ **[NEW]**

基于**动态稳定性**的成功判定。不要求完美收敛至圆心，只要求在目标区域内保持“足够好”的状态。

*   **触发条件** (需同时满足保持 `min_success_steps` 步):
    1.  **位置**：$d \le 2.0$m
    2.  **速度**：$||\mathbf{v}|| \le 0.35$m/s
    3.  **势能**：$J_t \le J_{best} + 0.03$ (势能未明显恶化，相对于进入圈内后的最优值)

*   **奖励值**:
    *   成功触发时一次性给予 **+50.0** 并立即结束 Episode。

---

### D. 碰撞惩罚 (Collision Penalty) - $R_{crash}$

$$
R_{crash} = \begin{cases} -C_{crash} & \text{if collision} \\ 0 & \text{otherwise} \end{cases}
$$

*   **参数配置**:
    *   `collision_penalty` ($C_{crash}$): **30.0** (已增大惩罚力度)

---

### E. 安全奖励 (Safety Reward) - $R_{safe}$

惩罚型 Log-Barrier，仅当 $d < d_{thresh}$ 时生效。

$$
R_{safe} = k_{safe} \cdot \frac{1}{N} \sum_{i=1}^{N} \min\left( \log(d_i) - \log(d_{thresh}), \ 0 \right)
$$

*   **参数配置**:
    *   `safety_reward_magnitude`: **0.1**
    *   `safety_dist_threshold`: **1.0 m**

---

### F. 其他 (Others)

*   **动作平滑**: $k_a=0.05, k_{\Delta a}=0.1$
*   **方向对齐**: $k_{dir}=0.0$ (禁用)

---

## 3. 环境配置 (Environment Config)

*   **dt**: 0.01s
*   **Episode Length**: **1000 steps** (10.0s)
*   **GMM Force**: Enabled ($k=0.05$)
