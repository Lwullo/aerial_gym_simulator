# 实施方案 - 方案 B 终端奖励与日志记录 (Scheme B)

## 目标描述
实施“方案 B” (Scheme B) 动态稳定性终端奖励逻辑，取代现有的成功判定标准。
该方案旨在奖励智能体在目标区域附近找到并保持“足够好”的位置，而不是要求完美的收敛。同时，增加失败案例的日志记录和成功率的可视化。

## 拟定修改

### [MODIFY] [navigation_task_gmm_noise_config.py](file:///home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/config/task_config/navigation_task_gmm_noise_config.py)

1.  **更新 `success_config`**:
    *   `success_reward = 50.0` (终端奖励)
    *   `success_radius = 2.0` (目标半径，保持现有的 2.0m)
    *   `stability_velocity_threshold = 0.35` (速度宽松阈值 $v_{hold}$)
    *   `stability_potential_delta = 0.03` (势能容忍度 $\delta_J$)
    *   `min_success_steps = 60` (保持步数 $N_{hold}$，对应 0.6秒)
    *   *(注意：取消原先可能存在的 +30 一次性抵达奖励计划)*

### [MODIFY] [navigation_task_gmm_noise.py](file:///home/lwulo/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/task/navigation_task_gmm_noise/navigation_task_gmm_noise.py)

1.  **初始化 (`reset_idx` & `__init__`)**:
    *   新增 Buffer `self.J_best_in_goal` (大小: num_envs, 初始化为无穷大)。
    *   新增 Buffer `self.hold_counter` (大小: num_envs, 初始化为 0)。
    *   在 `reset_idx` 中正确重置这些变量。

2.  **核心逻辑更新 (`post_physics_step` / `_check_success`)**:
    *   **更新最优值**: 若 `dist < 2.0`，则 `J_best_in_goal = min(J_best_in_goal, J_current)`。
    *   **稳定性判定 (Stability Check)**:
        *   `cond_dist`: `dist <= 2.0` (在圈内)
        *   `cond_vel`: `norm(vel) <= 0.35` (速度慢)
        *   `cond_pot`: `J_current <= J_best_in_goal + 0.03` (势能未恶化)
        *   `is_stable = cond_dist & cond_vel & cond_pot`
    *   **计数器更新**: 若 `is_stable` 为真，`hold_counter += 1`；否则重置为 `0`。
    *   **触发成功**: 若 `hold_counter >= 60`:
        *   `reward += 50.0` (发放终端奖励)
        *   `reset_buf = 1`
        *   `success_buf = 1`

3.  **失败日志记录 (Failure Logging)**:
    *   在 `reset_idx` 中，如果 `reset_buf[i]` 为真 但 `success_buf[i]` 为假 (即失败重置，包括碰撞或超时):
    *   将该回合结束时的 `dist` (距离), `speed` (速度), `max_hold_count` (最大保持步数) 记录到全局日志文件 `failure_log.txt` 中。
    *   *注：将在文件开头定义文件路径。*

4.  **TensorBoard 可视化**:
    *   确保 `success_buf` 的平均值 (即成功率) 被记录到 TensorBoard，标签为 `success_rate`。

## 验证计划
1.  **代码审查**: 检查逻辑与公式是否一致。
2.  **运行仿真**: 确保无报错崩溃。
3.  **检查日志**: 验证 `failure_log.txt` 是否在碰撞后生成并写入数据。
4.  **检查 TensorBoard**: 验证 `success_rate` 曲线是否出现。


![[Pasted image 20260118165648.png]]