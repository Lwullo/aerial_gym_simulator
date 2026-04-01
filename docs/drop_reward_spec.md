# DROP Reward Spec (Current Implementation)

同步日期：2026-03-20  
主文档：`drop_reward_function_spec.md`

## 1) WAIT 阶段

- `R_wait = -time_penalty`
- `R_progress = direction_reward_weight * (d_prev_xy - d_curr_xy)`（仅未 DROP 且距离大于阈值时生效）

## 2) DROP 当步

- `R_score = score_reward_weight * score_max * exp(-(landing_error_xy/score_d0)^score_p)`
- `impulse_metric = impulse_alpha * Delta_theta + impulse_beta * Delta_omega`
- `R_impulse = -impulse_penalty_weight * impulse_metric`
- `R_att = R_posture + R_angvel`
  - `R_posture = attitude_reward_weight * exp(-(theta_drop/attitude_theta0)^2)`
  - `R_angvel = attitude_reward_weight * exp(-(omega_drop_xy/attitude_theta0)^2)`
  - `theta_drop = ||[roll_before, pitch_before]||_2`
  - `omega_drop_xy = ||[w_x, w_y]||_2`（DROP 前瞬间）

若 `landing_error_xy` 超过外圈阈值，则 DROP 当步奖励强制覆盖为：

- `R_drop = -outside_region_penalty`

## 3) 终止附加惩罚

- 若 episode 结束且从未 DROP：`R_no_drop = -no_drop_penalty`

## 4) 说明

- `R_att` 第二项已经不再使用速度方向夹角；改为 DROP 前角速度稳定奖励。
- 角速度项与姿态项共享同一权重与尺度：`attitude_reward_weight`、`attitude_theta0`。
