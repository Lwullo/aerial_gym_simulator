# 对比实验结果分析：当前合并结果

本文档基于当前最新实验结果进行分析。实验 1、实验 2、实验 3 和实验 5 来自原完整结果目录；实验 4 已按“多高度 + 多风场 + 无障碍”重新运行，并使用新结果替换旧的“固定高度 + 多风场 + 有障碍”版本。

原结果目录：

`aerial_gym/rl_training/rl_games/runs/result_with_saved_traces`

新实验 4 结果目录：

`aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs`

数据来源包括每个实验条件下的：

- `raw/combined_drop_metrics.csv`
- `stats/key_metrics.csv`
- `stats/attitude_tail_metrics.csv`
- `figures/precision_cdf.pdf`
- `figures/roll_pitch_case.pdf`
- `figures/impact_cdf.pdf`
- `figures/summary_table.pdf`

实验 4 额外使用：

- `figures/post_drop_error_evolution_wind.pdf`
- `stats/post_drop_wind_summary.csv`

重新合并后的统计文件保存在：

`aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/combined_analysis_stats`

说明：原始结果文件中的 `ppo_gru` 标签对应本文中的 `Proposed`，即本研究提出的 MLP+GRU 投放决策算法；原始结果文件中的 `ppo_rnn` 标签在本文中统一命名为 `PPO-GRU`，表示不含 MLP 前端的 GRU baseline。

## 1. 指标口径

本文主要使用以下指标：

| 指标 | 含义 | 解释 |
|---|---|---|
| `Drop Rate` | 投放率 | 全部 case 中完成投放的比例 |
| `Mean Error` | 已投放样本的平均落点误差 | 越小越好 |
| `P50 / Median` | 已投放样本的中位误差 | 反映多数样本表现 |
| `P90` | 已投放样本 90 分位误差 | 反映尾部大误差风险 |
| `R@2m / R@3m` | 已投放样本中误差小于 2m / 3m 的比例 | 越高越好 |
| `Overall@2m / Overall@3m` | 把未投放也视为失败后的整体命中率 | 更能反映任务整体成功性 |
| `Abs Roll / Abs Pitch Mean` | 投放瞬间姿态绝对角均值 | 越小表示释放姿态越稳定 |
| `Impact Mean` | 释放冲击指标均值 | 越小表示释放冲击越小 |

`precision_cdf.pdf` 用于分析投放精度分布；`roll_pitch_case.pdf` 用于分析投放瞬间姿态；`impact_cdf.pdf` 用于分析冲击分布；`summary_table.pdf` 用于查验数值统计和未投放/撞毁统计。

## 2. 总体结果

全部 8 个实验条件合并后，四种方法的总体表现如下：

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@2m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 98.99% | 8.061m | 2.748m | 20.470m | 43.57% | 51.02% | 43.13% | 50.51% | 10.24deg | 6.47deg | 0.465 |
| Proposed | 97.04% | 2.669m | 1.034m | 7.669m | 75.63% | 82.97% | 73.39% | 80.51% | 15.51deg | 7.81deg | 0.533 |
| PPO-GRU | 97.32% | 3.275m | 1.286m | 11.108m | 67.21% | 76.95% | 65.41% | 74.89% | 15.73deg | 7.79deg | 0.510 |
| MPC | 52.69% | 5.101m | 2.705m | 15.004m | 18.76% | 62.88% | 9.89% | 33.13% | 2.54deg | 1.49deg | 0.146 |

依据图表：

- 总体数值来自所有 `raw/combined_drop_metrics.csv` 的合并统计。
- 各单条件数值可在对应实验的 `summary_table.pdf` 和 `stats/key_metrics.csv` 中核对。
- 精度分布形态由各条件的 `precision_cdf.pdf` 支撑。
- 姿态稳定性由各条件的 `roll_pitch_case.pdf` 支撑。
- 冲击大小由各条件的 `impact_cdf.pdf` 支撑。

总体结论：

- `Proposed` 在综合精度上最好，平均误差为 `2.669m`，`P90=7.669m`，`R@2m=75.63%`，`R@3m=82.97%`，均优于 `PPO` 和 `PPO-GRU`。
- `PPO-GRU` 排第二，整体优于 `PPO`，但尾部误差高于 `Proposed`。
- 新实验4引入随机高度后，`PPO` 在多风强条件下明显退化，导致总体 `Mean Error=8.061m`、`P90=20.470m`。
- `MPC` 的姿态和冲击指标最好，但投放率只有 `52.69%`，大量 case 在投放前撞毁，因此整体任务成功率最低。

### 2.1 释放姿态是否由长尾导致

为区分“姿态本身整体偏差较大”与“少数极端样本拉高均值”，额外统计释放瞬间合成姿态角：

```text
theta = sqrt(roll^2 + pitch^2)
```

对应统计文件：

[Combined Attitude Tail Metrics CSV](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/combined_analysis_stats/combined_attitude_tail_metrics.csv)

总体 dropped-only 姿态尾部统计如下：

| 方法 | Mean theta | Median theta | P90 theta | P95 theta | P99 theta | CVaR90 theta | theta>20deg | theta>30deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 13.34deg | 6.92deg | 31.54deg | 47.74deg | 102.33deg | 59.19deg | 18.06% | 10.77% |
| Proposed | 18.96deg | 12.03deg | 34.82deg | 66.40deg | 138.44deg | 79.13deg | 25.28% | 12.57% |
| PPO-GRU | 18.80deg | 10.45deg | 37.33deg | 69.33deg | 144.29deg | 84.31deg | 22.86% | 13.24% |
| MPC | 3.22deg | 2.81deg | 4.06deg | 5.27deg | 12.97deg | 8.81deg | 0.31% | 0.25% |

判断：

- 学习方法的姿态问题不是单纯由均值/中位数误读造成的。`Proposed` 的 median theta 为 `12.03deg`，高于 MPC 的 `2.81deg`，说明其常规释放姿态本身就比 MPC 更激进。
- 但学习方法姿态均值偏大的主要放大因素是尾部事件。以 `Proposed` 为例，mean theta 为 `18.96deg`，median theta 为 `12.03deg`，而 P95 达到 `66.40deg`，CVaR90 达到 `79.13deg`。这说明最差 10% 样本显著拉高了平均姿态角。
- `Proposed` 相比 `PPO-GRU` 的优势主要体现在精度尾部，而姿态尾部只略好于或接近 `PPO-GRU`：`P90=34.82deg` 低于 `PPO-GRU=37.33deg`，`CVaR90=79.13deg` 低于 `PPO-GRU=84.31deg`。两者都明显高于 MPC。
- `theta>20deg` 不宜直接定义为极端失稳，因为学习方法有约 20% 到 25% 的样本超过该阈值。更合理的解释是“中等姿态风险”。`theta>30deg` 更适合作为极端姿态事件指标。
- `MPC` 的姿态分布最集中，P90 仅 `4.06deg`，说明其释放姿态稳定性确实最好。但该结论必须结合投放率解释，因为 MPC 总体投放率只有 `52.69%`。

按条件看，`Proposed` 的姿态风险主要来自中风/强风/复杂场景中的尾部样本：

| 条件 | Proposed Mean theta | Proposed Median theta | Proposed P90 theta | Proposed theta>30deg |
|---|---:|---:|---:|---:|
| Exp4 Weak | 9.01deg | 8.16deg | 15.32deg | 0.65% |
| Exp1 Fixed | 18.87deg | 12.99deg | 32.50deg | 12.01% |
| Exp2 NoObs | 18.92deg | 12.59deg | 34.90deg | 13.34% |
| Exp2 RandObs | 18.64deg | 12.92deg | 33.31deg | 11.73% |
| Exp3 RandH | 18.96deg | 12.55deg | 32.66deg | 11.52% |
| Exp4 Medium | 18.95deg | 13.19deg | 33.73deg | 12.73% |
| Exp4 Strong | 29.96deg | 16.51deg | 83.00deg | 27.59% |
| Exp5 Complex | 19.66deg | 12.02deg | 35.55deg | 12.69% |

结论：

- 弱风下 Proposed 的释放姿态并不差，median theta 为 `8.16deg`，P90 为 `15.32deg`。
- 中风、随机高度和综合复杂场景下，Proposed 的 median theta 约为 `12deg` 到 `13deg`，说明常规释放姿态是中等偏大的；同时 P90 约为 `33deg` 到 `36deg`，说明存在稳定的尾部风险。
- 强风是姿态退化最明显的条件，Proposed 的 median theta 上升到 `16.51deg`，P90 达到 `83.00deg`，说明强风下既存在整体姿态变差，也存在严重长尾。
- 因此，当前姿态问题应表述为：学习方法为了提高投放精度，释放姿态相对 MPC 更激进；其平均姿态变差既有常规姿态偏大的因素，也明显受到长尾极端释放事件放大，其中强风贡献最大。

## 3. 实验一：固定高度、固定中风、无障碍

结果目录：

`aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp1_fixedH_fixedW_noObs/main`

依据图表：

- [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp1_fixedH_fixedW_noObs/main/figures/precision_cdf.pdf)
- [Roll/Pitch Case](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp1_fixedH_fixedW_noObs/main/figures/roll_pitch_case.pdf)
- [Impact CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp1_fixedH_fixedW_noObs/main/figures/impact_cdf.pdf)
- [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp1_fixedH_fixedW_noObs/main/figures/summary_table.pdf)
- [Key Metrics CSV](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp1_fixedH_fixedW_noObs/main/stats/key_metrics.csv)

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 99.40% | 2.273m | 1.258m | 4.946m | 73.24% | 84.71% | 84.20% | 14.42deg | 9.57deg | 0.561 |
| Proposed | 99.05% | 2.359m | 1.028m | 3.971m | 78.95% | 86.98% | 86.15% | 15.49deg | 7.52deg | 0.536 |
| PPO-GRU | 99.55% | 2.890m | 1.298m | 9.229m | 71.22% | 83.53% | 83.15% | 17.62deg | 8.01deg | 0.528 |
| MPC | 52.65% | 5.891m | 3.049m | 15.983m | 17.57% | 47.58% | 25.05% | 3.01deg | 1.58deg | 0.166 |

分析：

- `PPO` 的平均误差最低，但 `Proposed` 的 `P90`、`R@2m` 和 `R@3m` 更好。依据是 `summary_table.pdf` 中的 dropped-only quality 表，以及 `precision_cdf.pdf` 中 `Proposed` 曲线在 2m 到 3m 区间更靠上。
- `Proposed` 的中位误差 `1.028m` 明显低于 `PPO` 的 `1.258m`，说明多数样本投放更集中，但平均误差略高，说明仍存在少量偏大样本。
- `PPO-GRU` 投放率最高，但 `P90=9.229m`，尾部误差明显大于 `Proposed`。这一点由 `precision_cdf.pdf` 的尾部和 `summary_table.pdf` 的 `P90` 支撑。
- `MPC` 的 `|Roll|`、`|Pitch|` 和 `Impact Mean` 最小，说明释放姿态和冲击最好；但投放率只有 `52.65%`，`Overall@3m=25.05%`，总体任务效果最差。依据是 `summary_table.pdf` 的 no-drop breakdown 和 `impact_cdf.pdf`。

## 4. 实验二：固定高度、固定中风、有无随机障碍

### 4.1 无障碍条件

结果目录：

`aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/no_obstacle`

依据图表：

- [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/no_obstacle/figures/precision_cdf.pdf)
- [Roll/Pitch Case](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/no_obstacle/figures/roll_pitch_case.pdf)
- [Impact CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/no_obstacle/figures/impact_cdf.pdf)
- [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/no_obstacle/figures/summary_table.pdf)

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 99.15% | 2.421m | 1.308m | 5.796m | 68.99% | 81.09% | 80.40% | 13.92deg | 9.05deg | 0.572 |
| Proposed | 98.60% | 2.391m | 1.043m | 4.079m | 79.41% | 87.07% | 85.85% | 15.43deg | 7.63deg | 0.542 |
| PPO-GRU | 99.40% | 2.815m | 1.279m | 8.661m | 71.03% | 83.40% | 82.90% | 17.46deg | 7.76deg | 0.525 |
| MPC | 55.45% | 6.032m | 3.090m | 16.630m | 16.14% | 46.71% | 25.90% | 2.93deg | 1.56deg | 0.148 |

分析：

- `Proposed` 在无障碍条件下综合最好：平均误差、P90、R@2m、R@3m 均优于 `PPO` 和 `PPO-GRU`。依据是 `summary_table.pdf` 与 `precision_cdf.pdf`。
- `PPO-GRU` 的投放率略高，但尾部误差 `P90=8.661m`，明显高于 `Proposed`。
- `PPO` 的姿态 roll 小于两个循环策略，但 pitch 更大，说明释放姿态并非全面更稳定。依据是 `roll_pitch_case.pdf` 和 `summary_table.pdf`。
- `MPC` 的姿态与冲击仍最好，但投放率不足，导致 `Overall@3m` 只有 `25.90%`。

### 4.2 随机障碍条件

结果目录：

`aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/random_obstacle`

依据图表：

- [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/random_obstacle/figures/precision_cdf.pdf)
- [Roll/Pitch Case](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/random_obstacle/figures/roll_pitch_case.pdf)
- [Impact CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/random_obstacle/figures/impact_cdf.pdf)
- [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/random_obstacle/figures/summary_table.pdf)

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 99.25% | 2.304m | 1.312m | 4.923m | 70.38% | 82.42% | 81.80% | 14.18deg | 8.59deg | 0.562 |
| Proposed | 98.85% | 2.302m | 0.962m | 4.139m | 80.17% | 87.15% | 86.15% | 15.35deg | 7.46deg | 0.532 |
| PPO-GRU | 99.50% | 2.737m | 1.157m | 9.105m | 72.76% | 82.76% | 82.35% | 14.35deg | 7.34deg | 0.465 |
| MPC | 51.50% | 5.882m | 3.083m | 15.821m | 19.22% | 45.83% | 23.60% | 2.88deg | 1.55deg | 0.147 |

分析：

- 随机障碍加入后，`Proposed` 仍保持最好的精度统计，说明其对障碍扰动没有明显退化。依据是 `precision_cdf.pdf` 和 `summary_table.pdf`。
- `Proposed` 的中位误差 `0.962m` 是四种方法中最低的，说明多数投放样本精度较高。
- `PPO-GRU` 的冲击均值最低于学习方法，但精度尾部较差，`P90=9.105m`。
- `MPC` 姿态/冲击稳定，但投放率只有 `51.50%`，其 `summary_table.pdf` 中 no-drop breakdown 是判断总体任务能力不足的主要依据。

## 5. 实验三：固定中风、固定障碍、随机高度

结果目录：

`aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp3_randomH_fixedW_fixedObs/main`

依据图表：

- [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp3_randomH_fixedW_fixedObs/main/figures/precision_cdf.pdf)
- [Roll/Pitch Case](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp3_randomH_fixedW_fixedObs/main/figures/roll_pitch_case.pdf)
- [Impact CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp3_randomH_fixedW_fixedObs/main/figures/impact_cdf.pdf)
- [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp3_randomH_fixedW_fixedObs/main/figures/summary_table.pdf)

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 99.65% | 11.243m | 13.602m | 21.726m | 30.36% | 35.27% | 35.15% | 8.17deg | 5.00deg | 0.417 |
| Proposed | 98.10% | 2.314m | 0.946m | 4.264m | 80.84% | 87.16% | 85.50% | 15.62deg | 7.65deg | 0.541 |
| PPO-GRU | 99.10% | 2.692m | 1.102m | 8.642m | 73.97% | 84.31% | 83.55% | 14.62deg | 7.54deg | 0.469 |
| MPC | 52.45% | 5.634m | 2.870m | 15.707m | 18.11% | 55.96% | 29.35% | 2.60deg | 1.53deg | 0.145 |

分析：

- 这是区分算法鲁棒性的关键实验。`PPO` 在随机高度下明显失效，平均误差达到 `11.243m`，中位误差也达到 `13.602m`。依据是 `precision_cdf.pdf` 中 PPO 曲线显著右移，以及 `summary_table.pdf`。
- `Proposed` 在随机高度条件下明显优于其他学习方法，平均误差 `2.314m`，`R@2m=80.84%`，`R@3m=87.16%`。
- `PPO-GRU` 也显著优于 `PPO`，但 `P90=8.642m`，尾部风险仍高于 `Proposed`。
- `MPC` 在 dropped-only 的 `R@3m` 高于 `PPO`，但投放率仅 `52.45%`，整体 `Overall@3m=29.35%`，因此不能认为整体优于学习方法。
- 姿态/冲击方面，`MPC` 仍是最稳定方法；学习方法中 `PPO` 姿态角较小，但精度明显不足。

## 6. 实验四：多高度、无障碍、多风场强度

实验 4 的完整风强对比同时参考：

- [Post-DROP Error Evolution](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/figures/post_drop_error_evolution_wind.pdf)
- [Post-DROP Wind Summary CSV](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/stats/post_drop_wind_summary.csv)

说明：该实验用于单独分析“随机高度 + 多风场”的影响，并关闭障碍物。因此，实验 4 中的性能变化主要由高度变化和风场强度造成，而不是由障碍物碰撞风险造成。

### 6.1 弱风

结果目录：

`aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/weak_wind`

依据图表：

- [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/weak_wind/figures/precision_cdf.pdf)
- [Roll/Pitch Case](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/weak_wind/figures/roll_pitch_case.pdf)
- [Impact CDF](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/weak_wind/figures/impact_cdf.pdf)
- [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/weak_wind/figures/summary_table.pdf)

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 100.00% | 10.465m | 13.345m | 21.405m | 39.50% | 41.75% | 41.75% | 4.24deg | 2.54deg | 0.284 |
| Proposed | 99.55% | 1.493m | 0.707m | 1.562m | 93.02% | 94.83% | 94.40% | 6.60deg | 4.64deg | 0.298 |
| PPO-GRU | 98.35% | 1.522m | 0.922m | 2.660m | 88.26% | 90.70% | 89.20% | 8.29deg | 3.89deg | 0.318 |
| MPC | 87.10% | 2.354m | 2.444m | 2.744m | 18.77% | 100.00% | 87.10% | 1.60deg | 1.18deg | 0.134 |

分析：

- 弱风且无障碍时，`Proposed` 精度最好，`Mean Error=1.493m`，`P90=1.562m`，`R@2m=93.02%`，`R@3m=94.83%`。依据是 `precision_cdf.pdf` 和 `summary_table.pdf`。
- `PPO` 在随机高度条件下明显失效，即使弱风且无障碍，平均误差也达到 `10.465m`，中位误差达到 `13.345m`。这说明 PPO 的主要弱点不是障碍物，而是随机高度下的释放时机泛化不足。
- `PPO-GRU` 明显优于 `PPO`，但 `P90=2.660m` 高于 `Proposed=1.562m`，说明仅使用 GRU 的尾部精度仍弱于 MLP+GRU。
- `MPC` 的 dropped-only `R@3m=100.00%`，但 `R@2m=18.77%`，说明多数有效投放落在 2m 到 3m 范围内；其投放率为 `87.10%`，低于学习方法。

### 6.2 中风

结果目录：

`aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/medium_wind`

依据图表：

- [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/medium_wind/figures/precision_cdf.pdf)
- [Roll/Pitch Case](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/medium_wind/figures/roll_pitch_case.pdf)
- [Impact CDF](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/medium_wind/figures/impact_cdf.pdf)
- [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/medium_wind/figures/summary_table.pdf)

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 99.50% | 11.141m | 13.692m | 21.601m | 30.30% | 36.18% | 36.00% | 7.67deg | 4.70deg | 0.413 |
| Proposed | 97.80% | 2.561m | 1.068m | 5.755m | 77.20% | 85.12% | 83.25% | 15.37deg | 7.94deg | 0.557 |
| PPO-GRU | 97.35% | 3.539m | 1.455m | 10.802m | 61.27% | 70.42% | 68.55% | 14.96deg | 7.86deg | 0.557 |
| MPC | 53.25% | 5.941m | 2.904m | 16.357m | 17.46% | 53.80% | 28.65% | 2.69deg | 1.48deg | 0.140 |

分析：

- 中风下 `Proposed` 仍然是精度最好的方法，`Mean Error=2.561m`，`P90=5.755m`，`R@3m=85.12%`，均优于 `PPO` 和 `PPO-GRU`。
- `PPO` 在多高度中风条件下继续保持大误差，`Mean Error=11.141m`，`P50=13.692m`，说明其不是少数长尾造成，而是多数样本都偏离目标。
- `PPO-GRU` 处于中间水平，明显优于 PPO，但 `P90=10.802m`，尾部误差约为 `Proposed` 的 1.88 倍。
- `MPC` 投放率为 `53.25%`，姿态与冲击仍然最好，但 `Overall@3m=28.65%`，整体任务成功率低于学习方法。

### 6.3 强风

结果目录：

`aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/strong_wind`

依据图表：

- [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/strong_wind/figures/precision_cdf.pdf)
- [Roll/Pitch Case](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/strong_wind/figures/roll_pitch_case.pdf)
- [Impact CDF](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/strong_wind/figures/impact_cdf.pdf)
- [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/strong_wind/figures/summary_table.pdf)

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 96.10% | 13.287m | 14.664m | 21.805m | 8.32% | 14.00% | 13.45% | 10.97deg | 6.98deg | 0.501 |
| Proposed | 87.90% | 5.490m | 2.845m | 14.484m | 38.91% | 50.91% | 44.75% | 25.08deg | 12.25deg | 0.757 |
| PPO-GRU | 88.70% | 7.489m | 5.640m | 16.442m | 25.59% | 36.02% | 31.95% | 24.61deg | 13.02deg | 0.758 |
| MPC | 22.10% | 7.053m | 2.745m | 22.489m | 28.28% | 54.07% | 11.95% | 2.90deg | 2.13deg | 0.155 |

分析：

- 强风下所有算法退化，其中 `Proposed` 仍是学习方法中最优：`Mean Error=5.490m`，`P90=14.484m`，`R@3m=50.91%`，`Overall@3m=44.75%`。
- `PPO` 的投放率仍高，但投放质量很差，`Mean Error=13.287m`，`P50=14.664m`，`R@3m=14.00%`。这说明 PPO 倾向于投放，但在强风随机高度下难以选择有效释放时机。
- `PPO-GRU` 的强风表现弱于 `Proposed`，平均误差高出约 `2.000m`，`Overall@3m` 低 `12.80` 个百分点。
- `MPC` 的 dropped-only `R@3m=54.07%` 接近或略高于学习方法，但投放率只有 `22.10%`，`Overall@3m=11.95%`，说明其可行释放窗口过窄。
- 强风下 `Proposed` 和 `PPO-GRU` 的姿态代价明显增大，`roll_pitch_case.pdf` 与姿态尾部统计均显示二者存在更强的极端姿态风险。

### 6.4 投放后误差随风强演化

依据图表：

- [Post-DROP Error Evolution](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/figures/post_drop_error_evolution_wind.pdf)
- [Post-DROP Wind Summary CSV](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/stats/post_drop_wind_summary.csv)

关键统计：

| 方法 | 弱风 Mean / P90 | 中风 Mean / P90 | 强风 Mean / P90 |
|---|---:|---:|---:|
| PPO | 10.465 / 21.405 | 11.141 / 21.601 | 13.287 / 21.805 |
| Proposed | 1.493 / 1.562 | 2.561 / 5.755 | 5.476 / 14.455 |
| PPO-GRU | 1.522 / 2.660 | 3.539 / 10.804 | 7.482 / 16.396 |
| MPC | 2.354 / 2.744 | 5.941 / 16.357 | 7.053 / 22.489 |

分析：

- `post_drop_error_evolution_wind.pdf` 显示，随风强从弱风增加到强风，`Proposed`、`PPO-GRU` 和 `MPC` 的投放后误差均上升，符合风扰增强导致落点误差增大的预期。
- `PPO` 在三个风强下都维持较高误差，弱风 `Mean=10.465m`，中风 `Mean=11.141m`，强风 `Mean=13.287m`。这说明在该实验中，PPO 的主要失效因素是随机高度，而风强进一步加重误差。
- `Proposed` 在三种风强下均优于 `PPO-GRU`，尤其是中风和强风下，P90 分别降低 `5.049m` 和 `1.941m`。
- `MPC` 的曲线需要结合 `n_valid_drop` 解释。强风下 MPC 只有 `442` 个有效投放样本，投放率 `22.10%`，所以其 dropped-only 曲线不能代表整体任务成功能力。

## 7. 实验五：有障碍、多高度、多风强综合复杂场景

结果目录：

`aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp5_complex_multiH_multiW_obs/complex_mix`

依据图表：

- [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp5_complex_multiH_multiW_obs/complex_mix/figures/precision_cdf.pdf)
- [Roll/Pitch Case](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp5_complex_multiH_multiW_obs/complex_mix/figures/roll_pitch_case.pdf)
- [Impact CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp5_complex_multiH_multiW_obs/complex_mix/figures/impact_cdf.pdf)
- [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp5_complex_multiH_multiW_obs/complex_mix/figures/summary_table.pdf)

| 方法 | Drop Rate | Mean Error | P50 | P90 | R@2m | R@3m | Overall@3m | Abs Roll Mean | Abs Pitch Mean | Impact Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 98.85% | 11.497m | 13.538m | 21.933m | 26.35% | 31.66% | 31.30% | 8.46deg | 5.41deg | 0.408 |
| Proposed | 96.50% | 2.764m | 1.067m | 8.646m | 72.33% | 80.88% | 78.05% | 16.31deg | 7.93deg | 0.529 |
| PPO-GRU | 96.65% | 2.948m | 1.186m | 9.417m | 69.27% | 80.19% | 77.50% | 14.72deg | 7.44deg | 0.486 |
| MPC | 47.05% | 4.891m | 2.626m | 14.349m | 20.40% | 71.09% | 33.45% | 2.50deg | 1.51deg | 0.143 |

分析：

- 综合复杂场景是最能体现时序策略优势的实验。`PPO` 在该场景下明显失效，平均误差 `11.497m`，中位误差 `13.538m`，说明多数样本已经偏离目标较远。
- `Proposed` 在复杂场景中表现最好，平均误差 `2.764m`，`R@2m=72.33%`，`R@3m=80.88%`，均高于 `PPO-GRU`。
- `PPO-GRU` 与 `Proposed` 接近，但 `Mean/P90/R@2m/R@3m` 均略差，说明仅使用 GRU 仍不足以完全替代 MLP+GRU 的瞬时特征融合能力。
- `MPC` 的 dropped-only `R@3m=71.09%` 不低，但投放率只有 `47.05%`，整体 `Overall@3m=33.45%`，接近 PPO 但远低于两个循环策略。
- 姿态和冲击方面，`MPC` 仍然最好；学习方法中 `PPO` 姿态角较小，但这是以严重精度退化为代价。

## 8. 方法级总结

### 8.1 PPO

依据图表：

- 实验4弱风的 [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/weak_wind/figures/precision_cdf.pdf)
- 实验4中风的 [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/medium_wind/figures/summary_table.pdf)
- 实验3的 [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp3_randomH_fixedW_fixedObs/main/figures/summary_table.pdf)
- 实验5的 [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp5_complex_multiH_multiW_obs/complex_mix/figures/precision_cdf.pdf)

`PPO` 在固定高度的简单场景中仍有一定竞争力，例如实验一 `R@3m=84.71%`。但一旦引入随机高度，PPO 明显退化：实验三平均误差为 `11.243m`，实验五平均误差为 `11.497m`，新实验四在无障碍条件下也出现弱风 `Mean Error=10.465m`、中风 `Mean Error=11.141m`、强风 `Mean Error=13.287m`。因此 PPO 的主要问题不是障碍物，而是缺少时序记忆后对高度变化和风场变化的释放时机泛化不足。

### 8.2 Proposed

依据图表：

- 实验3的 [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp3_randomH_fixedW_fixedObs/main/figures/precision_cdf.pdf)
- 实验5的 [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp5_complex_multiH_multiW_obs/complex_mix/figures/summary_table.pdf)
- 实验4的 [Post-DROP Error Evolution](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/figures/post_drop_error_evolution_wind.pdf)

`Proposed` 是当前结果中综合精度最好的方法。合并统计中，`Proposed` 的 `Mean Error=2.669m`、`P90=7.669m`、`R@2m=75.63%`、`R@3m=82.97%`，均优于 `PPO` 和 `PPO-GRU`。在新实验四中，`Proposed` 在弱风、中风、强风下的平均误差分别为 `1.493m`、`2.561m`、`5.490m`，均低于 `PPO-GRU` 和 `PPO`，说明 MLP+GRU 对多高度和多风强的释放判断更稳定。

需要注意的是，`Proposed` 的姿态角和冲击并不是最小的，强风下姿态角明显增大。这说明它更倾向于为了保持投放精度而接受一定释放姿态代价。

### 8.3 PPO-GRU

依据图表：

- 实验2随机障碍的 [Precision CDF](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp2_fixedH_fixedW_obs_ablation/random_obstacle/figures/precision_cdf.pdf)
- 实验3的 [Summary Table](../aerial_gym/rl_training/rl_games/runs/result_with_saved_traces/exp3_randomH_fixedW_fixedObs/main/figures/summary_table.pdf)
- 实验4的 [Post-DROP Error Evolution](../aerial_gym/rl_training/rl_games/runs/result_exp4_multiH_multiW_noObs/exp4_multiH_multiW_noObs/figures/post_drop_error_evolution_wind.pdf)

`PPO-GRU` 整体优于 PPO，但不如 `Proposed`。总体平均误差为 `3.275m`，`P90=11.108m`，`R@3m=76.95%`。在实验三和实验五中，`PPO-GRU` 明显优于 PPO，说明循环记忆本身有效；但相比 `Proposed`，其尾部更重。新实验四中，`PPO-GRU` 在中风和强风下的平均误差分别为 `3.539m` 和 `7.489m`，高于 `Proposed` 的 `2.561m` 和 `5.490m`，说明仅使用 GRU 不足以完全替代 MLP 前端的瞬时特征提取。

### 8.4 MPC

依据图表：

- 各实验的 `impact_cdf.pdf`
- 各实验的 `roll_pitch_case.pdf`
- 各实验的 `summary_table.pdf`

`MPC` 的姿态和冲击指标始终最好，总体 `Abs Roll Mean=2.54deg`，`Abs Pitch Mean=1.49deg`，`Impact Mean=0.146`。这说明 MPC 在成功投放样本中具有最平稳的释放姿态。

但 MPC 的主要问题是投放率低和撞毁未投放多。总体投放率只有 `52.69%`，`Crash+NoDrop=7569`，导致 `Overall@3m=33.13%`。因此 MPC 可作为姿态稳定和低冲击基线，但不能作为整体任务成功率最优方法。

## 9. 论文可用结论

基于当前合并结果，可以写成以下结论：

- 在固定高度、固定中风和无障碍条件下，`Proposed` 的尾部误差和命中率优于 PPO，说明 MLP+GRU 对投放时机判断更稳定。
- 在随机障碍条件下，`Proposed` 的精度指标基本不退化，说明其对障碍扰动具有较好鲁棒性。
- 在随机高度条件下，PPO 明显失效，而 `Proposed` 和 `PPO-GRU` 保持较低误差，说明时序建模对高度变化下的释放决策是必要的。
- 在多高度、多风强、无障碍条件下，PPO 在弱风下也明显退化，说明高度变化本身就是主要挑战；`Proposed` 在弱风、中风和强风下均保持学习方法中最好的综合精度。
- 在综合复杂场景中，`Proposed` 取得最佳精度与整体命中率，证明 MLP+GRU 结构在多高度、多风强、有障碍条件下具有最好的综合鲁棒性。
- `MPC` 姿态与冲击最优，但投放率和整体成功率显著低于学习方法，说明传统控制基线在复杂扰动下受限于稳定性和可行释放窗口。

## 10. 后续建议

- 在最终论文图表中，建议同时给出 `Drop-only` 和 `Overall` 两类指标，避免只看已投放样本导致偏乐观。
- 对 `MPC` 的分析必须同时报告 `N(drop)` 和 `Drop Rate`，否则其姿态/冲击优势容易被误解为整体性能优势。
- 对 `Proposed` 与 `PPO-GRU` 的比较，建议重点强调 `P90` 和综合复杂场景结果，因为这最能体现 MLP+GRU 对尾部风险的改善。
- 图注和正文中统一使用 `Proposed` 表示本文方法，避免再出现工程化 checkpoint 名称。
