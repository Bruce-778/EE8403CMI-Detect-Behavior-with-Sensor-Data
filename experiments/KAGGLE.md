# Kaggle 线上提交记录

日期：2026-10-07（America/Los_Angeles）。两次冻结 CNN 方案均完成 Late Submission，Kaggle 返回真实 Public / Private 分数。

## 最新：运动特征与 Mixup 的固定五折 CNN

| 方案 | Public | Private |
| --- | ---: | ---: |
| 原始 / 分层 CNN 固定融合（旧方案） | 0.806408 | 0.796378 |
| `cnn_dynamics_mixup`（新方案） | **0.839556** | **0.833362** |
| 新方案变化 | **+0.033148** | **+0.036984** |

[官方评分页](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/submissions) 对 [CMI Winner Inspired CNN Inference v1](https://www.kaggle.com/code/mingweiwei03/cmi-winner-inspired-cnn-inference?scriptVersionId=356204022) 返回 `Succeeded (after deadline)`。新 [权重数据集](https://www.kaggle.com/datasets/mingweiwei03/cmi-winner-cnn-weights-20261007) 和 Notebook 保持私有；旧版本不变。公开示例 Notebook 用时 38.2 秒，随后 Kaggle 用隐藏测试重跑评分；上述数值来自最终 Private / Public 列，不是 Notebook 描述中的本地 CV。

新方案有十个 checkpoint：IMU-only / 多传感器各五折，固定输入可用性路由和五折等权概率。使用与旧方案相同的 subject-wise folds 和官方指标。本地开发 CV 为 0.852029 ± 0.008719，固定约半数辅助模态额外缺失为 0.821455 ± 0.011675；没有根据隐藏测试拟合、选择模型或调整权重。

按同一份[官方最终 Private 榜单](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/leaderboard)估算：排除 Rank 0 的 benchmark 后，2,657 支队伍中有 124 支高于新 Private 分数，插入位置约第 **125 名**，比旧方案约第 1,363 名明显提升。这是约在前 4.7% 的成绩参考，Late Submission 没有正式最终名次。冠军分数 0.886193，当前差距 0.052831；新方案已提升，但距离前排最高分仍有差距。

完整线上记录为 `experiments/results/kaggle_winner_cnn.json`，上传与评分过程为 `KAGGLE_WINNER_STATUS.md`，实际分数截图为 `outputs/kaggle_winner_submission_v2/online_scores_verified.png`。成绩和榜单来源校验值均保留；成功结果本地 Git commit，未 push。线上跟进自动化 `cmi-cnn` 已删除。

## 首次提交：保留原始结果

| 评价 | 官方线上分数 |
| --- | ---: |
| Public Score | **0.806408** |
| Private Score | **0.796378** |

与 2,657 个参赛队伍的最终 Private 分数比较，有 1,362 支队伍高于本次成绩，约在第 **1,363 名附近**；这是插入位置估算，Late Submission 没有正式最终名次。分数略低于队伍中位数 0.798205，与冠军 0.886193 相差 0.089815，当前处于排行榜中间水平，距离前排还有明显差距。[官方最终排行榜](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/leaderboard)

Notebook：[CMI Selected CNN Five-Fold Ensemble](https://www.kaggle.com/code/mingweiwei03/cmi-selected-cnn-five-fold-ensemble)。权重数据集：[CMI Selected CNN Weights 20261007](https://www.kaggle.com/datasets/mingweiwei03/cmi-selected-cnn-weights-20261007)。二者保持私有。

固定推理规则：A、B 各自使用原始和分层损失 CNN 等权概率融合；每折根据 THM / ToF 的可用性路由，五个 fold 的输出概率等权平均。20 个 checkpoint 来自已完成的本地实验，test 不参与标准化拟合、checkpoint 选择或权重调整。

本地核对：每个 fold 取一条 held-out 原始 sequence，导出推理概率与最终路由 OOF 的最大误差为 2.48e-7；忽略输入中人为加入的 gesture 标签。两个公开 test 样例及辅助模态全缺失的输入都能返回合法 gesture。54 项自动检查通过。公开 test 没有标签，这些检查不产生线上分数。

| 版本 | 运行情况 | 说明 |
| --- | --- | --- |
| 1 | 失败，未提交评分 | 权重成功加载；当前 Kaggle 数据根目录变为 `/kaggle/input/competitions/...`，旧路径找不到 test.csv。 |
| 2 | Notebook 运行成功，隐藏测试评分成功 | 自动定位官方 `cmi_inference_server.py` 所在比赛目录。模型、预处理参数与融合权重保持一致。 |

Kaggle 对第 2 版（`scriptVersionId=356132739`）返回 `Succeeded (after deadline)`，隐藏测试运行约 18 分钟。提交文件是官方网关产生的 `submission.parquet`。[评分结果](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/submissions)。第一版错误没有用于本次评分。

本地原始可用性验证分数为 `0.823642 ± 0.016330`；固定约半数辅助模态额外遮挡的验证压力测试为 `0.788400 ± 0.013442`。官方 test 包含 IMU-only 和全部传感器的混合输入，所以原始本地 CV 与线上分数的输入分布不同，不能把前者当成预期线上分数。压力测试也不是隐藏测试集的复刻；线上效果应以本页实际得分为准。

官方最终 Private 排行榜已下载，排除 `perfect_submission.csv` benchmark 后有 2,657 个队伍。第一名分数为 0.886193，第三名 0.878261，第 100 名 0.834622，第 500 名 0.820201，队伍分数中位数 0.798205。比较使用同一 Private 指标；Late Submission 不改变已结束比赛的正式名次。

可追踪的小型结果摘要：`experiments/results/kaggle_selected_cnn.json`。完整排行榜 CSV、线上结果截图与原始导出包保存在 `outputs/kaggle_submission/`；修复后的 Notebook 文件保存在 `outputs/kaggle_submission_v2/cmi-cnn-selected.ipynb`。本轮代码、修复和结果均本地 Git 提交，未 push。
