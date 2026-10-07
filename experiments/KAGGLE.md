# Kaggle 线上提交记录

日期：2026-10-07。已对锁定的 CNN 方案完成 Late Submission，Kaggle 返回真实 Public / Private 分数。

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
