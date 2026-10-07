# Kaggle 线上提交记录

日期：2026-10-07。目标是对已锁定的 CNN 方案进行 Late Submission，获取真实 Public / Private 分数。

Notebook：[CMI Selected CNN Five-Fold Ensemble](https://www.kaggle.com/code/mingweiwei03/cmi-selected-cnn-five-fold-ensemble)。权重数据集：[CMI Selected CNN Weights 20261007](https://www.kaggle.com/datasets/mingweiwei03/cmi-selected-cnn-weights-20261007)。二者保持私有。

固定推理规则：A、B 各自使用原始和分层损失 CNN 等权概率融合；每折根据 THM / ToF 的可用性路由，五个 fold 的输出概率等权平均。20 个 checkpoint 来自已完成的本地实验，test 不参与标准化拟合、checkpoint 选择或权重调整。

本地核对：每个 fold 取一条 held-out 原始 sequence，导出推理概率与最终路由 OOF 的最大误差为 2.48e-7；忽略输入中人为加入的 gesture 标签。两个公开 test 样例及辅助模态全缺失的输入都能返回合法 gesture。54 项自动检查通过。公开 test 没有标签，这些检查不产生线上分数。

| 版本 | 运行情况 | 说明 |
| --- | --- | --- |
| 1 | 失败，未提交评分 | 权重成功加载；当前 Kaggle 数据根目录变为 `/kaggle/input/competitions/...`，旧路径找不到 test.csv。 |
| 2 | Notebook 运行成功，已提交隐藏测试评分 | 自动定位官方 `cmi_inference_server.py` 所在比赛目录。模型、预处理参数与融合权重保持一致。 |

Kaggle 已接收第 2 版，`scriptVersionId=356132739`，状态为 `Notebook Running (after deadline)`。提交文件是官方网关产生的 `submission.parquet`。[提交状态](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/submissions)。尚未获取真实线上分数。本地 `0.823642 ± 0.016330` 是开发 CV，不能填写为 Kaggle 分数。

官方最终 Private 排行榜已下载，排除 `perfect_submission.csv` benchmark 后有 2,657 个队伍。第一名分数为 0.886193，第三名 0.878261，第 100 名 0.834622，第 500 名 0.820201，队伍分数中位数 0.798205。来源：[最终排行榜](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/leaderboard)。实际模型得分出来后再按同一 Private 指标比较；Late Submission 不改变已结束比赛的正式名次。
