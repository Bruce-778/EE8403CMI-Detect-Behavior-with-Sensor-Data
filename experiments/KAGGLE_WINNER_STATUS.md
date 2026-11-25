> 实验档案说明（2026-10-07 清理）：本页保留当时的结果、决策与命令。当前运行方案为 `cnn_dynamics_mixup`，Public 0.839556 / Private 0.833362；请以[当前项目入口](../README.md)为准。旧配置原样存于 `experiments/configs/`，旧运行代码可从 Git 历史恢复。

# 新 CNN 方案线上评分

用户在 2026-10-07 授权将已核验的新方案提交 Kaggle 查看真实线上成绩。此前“不新增比赛提交”的约束针对上一轮本地验证；本次只提交这一份冻结的新方案，不 push。

- 私有权重数据集：[CMI Winner CNN Weights 20261007](https://www.kaggle.com/datasets/mingweiwei03/cmi-winner-cnn-weights-20261007)，版本 1。上传的是 `outputs/kaggle_winner_submission_v2/cmi-cnn-selected.zip`，Kaggle 自动解压为十个权重、源码与 bundle.json；旧数据集保留。
- 私有推理 Notebook：[CMI Winner Inspired CNN Inference v1](https://www.kaggle.com/code/mingweiwei03/cmi-winner-inspired-cnn-inference?scriptVersionId=356204022)。挂载官方 CMI 比赛输入及上述权重，互联网关闭，CPU 推理。38.2 秒运行成功，官方网关生成 `submission.parquet`（两个公开示例的文件为 998 字节）。
- [比赛提交页](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/submissions) 已完成隐藏测试评分，状态为 `Succeeded (after deadline)`。已核对新 Notebook 版本 356204022 与 Private / Public 列：**Public 0.839556、Private 0.833362**。这是官方线上成绩，独立于公开示例和本地开发 CV。
- 原方案保留：Public 0.806408、Private 0.796378，原推理版本 356132739。新 Public 提升 **0.033148**，Private 提升 **0.036984**。
- 同一份最终 Private 排行榜排除 Rank 0 benchmark 后有 2,657 支队伍；124 支队伍高于 0.833362、没有同分队伍，插入位置约为 **第 125 名**，旧方案约第 1,363 名。Late Submission 没有正式最终名次。冠军为 0.886193，当前差距为 0.052831，不能声称达到冠军水平。

冻结方案为 `cnn_dynamics_mixup`：IMU-only / 多传感器各五折，输入可用性路由，五折等权概率。没有根据测试集拟合标准化、选择 checkpoint 或调整融合权重。本地开发 CV 为 0.852029 ± 0.008719，不等于预期线上分数。

紧凑、可追踪记录：`experiments/results/kaggle_winner_cnn.json`。新提交的成功状态与两项分数截图为 `outputs/kaggle_winner_submission_v2/online_scores_verified.png`；同时包含新旧提交、版本和评分列的可见页面文字证据为 `online_scores_verified.txt`。已更新本文件与 `KAGGLE.md`，旧结果保留；本次只新增这一份提交，没有修改模型或按线上分数调参。

同线程跟进自动化 `cmi-cnn` 已在实际评分返回后删除。成功结果本地 Git commit，不 push；本轮线上评分跟进完成。
