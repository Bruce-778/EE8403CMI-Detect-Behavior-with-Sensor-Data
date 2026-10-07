# 新 CNN 方案线上评分

用户在 2026-10-07 授权将已核验的新方案提交 Kaggle 查看真实线上成绩。此前“不新增比赛提交”的约束针对上一轮本地验证；本次只提交这一份冻结的新方案，不 push。

- 私有权重数据集：[CMI Winner CNN Weights 20261007](https://www.kaggle.com/datasets/mingweiwei03/cmi-winner-cnn-weights-20261007)，版本 1。上传的是 `outputs/kaggle_winner_submission_v2/cmi-cnn-selected.zip`，Kaggle 自动解压为十个权重、源码与 bundle.json；旧数据集保留。
- 私有推理 Notebook：[CMI Winner Inspired CNN Inference v1](https://www.kaggle.com/code/mingweiwei03/cmi-winner-inspired-cnn-inference?scriptVersionId=356204022)。挂载官方 CMI 比赛输入及上述权重，互联网关闭，CPU 推理。38.2 秒运行成功，官方网关生成 `submission.parquet`（两个公开示例的文件为 998 字节）。
- [比赛提交页](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/submissions) 已确认接收新版本，状态为 `Notebook Running (after deadline)`。Kaggle 正用隐藏测试重新运行，当前 Public / Private 分数为空。不要将公开示例运行成功或本地 CV 写成新的线上分数。
- 原方案保留：Public 0.806408、Private 0.796378，原推理版本 356132739。新结果返回后分别比较两个指标，并用最终 Private 排行榜估算插入位置；Late Submission 没有正式最终名次。

冻结方案为 `cnn_dynamics_mixup`：IMU-only / 多传感器各五折，输入可用性路由，五折等权概率。没有根据测试集拟合标准化、选择 checkpoint 或调整融合权重。本地开发 CV 为 0.852029 ± 0.008719，不等于预期线上分数。

紧凑、可追踪记录：`experiments/results/kaggle_winner_cnn.json`。待评分结束后更新本文件、`KAGGLE.md` 与该 JSON，分别保留成功或失败证据并本地 Git commit；没有原因不要重复提交。

已启用同线程安静跟进自动化 `cmi-cnn`，每 5 分钟检查这一次提交。状态未变化时不重复通知；实际分数或失败返回后保存结果、通知用户并删除自动化。该自动化不新增提交、不训练模型、不 push。
