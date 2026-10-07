# 当前实验与结果回收

用户要求按固定本地验证协议判断优化效果，每项完成后本地 Git commit，不 push。本轮不新增线上比赛提交。

## 已完成

| IMU，固定 fold 0 | CMI 分数 | 来源 |
| --- | ---: | --- |
| 旧原始/分层 CNN 等权融合 | 0.746375 | 已有固定验证 |
| 分组 CNN + masked BatchNorm + SE | 0.773586 | CPU |
| 34 通道运动特征 + Mixup | 0.775116 | CPU |
| 分组 CNN + 双向 GRU | 0.775714 | CPU |
| 分组 CNN + masked BatchNorm + SE | 0.770843 | GPU |
| 相同分组 CNN + Mixup | 0.772690 | GPU |
| 34 通道运动特征 + Mixup | 0.775652 | GPU |

CPU GRU 已完成，不能继续监控旧进程。所有尝试已核对同一 1,627 条验证 sequence、训练 subject、checkpoint 与实际预测分数，并分别本地 commit。证据为 `experiments/results/*_cpu_pilot.json` 和 `*_gpu_pilot.json`。GPU 三项选定一个固定配置 `cnn_dynamics_mixup`，CPU GRU 不混入 GPU 筛选；禁止按每折最高结果拼接。

这些是单折开发 CV，尚不能证明五折或线上提升。63 项项目测试、两项续跑/数据身份专项测试与真实 GPU 产物恢复检查通过。

## 两次基础设施失败

1. [原训练版本 2](https://www.kaggle.com/code/mingweiwei03/cmi-winner-inspired-cnn-training?scriptVersionId=356144886) 已完成三个 pilot，进入五折时因原生 tuple 与 JSON list 的表示差异被误拒绝。已修复为精确 JSON 值比较，统计值变化仍拒绝。原始产物已下载到 `outputs/kaggle_training/recovered_v2/cmi_worktree`，不能再把这个版本当作运行中任务。
2. [续跑版本 1](https://www.kaggle.com/code/mingweiwei03/cmi-winner-cnn-fixed-fold-validation?scriptVersionId=356157899) 在新 Notebook 挂载处复制缓存，被来源身份检查拒绝，未训练新 fold。失败记录为 `experiments/results/cnn_gpu_continuation_v1_failure.json`。已改为保留筛选结果，重建缓存及选定配置的全部模型；没有放宽缓存检查。

## 训练完成与结果回收

[CMI Winner CNN Fixed Fold Validation 版本 2](https://www.kaggle.com/code/mingweiwei03/cmi-winner-cnn-fixed-fold-validation?scriptVersionId=356161432) 已成功完成，不要重新训练。训练使用免费 T4、关闭互联网，是我们的固定 subject-wise CV，不是线上比赛分数。当前仅等待下述 CPU 结果导出任务以回收最终 ZIP。

云端已经通过 `train.csv` 和 `train_demographics.csv` 的字节 SHA256 校验，与 `configs/data_source_hashes.json` 固定的本地输入一致；固定 folds 文件 SHA256 也一致。已恢复原三项筛选结果，开始仅在 fold 0 的 6,524 条训练 sequence 上拟合新缓存。需要重新训练选定配置的 IMU-only 和多传感器各五折，包括重新训练 fold 0。旧筛选阶段 fold 0 产物独立保存于 `outputs/pilot_artifacts/`，不能替代新运行的 fold 0。

已确认新运行越过两次失败点，训练日志报告的已完成 checkpoint 分数如下（下载后仍需核对实际预测）：

| Fold | IMU-only | 多传感器 |
| --- | ---: | ---: |
| 0 | 0.775652 | 0.843238 |
| 1 | 0.785888 | 0.849067 |
| 2 | 0.785455 | 0.841385 |
| 3 | 0.811088 | 0.863590 |
| 4 | 0.777887 | 0.858618 |

十个模型及缺失场景评估已成功完成，训练版本 2 总用时 7,400.8 秒。日志显示 `COMPLETED FIVE FOLD EXPERIMENTS`，并生成 23,952,382 字节的 `winner_experiments.zip`。IMU fold 0 重训与原筛选结果在日志显示精度下一致。模型训练已结束；完整五折预测核验和新旧比较仍未完成，不能用训练日志替代最终检查。

单文件 Download 按钮打开了二进制文件的新标签页，下载事件没有返回。为避免下载原 working directory 的训练缓存，启动了 [仅导出结果 ZIP 的 CPU 任务](https://www.kaggle.com/code/mingweiwei03/cmi-fixed-fold-results-export?scriptVersionId=356184726)：只复制原版本的 `winner_experiments.zip`，不重新训练、不预测测试集、不提交比赛。其输出只有结果 ZIP 及 Notebook 自身文件；可通过 Output actions 的 Download output 回收，再提取内层 ZIP。原训练版本的结果来源 URL 保持 356161432。导出任务完成后还需核对复制的文件大小与 SHA256。

## 完成后的操作

1. 成功后从 Notebook 的 Output 下载 `winner_experiments.zip`。不要下载包含缓存的整个 working directory。部分失败也会保存紧凑 ZIP；失败 ZIP 不可当作完整结果。
2. 导入到 `outputs/kaggle_training/imported/` 下的新目录，勿覆盖 CPU 或旧 GPU 运行。执行 `scripts/import_winner_experiments.py --archive ZIP --output-dir NEW_DIRECTORY --source-url CURRENT_VERSION_URL`，核对输入字节校验值、固定 folds、训练 subject、全部 checkpoint 和实际预测分数。
3. 对导入的 selected run 执行 `scripts/compare_winner_experiments.py --run SELECTED_RUN_DIRECTORY`。核对 8,151 条固定验证 sequence，比较 IMU-only、多传感器及按可用性路由；查看原始输入、全部辅助传感器缺失、固定约 50% 辅助缺失的各折变化、五折均值与样本标准差。
4. 保存紧凑结果与 OOF，更新 `WINNER_INSPIRED.md`、`WINNER_RESULTS.md` 和本文件；每项已完成尝试或失败分别本地 commit，不 push。按完整证据判断是否保留新方案。
5. 如果完整验证支持新方案，执行 `scripts/export_kaggle_submission.py --runs SELECTED_RUN_DIRECTORY --validation-report SCENARIOS_JSON --output-dir NEW_EXPORT_DIRECTORY`，再执行 `scripts/check_kaggle_submission.py --directory NEW_EXPORT_DIRECTORY --oof-path NEW_ROUTED_OBSERVED_OOF` 检查推理包。不要新增比赛提交。
6. 全部工作完成后删除跟进自动化 `cmi`。未变化且无需行动时保持安静；若出现无法解决的训练失败或必须由用户处理的问题，报告具体证据，停止继续训练。

保留旧方案与首次线上成绩，不搜索测试标签或按验证样本定制规则。开发 CV 不等同于新的线上比赛成绩。
