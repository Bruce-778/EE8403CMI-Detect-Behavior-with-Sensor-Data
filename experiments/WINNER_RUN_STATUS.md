# 当前实验与结果回收

用户要求按固定本地验证协议判断优化效果，每项完成后本地 Git commit，不 push。该验证阶段没有新增线上比赛提交；其后用户授权的一次线上评分已完成，Public 0.839556、Private 0.833362，详见 `KAGGLE.md` 和 `KAGGLE_WINNER_STATUS.md`。

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

上述 pilot 是单折开发 CV，用于选择固定配置。后续完整五折提升已在下文核验，线上提升尚未验证。63 项项目测试、两项续跑/数据身份专项测试与真实 GPU 产物恢复检查通过。

## 两次基础设施失败

1. [原训练版本 2](https://www.kaggle.com/code/mingweiwei03/cmi-winner-inspired-cnn-training?scriptVersionId=356144886) 已完成三个 pilot，进入五折时因原生 tuple 与 JSON list 的表示差异被误拒绝。已修复为精确 JSON 值比较，统计值变化仍拒绝。原始产物已下载到 `outputs/kaggle_training/recovered_v2/cmi_worktree`，不能再把这个版本当作运行中任务。
2. [续跑版本 1](https://www.kaggle.com/code/mingweiwei03/cmi-winner-cnn-fixed-fold-validation?scriptVersionId=356157899) 在新 Notebook 挂载处复制缓存，被来源身份检查拒绝，未训练新 fold。失败记录为 `experiments/results/cnn_gpu_continuation_v1_failure.json`。已改为保留筛选结果，重建缓存及选定配置的全部模型；没有放宽缓存检查。

## 训练完成与结果回收

[CMI Winner CNN Fixed Fold Validation 版本 2](https://www.kaggle.com/code/mingweiwei03/cmi-winner-cnn-fixed-fold-validation?scriptVersionId=356161432) 已成功完成，不要重新训练。训练使用免费 T4、关闭互联网，是我们的固定 subject-wise CV，不是线上比赛分数。用户已手动下载并解压结果，十个模型和完整 OOF 已在本机核验；新方案在全部五折和固定缺失场景中均提升，保留 `cnn_dynamics_mixup`。两个导出会话均已取消，没有运行中的训练或导出任务。

云端通过了 `train.csv` 和 `train_demographics.csv` 的字节 SHA256 校验，与 `configs/data_source_hashes.json` 固定的本地输入一致；固定 folds 文件 SHA256 也一致。原三项筛选结果保留，选定配置的 IMU-only 和多传感器各五折已从头训练，缓存仅在各折训练 sequence 上拟合（fold 0 为 6,524 条）。旧筛选阶段 fold 0 产物独立保存于 `outputs/pilot_artifacts/`，未替代新运行的 fold 0。

新运行越过两次失败点，以下十个 checkpoint 的日志分数已由本机实际保存预测重算核对：

| Fold | IMU-only | 多传感器 |
| --- | ---: | ---: |
| 0 | 0.775652 | 0.843238 |
| 1 | 0.785888 | 0.849067 |
| 2 | 0.785455 | 0.841385 |
| 3 | 0.811088 | 0.863590 |
| 4 | 0.777887 | 0.858618 |

十个模型及缺失场景评估已成功完成，训练版本 2 总用时 7,400.8 秒。日志显示 `COMPLETED FIVE FOLD EXPERIMENTS`，并生成 23,952,382 字节的 `winner_experiments.zip`。本机已核对每个 checkpoint 的训练 sequence / subject、固定 folds 和实际预测分数；以下最终分数由全部 8,151 条 OOF 重算，证据见 `WINNER_RESULTS.md` 与 `experiments/results/cnn_winner_comparison.json`。

单文件 Download 按钮打开了二进制文件的新标签页，下载事件没有返回。为避免下载原 working directory 的训练缓存，启动了 [仅导出结果 ZIP 的 CPU 任务版本 1](https://www.kaggle.com/code/mingweiwei03/cmi-fixed-fold-results-export?scriptVersionId=356184726)。输入页可见完整结果 ZIP，但运行约 30 分钟仍没有日志、输出为 0 B，已取消；失败记录为 `experiments/results/cnn_results_export_cpu_attempt.json`，已本地 commit。

[CMI Fixed Fold Results Export 版本 2](https://www.kaggle.com/code/mingweiwei03/cmi-fixed-fold-results-export?scriptVersionId=356189192) 改用 T4 环境执行同一份文件复制代码，互联网关闭。它同样在约 29 分钟后仍无执行日志、输出为 0 B，已取消。Active Events 已确认两个版本均为 Cancelled、0 Active Events。失败记录为 `experiments/results/cnn_results_export_t4_attempt.json`。不再启动新的导出或训练，不要重复等待这两个版本。

用户从原训练版本手动下载并解压到 `outputs/kaggle_training/recovery_inbox/winner_experiments`。原下载文件曾在 Downloads 中观测到 23,952,382 字节；随后已解压，原 ZIP 不在目录中。为复用导入检查，只将用户提供的 153 个文件重新打包为 `winner_experiments_repacked.zip`，导入全新目录 `outputs/kaggle_training/imported/winner_selected_v2`。保存了全部解压文件的 SHA256 清单及恢复记录。重新打包后的哈希不等于原 ZIP 的哈希；云端 ZIP SHA256 尚不可用，不能宣称二者已核对一致。训练输入的字节校验值已重新在本机计算，与固定参数和云端记录完全一致。结果来源 URL 保持 356161432。

## 最终验证与收尾

1. 导入检查通过：十个 checkpoint、训练 sequence / subject、81 个 subject 的固定五折与实际保存预测分数一致。IMU-only 与多传感器的完整五折证据分别本地 commit。
2. 对新旧两个方案的 18 个模型 / 缺失场景组合逐一重算官方指标，检查每个场景完整覆盖 8,151 条 sequence 一次、概率 argmax 与分类标签一致。固定额外缺失样本、每折 A/B 路由数量一致，辅助模态缺失不改变 IMU 分支概率。
3. 原始输入路由：`0.823642 ± 0.016330 → 0.852029 ± 0.008719`；约半数额外辅助缺失：`0.788400 ± 0.013442 → 0.821455 ± 0.011675`；全部辅助缺失：`0.748366 ± 0.016025 → 0.787194 ± 0.014102`。均为五折均值与样本标准差，每项均在 5/5 折提升，因此保留新配置；没有按折挑模型或调整权重。
4. OOF 与每折评价保存在 `outputs/winner_comparison/{reference,candidate}/`；原始权重和产物保存在新导入目录。紧凑检查为 `experiments/results/cnn_winner_import_verified.json`，完整比较为 `cnn_winner_comparison.json`。
5. 新方案已导出到独立的 `outputs/kaggle_winner_submission_v2`，共十个冻结权重，ZIP 为 8,694,116 字节。每折一条真实 held-out sequence 的推理概率与 GPU OOF 相符，最大绝对误差 `3.56e-7`；修改输入标签不影响预测。两个公开无标签示例及全部辅助模态缺失输入正常，ZIP CRC / 文件内容、打包源码和 Notebook 语法检查通过。证据为 `experiments/results/cnn_winner_inference_verified.json`。没有重新对全部 8,151 条原始序列运行 checkpoint 推理，也未执行线上 gateway / 隐藏测试。
6. 跟进自动化 `cmi` 已删除，本轮结果回收、固定五折比较和本机推理检查完成。旧推理包和原线上成绩保留；无 push、无新比赛提交，不需要重启训练。

保留旧方案与首次线上成绩，不搜索测试标签或按验证样本定制规则。开发 CV 不等同于新的线上比赛成绩。
