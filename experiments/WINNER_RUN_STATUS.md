# 当前实验与结果回收

用户要求按固定本地验证协议判断优化效果，每项完成后本地 Git commit，不 push。本轮不新增线上比赛提交。

## 已完成

- CPU fold 0 旧 IMU 等权融合：0.746375。
- CPU fold 0 分组 CNN + masked BatchNorm + SE：0.773586，最佳 epoch 29，运行 45 epochs。
- CPU fold 0 34 通道运动特征 + Mixup：0.775116，最佳 epoch 39，运行 55 epochs。
- CPU fold 0 分组 CNN + 双向 GRU：0.775714，最佳 epoch 25，运行 41 epochs；已核对固定 1,627 条验证 sequence 与 OOF。
- 两项 CPU 结果已分别提交；结果证据在 `experiments/results/*_cpu_pilot.json`。
- 63 项测试通过。GRU 代码、打包、导入校验与五折比较工具已经准备好。

## 运行中

CPU 的 `cnn_grouped_gru` 已完成，只做固定 fold 0 对照，输出在 `outputs/experiments/cnn_grouped_gru`。

[GPU 训练版本 2](https://www.kaggle.com/code/mingweiwei03/cmi-winner-inspired-cnn-training?scriptVersionId=356144886) 在 Kaggle 免费 T4 上计算我们的固定验证 folds，不是线上测试分数。版本 1 已取消，勿监控版本 1。

GPU 任务先比较 `cnn_grouped_se`、`cnn_grouped_mixup`、`cnn_dynamics_mixup` 的 fold 0，选一个完整配置，用于 A/B 的全部五折。首项 GPU pilot 已结束：0.770843。CPU/GPU 的训练数值存在差异，分别保留来源；不把两者按每折最高分拼接。

## 完成后

1. CPU GRU 完成后，用 `scripts/summarize_cnn_experiment.py` 保存 `cnn_grouped_gru_cpu_pilot`，更新研究记录并 commit；明确它只有单折。
2. GPU 成功后，从 Notebook 的 Output 下载单个 `winner_experiments.zip`。不要下载包含缓存的全部 working directory。当前版本尚未生成最终五折结果。
3. 导入到 `outputs/kaggle_training/imported/` 下的全新子目录，勿覆盖 CPU 同名运行。用 `scripts/import_winner_experiments.py --archive ZIP --output-dir NEW_DIRECTORY --source-url VERSION_2_URL` 核对各 checkpoint 的训练 subject、fold 指纹、验证覆盖与实际预测分数。
4. 对导入的 selected run 执行 `scripts/compare_winner_experiments.py --run SELECTED_RUN_DIRECTORY`，重新计算 8,151 条完全相同 sequence 的五折比较。查看原始输入、全部辅助传感器缺失、固定约 50% 辅助缺失的分数及每折变化。
5. 将 GPU 三个 pilot 及 selected 的紧凑结果放入 `experiments/results/`，区分 CPU/GPU 来源；每项已完成尝试分别 commit。保存 OOF 于 outputs，更新 `WINNER_INSPIRED.md`、`WINNER_RESULTS.md` 和本文件。
6. 只在完整验证证据支持时保留新方案。若新方案未提升，明确记录失败。若需要准备新方案的推理包，使用 `scripts/export_kaggle_submission.py --runs SELECTED_RUN_DIRECTORY --validation-report SCENARIOS_JSON --output-dir NEW_EXPORT_DIRECTORY`，再用 `scripts/check_kaggle_submission.py --directory NEW_EXPORT_DIRECTORY --oof-path NEW_ROUTED_OBSERVED_OOF` 核对原始输入预测。

保留旧方案和首次线上成绩，不搜索测试标签或每折权重。后续判断使用五折均值、标准差和固定缺失压力测试；开发 CV 不等同于新的线上比赛成绩。
