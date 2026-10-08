# 第一轮后训练：多传感器向 IMU 蒸馏

当前最佳上线方案保持 `cnn_dynamics_mixup`，Public 0.839556、Private 0.833362。
本轮新增可重复的后训练实验，先在固定 fold 0 筛选，不把单折结果当作五折或线上提升。

## 训练设计

每折分别加载该折的 IMU-only 学生和多传感器老师。检查两者的固定 folds 指纹、训练 sequence/subject、类别顺序、物理预处理与输入参数一致。
只使用完整训练 checkpoint；推理包删除了训练 IDs，不能作为缺少 provenance 的替代输入。
老师设为 `eval()` 并冻结梯度、BatchNorm 统计；仅为该训练 fold 的序列生成 logits，不生成 validation 的蒸馏目标。
学生保留训练时的传感器 dropout；老师看到原始可用的多传感器输入。辅助传感器全缺失的训练序列不贡献蒸馏损失。

固定两组对照：

1. `finetune`：原学生 checkpoint，小学习率监督微调。
2. `distill`：相同起点、训练 IDs、预处理、随机种子、数据顺序、增强与预算，增加老师的蒸馏损失。

共同监督目标为原来的 `CE18 + 0.5 CE9 + 0.1 BCE2`，label smoothing 为 0.03。
蒸馏目标为 `KL(teacher || student)`，softmax 温度 T=2，并乘 T²；总损失为 `监督损失 + 0.5 × 蒸馏损失`。
KL 按样本计算后，用辅助模态可用 mask 排除不适用的老师目标，再按整个 batch 平均。
两组都关闭 Mixup；lr=1e-4、batch=64、weight decay=3e-4、最多 8 epochs、early stopping patience=3。
源模型为已核验的 GPU checkpoint，本轮使用 CPU；先核对整个 held-out fold 的起点概率与原 OOF。
每个训练 fold 使用老师保存的精确标准化参数和长度，直接从原始 CSV 构建冻结缓存，不重新拟合。

第 0 轮原始 checkpoint 也参与选择。最佳分数没有超过起点时，保留第 0 轮；仅因为继续训练导致的变化不能归因于蒸馏，需比较 `distill` 和 `finetune`。
验证只使用真实标签监督评价，不使用老师作为验证目标。每轮报告统一官方 CMI score、binary F1、九类 macro F1。

## 运行入口

```powershell
conda activate cmi
python -s -u scripts/train_posttraining.py
```

配置为 `configs/posttraining.json`，默认使用新目录 `outputs/posttraining/pilot_v1/`，不覆盖原始权重、原始 OOF 或以前实验。
老师软标签保存于 `teacher_targets/fold_0/training_targets.npz`，只包含训练序列。
两组权重、history、预处理参数和逐序列验证预测分别保存于 `finetune/imu/fold_0/`、`distill/imu/fold_0/`。
紧凑记录保存于 `experiments/results/posttraining_pilot_v1_{finetune,distill,comparison}.json`。
额外使用原模型 B 的已核验预测，对相同原始输入、固定辅助模态缺失和全部辅助缺失场景计算路由分数。
这只替换 IMU 分支；B 的权重与预测保持一致。原始输入中不使用 A 的样本，其路由分数自然不会随 A 的优化变化。

单折筛选支持蒸馏后，锁定参数补齐五折时使用**新的输出目录和结果名**：

```powershell
python -s -u scripts/train_posttraining.py --fold 0 1 2 3 4 --device cuda --output-dir outputs/posttraining/fivefold_v1 --name posttraining_fivefold_v1
```

该命令要求实际可用的 CUDA 环境与完整源 checkpoint，不自动启动 Kaggle，也不会创建比赛提交。
五折完整时，统一评价脚本输出逐折分数、均值、样本标准差和 8,151 条完整 OOF；单折运行不输出五折均值。

报告阶段意外中断后，可以在配置完全一致时使用 `--resume`。已完成组会校验源老师/学生哈希、冻结输入、训练设置、checkpoint 指标与预测覆盖后复用；未完成训练不会被覆盖或当作成功结果。

## 本地 fold 0 实际结果（2026-10-07）

固定 6,524 条训练序列、1,627 条 validation 序列；只有 6,428 条辅助传感器可用的训练序列贡献蒸馏损失。
CPU 起点与原 GPU OOF 的全部 validation 概率最大差异为 1.674e-6，所有类别预测保持一致。

监督微调对照已完成：三轮 validation score 分别为 0.770324、0.771206、0.773992，均低于原始 IMU checkpoint 的 0.775652，early stopping 后保留 epoch 0。证据为 `results/posttraining_pilot_v1_finetune.json` 与对应输出目录的逐轮 history、逐序列预测。

蒸馏组已完成 7 epochs，第 4 轮最佳。两组采用相同最大训练预算与停止规则，实际运行轮数由各自验证曲线决定。

| 固定 fold 0 场景 | 原始方案 / 监督微调保留起点 | 蒸馏 | 变化 |
| --- | ---: | ---: | ---: |
| 全部辅助模态缺失，IMU-only | 0.775652 | 0.778765 | +0.003112 |
| 固定约 50% 序列移除辅助模态，再按可用性路由 | 0.812717 | 0.816301 | +0.003584 |
| 原始输入，全部 1,627 条验证序列使用未改变的 B | 0.843238 | 0.843238 | 0 |

IMU 的九类 macro F1 从 0.566538 提升到 0.575176，binary F1 从 0.984767 降到 0.982353。当前只能说明 fold 0 有小幅收益，不能推断五折或线上收益。锁定本轮参数后补齐其他 folds，不用新的 fold 结果继续调参数。

已重新计算两组保存预测的官方指标、核对 1,627 条完整验证覆盖和类别 argmax、确认 6,524 条老师目标与 validation 完全隔离，并核对老师及学生源权重 SHA256 均未变化。完整测试 61 项通过，包含 KL 梯度、冻结老师、同折隔离、固定预处理及路由报告检查。紧凑蒸馏与比较证据为 `results/posttraining_pilot_v1_distill.json`、`results/posttraining_pilot_v1_comparison.json`。

第一次训练在对照训练和预测保存完成后，路由报告调用遗漏了强制 fold 指纹检查参数而中断；已修复并增加报告回归检查。恢复时校验并复用完整对照组，没有重新训练或修改其权重、预测。

## 五折监督微调对照

fold 0 完成后锁定参数，以相同 CPU 环境复用该折完整结果，补齐 folds 1–4。输入缓存均从该折原 checkpoint 的精确预处理参数生成，不重新拟合。

监督微调五折 IMU 分数为 0.775652、0.786080、0.785455、0.811088、0.781670；均值 0.787989、样本标准差 0.013563。原始 IMU 均值 0.787194、样本标准差 0.014102，平均增量 +0.000795。
固定半数辅助模态缺失时，微调路由均值 0.821164，较原始 0.821455 下降 0.000292；原始输入路由均值保持 0.852029。不能仅根据 IMU 单分支的微小均值提升替换当前方案。

紧凑证据保存于 `results/posttraining_fivefold_v1_finetune.json`；完整 8,151 条 IMU OOF、每折分数和均值/样本标准差保存于 `outputs/posttraining/fivefold_v1/finetune/imu/evaluation/`，场景 OOF 位于该实验的 `scenarios/finetune/`。

## 实验边界

源设计和 checkpoint 曾使用 validation 选择；后训练也由同一 validation early stopping，所以这是开发 CV。
老师/学生都不使用其他折的模型，不把见过当前 validation subject 的四个老师混入蒸馏。
每个 completed attempt 保存紧凑结果并本地 Git commit；本轮不自动 push，也不更新已上线方案。
本轮先评估蒸馏，不同时改变模型结构或引入强化学习，方便判断收益来源。

方法参考：[知识蒸馏原论文](https://arxiv.org/abs/1503.02531)、[PyTorch 官方蒸馏教程](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)。实现复用本项目的 mask、物理特征、固定 folds 和官方评价。
