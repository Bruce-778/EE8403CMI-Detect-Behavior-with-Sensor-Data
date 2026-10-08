# 特征表达优化记录

## R0：错误分析与实验协议

上一轮蒸馏的 IMU-only 五折均值 0.789415，比原始 0.787194 提升 0.002221。
原始输入路由分数未变。原始 IMU OOF 的眉毛/睫毛、耳上/颈部等动作混淆较多；
优先测试动作阶段建模与跨受试者特征约束。所有结果是开发 CV，不是线上成绩。

固定 `configs/folds.csv`（8,151 sequences，81 subjects，fingerprint
`51ee93e4d6f817c10c143d203bbb4a4195dcf3b080b096585c230e0bf117c1f8`）。
原始同折 A 权重 warm start，复用 GPU 训练保存的精确预处理与输入缓存；不 refit。
两种方法分别测试，B 冻结；训练输入先验证原始 CSV SHA256。
复用上一轮已经完成且逐项核对参数/源权重/预处理/预测的监督微调作为对照。
新头初始化后恢复 Torch RNG，保证共享 dropout 随机流与对照一致。

共同预算：lr=1e-4、batch=64、最多 8 epochs、early stopping patience=3、seed=42+fold，
原始传感器 dropout，禁用 Mixup；原始 epoch 0 也参与 checkpoint 选择。
先固定 fold 0 筛选；只有超过同预算对照才扩展五折。五折扩展锁定参数，不逐折调参。
记录每折分数、均值和样本标准差、完整 OOF，以及 observed/aux_dropout50/imu_only 固定场景。
本轮保存本地 Git commit，不 push、不提交线上比赛。

R0 已完成：66 项现有与新增检查全部通过，包含两种新方法的实际小数据训练、
checkpoint 恢复和原始推理兼容性。真实 fold 0 输入的两个 CSV SHA256 已匹配，
冻结缓存和旧微调对照复用核验通过，正在进行 R1 实际训练。

## R1：动作阶段感知 CNN（实现与筛选）

方法来自[第二名公开模型的阶段预测与注意力设计](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution/blob/main/model.py)。
本项目实现为原始 IMU 编码器上的小型残差适配，不等同于该队伍完整方案。
训练 behavior 合并为移动/到位/执行三个阶段；逐帧标签按 counter 排序，尾部裁剪、左 padding
与传感器完全一致。训练文件中仅提取当前 fold 的 train 序列；全部 validation phase=-100。
阶段目标通过各卷积主路径的相同 kernel/stride/padding 聚合为软分布，辅助 CE 权重 0.1。
该目标是主路径输入计数；残差捷径的感受野更短，没有宣称它是精确物理分段。
网络从传感器预测阶段概率，结合可学习时间注意力得到三段池化特征；
零初始化适配层把特征加回原始 mean/max 表示，确保 epoch 0 与原始模型一致。
模型 forward 不读取真实 behavior、gesture 或 subject，验证 loader 不含训练阶段标签。

这只是短程 warm-start 筛选。新头只有有限训练时间，失败不能说明从头训练的阶段模型无效。
配置：`configs/representation.json`；入口：`scripts/train_representation.py --method phase`。
检查：阶段标签对齐、padding/空序列、无 oracle 推理、共享随机流、实际训练及 checkpoint 恢复。
实际结果与保留决定将在运行完成后追加，不能将未完成的实验写成提升。

## R2：跨 subject 监督对比学习（实现与筛选）

参考[SupCon 原论文](https://arxiv.org/abs/2004.11362)，针对 subject-wise 验证改为：
同 gesture 且不同 subject 为正对，不同 gesture 为负对；同 subject 同类与自身不进入分母。
原始 mean/max embedding 做 L2 normalization，不增加推理头；temperature=0.1、辅助权重 0.05。
无正对的 anchor 跳过，记录每轮具有跨 subject 正对的 anchor 比例；subject 仅用于训练 loss。
保持相同随机 batch、预算和对照，不改变采样器，也不与阶段 loss 或蒸馏同时启用。
这是一项需要验证的 SupCon 改写，论文并未证明它在本比赛上的收益。
入口：`scripts/train_representation.py --method cross_subject_supcon`。
检查：正负对定义、梯度、无正对 batch、训练/验证隔离、实际训练和原始推理兼容性。
实际结果与保留决定将在运行完成后追加。
