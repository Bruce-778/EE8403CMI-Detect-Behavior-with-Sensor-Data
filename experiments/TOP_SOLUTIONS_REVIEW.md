# 前五名方案复核与下一轮实验方向

本次复核日期：2026-10-07。使用参赛者自己的 writeup 与公开代码；没有
将他们不同 folds、样本处理或 ensemble 的 CV 当作我们的可比成绩。
以下公开分数是作者报告值，不是本项目复现实验。

## 前排的设计与当前差异

| 排名与一手来源 | 公开方案要点 | 对本项目的启发 |
| --- | --- | --- |
| [第一名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution)，[代码](https://github.com/statist-bhfz/kaggle_cmi_1st_place_solution) | 物理特征分组、不同 CNN/循环/attention 模型与时间增强，融合后做 subject 历史约束；部分 ToF 3D 模型作为互补成员 | 我们已经有分组、运动衍生特征与 Mixup；缺少经过完整验证的时间增强和互补模型集成 |
| [第二名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/2nd-place-solution)，[代码](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution) | ToF 每帧 8×8 Conv2d 后接时间 CNN；阶段预测、分阶段 attention 和阶段对齐 Mixup；区分旋转/辅助模态可用性；联合标签与因果历史分配 | 完整 ToF 空间、充分训练的阶段方法、orientation 辅助监督是更大的结构差异 |
| [第三名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/3rd-place-solution) | 多种 CNN/Transformer，部分模型将时间×特征矩阵当图像；阶段/orientation 辅助任务、物理一致增强、异构 ensemble 与 subject 后处理 | 多样性要通过 OOF 错误互补证明；时间×特征 2D CNN 与 ToF 空间 CNN 是两种不同设计 |
| [第四名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/4th-place-solution) | 独立传感器 stem、CNN/attention/BERT、按阶段裁切与同类阶段 Mixup、联合标签历史约束 | 需要保留动作阶段而不是随意裁切；联合标签训练必须保证验证真值不进入推理 |
| [第五名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/5th-solution) | sequence 模型之外，使用同一 subject 的历史 embedding Transformer；短历史时更多依赖独立 sequence 模型 | 应把历史模型列为独立输入协议；历史不足时要有回退 |

第二名报告，无伪标签时 Private 从 0.854 提到后处理后的 0.875；第三名报告
ensemble 从 0.864 提到后处理后的 0.878。这提示历史信息值得单独测试，
也说明我们 0.833362 的 Private 与前排无后处理模型仍有差距。上述差值
不是我们可以直接获得的增量。[第二名来源](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/2nd-place-solution)，[第三名来源](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/3rd-place-solution)。

第二名公开 [model.py](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution/blob/main/model.py)
实际使用预测阶段概率加权的三个 attention pooling，并把每个 ToF sensor
的逐帧 Conv2d 特征送入时间 CNN。
[train.py](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution/blob/main/train.py)
默认训练 50 epoch，Adam、lr=1e-3、batch=32、10% warmup 与 cosine schedule。
其划分分层变量是 handedness；我们保持原来的 gesture 分层、subject 分组
五折，不替换 folds。我们的 3 epoch warm-start 阶段筛选，不能等价评价该
训练方法。未运行他们的整套训练或声称精确复现其分数。

## 从自己的 OOF 找主要瓶颈

以下是原权重各分支完整 8,151 条 OOF 拼接后的指标，**不是五折均值**：

| 分支 | binary F1 | 九类 macro F1 | CMI score |
| --- | ---: | ---: | ---: |
| IMU-only | 0.980323 | 0.595143 | 0.787733 |
| 多传感器 | 0.986557 | 0.716139 | 0.851348 |

IMU 已能较好区分 target/non-target，困难在具体目标动作。眉毛拔毛与睫毛
拔毛互相误判 127/113 条；颈部捏皮与耳上拔毛互相误判 108/98 条。多传感器
仍有眉毛/睫毛 113/109 条互相误判。逐类 F1 与混淆计数已保存到
[ablation_study.json](results/ablation_study.json)，不凭整体分数猜测瓶颈。

官方说明隐藏测试约半数只有 IMU，因此单独改善 ToF 不能覆盖全部测试
输入。[官方数据说明](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data)。
当前原始输入 CV 0.852029 与固定半数辅助缺失 CV 0.821455 的差异，说明
缺失场景必须作为主要观察项；后者仍不是隐藏测试的精确复刻或线上分数。

## 本次已经完成的 H1 可行性检验

没有重训：原模型、原 folds、原 OOF 完全不变。仅增加 train-fold 学得的
每类 gesture 容量，用当前 subject 已到达的概率做因果 Hungarian 分配。
不用真实 orientation/phase，也不看未来行。三个预先固定顺序全部保留。

原始输入三顺序平均 +0.004673，固定半数辅助缺失 +0.003803，IMU-only
+0.004525。全部是开发 OOF 结果。容量只基于 gesture，弱于前排预测的
gesture×orientation×initial-behavior 联合标签约束；不能声称复现他们的
0.02 增量。实时顺序、可见历史长度和采集结构影响有效性。保留为评价
原型，暂未接入线上推理。[方法与全部结果](HISTORY_POSTPROCESSING.md)。

## 下一轮按单项验证，而不是同时改一堆参数

1. **先补 IMU 表示学习的充分训练对照。** 新初始化、相同训练预算/种子/
   scaler/长度/数据增强，先跑原 grouped CNN 对照，再加预测阶段 pooling。
   同预算用完整训练计划，而不是成熟模型上 3 个 epoch。阶段真值只用于
   训练辅助 CE。先隔离 pooling 的效果，再单独加阶段 CE，最后才测阶段
   对齐 Mixup；主要观察 IMU-only 和固定 aux_dropout50。
2. **独立测试完整 ToF 空间。** 先将原 2×2 区域统计替换为每 sensor 的
   8×8 distance + validity map 逐帧 Conv2d/空间 pooling，再接现有时间 CNN。
   将 3D CNN 作为后续独立对照。保持其他分支和训练计划相同；无效像素、
   时间 padding 必须从统计和 pooling 中排除，不把填零当有效距离。
3. **物理一致时间增强。** 在原始 IMU 时间序列做 shift/stretch，再计算
   rotation、omega、linear acceleration 和动态特征，最后使用该训练
   fold 保存的 scaler。不能在标准化缓存里分别扰动这些物理相关通道。
   quaternion 插值及导数时间尺度要同步处理，单独对比无增强组。
4. **orientation 辅助任务与异构融合。** orientation 只能是训练标签，
   推理用预测值；先隔离辅助监督效应，再评估联合标签/history。新增 CNN+GRU
   或时间×特征 2D 模型须有相同 folds 的完整 OOF，并证明与当前模型的
   错误互补后再融合。融合超参数选择应标为开发选择，不能伪装独立验证。

这些是按现有证据排列的实验设计，不是已经获得的分数。本次没有启动新
长训练、修改已上线权重或新增线上提交。最终保留需要完整五折和缺失场景
核对，重要候选再做额外随机种子；所有尝试包括失败均记录并单独 commit。

## 不直接复制的做法

部分公开 pipeline 删除验证中的佩戴异常/短动作 sequence，或使用不同
划分与采集协议；这些会改变评价样本，我们保留全部固定验证 sequence。
不把缺失旋转插值后直接标为真实有效，不使用验证真值或未来历史来做
后处理，也不通过反复线上提交挑选幸运输入顺序。历史策略只在采集/调用
协议支持时适用，应与独立 sequence 识别结果分开报告。
