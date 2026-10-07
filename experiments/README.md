# 主实验记录

仅研究 Model A（IMU-only 1D CNN）与 Model B（IMU + THM + ToF，多分支 1D CNN）。已有树模型 baseline 保持原样。

所有实验复用 `configs/folds.csv`，subject 不跨 training / validation。先在固定 fold 0 筛选少量方案，再锁定配置补齐五折；fold 0 筛选和 checkpoint 选择都使用 validation，因此分数是开发 CV，不能当作独立测试集表现。单折结果只与相同折比较。每项完成后本地 Git 提交，不 push。

## 本轮结论

已完成六项尝试，并为原始 CNN、分层损失 CNN 及两者的固定融合补齐 A/B 五折。下方历史记录保留当时的观察和决策；完整比较见 [实验汇总](RESULTS.md)。

| 配置 | A：IMU-only，五折均值 ± 标准差 | B：多传感器，五折均值 ± 标准差 |
| --- | ---: | ---: |
| 原始 CNN | 0.730773 ± 0.010601 | 0.816032 ± 0.010804 |
| 分层损失 CNN | 0.742452 ± 0.019493 | 0.817443 ± 0.021038 |
| 固定 50% / 50% 概率融合 | **0.748366 ± 0.016025** | **0.822268 ± 0.017113** |

相对原始配置，融合 A 提升 0.017593，B 提升 0.006237。B 的单模型分层损失收益较小，且部分折下降；没有把各折最高配置拼成结果，也没有搜索融合权重。当前 A/B 各由两个 CNN 组成，不是单个网络的成绩。全部 8,151 条 sequence 各有一次 held-out 预测，标准差采用 ddof=1。

当前使用固定可用性路由：THM 或 ToF 至少一种可用时使用融合 B，两者都不可用时使用融合 A。原始验证输入的路由分数为 **0.823642 ± 0.016330**，pooled OOF **0.823679**；完全遮掉 THM、ToF 时为 **0.748366 ± 0.016025**。约半数序列额外失去辅助模态时，单独融合 B 为 **0.767681 ± 0.015817**，路由为 **0.788400 ± 0.013442**。这些遮挡是标签无关、可重现的压力测试，不是隐藏测试集分布或真实测试成绩。

验证：54 项现有自动检查通过；额外核对最终 9 份场景 OOF，各覆盖同一批 8,151 条 sequence，ID 不重复，subject、fold、标签与固定索引一致，概率有限且和为 1，重新计算的官方分数与记录一致。实际加载模型的概率融合与已保存的融合 OOF 一致；遮挡辅助模态不改变 A 的预测。五折 training / validation subject 无交集。

Attention pooling、5/9/13 时间 kernel、B 恢复原始训练参数均未通过同一 fold 0 的筛选，没有继续投入五折。通过已有论文启发进行适配和验证，但本轮结果不足以宣称新的方法创新。归一化保持训练 fold 拟合，长度继续使用训练 fold 的 95% 分位数（五折分别 127/128/124/125/130），batch=64；没有做完整的长度和 batch size 网格搜索。ToF 当前采用 2×2 区域聚合输入 1D CNN，8×8 的 3D CNN 不在本轮实现中。

## 最终方案复现

```powershell
# 已完成的对应 fold 会进行配置、数据、预测一致性检查后复用。
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --config configs/cnn_v1.json --fold 0 1 2 3 4 --resume
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --config configs/cnn_hierarchical.json --fold 0 1 2 3 4 --resume
& 'D:\anaconda\envs\cmi\python.exe' -s scripts/ensemble_cnn_oof.py --runs outputs/experiments/cnn_v1 outputs/experiments/cnn_hierarchical --model imu --name cnn_equal_blend
& 'D:\anaconda\envs\cmi\python.exe' -s scripts/ensemble_cnn_oof.py --runs outputs/experiments/cnn_v1 outputs/experiments/cnn_hierarchical --model multisensor --name cnn_equal_blend
# 实际加载两个成员模型进行推理，核对融合和缺失模态路由。
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/evaluate_cnn_scenarios.py outputs/experiments/cnn_hierarchical --blend-imu-with outputs/experiments/cnn_v1 --blend-multisensor-with outputs/experiments/cnn_v1 --name cnn_final_selected
& 'D:\anaconda\envs\cmi\python.exe' -s scripts/compare_main_experiments.py --runs cnn_v1 cnn_tuned cnn_hierarchical cnn_attention cnn_large_kernel cnn_multisensor_hier_v1train
```

原始及分层损失的 checkpoint 与各折输入预处理参数位于 `outputs/experiments/<配置>/<模型>/fold_<N>/`。融合完整 OOF 位于 `outputs/experiments/cnn_equal_blend/<模型>/evaluation/`；最终路由及三种输入场景的完整 OOF 位于 `outputs/experiments/cnn_hierarchical/scenarios/cnn_final_selected/<模型或routed>/<场景>/`。`experiments/results/cnn_final_selected_scenarios.json` 是对应的小型结果摘要。

## 初始观察

v1 fold 0：A = 0.722778，B = 0.811524。A 最佳 epoch 28，之后训练 loss 从约 0.91 下降到 0.72，validation loss 反而上升到 1.60，提示过拟合。B 的 binary F1 已达 0.97935，9 类 macro F1 只有 0.64370；目标手势中眉毛、睫毛、颈部类别最难。应优先改善目标手势区分，而不是继续优化已经很高的 binary F1。

## 资料与采用理由

- [CMI 第一名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution)：独立传感器分支、CNN、较小学习率、模态遮挡等。首先测试 learning rate 5e-4 / dropout 0.3 / 最多 70 epochs / patience 15，padding、归一化、长度和输入特征保持一致。
- [CMI 第三名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/3rd-place-solution)：报告过 18 类、9 类与二分类联合损失，以及关注动作时段的 pooling。考虑把 18 类 logits 聚合为官方 9 类和二分类，增加相应监督；无需额外标注。
- [Attentive Statistics Pooling](https://arxiv.org/abs/1803.10963)：借鉴学习时间权重的方式，测试带 mask 的 attention pooling，减少无关时段稀释动作特征。这是把已有方法适配到本项目的实验假设，尚不能宣称创新性或必然提升。
- [ModernTCN 官方实现](https://github.com/luodhhh/ModernTCN)与 [ConvTimeNet 论文](https://arxiv.org/abs/2403.01493)：较新的纯卷积时序方法强调扩大时间感受野。提供逐层 kernel 配置，拟测试 5 / 9 / 13 的 CNN；这是受这些工作启发的简化实验，并非复现它们的完整网络。

不使用 subject 身份作为模型输入，不读取 test 标签，不依据排名方案删掉困难验证样本。所有验证 sequence 保留，传感器缺失仍由 mask 表示。每次改动、失败与提升均记录；最终配置按完整五折而非单折最高分判断。

## 缺失模态诊断

同一 fold 0、同一批 1,627 条验证序列：v1 B 原始输入 0.811524；完全遮掉 THM + ToF 后 0.703605；按 sequence_id 的固定 hash 对约半数序列额外遮掉这两种模态后 0.752229。A 三种情况下都为 0.722778，确认 A 不依赖辅助模态。

这个结果提示不能只选全传感器分数最高的 B；需要保留独立 A，并比较在辅助模态全缺时路由到 A。`aux_dropout50` 只是一致、标签无关的压力测试，原有自然缺失仍保留，不等同于隐藏测试集分布。

v1 固定可用性路由（只看 THM / ToF 是否至少一种存在）在 `aux_dropout50` 上为 **0.763129**，比单独 B 的 0.752229 提高约 0.0109；全辅助缺失时恢复 A 的 0.722778。路由不调分类阈值、不使用 gesture 真值，不对不同 sequence 事后挑选分数较高的模型。

## 重现与续跑

```powershell
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --config configs/cnn_tuned.json
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --config configs/cnn_hierarchical.json
# 只复用配置、fold、数据和预测检查全部通过的已完成折，补齐五折。
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --config configs/cnn_hierarchical.json --fold 0 1 2 3 4 --resume
& 'D:\anaconda\envs\cmi\python.exe' -s scripts/summarize_cnn_experiment.py outputs/experiments/cnn_hierarchical --name cnn_hierarchical
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/evaluate_cnn_scenarios.py outputs/experiments/cnn_hierarchical --name cnn_hierarchical
```

`--resume` 复用完整的 fold，不恢复中断 epoch；不匹配的参数或不完整 fold 会报错，避免把不同实验混成一份结果。配置和小型结果摘要进入 Git，checkpoint、缓存、完整 OOF、曲线继续留在本地 `outputs/`。

## 尝试 1：学习率和正则化

`cnn_tuned` 已完成同一 fold 0：A **0.726348**（v1 0.722778）；B **0.809248**（v1 0.811524）。A 最佳 epoch 45 / 60，B 最佳 epoch 51 / 66。较小学习率加较强 dropout 没有同时改善两个模型，不能据此替换原始 B。下一步在这组固定训练参数上单独比较分层损失，随后比较 attention pooling。

输入检查：训练长度中位数 59，95% 分位数 127；有效 IMU 读数约 0.04% 达到截断边界。fold 0 validation 没有自然整段辅助模态缺失，而 training 有 96 条，因此需要其他折及固定遮挡压力测试确认泛化。

## 尝试 2：官方分组的联合损失

在尝试 1 的参数上，增加 `0.5 × CE9 + 0.1 × BCE2`（由同一 18 类 logits 用 logsumexp 聚合）。fold 0：A **0.742987**，B **0.817910**，最佳 epoch 分别为 61、67 / 70。相比尝试 1，两模型均提升；A 相比 v1 提升约 0.0202，B 提升约 0.0064。这是单折候选，尚未证明完整五折都有效。

Attention pooling 单独增加在该配置上，A 暂为 0.730267，低于 0.742987；说明“更复杂”没有保证提升，保留失败结果。另一个候选只把逐层时间 kernel 从 5/5/5 改为 5/9/13，以检验较长上下文；不同时启用 attention。

该配置 A 的 neck pinch / neck scratch F1 相比 v1 分别提高约 0.110 / 0.097，但眉毛 F1 降约 0.028；B 的 eyelash F1 提高约 0.069，forehead scratch 降约 0.034。改进是整体指标收益，不是每类均提升。遮挡压力测试中固定路由为 0.782336（v1 0.763129）。

## Model A 配置锁定

四个候选使用同一 fold 0：原始 0.722778，调参 0.726348，分层损失 **0.742987**，分层损失 + attention 0.730267，分层损失 + 5/9/13 kernels 0.731901。因此 A 锁定 `cnn_hierarchical`，补齐五折；后续不按各折结果选择不同配置或随机种子。B 的候选选择独立进行。

## 尝试 3：可学习的时间 pooling

在尝试 2 上仅用 attention 加权时间均值替代普通均值，仍保留 max pooling。fold 0 A **0.730267**，B **0.809325**，都低于尝试 2 的 0.742987 / 0.817910。B 第 58 轮 early stop，最佳第 43 轮；本轮不进入最终配置。padding、整段缺失分支、NaN 污染不影响输出及有限梯度的检查均通过。不同 objective 的 loss 数值不可直接横向比较，决策使用相同官方分数。

## 尝试 5：固定等权概率融合

参考冠军的多模型组合，但不搜索权重。v1 与分层损失 CNN 的 18 类概率固定各 50%，按 sequence_id 对齐。fold 0：A **0.746375**（单独分层 A 0.742987），B **0.817826**（单独分层 B 0.817910），因此暂只有 A 显示融合收益。该权重在运行其他折之前固定；待源模型五折都齐全后再计算完整 OOF，不因某一折结果改权重。

## 尝试 4：较大时间感受野

在尝试 2 上仅将逐层 kernel 改为 5/9/13。fold 0 A **0.731901**，B **0.811549**，均低于尝试 2；参数量增加到 A 183,442 / B 289,554，暂没有带来相应收益。B 也锁定 `cnn_hierarchical`，其余四折不再改参数。

为并行训练 A/B 且避免共享报告文件的写入冲突，B 续跑放在 `outputs/experiments/cnn_hierarchical_b5`，复用检查通过的原始 B fold 0。所有输入、folds、参数相同；完成后将完整结果汇总到主记录。续跑命令为：

```powershell
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --config configs/cnn_hierarchical.json --model multisensor --fold 0 1 2 3 4 --output-dir outputs/experiments/cnn_hierarchical_b5 --resume
```

## 尝试 6：B 使用独立训练参数

完整训练继续推进后，A 在 fold 1 为 0.727996（v1 0.728407，基本持平），fold 2 为 0.763863（v1 0.739067），fold 3 为 0.759007（v1 0.744238）。B 分层损失 + 强正则化配置在 fold 1 却为 **0.791769**（v1 **0.807482**），主要是 9 类 macro F1 下降，binary F1 基本相同。这提示 B 不应仅因 fold 0 提升而沿用 A 的参数。

新增独立候选 `cnn_multisensor_hier_v1train`：保留分层损失，恢复 B 原始 lr=1e-3 / dropout=0.2 / patience=10 / scheduler patience=3，最多 70 epochs。先检查固定 fold 0，再决定是否投入完整五折；此前已锁定的五折实验照常完成，不在其中某一折单独换参数。这个新增假设参考了 fold 1，所以最终比较仍属于开发 CV。

```powershell
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --config configs/cnn_multisensor_hier_v1train.json --model multisensor
```

该候选 fold 0 最终为 **0.807612**，低于 v1 0.811524 与分层损失 + 强正则化 0.817910；本轮不进入五折，保留结果。没有再搜索随机种子或验证阈值。

## A 的完整五折

分层损失 A：fold 0–4 为 **0.742987 / 0.727996 / 0.763863 / 0.759007 / 0.718408**，均值 **0.742452 ± 0.019493**，pooled OOF **0.742757**。全部 8,151 条 sequence 各验证一次，完整 OOF 概率保存于 `outputs/experiments/cnn_hierarchical/imu/evaluation/oof_predictions.csv`。原始 A 及固定等权融合在源模型全部五折完成后比较，不能拿这张表与单折筛选分数直接比较。
