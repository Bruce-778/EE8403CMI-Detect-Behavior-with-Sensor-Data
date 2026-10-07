# 主实验记录

仅研究 Model A（IMU-only 1D CNN）与 Model B（IMU + THM + ToF，多分支 1D CNN）。已有树模型 baseline 保持原样。

所有实验复用 `configs/folds.csv`，subject 不跨 training / validation。先在固定 fold 0 筛选少量方案，再锁定配置补齐五折；fold 0 筛选和 checkpoint 选择都使用 validation，因此分数是开发 CV，不能当作独立测试集表现。单折结果只与相同折比较。每项完成后本地 Git 提交，不 push。

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
