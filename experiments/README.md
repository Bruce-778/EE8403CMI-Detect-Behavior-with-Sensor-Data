# 主实验记录

仅研究 Model A（IMU-only 1D CNN）与 Model B（IMU + THM + ToF，多分支 1D CNN）。已有树模型 baseline 保持原样。

所有实验复用 `configs/folds.csv`，subject 不跨 training / validation。先在固定 fold 0 筛选少量方案，再锁定配置补齐五折；fold 0 筛选和 checkpoint 选择都使用 validation，因此分数是开发 CV，不能当作独立测试集表现。单折结果只与相同折比较。每项完成后本地 Git 提交，不 push。

## 初始观察

v1 fold 0：A = 0.722778，B = 0.811524。A 最佳 epoch 28，之后训练 loss 从约 0.91 下降到 0.72，validation loss 反而上升到 1.60，提示过拟合。B 的 binary F1 已达 0.97935，9 类 macro F1 只有 0.64370；目标手势中眉毛、睫毛、颈部类别最难。应优先改善目标手势区分，而不是继续优化已经很高的 binary F1。

## 资料与采用理由

- [CMI 第一名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution)：独立传感器分支、CNN、较小学习率、模态遮挡等。首先测试 learning rate 5e-4 / dropout 0.3 / 最多 70 epochs / patience 15，padding、归一化、长度和输入特征保持一致。
- [CMI 第三名](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/3rd-place-solution)：报告过 18 类、9 类与二分类联合损失，以及关注动作时段的 pooling。考虑把 18 类 logits 聚合为官方 9 类和二分类，增加相应监督；无需额外标注。
- [Attentive Statistics Pooling](https://arxiv.org/abs/1803.10963)：借鉴学习时间权重的方式，测试带 mask 的 attention pooling，减少无关时段稀释动作特征。这是把已有方法适配到本项目的实验假设，尚不能宣称创新性或必然提升。

不使用 subject 身份作为模型输入，不读取 test 标签，不依据排名方案删掉困难验证样本。所有验证 sequence 保留，传感器缺失仍由 mask 表示。每次改动、失败与提升均记录；最终配置按完整五折而非单折最高分判断。

## 缺失模态诊断

同一 fold 0、同一批 1,627 条验证序列：v1 B 原始输入 0.811524；完全遮掉 THM + ToF 后 0.703605；按 sequence_id 的固定 hash 对约半数序列额外遮掉这两种模态后 0.752229。A 三种情况下都为 0.722778，确认 A 不依赖辅助模态。

这个结果提示不能只选全传感器分数最高的 B；需要保留独立 A，并比较在辅助模态全缺时路由到 A。`aux_dropout50` 只是一致、标签无关的压力测试，原有自然缺失仍保留，不等同于隐藏测试集分布。
