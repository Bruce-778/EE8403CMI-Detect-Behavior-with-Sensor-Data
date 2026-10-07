# 前排方案启发的 CNN 优化

起点为首次线上提交：Public 0.806408、Private 0.796378。旧方案的五折 IMU-only 融合为 0.748366 ± 0.016025，原始输入的 A/B 路由为 0.823642 ± 0.016330；额外遮掉约半数序列的 THM+ToF 时为 0.788400 ± 0.013442。

## 参考与取舍

- [冠军公开说明](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution)与[训练代码](https://github.com/statist-bhfz/kaggle_cmi_1st_place_solution)：参考特征分组、运动变化特征和 Mixup 的设计方向。实际查看 `public_solution_ogurtsov/IMU_CNN_cross_attention_exp_267.ipynb`；仅用于阅读，没有执行或复制其训练实现。
- [亚军说明](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/2nd-place-solution)：参考独立 IMU 分支及缺失模态适配。
- [季军说明](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/3rd-place-solution)：其团队将模型本身与采集规则后处理的分数分别报告。我们也将区分通用模型收益和比赛专用规则收益。

本轮保留原有训练 fold 标准化、6D 旋转和严格缺失 mask。所有验证 subject 和 sequence 均保留，不因装置佩戴问题或 sequence 长度删除验证样本；不使用测试伪标签、测试统计或按 subject 的类别数量约束。没有声称完整复现冠军，更没有承诺冠军分数。

## 三项固定尝试

1. `cnn_grouped_se`：加速度、6D 旋转、角速度、去重力加速度各有独立 CNN stem，再融合。残差块使用只统计有效时间 token 的 BatchNorm 和基于有效时间 pooling 的 SE 通道门控。
2. `cnn_grouped_mixup`：在相同网络上增加训练 batch Mixup，alpha=0.4、概率=0.5。读数先按 mask 置零，再插值；有效时间和传感器 mask 同步组合；两个硬标签分别计算分层损失并按同一权重加权。Validation 不做 Mixup。
3. `cnn_dynamics_mixup`：增加到 34 个 IMU 特征，包括幅值、加速度/去重力加速度的后向差分、差分幅值、幅值变化和 9 帧局部轴间相关性。所有物理特征在标准化之前、单个 sequence 内生成；差分不跨 counter gap。旋转缺失不影响加速度独有的衍生特征。新增通道也只在训练 fold 拟合标准化。

共同设置：seed=42、batch=64、lr=0.001、weight decay=0.0003、dropout=0.2、最多 80 epochs、early stopping patience=16、CE18+0.5 CE9+0.1 BCE2。长度沿用各训练 fold 的 95% 分位数，ToF 沿用 2×2 有效区域聚合。

先在固定 fold 0 对比三项 IMU 方案，选择一个完整配置，随后将该配置用于 A/B 全部五折。禁止按每折最高结果拼接实验。fold 0 参与配置选择、各 validation fold 参与 checkpoint 选择，因此结果为开发 CV。CPU 与 GPU 训练可能有数值差异；来源和环境随结果保存。

## 复现

本地：`D:\anaconda\envs\cmi\python.exe -s -u scripts/train_cnn.py --config configs/cnn_grouped_se.json --model imu --fold 0`。

`scripts/export_kaggle_training.py` 将本项目代码和固定 `folds.csv` / metadata 打包进离线 Notebook。挂载官方比赛数据、开启免费 T4 GPU 后，`scripts/run_winner_experiments.py` 执行三项 pilot、统一配置五折 A/B、OOF 与缺失场景检查，最后导出 `winner_experiments.zip`。

训练 Notebook：[CMI Winner Inspired CNN Training](https://www.kaggle.com/code/mingweiwei03/cmi-winner-inspired-cnn-training)。结果返回后记录每次尝试及固定五折比较；新提交成绩与首次线上成绩分别保留。本轮不会 push。

## 已完成的本地结果

| IMU-only，固定 fold 0 | CMI 分数 | 最佳 epoch | 范围 |
| --- | ---: | ---: | --- |
| 旧原始/分层 CNN 的固定等权融合 | 0.746375 | — | 同一验证样本 |
| 分组 CNN + SE + masked BatchNorm | **0.773586** | 29 / 45 | CPU 单折 pilot |

第一项本地尝试提升 0.027211；二值 F1 与 9 类 macro F1 按官方方式计算。它仍为单折开发结果，不能当作五折或线上提升。60 项检查通过，包含 padding 对 BatchNorm 统计的隔离、空分支、恢复模型、Mixup 的 mask 以及衍生特征的缺失语义。完整证据见 `experiments/results/cnn_grouped_se_cpu_pilot.json`。
