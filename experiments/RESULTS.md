# 主实验结果

所有数值使用相同 subject folds 和官方 CMI 指标。单折筛选与 early stopping 使用 validation，结果属于开发 CV，不能代替独立测试集成绩。

## 当前采用的方案

A 与 B 各自融合原始 CNN 和分层损失 CNN 的预测概率，权重固定为 50% / 50%。THM、ToF 至少一种可用时使用 B，两者都不可用时使用 A。路由只看输入可用性。

| 模型 | 原始五折均值 | 当前五折均值 ± 标准差 | 均值提升 |
| --- | ---: | ---: | ---: |
| A：IMU-only | 0.73077 | 0.74837 ± 0.01602 | +0.01759 |
| B：多传感器 | 0.81603 | 0.82227 ± 0.01711 | +0.00624 |

固定路由在原始验证输入上为 **0.82364 ± 0.01633**；约半数序列额外失去 THM、ToF 时为 **0.78840 ± 0.01344**。

本轮完成六项尝试。Attention pooling、较大时间 kernel、B 的独立训练参数没有通过固定 fold 0 筛选，保留结果。分层损失 B 的单模型五折收益较小，当前提升主要来自固定概率融合。

## 固定 fold 0 筛选

| 配置 | Model A：IMU | Model B：多传感器 |
| --- | ---: | ---: |
| cnn_v1 | 0.722778 | 0.811524 |
| cnn_tuned | 0.726348 | 0.809248 |
| cnn_hierarchical | 0.742987 | 0.817910 |
| cnn_attention | 0.730267 | 0.809325 |
| cnn_large_kernel | 0.731901 | 0.811549 |
| cnn_multisensor_hier_v1train | 未完成 | 0.807612 |

## 完整五折

只有五折全部完成的配置才进入这张表。标准差使用 ddof=1。

| 配置 | 模型 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 均值 ± 标准差 | Pooled OOF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnn_v1 | imu | 0.72278 | 0.72841 | 0.73907 | 0.74424 | 0.71938 | 0.73077 ± 0.01060 | 0.73099 |
| cnn_v1 | multisensor | 0.81152 | 0.80748 | 0.80803 | 0.83318 | 0.81994 | 0.81603 ± 0.01080 | 0.81629 |
| cnn_hierarchical | imu | 0.74299 | 0.72800 | 0.76386 | 0.75901 | 0.71841 | 0.74245 ± 0.01949 | 0.74276 |
| cnn_hierarchical | multisensor | 0.81791 | 0.79177 | 0.80249 | 0.84387 | 0.83118 | 0.81744 ± 0.02104 | 0.81774 |

## 固定等权概率融合

两个来源均为对应 fold 的 held-out 预测，各 50%；不逐折调权。

| 模型 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 均值 ± 标准差 | Pooled OOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| imu | 0.74637 | 0.74179 | 0.76683 | 0.76060 | 0.72624 | 0.74837 ± 0.01602 | 0.74849 |
| multisensor | 0.81783 | 0.80380 | 0.81190 | 0.84754 | 0.83028 | 0.82227 ± 0.01711 | 0.82234 |

## 最终方案的模态缺失压力测试

同一份验证 sequence；额外遮挡由 sequence_id hash 决定，与标签无关。这是验证压力测试，不是隐藏测试集成绩。

| 输入 | 模型 | 五折均值 ± 标准差 | Pooled OOF |
| --- | --- | ---: | ---: |
| observed | imu | 0.74837 ± 0.01602 | 0.74849 |
| imu_only | imu | 0.74837 ± 0.01602 | 0.74849 |
| aux_dropout50 | imu | 0.74837 ± 0.01602 | 0.74849 |
| observed | multisensor | 0.82227 ± 0.01711 | 0.82234 |
| imu_only | multisensor | 0.70789 ± 0.01855 | 0.70830 |
| aux_dropout50 | multisensor | 0.76768 ± 0.01582 | 0.76748 |
| observed | routed | 0.82364 ± 0.01633 | 0.82368 |
| imu_only | routed | 0.74837 ± 0.01602 | 0.74849 |
| aux_dropout50 | routed | 0.78840 ± 0.01344 | 0.78829 |

## 参数

- **cnn_v1**：lr=0.001，batch=64，dropout=0.2，epochs≤45，patience=10，macro loss weight=0.0，binary loss weight=0.0，pooling=mean_max，stage kernels=[5, 5, 5]。
- **cnn_tuned**：lr=0.0005，batch=64，dropout=0.3，epochs≤70，patience=15，macro loss weight=0，binary loss weight=0，pooling=mean_max，stage kernels=[5, 5, 5]。
- **cnn_hierarchical**：lr=0.0005，batch=64，dropout=0.3，epochs≤70，patience=15，macro loss weight=0.5，binary loss weight=0.1，pooling=mean_max，stage kernels=[5, 5, 5]。
- **cnn_attention**：lr=0.0005，batch=64，dropout=0.3，epochs≤70，patience=15，macro loss weight=0.5，binary loss weight=0.1，pooling=attention_max，stage kernels=[5, 5, 5]。
- **cnn_large_kernel**：lr=0.0005，batch=64，dropout=0.3，epochs≤70，patience=15，macro loss weight=0.5，binary loss weight=0.1，pooling=mean_max，stage kernels=[5, 9, 13]。
- **cnn_multisensor_hier_v1train**：lr=0.001，batch=64，dropout=0.2，epochs≤70，patience=10，macro loss weight=0.5，binary loss weight=0.1，pooling=mean_max，stage kernels=[5, 5, 5]。

## 输出与复现

配置位于 `configs/cnn_*.json`；完整 checkpoint、OOF、曲线在 `outputs/experiments/<配置名称>/`。`experiments/results/` 保存可追踪的小型结果摘要。命令和资料来源见 [实验记录](README.md)。真实线上分数与最终排行榜比较见 [Kaggle 提交记录](KAGGLE.md)。
