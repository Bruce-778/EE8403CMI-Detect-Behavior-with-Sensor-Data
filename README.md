# CMI：第二名 base + IMU 辅助损失

当前主实验是第二名公开 base 方法，加上我们的官方指标分组辅助损失。
主配置：[configs/main_experiment.json](configs/main_experiment.json)。
用户已选择这一方案作为后续研究基础；旧 `cnn_dynamics_mixup` 代码移入实验档案。

本方案已经完成固定五折、20 个分支模型与 8,151 条完整 OOF 的严格核验：
十个 IMU/旋转缺失 IMU 分支各训练 50 epoch、取最后权重；十个 ToF 分支严格复用
同 fold、同 trainIDs、同联合类别轴的原始 base 权重。辅助损失只在 IMU 分支测试。

| 固定五折开发验证 | 第二名原 base | 当前混合主实验 |
|---|---:|---:|
| 原始输入 | 0.891362 ± 0.012919 | **0.891716 ± 0.012698** |
| 固定约半数辅助模态缺失 | 0.861310 ± 0.013242 | **0.861624 ± 0.011197** |
| IMU-only | 0.831799 ± 0.013723 | **0.832369 ± 0.012404** |

数值是五折均值 ± 样本标准差（ddof=1）。提升很小，历史后处理结果有升有降，
不将主实验选择解释为稳健普遍改善。没有该混合方案的线上 Private 成绩。
原方案线上 Private **0.833362** 属于保留的旧冻结权重，不是本方案线上成绩。

## 主代码与入口

```text
configs/main_experiment.json       当前主实验、审计记录与权重来源
scripts/main_experiment.py         默认状态/核验、显式训练、离线 Notebook 导出
scripts/train_second_place_hybrid.py  五折 IMU 两分支、50 epoch 辅助损失训练
scripts/evaluate_second_place_hybrid.py  新 IMU + 原 ToF 严格导入、合并与固定验证
src/cmi_project/second_place.py    物理特征、训练缓存、train-only 联合类别及安全加载
src/cmi_project/second_place_hybrid.py  joint CE + official9 CE + binary BCE
src/cmi_project/vendor/second_place/  固定上游源码，原作者署名和 SHA 保留
experiments/legacy/cnn_dynamics_mixup/  旧方案原始代码、配置、测试和项目说明
experiments/results/              每一步成功/失败与逐折比较的历史记录
```

上游固定 commit `cb53f8d0cc82f33b86403ba35c5a580930475528`，
[原作者仓库](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution)。
保留作者模型结构、原特征与 phase CE；加入 `0.5 × official9 soft CE + 0.1 × binary BCE`，
分组 target 来自训练 joint Mixup soft targets。IMU 的 joint CE 和 phase CE 权重均为 1。
保留 Mixup0.5、Adam lr0.001/wd0.0001、batch32、clip1、10% warmup/cosine、
200 帧尾部截断/右 padding、seed42+fold、固定 50 epoch 最后权重。

## 环境与使用

使用现有 Conda `cmi` 环境。本机 Python 为 `D:\anaconda\envs\cmi\python.exe`，
直接调用加 `-s`。所有实验复用 `configs/folds.csv`，训练/验证 subjects 不重叠；
预处理与联合类别拟合只使用训练 fold。原始数据放在 `data/`。

```powershell
# 默认只显示已核验主实验，不启动训练。
python -s scripts/main_experiment.py
# 检查所有主模型复制文件及 15 份完整 OOF 的原始字节 SHA。
python -s scripts/main_experiment.py status --verify
python -s -m unittest discover -s tests -v

# 导出一个新离线训练 Notebook；只导出，不启动 Kaggle。
python -s scripts/main_experiment.py export --run-name hybrid_imu_next --notebook-name hybrid-imu-next.ipynb

# 仅在完整计算预算允许时显式训练，必须使用新目录；保留全部十个 IMU 模型的 50 epoch。
python -s -u scripts/main_experiment.py train --device cuda --output outputs/second_place/hybrid_imu_next
```

完整路由评估仍需 `evaluate_second_place_hybrid.py` 将实际新 ZIP 与原 ToF 控制严格合并；
该脚本的导入命令见 [SECOND_PLACE_HYBRID.md](experiments/SECOND_PLACE_HYBRID.md)。
不能仅凭重新训练的 IMU 输出宣称全模态或完整五折路由结果。
现有主实验权重保留在 `outputs/kaggle_training/imported/second_place_hybrid_imu_group_loss_v1/`。

## 实验记录与独立控制

- [第二名逐步复现](experiments/SECOND_PLACE_REPRODUCTION.md)：原 base、固定五折、来源适配及差异。
- [混合方案每一步](experiments/SECOND_PLACE_HYBRID.md)：融合筛查、训练批次、GPU 运行、ZIP 导入、完整审计与结果。
- [主实验完整结果](experiments/results/second_place_hybrid_imu_group_loss_v1.json)：各折、三场景、全部固定历史顺序及 provenance。
- [此次迁移记录](experiments/MAIN_EXPERIMENT_MIGRATION.md)：清理前 Git 快照、迁移 SHA 和保留范围。
- [原方案实验索引](experiments/README.md)、[旧实现档案](experiments/legacy/cnn_dynamics_mixup/README.md)：每步历史结果保留，旧方案不再是当前入口。

概率融合、因果历史和在线伪标签是独立研究控制，不自动加入主模型。
在线伪标签比较继续使用原 base 的冻结权重与匹配 CPU 逐条实际长度推理对照，
未切换成 hybrid；状态、模型、buffer、history 与 RNG 按同 subject/fold/场景/顺序隔离，
不读取验证真 gesture/orientation/phase、不使用未来、不回改返回、不挑最佳顺序。
正在运行的比较完成后单独记录，不提前报告伪标签收益。

simple/deep、作者原十折 ensemble、dynamic34D/THM 和 ToF 辅助损失重训尚未完成。
现有 GPU 剩余额度不足以支持新的完整训练，不缩短 50 epoch。
`data/`、`outputs/` 的大数据、权重、缓存与完整 OOF 留在本机，不随 Git 上传；
此次仅切换研究主代码，旧生产权重保持冻结，不新增比赛提交。
