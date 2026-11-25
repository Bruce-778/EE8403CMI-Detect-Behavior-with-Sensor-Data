> 实验档案说明（2026-10-07 清理）：本页保留当时的结果、决策与命令。当前运行方案为 `cnn_dynamics_mixup`，Public 0.839556 / Private 0.833362；请以[当前项目入口](../README.md)为准。旧配置原样存于 `experiments/configs/`，旧运行代码可从 Git 历史恢复。

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
| 分组 CNN + SE + masked BatchNorm | 0.773586 | 29 / 45 | CPU 单折 pilot |
| 34 通道运动特征 + 同一网络 + Mixup | 0.775116 | 39 / 55 | CPU 单折 pilot |
| 分组 CNN + 同一输入 + 双向 GRU | **0.775714** | 25 / 41 | CPU 单折 pilot |

第一项本地尝试提升 0.027211；二值 F1 与 9 类 macro F1 按官方方式计算。它仍为单折开发结果，不能当作五折或线上提升。60 项检查通过，包含 padding 对 BatchNorm 统计的隔离、空分支、恢复模型、Mixup 的 mask 以及衍生特征的缺失语义。完整证据见 `experiments/results/cnn_grouped_se_cpu_pilot.json`。

第二项本地尝试为 0.775116，比第一项高 0.001530，比旧融合高 0.028741。新增运动特征和 Mixup 的联合收益较小，尚不能分别归因；单独 Mixup 的 GPU pilot 用于进一步区分。55 epochs 后 early stop，最佳第 39 轮，完整证据见 `experiments/results/cnn_dynamics_mixup_cpu_pilot.json`。

补充 CPU 对照 `cnn_grouped_gru`：保持第一项的输入、损失和训练参数，仅将 CNN 的 mean/max pooling 换成双向 GRU 的有效时间均值与末状态。先将有效 CNN token 按原时间顺序打包，排除 padding，整段不可用分支输出零。结果为 0.775714，比第一项高 0.002128；最佳 epoch 25，41 epochs 后 early stop。已核对同一 1,627 条验证 sequence、训练 subject 与保存的 OOF 分数，证据为 `experiments/results/cnn_grouped_gru_cpu_pilot.json`。仅为 CPU 单折对照，不参与已启动的三项 GPU 配置筛选；与运动特征 + Mixup 的差距不足 0.001，不能据此认定 GRU 更优。

## GPU 配置筛选

GPU 三项均在同一 fold 0、同一环境运行：分组 CNN 0.770843（60 / 76 epochs）、仅增加 Mixup 0.772690（56 / 60）、运动特征 + Mixup **0.775652**（38 / 43）。因此统一五折配置为 `cnn_dynamics_mixup`；CPU GRU 不混入该硬件条件下的筛选。已将三项预测、训练 subject 和 checkpoint / 缓存参数逐项核对，紧凑证据为 `experiments/results/*_gpu_pilot.json`。

首轮五折续跑因 Python tuple 与 JSON list 的表示差异被一致性检查误拒绝，未开始新的五折训练。修复为比较精确 JSON 保存值，不放宽参数、标准化统计或训练 ID 的检查；实测三项原始 GPU checkpoint 均能复用，回归检查也确认统计值改变仍会拒绝。已完成 pilot 作为下一轮 Notebook 输入保留，避免重复训练筛选阶段。

跨 Notebook 续跑版本 1 又被缓存来源身份检查拒绝，未训练任何新 fold。该检查包含文件路径、大小和时间戳，不适合直接复制到新的数据挂载。版本 2 保留相同三项筛选结果和 `cnn_dynamics_mixup` 选择，把旧 pilot 权重放入独立证据目录；重新构建训练 fold 缓存，并从头训练选定配置的十个模型。原始训练 CSV 的 SHA256 已固定在 `configs/data_source_hashes.json`，云端两份文件的校验值与本地完全一致。缓存检查保持原样，没有绕过数据来源限制。

完整验证版本为 [CMI Winner CNN Fixed Fold Validation v2](https://www.kaggle.com/code/mingweiwei03/cmi-winner-cnn-fixed-fold-validation?scriptVersionId=356161432)，十个模型及全部五折 OOF 已完成并回收。本机核对了原始训练 CSV SHA256、固定训练 sequence / subject、checkpoint 和官方指标；没有用筛选阶段的 fold 0 替代新运行。新增两项针对性检查及真实 GPU 产物恢复检查通过，已完成的 63 项项目测试仍保留记录。

## 完整五折结论

IMU-only：`0.748366 ± 0.016025 → 0.787194 ± 0.014102`；多传感器：`0.822268 ± 0.017113 → 0.851180 ± 0.009654`；按输入可用性路由：`0.823642 ± 0.016330 → 0.852029 ± 0.008719`。三个比较均提升全部五折，使用完全相同的 8,151 条验证 sequence，标准差为样本标准差。

固定约半数 sequence 额外缺失 THM+ToF 时，路由分数为 `0.821455 ± 0.011675`，比旧方案高 `0.033055`；全部辅助模态缺失时为新 IMU 分支的 `0.787194 ± 0.014102`。新旧缺失样本和每折 A/B 路由数量已核对一致，18 个模型 / 场景组合的概率 argmax、官方指标及完整 OOF 均通过检查。

因此保留一个统一的 `cnn_dynamics_mixup` 配置和十个 A/B 五折权重。结果与每折变化见 `WINNER_RESULTS.md`，完整配置和训练记录见 `experiments/results/cnn_winner_selected.json`。收益属于分组 CNN、masked BatchNorm、SE、34 通道运动特征及 Mixup 的联合效果；现有五折证据无法分别归因。fold 0 参与配置筛选、各折 validation 参与 early stopping，仍是开发 CV，不是独立测试或新的线上成绩。

新推理包已保存于 `outputs/kaggle_winner_submission_v2`，十个冻结 checkpoint，五折等权及原可用性路由。CPU 实际推理核对每折一条 held-out sequence，与 GPU OOF 的最大概率误差为 `3.56e-7`，标签字段被忽略；公开无标签示例和全部辅助模态缺失输入均通过。包内容与源码一致，Notebook 语法有效；推理核验证据已本地保存并 commit，本地跟进自动化 `cmi` 已删除。

用户后续授权将该冻结包提交线上评分，实际 Public 为 0.839556（提升 0.033148），Private 为 0.833362（提升 0.036984）。按最终 Private 榜单估算插入位置约第 125 / 2,657 名，Late Submission 没有正式最终名次。没有在查看线上分数后改模型或调权重；评分过程与来源见 `KAGGLE.md`、`KAGGLE_WINNER_STATUS.md` 和 `experiments/results/kaggle_winner_cnn.json`。线上跟进自动化 `cmi-cnn` 也已删除。
