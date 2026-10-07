# 前排方案优化的固定五折比较

Paired development CV on the exact same 8,151 sequences; checkpoint/design selection uses validation

| 输入 / 模型 | 原方案均值 ± 标准差 | 新方案均值 ± 标准差 | 均值变化 | 提升折数 |
| --- | ---: | ---: | ---: | ---: |
| observed / imu | 0.748366 ± 0.016025 | 0.787194 ± 0.014102 | +0.038828 | 5/5 |
| observed / multisensor | 0.822268 ± 0.017113 | 0.851180 ± 0.009654 | +0.028911 | 5/5 |
| observed / routed | 0.823642 ± 0.016330 | 0.852029 ± 0.008719 | +0.028387 | 5/5 |
| aux_dropout50 / routed | 0.788400 ± 0.013442 | 0.821455 ± 0.011675 | +0.033055 | 5/5 |
| imu_only / routed | 0.748366 ± 0.016025 | 0.787194 ± 0.014102 | +0.038828 | 5/5 |

| 输入 / 模型 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | Pooled OOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| observed / imu | 0.775652 | 0.785888 | 0.785455 | 0.811088 | 0.777887 | 0.787733 |
| observed / multisensor | 0.843238 | 0.849067 | 0.841385 | 0.863590 | 0.858618 | 0.851348 |
| observed / routed | 0.843238 | 0.849067 | 0.845631 | 0.863590 | 0.858618 | 0.852202 |
| aux_dropout50 / routed | 0.812717 | 0.815806 | 0.817661 | 0.841888 | 0.819203 | 0.821583 |
| imu_only / routed | 0.775652 | 0.785888 | 0.785455 | 0.811088 | 0.777887 | 0.787733 |

aux_dropout50 为标签无关的固定缺失压力测试。上述标准差为五折样本标准差（ddof=1），Pooled OOF 是将全部验证预测合并后的分数，与五折均值分别报告。结果为开发 CV，不能当作新的线上比赛分数。

保留统一配置 `cnn_dynamics_mixup`：五个比较均在全部五折提升，原始输入路由的均值提升 0.028387。未按折挑选配置、未搜索路由阈值或融合权重。IMU-only 和多传感器各五个 checkpoint 分别本地保存并 commit。

本机已核对原始训练 CSV 的字节 SHA256、每个模型训练 sequence / subject 与固定 folds。新旧共 18 个模型 / 场景组合均覆盖同一 8,151 条 sequence 一次，概率 argmax 与分类标签一致，官方二值 F1 和九类 macro F1 重算一致。固定缺失序列、每折路由数量相同；全部辅助缺失时路由到 IMU，IMU 概率不受辅助缺失影响。

原训练结果来自 [Kaggle 固定五折训练版本 2](https://www.kaggle.com/code/mingweiwei03/cmi-winner-cnn-fixed-fold-validation?scriptVersionId=356161432)。用户手动下载并解压结果；本地重打包仅用于导入，不能将重打包哈希当作原云端 ZIP 哈希。详细恢复与核验证据为 `experiments/results/cnn_winner_import_verified.json`；OOF 和每折评价保存在 `outputs/winner_comparison/{reference,candidate}/`。

模型和 checkpoint 选择使用 validation，以上提升不能保证线上提升。本轮没有新的比赛提交。
