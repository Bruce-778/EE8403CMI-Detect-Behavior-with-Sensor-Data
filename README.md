# CMI：运动特征 + Mixup 分组 CNN

当前代码保留已通过固定五折验证和 Kaggle 线上评分的 `cnn_dynamics_mixup` 方案：
IMU-only Model A 与 IMU + THM + ToF Model B，各训练五折，共十个 checkpoint。
按每条输入的传感器可用性选择 A/B，再对五折概率等权平均。

| 评价范围 | 当前成绩 |
| --- | ---: |
| Kaggle Public | **0.839556** |
| Kaggle Private | **0.833362** |
| 固定五折开发 CV，原始输入路由 | **0.852029 ± 0.008719** |
| 固定五折开发 CV，额外约半数辅助模态缺失 | 0.821455 ± 0.011675 |
| 固定五折开发 CV，辅助模态全部缺失 | 0.787194 ± 0.014102 |

线上结果来自 [推理 Notebook v1 / 356204022](https://www.kaggle.com/code/mingweiwei03/cmi-winner-inspired-cnn-inference?scriptVersionId=356204022)，状态 `Succeeded (after deadline)`。
这是 Late Submission，没有正式最终名次。CV 标准差为五折样本标准差（ddof=1）；fold 0 参与过设计选择，validation 参与 early stopping，因此是开发 CV。
线上来源、权重哈希和旧方案对照保存在 [评分记录](experiments/results/kaggle_winner_cnn.json)，逐折结果见 [固定验证比较](experiments/WINNER_RESULTS.md)。

## 代码与实验档案

```text
configs/                     # 当前配置、不可变 folds、原始训练数据 SHA256
src/cmi_project/
  preprocessing.py           # 坐标、物理特征、缺失 mask、训练 fold 标准化
  cnn.py                     # 分组残差 1D CNN、masked BatchNorm、SE、mean/max pooling
  cnn_data.py                # 训练 fold 缓存、ToF 有效区域聚合、训练传感器 dropout
  cnn_training.py            # Mixup、分层损失、early stopping、checkpoint 校验
  inference.py               # 十个冻结模型的可用性路由与五折概率平均
  validation.py              # sequence 索引、subject folds 和一致性检查
  evaluation.py              # 官方指标及 OOF 汇总
scripts/                     # 数据审计、训练、评价、结果导入、离线 Notebook 导出
tests/                       # padding、缺失、泄漏隔离、损失与推理检查
experiments/                 # 全部历史实验记录和当前结果
  configs/                   # 原样存档的旧实验配置，不是当前默认入口
data/                        # 原始比赛数据，Git 忽略
outputs/                     # 缓存、权重、完整 OOF、Notebook 与证据，Git 忽略
```

旧 baseline、attention/GRU、旧模型融合和 pilot 续跑代码已从当前运行路径移除。
**全部实验文档、成功/失败记录、旧结果 JSON 和旧配置保留**；旧实现仍能从 Git 历史恢复。
实验档案中的旧命令用于说明当时的尝试，运行当前方案请使用下方入口。
清理前项目说明另存为 [PROJECT_HISTORY.md](experiments/PROJECT_HISTORY.md)。

## 环境和数据

在项目根目录运行：

```powershell
conda env create -f environment.yml
conda activate cmi
```

已存在 `cmi` 环境时用 `conda env update -n cmi -f environment.yml`。
本机解释器为 `D:\anaconda\envs\cmi\python.exe`；直接调用时加 `-s`，环境已设 `PYTHONNOUSERSITE=1`。
将官方 `train.csv`、`train_demographics.csv`、`test.csv` 和 `test_demographics.csv` 放入 `data/`。
数据和模型权重不随 Git 上传；已有线上权重保存在 [私有 Kaggle 数据集 v1](https://www.kaggle.com/datasets/mingweiwei03/cmi-winner-cnn-weights-20261007)，访问需要账号权限。

## 固定验证与模型

所有模型和 ablation 复用 `configs/folds.csv`：每个 sequence 一行，gesture 分层、subject 分组，
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`。全部 8,151 条 sequence 各验证一次，81 个 subjects。
每折训练和验证 subject 不重叠；标准化、长度阈值只由该训练 fold 拟合。
统一官方 CMI 指标：binary F1 与九类 macro F1 的平均。报告逐折分数、五折均值/样本标准差及完整 OOF。

IMU 共 34 个通道，包含原始加速度、6D 旋转、角速度、去重力加速度和 sequence 内运动变化特征。
加速度、旋转、角速度、去重力加速度分别经过 CNN stem，再融合；THM 和 ToF 分别有独立 encoder。
ToF 使用每个传感器 2×2 有效区域距离、有效像素比例及硬件存在标记，沿时间做 1D CNN。
残差块使用 masked BatchNorm 和 SE，聚合使用有效时间 mean/max pooling。
长度取训练 fold 的 95% 分位数，保留尾部、左侧 padding；无效位置与 padding 排除在统计和聚合外。
训练使用传感器 dropout、Mixup、`CE18 + 0.5 CE9 + 0.1 BCE2` 和 early stopping。

## 运行当前方案

```powershell
# 审计并复用已有固定划分，不覆盖它。
python -s scripts/create_folds.py

# 一次训练 A/B 全五折；原始输入 SHA256 必须匹配已记录的数据。
python -s -u scripts/run_winner_experiments.py --data-dir data --device cuda
# 无 GPU 时可使用 --device cpu；同一工作区复用已完成且校验一致的 folds 可加 --resume。

# 也可分别使用训练、汇总和场景评价入口。
python -s -u scripts/train_cnn.py --config configs/cnn_dynamics_mixup.json --model both
python -s scripts/summarize_cnn_experiment.py outputs/experiments/cnn_dynamics_mixup --name cnn_winner_selected
python -s -u scripts/evaluate_cnn_scenarios.py outputs/experiments/cnn_dynamics_mixup --name cnn_winner_selected

# 生成离线 GPU 训练 Notebook；本命令只导出，不启动云端训练。
python -s scripts/export_kaggle_training.py
```

训练结果位于 `outputs/experiments/cnn_dynamics_mixup/`，各分支 `fold_0` 至 `fold_4` 保存权重、训练预处理参数、逐折预测和指标。
场景评价保存完整 OOF，并使用相同序列、相同固定遮挡规则对比原始输入、全部辅助模态缺失和约半数辅助模态缺失。
GPU 输出 `winner_experiments.zip` 可用 `scripts/import_winner_experiments.py --help` 查看导入参数；导入检查原始数据哈希、固定 folds、训练 subjects、十个模型和逐折指标。
重训练存在硬件和运行库数值差异，已记录的官方成绩属于上面的冻结权重版本。

## 导出和检查推理

```powershell
python -s scripts/export_kaggle_submission.py
python -s scripts/check_kaggle_submission.py
python -s -m unittest discover -s tests -v
```

默认导出到全新的 `outputs/kaggle_submission_current/`。`--runs` 可指定已导入的 `cnn_dynamics_mixup` 目录，
`--output-dir` 可选择新的导出位置；对应检查须指定 `--directory` 和匹配的 `--oof-path`。
导出要求 A/B 各五折和匹配的场景评价报告，检查每折真实 held-out 输入与保存 OOF 概率、标签字段隔离、公开无标签示例及辅助模态全缺失推理。
公开示例没有标签，只用于验证推理接口。
推理 Notebook 需挂载比赛输入和冻结权重，关闭互联网；本地导出、检查不会自动创建比赛提交。
