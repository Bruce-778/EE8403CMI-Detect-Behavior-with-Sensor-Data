# CMI 项目框架

这是用于 CMI 传感器手势分类实验的 Python 项目，包含数据检查、按 subject fold 拟合的预处理和固定的五折验证。

## 目录结构

```text
project/
├── configs/
│   ├── dataset_analysis.json       # 数据路径、分块大小与类别数量
│   ├── preprocessing.json          # fold、坐标、归一化及 dropout 配置
│   ├── validation.json             # 五折验证配置
│   ├── baseline.json               # IMU-only LightGBM 配置
│   ├── baseline_xgboost.json       # IMU-only XGBoost 配置
│   ├── folds.csv                   # 所有实验共用的 sequence 划分
│   └── folds.meta.json             # 划分参数与指纹
├── data/                           # 原始数据；不纳入 Git
├── scripts/
│   ├── analyze_dataset.py          # Step 1 入口
│   ├── preprocess_data.py          # Step 2 入口
│   ├── create_folds.py             # Step 3 划分与检查入口
│   ├── evaluate_oof.py             # 统一评分和 OOF 汇总
│   └── train_baseline.py           # IMU-only baseline 训练
├── src/
│   └── cmi_project/
│       ├── __init__.py
│       ├── dataset_analysis.py     # 可复用的数据分析逻辑
│       ├── preprocessing.py        # 物理特征、mask 和 fold 参数
│       ├── preprocess_cli.py       # 读取固定划分和流式导出
│       ├── torch_inputs.py         # Dataset、3D CNN 和时间 mask
│       ├── validation.py           # sequence 索引、subject 划分及检查
│       ├── evaluation.py           # CMI 指标和 OOF 预测导出
│       └── baseline.py             # IMU 统计特征和两种树模型
├── tests/
│   ├── test_dataset_analysis.py
│   ├── test_preprocessing.py
│   ├── test_torch_inputs.py
│   ├── test_validation.py
│   └── test_baseline.py
├── .gitignore
├── environment.yml                # cmi Conda 环境与已验证的依赖版本
├── requirements.txt
└── README.md
```

`src/` 负责实现，`scripts/` 负责运行，`configs/` 管理参数，`outputs/` 保存结果。

## Python 环境

项目使用 Conda 环境 `cmi`。本机解释器路径为 `D:\anaconda\envs\cmi\python.exe`，在项目根目录的 PowerShell 中运行：

```powershell
conda activate cmi
python -c "import sys; print(sys.executable)"
```

输出应为 `D:\anaconda\envs\cmi\python.exe`。下文的 `python` 命令均在这个环境中执行，也可以使用 `conda run -n cmi python ...`，或直接使用 `& 'D:\anaconda\envs\cmi\python.exe' -s ...`。

首次配置或在其他机器上恢复依赖时运行：

```powershell
conda env update -n cmi -f environment.yml
conda activate cmi
```

`environment.yml` 固定了迁移时使用的核心包版本；`requirements.txt` 保留项目的一般依赖要求。工作区 `.vscode/settings.json` 已将默认 Python 解释器设为 `cmi`。如果编辑器此前已选择 `.venv`，需通过 `Python: Select Interpreter` 选择上述解释器，并新建终端。

`cmi` 设置了 `PYTHONNOUSERSITE=1`，避免加载 Windows 用户目录下其他项目的 Python 包。环境设置变更后需重新激活；直接调用解释器时使用 `-s` 保持同样的隔离。

## 1. Dataset Analysis：数据读取与检查

将 `train.csv` 和 `train_demographics.csv` 放入 `data/`，在项目根目录（README 所在目录）运行：

```powershell
python scripts/analyze_dataset.py
```

运行后，结果保存在 `outputs/dataset_analysis/`：

- `tables/basic_statistics.csv`：基础数据统计表，包含 sequence、subject、gesture 数量和 sequence 长度摘要；其余 CSV 提供类别分布、传感器缺失值及标签一致性检查。
- `figures/`：6 张 PNG，展示 gesture 样本分布、sequence 长度分布、subject 样本数量、传感器缺失情况和 ToF 像素不可用比例。
- `dataset_analysis.html`：汇总统计表与图表的报告，可直接用浏览器打开，用于后续实验和报告中的 Dataset Analysis。

类别样本按 sequence 计数；ToF 的 `-1` 无效值与 NaN 分开统计。数据路径和分块大小可在 `configs/dataset_analysis.json` 中调整。`outputs/` 为本地生成目录，不纳入 Git。

## 2. 数据预处理

预处理读取 `configs/folds.csv`，首次运行先执行下一节的 `create_folds.py` 生成固定划分。

```powershell
python scripts/preprocess_data.py --fold 0
```

实现左右手统一、15 维 IMU 物理特征、THM 段内填充、ToF 距离/validity map、左侧 padding 与尾部裁剪。标准化参数和长度阈值只由训练 subject 拟合，验证和 test 复用保存的参数。训练缓存读取时可加入传感器 dropout。

默认输出 `outputs/preprocessing/fold_0/` 中的参数、样本 NPZ 和 manifest；`--fit-only` 可只拟合参数。多个 fold 使用 `--fold 0 1 2 3 4`。

fold 可以理解为一组用来轮流做验证的数据。
我们的代码把所有 subject（参与者）分成 5 组：

- fold 0：第 0 组做验证集，其余 4 组做训练集。
- fold 1：第 1 组做验证集，其余 4 组做训练集。
- 依此类推，共可以进行 5 轮训练和验证。

这里按 人 分组，同一个人的所有 sequence 都在同一组，避免模型在训练时见过这个人，导致验证结果过于乐观。
因此，--fold 0 表示只准备第 0 轮的数据

配置位于 `configs/preprocessing.json`。没有确认采样间隔时角速度为 rad/采样步；实际间隔可通过 `sample_period_seconds` 设置。

## 3. 建立正确的验证方式

**推荐：5-fold StratifiedGroupKFold，`group = subject`，分层标签为 `gesture`。** 每个 sequence 作为一个样本，同一个 subject 的所有 sequence 分到同一折，避免模型记住个人动作习惯而使验证分数虚高。分层用于尽量平衡各折的类别比例。

在项目根目录运行一次：

```powershell
python scripts/create_folds.py
```
```powershell
python scripts/evaluate_oof.py --prediction-dir outputs/experiments/baseline/predictions --output-dir outputs/experiments/baseline/evaluation --experiment-name baseline
```

输出 `fold_scores.csv`、`metrics.json` 和 `oof_predictions.csv`，包含每折分数、五折均值与样本标准差（`ddof=1`），以及合并 OOF 的分数。统一使用 [CMI 官方指标](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/overview/evaluation)：target/non-target 的 Binary F1 与合并 non-target 后的 9 类 Macro F1 的平均值。每条 OOF 预测必须来自没有训练过该 subject 的模型。

参考：[CMI 第一名方案](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution)采用按 subject 分组的五折验证；[scikit-learn 文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)说明了分组隔离与类别分层的行为。当前环境的 scikit-learn 1.6.1 存在 shuffle 时分组计数映射问题，`validation.py` 根据[上游修复](https://github.com/scikit-learn/scikit-learn/pull/32540)做了小型兼容处理，并保留上述配置。

## 4. IMU-only Baseline

LightGBM 与 XGBoost 使用相同的 108 个 sequence 统计特征、`configs/folds.csv` 和官方 CMI 指标。统计包括 mean、std、min、max、range、向量模长及有效读数比例；缺失读数排除后，无法计算的统计量保留为 NaN。两种模型默认均训练 300 轮，随机种子为 42。

在已安装项目依赖的环境中运行：

```powershell
python scripts/train_baseline.py
python scripts/train_baseline.py --config configs/baseline_xgboost.json
```

在 `cmi` Conda 环境中执行上述命令。每个实验只输出 `outputs/baseline/<模型名>_imu/evaluation/fold_scores.csv` 和 `metrics.json`，包含每折分数、五折均值与标准差及合并 OOF 分数；OOF 预测仅在内存中计算。再次训练可使用 `--output-dir` 指定新目录。

XGBoost 使用 CPU `hist` 和 `multi:softprob` 输出类别概率，参数见 [官方文档](https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html)。

当前固定配置的五折官方分数：LightGBM **0.70368 ± 0.00752**，XGBoost **0.69480 ± 0.00725**；标准差使用 `ddof=1`。两次实验使用完全相同的 IMU 特征与 folds。

## 5. 主实验 v1：1D CNN

入口是 `scripts/train_cnn.py`，参数在 `configs/cnn_v1.json`。两个模型共用固定 subject folds 和每个训练折专属的数据缓存：

- **Model A (`imu`)**：15 个 IMU 物理特征及其有效性 mask → residual 1D CNN → masked mean/max pooling → 全连接分类。
- **Model B (`multisensor`)**：IMU、THM、ToF 分别经过独立的 residual 1D CNN encoder；池化后的特征 concatenate，再通过全连接层预测 18 个 gesture。
- ToF 默认将每个传感器的 8×8 图划为 2×2 区域，只对有效像素求均值，同时输入区域有效像素比例和硬件存在 mask。1D 卷积沿时间轴计算。`tof_regions` 可选 1、2、4、8；8 表示保留全部像素。
- IMU / THM / ToF 逐通道标准化只在训练 subjects 上拟合；标准化后默认截断到 ±8。每个卷积层使用按时间 token 的 channel LayerNorm，padding 不参与其他 token 的 normalization；每层传播 mask，池化忽略不可用位置。缺失模态的 encoder 输出保持为零。

v1 默认以训练序列长度的 95% 分位数确定长度，左侧 padding、保留序列末尾；可以通过 `sequence_length` 或 `--sequence-length` 指定固定长度。默认 AdamW、learning rate `1e-3`、batch size `64`、dropout `0.2`、weight decay `1e-3` 和 label smoothing `0.03`。训练读取时应用传感器 dropout；验证时关闭 augmentation 和 dropout。连续 10 个 epoch 的官方 CMI 分数没有超过 `min_delta=1e-4` 的提升时 early stop，分数停滞时自动降低 learning rate；最终评估恢复最高验证分数的 checkpoint。

在项目根目录运行，默认先在 fold 0 比较两个模型：

```powershell
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py
```

默认输出在 `outputs/experiments/cnn_v1/`：`report.html`、模型对照表和图，每个模型的 `best.pt`、`history.csv`、训练曲线、逐类指标、混淆矩阵和验证预测概率。checkpoint 包含权重、全局 class order、preprocessor、输入配置和训练配置；可以用 `cmi_project.cnn_training.load_cnn_checkpoint()` 恢复。共享缓存位于 `outputs/cnn_cache/v1/`，参数或原始数据变化后需指定新的 `--cache-dir`；再次训练需使用新的 `--output-dir`，避免覆盖已有实验。

```powershell
# 完整五折；两模型全部训练后才汇总完整 OOF、折均值和标准差。
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --fold 0 1 2 3 4 --output-dir outputs/experiments/cnn_v1_5fold

# 示例：单模型的新实验，修改 learning rate、batch size 和 dropout。
& 'D:\anaconda\envs\cmi\python.exe' -s -u scripts/train_cnn.py --model imu --learning-rate 0.0005 --batch-size 32 --dropout 0.3 --output-dir outputs/experiments/cnn_v1_imu_lr0005
```

单折预览只与相同 fold 的 baseline 比较，不与 baseline 五折均值直接比较。验证数据用于 early stopping / checkpoint 选择，所以这些是开发阶段结果；调参后应锁定配置再运行五折。

v1 首次 CPU 训练的 **fold 0 预览**（6,524 条训练序列 / 1,627 条验证序列，长度 127，seed 42）：

| 模型 | 官方 CMI 分数 | Binary F1 | 9 类 Macro F1 | 最佳 epoch |
| --- | ---: | ---: | ---: | ---: |
| LightGBM IMU baseline | 0.71012 | 0.96303 | 0.45720 | — |
| XGBoost IMU baseline | 0.69535 | 0.95985 | 0.43086 | — |
| Model A：IMU-only 1D CNN | 0.72278 | 0.96711 | 0.47845 | 28 |
| Model B：多传感器 1D CNN | **0.81152** | **0.97935** | **0.64370** | 38 |

Model A 在第 38 轮 early stop；Model B 运行到 45 轮上限，最终都恢复最佳 checkpoint。Model B 相比 Model A 的 CMI 分数提高 0.08875，主要来自目标手势细分类改善。结果和曲线见 `outputs/experiments/cnn_v1/report.html`。上述配置是 v1 首轮配置，并非搜索得到的全局最优超参数。

运行全部检查：

```powershell
& 'D:\anaconda\envs\cmi\python.exe' -s -m unittest discover -s tests -v
```
