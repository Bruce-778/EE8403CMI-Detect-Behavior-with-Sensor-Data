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
