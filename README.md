# CMI 项目框架

这是用于 CMI 传感器手势分类实验的 Python 项目。当前实现 Step 1：数据读取与检查，后续数据预处理、模型与实验代码可以逐步加入 `src/cmi_project/`。

## 目录结构

```text
project/
├── configs/
│   └── dataset_analysis.json       # 数据路径、分块大小与类别数量
├── data/                           # 原始数据；不纳入 Git
├── docs/                           # 本地文档；不纳入 Git
├── scripts/
│   └── analyze_dataset.py          # 命令行入口
├── src/
│   └── cmi_project/
│       ├── __init__.py
│       └── dataset_analysis.py     # 可复用的数据分析逻辑
├── tests/
│   └── test_dataset_analysis.py    # 小数据正确性验证
├── outputs/
│   └── dataset_analysis/           # 运行后生成报告、表格与图表
├── .gitignore
├── requirements.txt
└── README.md
```

`src/` 负责实现，`scripts/` 负责运行，`configs/` 管理参数，`outputs/` 保存结果。当前无需单独的训练/评估脚本、模型 checkpoint 目录或环境测试工程；实验扩展到这些阶段时再添加。

Step 1 完整扫描训练数据，采用分块读取，内存使用由 `chunksize` 控制，原始 CSV 保持不变。

## 运行

推荐 Python 3.10 或更高版本。在项目根目录（README 所在目录）运行：

```powershell
python -m pip install -r requirements.txt
python scripts/analyze_dataset.py
```

默认配置为 `configs/dataset_analysis.json`，读取 `data/train.csv` 和 `data/train_demographics.csv`。克隆仓库后，需自行创建 `data/` 并放入这两个数据文件。配置文件中的相对路径按项目根目录解析；CLI 显式指定的相对路径按当前工作目录解析。输出默认是项目根目录下的 `outputs/dataset_analysis/`，不依赖启动目录。

指定路径或调整内存使用：

```powershell
python scripts/analyze_dataset.py --chunksize 10000
python scripts/analyze_dataset.py --data-dir ../cmi-data
python scripts/analyze_dataset.py --output-dir outputs/my_analysis
```

默认核对 18 个实际观测 gesture。`--no-plots` 可跳过绘图。成功返回 0；检查发现完整性问题时，保留诊断文件并返回 1；必需文件或列缺失、参数错误时返回 2。

## 统计口径

- 每个 `sequence_id` 是一个分类样本，gesture 样本分布按 sequence 数量统计；`row_count` 是时间步数量。
- sequence 长度为该 ID 的全部 CSV 行数，跨分块、非连续出现的同一 ID 也会正确合并。未假设采样频率，不将行数转换为秒。
- IMU：`acc_x/y/z` 和 `rot_w/x/y/z`，7 个特征；THM：`thm_1`–`thm_5`，5 个特征；ToF：`tof_1`–`tof_5` 各含 `v0`–`v63`，320 个特征。
- `nan_ratio = NaN 单元格数 / (总行数 × 特征数)`。CSV 空字段按 NaN 读取。
- 仅 ToF 的 `-1` 视为无效。`minus1_ratio` 以全部单元格为分母；`minus1_ratio_of_non_nan` 以非 NaN 单元格为分母。`unavailable_ratio` 是 NaN 与 ToF `-1` 的并集比例。IMU 负值不会被当作缺失。
- `sequences_fully_unavailable` 表示整个 sequence 的该模态均不可用；部分像素缺失与整段缺失分别统计。
- 每个 sequence 必须只有一个非空 gesture 和一个非空 subject。不会用第一行标签掩盖冲突；异常 sequence 保留所有观测标签和 subject，最终标签留空。若有异常，gesture 分布只计标签有效的 sequence，比例仍以全部 sequence 为分母，检查表记录未对齐计数。
- demographics 核对 subject 唯一性、空键、训练 subject 覆盖与逐列缺失情况。缺失统计阶段不填补数据。

## 输出

打开 `outputs/dataset_analysis/dataset_analysis.html` 查看中文报告。报告链接至 CSV 明细与图表，移动报告时应保留整个输出目录。

`tables/` 包含：

| 文件 | 内容 |
| --- | --- |
| `basic_statistics.csv` | 行数、sequence/subject/gesture 数量、特征数量、长度摘要和检查结果 |
| `gesture_distribution.csv` | 每类 gesture 的 sequence 数量、比例、时间步数和 subject 数 |
| `sequence_summary.csv` | 每个 sequence 的长度、标签、subject 和传感器可用情况 |
| `sequence_length_summary.csv` | 长度的均值、标准差、最小值、分位数和最大值 |
| `sequence_length_by_gesture.csv` | 各 gesture 的长度统计 |
| `subject_distribution.csv` | 每个 subject 的 sequence 数、行数和 demographics 匹配数 |
| `sensor_missingness.csv` | IMU、THM、ToF 的模态级 NaN/-1/不可用统计 |
| `sensor_feature_missingness.csv` | 332 个特征的逐列 NaN/-1/不可用统计 |
| `tof_device_missingness.csv` | 5 个 ToF 传感器分别的缺失与无效统计 |
| `demographics_missingness.csv` | demographics 逐列缺失情况 |
| `subject_demographics_coverage.csv` | subject 覆盖核对 |
| `feature_schema.csv` | 三类传感器的特征清单 |
| `integrity_checks.csv` / `integrity_issues.csv` | 全部检查 / 未通过的检查 |

`figures/` 包含 6 张 PNG：gesture 数量、sequence 长度直方图与 ECDF、各 gesture 长度箱线图、subject 样本数量、模态可用性、ToF 像素不可用比例。ToF 像素图按索引排列为 8×8，未推断物理朝向。`summary.json` 保存摘要、传感器统计、来源路径、输入文件大小和运行版本。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试用可手工核对的小数据检查跨分块聚合、NaN 与 ToF `-1` 的口径、冲突标签、缺失标签和 demographics 重复/缺失键，临时结果自动清理。
