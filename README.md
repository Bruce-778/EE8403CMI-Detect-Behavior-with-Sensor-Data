# CMI 项目框架

这是用于 CMI 传感器手势分类实验的 Python 项目。

## 目录结构

```text
project/
├── configs/
│   └── dataset_analysis.json       # 数据路径、分块大小与类别数量
├── data/                           # 原始数据；不纳入 Git
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

## 1. Dataset Analysis：数据读取与检查

将 `train.csv` 和 `train_demographics.csv` 放入 `data/`，在项目根目录（README 所在目录）运行：

```powershell
python -m pip install -r requirements.txt
python scripts/analyze_dataset.py
```

运行后，结果保存在 `outputs/dataset_analysis/`：

- `tables/basic_statistics.csv`：基础数据统计表，包含 sequence、subject、gesture 数量和 sequence 长度摘要；其余 CSV 提供类别分布、传感器缺失值及标签一致性检查。
- `figures/`：6 张 PNG，展示 gesture 样本分布、sequence 长度分布、subject 样本数量、传感器缺失情况和 ToF 像素不可用比例。
- `dataset_analysis.html`：汇总统计表与图表的报告，可直接用浏览器打开，用于后续实验和报告中的 Dataset Analysis。

类别样本按 sequence 计数；ToF 的 `-1` 无效值与 NaN 分开统计。数据路径和分块大小可在 `configs/dataset_analysis.json` 中调整。`outputs/` 为本地生成目录，不纳入 Git。
