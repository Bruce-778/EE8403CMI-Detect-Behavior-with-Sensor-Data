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

