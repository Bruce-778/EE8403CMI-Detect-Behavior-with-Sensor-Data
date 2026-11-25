# 当前代码清理与实验档案保留

2026-10-07 按用户要求，将当前运行代码统一到已验证的 `cnn_dynamics_mixup`，保留全部实验记录。

- 移除旧树模型 baseline 的训练实现、LightGBM/XGBoost 依赖、旧模型对比/融合脚本、旧 pilot 筛选和跨 Notebook 续跑逻辑。
- 移除旧 joint/token LayerNorm CNN、attention、GRU 和未采用的 ToF 3D encoder 分支及其专用测试；保留当前分组 masked SE CNN、物理预处理、固定 subject folds、官方指标、Mixup、训练和离线推理流程。
- 当前配置默认训练 A/B 全五折，推理导出和校验默认使用当前方案。没有重新训练或新增线上比赛提交。
- `experiments/results/` 中已有结果 JSON 全部原样保留；所有旧实验文档保留，部分页面增加“历史档案”提示，原内容未删减。
- 十二份旧配置原样存入 `experiments/configs/`；清理前 README 原文保存于 `experiments/PROJECT_HISTORY.md`。档案中的旧命令对应旧 Git 版本，当前入口见根目录 README。
- 原始数据、旧/新权重、缓存、完整 OOF 和线上截图继续保存在本地 `data/`、`outputs/`；这些大文件不加入 Git。

当前已评分权重的官方 Public 为 **0.839556**，Private 为 **0.833362**，来源版本为 356204022。
固定五折开发 CV 为 **0.852029 ± 0.008719**，采用样本标准差；线上成绩和开发 CV 分别保存，清理后的代码没有获得新的线上评分。

已执行清理后的 55 项项目测试和真实权重推理检查，全部通过；另对最终训练 Notebook 导出执行了两项针对性检查。
39 份既有结果 JSON 未改动，十二份旧配置原样归档，历史文档原文完整保留。
十个冻结 checkpoint 的 2,160 个权重/状态张量与原已评分包逐个完全一致，预处理参数一致。
每折一条真实 held-out sequence 的概率与原 GPU OOF 对齐，最大绝对误差为 `3.56e-7`。
标签字段不影响推理，两个公开无标签示例及强制辅助模态全缺失输入通过。公开示例用于接口检查，没有测试标签或测试分数。
最终离线训练 Notebook 与推理 Notebook 的语法、训练源码载荷、推理包源码一致性和 ZIP 完整性检查通过。
清理后的推理包保存在 `outputs/kaggle_submission_current/`；原已评分包继续保留。
验证明细见 [`results/code_cleanup_verified.json`](results/code_cleanup_verified.json)。
