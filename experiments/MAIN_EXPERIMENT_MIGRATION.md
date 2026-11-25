# 主实验切换与代码清理：2026-10-09

用户明确指定第二名 base + 我们的辅助损失作为主实验，并授权清理旧代码、提交和 push。
当前混合结果在 `second_place_hybrid_imu_group_loss_v1.json` 已完整核验，
不会因主代码选择而更改已有评分、上游出处、原配方适配说明或未完成项。

1. 清理前工作区干净，全部实际实验已提交在 `b4de175`。建立 annotated tag
   `pre-hybrid-main-cleanup-20261009`，保留完整清理前源树。
2. 原 CNN/预处理/推理、后训练/表示学习入口、旧配置和旧测试共 37 文件逐字迁移到
   `experiments/legacy/cnn_dynamics_mixup/`。每个原路径、新路径与 SHA 记录在配套 JSON。
3. 数据字节 SHA 校验和安全 ZIP 提取抽到中性 `artifact_io.py`。两份旧命名入口只保留
   导入兼容层，避免复现/CPU 工具依赖旧模型实现；其旧完整代码已存档。
4. `configs/main_experiment.json` 指向已核验的 hybrid 结果和 20 模型来源。
   `scripts/main_experiment.py` 默认只显示状态；完整文件 SHA 核验、显式训练和新 Notebook
   导出分开。原 hybrid 训练脚本、导出器、辅助损失函数和上游源码未改。
5. 全部历史结果 JSON 保留原路径与字节。旧数据、原模型权重、OOF、证据、下载 ZIP、
   正在运行的 CPU 源码及固定 folds 保留。在线比较仍使用原 base，不能混入 hybrid 权重。

原实验说明中的命令反映当时的尝试；旧实现独立重跑应使用清理前 Git 快照。
主方案是开发研究入口，不是线上部署替换。新 Git 仓库内容不含 data/outputs 权重缓存。
核验与 push 状态保存于迁移 JSON；此操作不启动新的训练或比赛提交。

核验完成：33 项测试通过；改用随仓库保存的固定上游源码后，7 项相关测试再次通过。
37 个旧文件在本地和提交后的 Git blob 均匹配原 SHA；79 个受保护的历史结果、
正在运行源码与固定 folds 保持原字节。原始 CSV SHA、主模型文件、15 份 OOF 与离线
Notebook 嵌入源码/编译及防覆盖检查全部通过。

主代码提交 `c30642c` 和清理前快照标签已成功原子 push 到 `origin`，
随后通过 `git ls-remote` 核对远端 main 与标签指向。此次也同步了此前仅在本地的
逐步实验提交。大数据与权重仍在原本机目录；原 base CPU 在线实验继续运行。
