# 旧 cnn_dynamics_mixup 实现档案

用户于 2026-10-09 指定第二名 base + 我们的 IMU 辅助损失作为主实验。
此目录保留原模型、预处理、训练、推理、后训练/表示学习、配置及测试的原始字节。
迁移映射和每个文件的 SHA256 见 [MAIN_EXPERIMENT_MIGRATION.json](../../MAIN_EXPERIMENT_MIGRATION.json)。

原项目说明保存为 [PROJECT_README.md](PROJECT_README.md)。全部逐步成功/失败记录仍在
`experiments/` 和 `experiments/results/` 的原位置；本地数据、旧冻结权重、OOF、下载 ZIP
和证据留在原路径。这里的旧配置及命令属于历史实验，不是当前默认入口。

完整清理前代码已保存于 commit `b4de175`、tag `pre-hybrid-main-cleanup-20261009`。
若要重新运行旧方案，可将该版本导出到独立目录，按当时的项目说明配置数据，
不要将档案脚本直接当成新项目入口运行。其历史相对路径保留，未静默改写 ROOT 或模型。

```powershell
git archive --format=zip --output=outputs/legacy_before_cleanup.zip pre-hybrid-main-cleanup-20261009
```

当前 `scripts/run_winner_experiments.py` 与 `scripts/import_winner_experiments.py` 仅保留
数据 SHA 检查、安全 ZIP 提取的兼容导入；公共实现位于 `src/cmi_project/artifact_io.py`。
旧模型的生产权重继续冻结，主实验切换不创建比赛提交，也不替换线上部署。
