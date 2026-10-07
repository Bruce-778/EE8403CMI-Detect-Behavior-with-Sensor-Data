"""Generate a concise Chinese report from completed, matching CNN folds."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.validation import load_fold_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    lines = ["# 主实验结果", "", "所有数值使用相同 subject folds 和官方 CMI 指标。"
        "单折筛选与 early stopping 使用 validation，结果属于开发 CV，不能代替独立测试集成绩。", "",
        "## 固定 fold 0 筛选", "", "| 配置 | Model A：IMU | Model B：多传感器 |", "| --- | ---: | ---: |"]
    rows, configurations = [], {}
    for name in args.runs:
        directory = ROOT / "outputs/experiments" / name
        if not (directory / "run_config.json").is_file():
            continue
        run = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        if run["folds_sha256"] != manifest.fingerprint:
            raise ValueError("Cannot compare different folds.")
        configurations[name] = run
        scores = []
        for model in ("imu", "multisensor"):
            for path in sorted((directory / model).glob("fold_*/metrics.json")):
                metric = json.loads(path.read_text(encoding="utf-8"))
                if metric["folds_sha256"] != manifest.fingerprint:
                    raise ValueError("Fold metrics fingerprint mismatch.")
                rows.append({"experiment": name, "model": model, "fold": metric["fold"],
                    "best_epoch": metric["best_epoch"], "epochs_run": metric["epochs_run"],
                    "max_length": metric["max_length"], **metric["validation"]})
            path = directory / model / "fold_0/metrics.json"
            scores.append(f"{json.loads(path.read_text(encoding='utf-8'))['validation']['score']:.6f}" if path.is_file() else "未完成")
        lines.append(f"| {name} | {' | '.join(scores)} |")
    lines.extend(["", "## 完整五折", "", "只有五折全部完成的配置才进入这张表。标准差使用 ddof=1。", "",
        "| 配置 | 模型 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 均值 ± 标准差 | Pooled OOF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    table = pd.DataFrame(rows)
    if len(table):
        for (name, model), group in table.groupby(["experiment", "model"], sort=False):
            if set(group["fold"]) != set(range(manifest.n_splits)):
                continue
            group = group.sort_values("fold")
            values = group["score"].to_numpy()
            path = ROOT / "outputs/experiments" / name / model / "evaluation/metrics.json"
            if not path.is_file():
                raise ValueError("Five folds exist but validated OOF evaluation is missing.")
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            if evaluation["folds_sha256"] != manifest.fingerprint:
                raise ValueError("OOF fingerprint mismatch.")
            pooled = evaluation["oof"]["score"]
            lines.append(f"| {name} | {model} | " + " | ".join(f"{v:.5f}" for v in values)
                + f" | {values.mean():.5f} ± {values.std(ddof=1):.5f} | {pooled:.5f} |")
    lines.extend(["", "## 参数", ""])
    for name, run in configurations.items():
        training, model = run["training"], run["model_parameters"]
        lines.append(f"- **{name}**：lr={training['learning_rate']}，batch={training['batch_size']}，"
            f"dropout={model['dropout']}，epochs≤{training['epochs']}，patience={training['early_stopping_patience']}，"
            f"macro loss weight={training.get('macro_loss_weight', 0)}，binary loss weight={training.get('binary_loss_weight', 0)}，"
            f"pooling={model.get('pooling', 'mean_max')}，stage kernels={model.get('stage_kernel_sizes') or [model['kernel_size']]*len(model['imu_channels'])}。")
    lines.extend(["", "## 输出与复现", "", "配置位于 `configs/cnn_*.json`；完整 checkpoint、OOF、曲线在"
        " `outputs/experiments/<配置名称>/`。`experiments/results/` 保存可追踪的小型结果摘要。"
        "命令和资料来源见 [实验记录](README.md)。", ""])
    output = ROOT / "experiments/RESULTS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
