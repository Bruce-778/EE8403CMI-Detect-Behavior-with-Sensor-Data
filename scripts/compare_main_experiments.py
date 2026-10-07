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
                # The joint A/B trainer writes OOF after both models finish.
                # Keep partial progress out of the complete-results table.
                continue
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            if evaluation["folds_sha256"] != manifest.fingerprint:
                raise ValueError("OOF fingerprint mismatch.")
            pooled = evaluation["oof"]["score"]
            lines.append(f"| {name} | {model} | " + " | ".join(f"{v:.5f}" for v in values)
                + f" | {values.mean():.5f} ± {values.std(ddof=1):.5f} | {pooled:.5f} |")
    lines.extend(["", "## 固定等权概率融合", "", "两个来源均为对应 fold 的 held-out 预测，各 50%；不逐折调权。", "",
        "| 模型 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 均值 ± 标准差 | Pooled OOF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for path in sorted((ROOT / "experiments/results").glob("cnn_equal_blend_*.json")):
        blend = json.loads(path.read_text(encoding="utf-8"))
        evaluation = blend.get("evaluation")
        if evaluation is None:
            continue
        if evaluation["folds_sha256"] != manifest.fingerprint or evaluation["oof_sequences"] != len(manifest.table):
            raise ValueError("Blend OOF coverage/fingerprint mismatch.")
        values = [row["score"] for row in sorted(blend["fold_scores"], key=lambda row: row["fold"])]
        lines.append(f"| {blend['settings']['model']} | " + " | ".join(f"{v:.5f}" for v in values)
            + f" | {evaluation['fold_mean']['score']:.5f} ± {evaluation['fold_std']['score']:.5f} | {evaluation['oof']['score']:.5f} |")
    scenario_path = ROOT / "experiments/results/cnn_final_selected_scenarios.json"
    if scenario_path.is_file():
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        lines.extend(["", "## 最终方案的模态缺失压力测试", "", "同一份验证 sequence；额外遮挡由 sequence_id hash 决定，与标签无关。"
            "这是验证压力测试，不是隐藏测试集成绩。", "", "| 输入 | 模型 | 五折均值 ± 标准差 | Pooled OOF |",
            "| --- | --- | ---: | ---: |"])
        for key, evaluation in scenario["five_fold_evaluation"].items():
            if evaluation["folds_sha256"] != manifest.fingerprint or evaluation["oof_sequences"] != len(manifest.table):
                raise ValueError("Scenario OOF coverage/fingerprint mismatch.")
            model, condition = key.split("/")
            lines.append(f"| {condition} | {model} | {evaluation['fold_mean']['score']:.5f} ± {evaluation['fold_std']['score']:.5f} | {evaluation['oof']['score']:.5f} |")
        selection = ["## 当前采用的方案", "",
            "A 与 B 各自融合原始 CNN 和分层损失 CNN 的预测概率，权重固定为 50% / 50%。"
            "THM、ToF 至少一种可用时使用 B，两者都不可用时使用 A。路由只看输入可用性。", "",
            "| 模型 | 原始五折均值 | 当前五折均值 ± 标准差 | 均值提升 |",
            "| --- | ---: | ---: | ---: |"]
        for model, label in (("imu", "A：IMU-only"), ("multisensor", "B：多传感器")):
            original = json.loads((ROOT / "outputs/experiments/cnn_v1" / model / "evaluation/metrics.json").read_text(encoding="utf-8"))
            if original["folds_sha256"] != manifest.fingerprint or original["oof_sequences"] != len(manifest.table):
                raise ValueError("Reference OOF coverage/fingerprint mismatch.")
            selected = scenario["five_fold_evaluation"][f"{model}/observed"]
            before, after = original["fold_mean"]["score"], selected["fold_mean"]["score"]
            selection.append(f"| {label} | {before:.5f} | {after:.5f} ± {selected['fold_std']['score']:.5f} | +{after-before:.5f} |")
        routed = scenario["five_fold_evaluation"]["routed/observed"]
        stress = scenario["five_fold_evaluation"]["routed/aux_dropout50"]
        selection.extend(["", f"固定路由在原始验证输入上为 **{routed['fold_mean']['score']:.5f} ± {routed['fold_std']['score']:.5f}**；"
            f"约半数序列额外失去 THM、ToF 时为 **{stress['fold_mean']['score']:.5f} ± {stress['fold_std']['score']:.5f}**。", "",
            "本轮完成六项尝试。Attention pooling、较大时间 kernel、B 的独立训练参数没有通过固定 fold 0 筛选，保留结果。"
            "分层损失 B 的单模型五折收益较小，当前提升主要来自固定概率融合。", ""])
        lines[4:4] = selection
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
