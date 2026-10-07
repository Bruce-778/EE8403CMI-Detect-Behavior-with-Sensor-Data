"""Recompute paired five-fold scores from predictions, then compare with the old ensemble."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.evaluation import evaluate_oof_frames
from cmi_project.validation import load_fold_manifest


def checked_scenario(manifest, path, output):
    frame = pd.read_csv(path, dtype={"sequence_id": str})
    summary = evaluate_oof_frames(manifest, {fold: frame.loc[frame["fold"] == fold]
        for fold in range(5)}, output, experiment_name=path.parent.name)
    scores = pd.read_csv(output / "fold_scores.csv").sort_values("fold")
    return summary, scores["score"].to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed selected experiment directory")
    args = parser.parse_args()
    directory = (ROOT / args.run).resolve()
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    old_root = ROOT / "outputs/experiments/cnn_hierarchical/scenarios/cnn_final_selected"
    new_root = directory / "scenarios/cnn_winner_selected"
    checks = ROOT / "outputs/winner_comparison"
    rows = []
    for model, scenario in (("imu", "observed"), ("multisensor", "observed"),
                            ("routed", "observed"), ("routed", "aux_dropout50"), ("routed", "imu_only")):
        old, old_folds = checked_scenario(manifest, old_root / model / scenario / "oof_predictions.csv",
            checks / "reference" / model / scenario)
        new, new_folds = checked_scenario(manifest, new_root / model / scenario / "oof_predictions.csv",
            checks / "candidate" / model / scenario)
        row = {"model": model, "scenario": scenario, "reference": old, "candidate": new,
            "reference_fold_scores": old_folds.tolist(), "candidate_fold_scores": new_folds.tolist(),
            "paired_fold_deltas": (new_folds - old_folds).tolist(),
            "mean_delta": float(np.mean(new_folds - old_folds)), "improved_folds": int((new_folds > old_folds).sum())}
        rows.append(row)
        print(f"{model}/{scenario}: {old['fold_mean']['score']:.6f} -> {new['fold_mean']['score']:.6f} "
              f"({row['mean_delta']:+.6f}, improved {row['improved_folds']}/5 folds)", flush=True)
    result = {"folds_sha256": manifest.fingerprint, "candidate_run": directory.name,
        "metric_scope": "Paired development CV on the exact same 8,151 sequences; checkpoint/design selection uses validation",
        "rows": rows}
    destination = ROOT / "experiments/results/cnn_winner_comparison.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lines = ["# 前排方案优化的固定五折比较", "", result["metric_scope"], "",
        "| 输入 / 模型 | 原方案均值 ± 标准差 | 新方案均值 ± 标准差 | 均值变化 | 提升折数 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        old, new = row["reference"], row["candidate"]
        lines.append(f"| {row['scenario']} / {row['model']} | {old['fold_mean']['score']:.6f} ± {old['fold_std']['score']:.6f} | "
            f"{new['fold_mean']['score']:.6f} ± {new['fold_std']['score']:.6f} | {row['mean_delta']:+.6f} | {row['improved_folds']}/5 |")
    lines += ["", "| 输入 / 模型 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | Pooled OOF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['scenario']} / {row['model']} | " + " | ".join(f"{score:.6f}" for score in row["candidate_fold_scores"])
                     + f" | {row['candidate']['oof']['score']:.6f} |")
    lines += ["", "aux_dropout50 为标签无关的固定缺失压力测试。上述结果为验证分数，不能当作新的线上比赛分数。", ""]
    (ROOT / "experiments/WINNER_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
