"""Save compact, Git-friendly evidence from CNN outputs (large artifacts stay local)."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.evaluation import TARGET_GESTURES
from cmi_project.validation import load_fold_manifest


def summarize(directory: Path) -> dict:
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    rows = []
    for path in sorted(directory.glob("*/fold_*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics["folds_sha256"] != manifest.fingerprint:
            raise ValueError("Experiment uses different folds.")
        history = pd.read_csv(path.parent / "history.csv")
        classes = pd.read_csv(path.parent / "class_scores.csv", index_col=0)
        confusion = pd.read_csv(path.parent / "confusion_matrix.csv", index_col=0)
        pairs = [(int(confusion.loc[a, b]), a, b) for a in TARGET_GESTURES
                 for b in TARGET_GESTURES if a != b]
        best_row = history.loc[history["epoch"].eq(metrics["best_epoch"])].iloc[0]
        rows.append({"model": metrics["model"], "fold": metrics["fold"],
            "validation": metrics["validation"], "best_epoch": metrics["best_epoch"],
            "epochs_run": metrics["epochs_run"], "max_length": metrics["max_length"],
            "training": metrics["training"], "model_metadata": metrics["model_metadata"],
            "sensor_dropout": metrics["sensor_dropout"], "versions": metrics["versions"],
            "best_epoch_train_loss": float(best_row["train_loss"]),
            "best_epoch_validation_loss": float(best_row["validation_loss"]),
            "last_epoch_train_loss": float(history.iloc[-1]["train_loss"]),
            "last_epoch_validation_loss": float(history.iloc[-1]["validation_loss"]),
            "target_class_f1": {g: float(classes.loc[g, "f1-score"]) for g in TARGET_GESTURES},
            "largest_target_confusions": [{"count": n, "truth": a, "prediction": b}
                                          for n, a, b in sorted(pairs, reverse=True)[:5]]})
    evaluations = {}
    for path in sorted(directory.glob("*/evaluation/metrics.json")):
        evaluations[path.parent.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return {"output_dir": directory.relative_to(ROOT).as_posix(),
        "folds_sha256": manifest.fingerprint, "results": rows, "five_fold_evaluation": evaluations,
        "scope": "development CV; held-out folds select checkpoints; fold 0 also used for configuration selection"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = summarize((ROOT / args.directory).resolve())
    output = ROOT / "experiments/results" / f"{args.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for row in result["results"]:
        print(f"{row['model']} fold {row['fold']}: {row['validation']['score']:.6f}; best epoch {row['best_epoch']}")
    print(output)


if __name__ == "__main__":
    main()
