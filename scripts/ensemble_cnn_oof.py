"""Fixed equal-probability CNN blending on exactly the same held-out sequences."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.evaluation import (ALL_GESTURES, PROBABILITY_COLUMNS, _checked_predictions,
    cmi_metrics, evaluate_oof_frames, write_fold_predictions)
from cmi_project.validation import load_fold_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--model", choices=("imu", "multisensor"), required=True)
    parser.add_argument("--fold", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    sources = [(ROOT / path).resolve() for path in args.runs]
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError("Need at least two distinct CNN runs.")
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    if not args.fold or len(set(args.fold)) != len(args.fold) or not set(args.fold).issubset(range(manifest.n_splits)):
        raise ValueError("Invalid fold selection.")
    output = ROOT / "outputs/experiments" / args.name
    predictions, scores = {}, []
    for fold in args.fold:
        frames = [_checked_predictions(manifest, fold, pd.read_csv(
            source / args.model / f"fold_{fold}/predictions.csv", dtype={"sequence_id": str}),
            require_fingerprint=True) for source in sources]
        reference_ids = frames[0]["sequence_id"].to_numpy()
        if any(not np.array_equal(frame["sequence_id"], reference_ids) for frame in frames):
            raise ValueError("Ensemble validation sequences do not align.")
        values = np.mean([frame[PROBABILITY_COLUMNS].to_numpy() for frame in frames], axis=0)
        values /= values.sum(axis=1, keepdims=True)
        frame = pd.DataFrame({"sequence_id": reference_ids,
            "predicted_gesture": np.asarray(ALL_GESTURES)[values.argmax(1)],
            "folds_sha256": manifest.fingerprint})
        frame = pd.concat([frame, pd.DataFrame(values, columns=PROBABILITY_COLUMNS)], axis=1)
        predictions[fold] = frame
        score = cmi_metrics(frames[0]["gesture"], frame["predicted_gesture"])
        scores.append({"fold": fold, **score})
        write_fold_predictions(manifest, fold, frame, output / args.model / f"fold_{fold}/predictions.csv")
        print(f"{args.model} equal blend fold {fold}: {score['score']:.6f}", flush=True)
    settings = {"model": args.model, "sources": [path.relative_to(ROOT).as_posix() for path in sources],
        "probability_weights": [1 / len(sources)] * len(sources), "folds_sha256": manifest.fingerprint,
        "selection_note": "equal weights fixed before five-fold comparison; no per-fold or per-sequence score selection"}
    evaluation = (evaluate_oof_frames(manifest, predictions, output / args.model / "evaluation",
        experiment_name=args.name, experiment_config=settings) if set(args.fold) == set(range(manifest.n_splits)) else None)
    target = ROOT / "experiments/results" / f"{args.name}_{args.model}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"settings": settings, "fold_scores": scores, "evaluation": evaluation},
        indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
