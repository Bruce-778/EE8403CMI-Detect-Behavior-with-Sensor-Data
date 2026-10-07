"""One CMI metric and strict, complete OOF evaluation for all experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .dataset_analysis import PROJECT_DIR
from .validation import FoldManifest, assert_preprocessor_matches, load_fold_manifest

# Label ontology matches the supplied data/kaggle_evaluation/cmi_gateway.py.
TARGET_GESTURES = (
    "Above ear - pull hair", "Cheek - pinch skin", "Eyebrow - pull hair", "Eyelash - pull hair",
    "Forehead - pull hairline", "Forehead - scratch", "Neck - pinch skin", "Neck - scratch",
)
NON_TARGET_GESTURES = (
    "Write name on leg", "Wave hello", "Glasses on/off", "Text on phone", "Write name in air",
    "Feel around in tray and pull out an object", "Scratch knee/leg skin", "Pull air toward your face",
    "Drink from bottle/cup", "Pinch knee/leg skin",
)
ALL_GESTURES = TARGET_GESTURES + NON_TARGET_GESTURES
PROBABILITY_COLUMNS = [f"prob_{label}" for label in ALL_GESTURES]
METRIC_NAME = "cmi_binary_f1_macro_f1_9class_v1"


def cmi_metrics(truth, prediction) -> dict[str, float]:
    truth, prediction = np.asarray(truth, dtype=str), np.asarray(prediction, dtype=str)
    if truth.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError("Expected equally sized, non-empty label vectors.")
    if not set(truth).issubset(ALL_GESTURES) or not set(prediction).issubset(ALL_GESTURES):
        raise ValueError("Unknown gesture in truth/prediction.")
    binary_truth, binary_prediction = np.isin(truth, TARGET_GESTURES), np.isin(prediction, TARGET_GESTURES)
    binary_f1 = f1_score(binary_truth, binary_prediction, zero_division=0)
    collapsed_truth = np.where(binary_truth, truth, "non_target")
    collapsed_prediction = np.where(binary_prediction, prediction, "non_target")
    macro_f1 = f1_score(collapsed_truth, collapsed_prediction,
                        labels=[*TARGET_GESTURES, "non_target"], average="macro", zero_division=0)
    return {"score": float((binary_f1 + macro_f1) / 2), "binary_f1": float(binary_f1),
            "macro_f1_9class": float(macro_f1)}


def _checked_predictions(manifest: FoldManifest, fold: int, predictions: pd.DataFrame,
                         *, require_fingerprint: bool) -> pd.DataFrame:
    _, validation = manifest.split(fold)
    if not {"sequence_id", "predicted_gesture"}.issubset(predictions.columns):
        raise ValueError("Predictions require sequence_id and predicted_gesture.")
    if predictions[["sequence_id", "predicted_gesture"]].isna().any().any():
        raise ValueError("Null OOF sequence ID/prediction.")
    predictions = predictions.copy()
    predictions["sequence_id"] = predictions["sequence_id"].astype(str)
    if predictions["sequence_id"].duplicated().any():
        raise ValueError("Duplicate OOF sequence_id.")
    if set(predictions["sequence_id"]) != set(validation["sequence_id"]):
        raise ValueError(f"Fold {fold} OOF coverage differs from fixed validation samples.")
    if not set(predictions["predicted_gesture"]).issubset(ALL_GESTURES):
        raise ValueError("Unknown predicted gesture.")
    if require_fingerprint and "folds_sha256" not in predictions:
        raise ValueError("OOF predictions must include the fixed fold fingerprint.")
    if "folds_sha256" in predictions and not predictions["folds_sha256"].eq(manifest.fingerprint).all():
        raise ValueError("OOF fold fingerprint mismatch.")
    expected = validation.set_index("sequence_id")
    aligned = expected.loc[predictions["sequence_id"]]
    for key in ("subject", "gesture", "fold"):
        if key in predictions and not np.array_equal(predictions[key].to_numpy(), aligned[key].to_numpy()):
            raise ValueError(f"OOF {key} disagrees with fixed folds.")
    probability_columns = [c for c in predictions if c.startswith("prob_")]
    if probability_columns:
        if set(probability_columns) != set(PROBABILITY_COLUMNS):
            raise ValueError("OOF probabilities must cover all 18 named gesture classes.")
        values = predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Invalid OOF probabilities; expected finite values in [0,1] summing to 1.")
    result = aligned.reset_index()[["sequence_id", "subject", "gesture", "fold"]].copy()
    result["predicted_gesture"] = predictions["predicted_gesture"].to_numpy()
    result["folds_sha256"] = manifest.fingerprint
    for key in PROBABILITY_COLUMNS if probability_columns else []:
        result[key] = predictions[key].to_numpy()
    return result.sort_values("sequence_id").reset_index(drop=True)


def write_fold_predictions(manifest: FoldManifest, fold: int, predictions: pd.DataFrame,
                           output_path: Path, *, preprocessor_state: dict | None = None) -> None:
    """Save predictions from the HELD-OUT fold model, aligning by ID not row order.

    Provide the fitted preprocessor state to verify its actual train IDs and
    subject groups. Training code remains responsible for using that same split.
    """
    if preprocessor_state is not None:
        assert_preprocessor_matches(preprocessor_state, manifest, fold)
    result = _checked_predictions(manifest, fold, predictions, require_fingerprint=False)
    output_path = Path(output_path)
    if output_path.resolve() in (manifest.path, manifest.path.with_suffix(".meta.json")):
        raise ValueError("Predictions cannot overwrite fixed folds.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")


def evaluate_oof(manifest: FoldManifest, prediction_dir: Path, output_dir: Path,
                 *, experiment_name: str) -> dict:
    """Require all five validation predictions, then export OOF and fold scores."""
    predictions = {fold: pd.read_csv(Path(prediction_dir) / f"fold_{fold}.csv",
        dtype={"sequence_id": "string", "subject": "string"}) for fold in range(manifest.n_splits)}
    return evaluate_oof_frames(manifest, predictions, output_dir, experiment_name=experiment_name)


def evaluate_oof_frames(manifest: FoldManifest, predictions: dict[int, pd.DataFrame], output_dir: Path,
                        *, experiment_name: str, save_oof: bool = True, experiment_config: dict | None = None) -> dict:
    """Apply the same coverage/metric checks to in-memory or saved predictions."""
    if set(predictions) != set(range(manifest.n_splits)):
        raise ValueError("Predictions must include every fixed validation fold exactly once.")
    parts, rows, probability_schema = [], [], None
    for fold in range(manifest.n_splits):
        checked = _checked_predictions(manifest, fold, predictions[fold], require_fingerprint=True)
        schema = tuple(c for c in checked if c.startswith("prob_"))
        if probability_schema is not None and schema != probability_schema:
            raise ValueError("All folds must export the same probability schema.")
        probability_schema = schema
        rows.append({"fold": fold, "validation_sequences": len(checked),
                     **cmi_metrics(checked["gesture"], checked["predicted_gesture"])})
        parts.append(checked)
    oof = pd.concat(parts, ignore_index=True).sort_values("sequence_id").reset_index(drop=True)
    if oof["sequence_id"].duplicated().any() or set(oof["sequence_id"]) != set(manifest.table["sequence_id"]):
        raise ValueError("OOF must cover every fixed validation sequence exactly once.")
    scores = pd.DataFrame(rows)
    metrics = ["score", "binary_f1", "macro_f1_9class"]
    summary = {"experiment_name": experiment_name, "metric": METRIC_NAME,
               "folds_sha256": manifest.fingerprint, "dataset_sha256": manifest.metadata["dataset_sha256"],
               "n_splits": manifest.n_splits, "oof_sequences": len(oof), "std_ddof": 1,
               "fold_mean": scores[metrics].mean().to_dict(), "fold_std": scores[metrics].std(ddof=1).to_dict(),
               "oof": cmi_metrics(oof["gesture"], oof["predicted_gesture"])}
    if experiment_config is not None:
        summary["experiment_config"] = experiment_config
    output_dir = Path(output_dir).resolve()
    protected = {manifest.path, manifest.path.with_suffix(".meta.json")}
    if any(output_dir / filename in protected for filename in ("oof_predictions.csv", "fold_scores.csv", "metrics.json")):
        raise ValueError("Evaluation cannot overwrite fixed folds.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_oof:
        oof.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(output_dir / "fold_scores.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate complete OOF with one shared CMI metric.")
    parser.add_argument("--folds-path", type=Path, default=PROJECT_DIR / "configs/folds.csv")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    try:
        summary = evaluate_oof(load_fold_manifest(args.folds_path), args.prediction_dir,
                               args.output_dir, experiment_name=args.experiment_name)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"CMI score: {summary['fold_mean']['score']:.6f} +/- {summary['fold_std']['score']:.6f} ({summary['n_splits']}-fold sample std)")
    print(f"Complete OOF score: {summary['oof']['score']:.6f}; saved to {args.output_dir}")
    return 0
