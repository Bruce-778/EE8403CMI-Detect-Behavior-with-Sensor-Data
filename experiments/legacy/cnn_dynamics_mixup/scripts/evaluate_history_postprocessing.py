"""Evaluation-only causal subject history prototype; no training or submission."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.evaluation import (ALL_GESTURES, PROBABILITY_COLUMNS, _checked_predictions,
    evaluate_oof_frames, write_fold_predictions)
from cmi_project.validation import load_fold_manifest


def training_capacities(table, fold, labels=ALL_GESTURES):
    """Maximum observed gesture count per training subject, never validation labels."""
    train, validation = table[table.fold != fold], table[table.fold == fold]
    if train.empty or validation.empty or not table.sequence_id.is_unique:
        raise ValueError("Require a unique fixed sequence index and nonempty split.")
    if set(train.subject) & set(validation.subject):
        raise ValueError("Training and validation subjects overlap.")
    if not set(train.gesture).issubset(labels):
        raise ValueError("Unknown training gesture.")
    counts = train.groupby(["subject", "gesture"]).size().unstack(fill_value=0)
    maximum = counts.reindex(columns=list(labels), fill_value=0).max().to_numpy(dtype=np.int64)
    if (maximum < 1).any():
        raise ValueError("Every gesture needs training support; do not infer a quota from validation.")
    return maximum


class CausalHistoryAssignment:
    """Re-optimize arrived probability rows; return ONLY the new row's assignment.

    Old returned labels are immutable. No orientation, behavior, true gesture,
    future probabilities or final subject sequence count enter this predictor.
    """
    def __init__(self, capacities):
        counts = np.asarray(capacities)
        if counts.ndim != 1 or not len(counts) or not np.issubdtype(counts.dtype, np.integer) or (counts < 1).any():
            raise ValueError("Use positive integer capacities.")
        self.slots = np.repeat(np.arange(len(counts)), counts)
        self.classes = len(counts)
        self.history = {}
        self.overflow = 0

    def predict_one(self, subject, probability):
        p = np.asarray(probability, dtype=np.float64)
        if p.shape != (self.classes,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
            raise ValueError("Invalid probability vector.")
        p = p / p.sum()
        arrived = self.history.setdefault(str(subject), [])
        if len(arrived) >= len(self.slots):
            self.overflow += 1
            return int(p.argmax())  # Keep every sequence; unsupported prefixes use baseline.
        arrived.append(p.copy())
        cost = -np.log(np.maximum(np.stack(arrived)[:, self.slots], 1e-12))
        row, column = linear_sum_assignment(cost)
        chosen = column[row == len(arrived) - 1]
        if len(chosen) != 1:
            raise ValueError("The new row was not assigned.")
        return int(self.slots[chosen[0]])


def decode_fold(frame, capacities, seed, *, labels=ALL_GESTURES, probability_columns=PROBABILITY_COLUMNS):
    """Arrival order is fixed without labels; frame metadata is for reporting only."""
    ordered = frame.sort_values("sequence_id").reset_index(drop=True)
    probabilities = ordered[list(probability_columns)].to_numpy(dtype=float)
    order = np.random.default_rng(seed).permutation(len(ordered))
    decoder = CausalHistoryAssignment(capacities)
    predicted = np.empty(len(ordered), dtype=np.int64)
    positions = np.empty(len(ordered), dtype=np.int64)
    for position, index in enumerate(order):
        predicted[index] = decoder.predict_one(ordered.subject.iloc[index], probabilities[index])
        positions[index] = position
    # Decoding changes hard labels, not calibrated probabilities. Do not attach
    # unchanged probabilities whose argmax no longer matches the returned label.
    result = pd.DataFrame({"sequence_id": ordered.sequence_id,
                          "predicted_gesture": np.asarray(labels)[predicted],
                          "folds_sha256": ordered.folds_sha256, "arrival_position": positions})
    return result, decoder.overflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/winner_comparison/candidate/routed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/history_postprocessing/pilot_v1"))
    parser.add_argument("--name", default="history_postprocessing_v1")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or any(c in args.name for c in "\\/:"):
        raise ValueError("Use a plain experiment name.")
    source, output = (ROOT / args.source).resolve(), (ROOT / args.output).resolve()
    data_dir = (ROOT / "data").resolve()
    if (output == source or source in output.parents or output in source.parents
            or output == data_dir or data_dir in output.parents or output in data_dir.parents):
        raise ValueError("Keep this evaluation separate from source results and raw data.")
    result_path = ROOT / f"experiments/results/{args.name}.json"
    if result_path.exists() or (output.exists() and any(output.iterdir())):
        raise ValueError("Use a fresh experiment name and output; preserve all prior attempts.")
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    output.mkdir(parents=True, exist_ok=True)
    seeds = [42, 142, 242]  # Predeclared orders; report all, never choose the best.
    capacities = {fold: training_capacities(manifest.table, fold) for fold in range(5)}
    result = {"folds_sha256": manifest.fingerprint, "method": "causal_gesture_capacity_assignment",
              "scope": "Competition-protocol development OOF; information differs from independent single-sequence inference",
              "seeds": seeds, "source": str(source), "scenarios": {},
              "constraints": {str(fold): dict(zip(ALL_GESTURES, map(int, values))) for fold, values in capacities.items()},
              "policy": "Train-fold class maxima only; past arrived probabilities only; no true orientation/behavior; no future rows; no training or submission"}
    for scenario in ("observed", "aux_dropout50", "imu_only"):
        path = source / scenario / "oof_predictions.csv"
        source_frame = pd.read_csv(path)
        checked = {fold: _checked_predictions(manifest, fold, source_frame[source_frame.fold == fold],
                                              require_fingerprint=True) for fold in range(5)}
        baseline_frames = {fold: frame.drop(columns=PROBABILITY_COLUMNS) for fold, frame in checked.items()}
        baseline = evaluate_oof_frames(manifest, baseline_frames, output / scenario / "baseline",
                                       experiment_name=f"{args.name}_{scenario}_baseline")
        baseline["fold_scores"] = pd.read_csv(output / scenario / "baseline/fold_scores.csv").to_dict("records")
        evaluations, changes = {}, []
        for seed in seeds:
            frames, overflow, changed = {}, 0, 0
            for fold, frame in checked.items():
                decoded, count = decode_fold(frame, capacities[fold], seed + fold)
                prediction_path = output / scenario / f"seed_{seed}/fold_{fold}/predictions.csv"
                write_fold_predictions(manifest, fold, decoded, prediction_path)
                decoded[["sequence_id", "arrival_position"]].to_csv(
                    prediction_path.with_name("arrival_order.csv"), index=False)
                decoded = _checked_predictions(manifest, fold, decoded, require_fingerprint=True)
                frames[fold] = decoded
                overflow += count
                original = frame.set_index("sequence_id").predicted_gesture
                changed += int((decoded.predicted_gesture.to_numpy() != original.loc[decoded.sequence_id].to_numpy()).sum())
            evaluation = evaluate_oof_frames(manifest, frames, output / scenario / f"seed_{seed}/evaluation",
                                             experiment_name=f"{args.name}_{scenario}_{seed}")
            delta = evaluation["fold_mean"]["score"] - baseline["fold_mean"]["score"]
            changes.append(delta)
            evaluations[str(seed)] = {"metrics": evaluation, "changed_predictions": changed,
                                      "fold_scores": pd.read_csv(output / scenario / f"seed_{seed}/evaluation/fold_scores.csv").to_dict("records"),
                                      "overflow_baseline_fallbacks": overflow, "delta_fold_mean": delta}
            print(f"{scenario} seed={seed}: mean={evaluation['fold_mean']['score']:.6f}, delta={delta:+.6f}, changed={changed}, overflow={overflow}", flush=True)
        result["scenarios"][scenario] = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "baseline": baseline, "orders": evaluations,
            "mean_delta_across_orders": float(np.mean(changes)),
            "min_delta_across_orders": float(min(changes)), "max_delta_across_orders": float(max(changes))}
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    result["decision"] = "Evaluation prototype only; no production promotion or online score claim. Review all arrival orders and protocols before deciding."
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Completed all three scenarios and all three predeclared arrival orders.", flush=True)


if __name__ == "__main__":
    main()
