"""Evaluate identical held-out sequences with observed or removed auxiliary sensors."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.cnn_data import CNNTensorDataset, validate_arrays
from cmi_project.cnn_training import (CMIHierarchicalLoss, TrainingConfig, evaluate_model,
                                      load_cnn_checkpoint, seed_everything)
from cmi_project.evaluation import ALL_GESTURES, PROBABILITY_COLUMNS, cmi_metrics, evaluate_oof_frames
from cmi_project.validation import assert_preprocessor_matches, load_fold_manifest

AUXILIARY_KEYS = ("thm", "thm_valid", "thm_observed", "tof", "tof_valid", "tof_fraction", "tof_sensor_present")


class ScenarioDataset(Dataset):
    def __init__(self, arrays, indices, scenario):
        if scenario not in ("observed", "imu_only", "aux_dropout50"):
            raise ValueError("Unknown missing-sensor scenario.")
        self.base = CNNTensorDataset(arrays, indices)
        self.drop = np.array([scenario == "imu_only" or (scenario == "aux_dropout50" and
            int(hashlib.sha256(str(arrays["sequence_id"][i]).encode()).hexdigest()[:8], 16) % 2 == 0)
            for i in indices], dtype=bool)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        if self.drop[index]:
            for key in AUXILIARY_KEYS:
                sample[key].zero_()
        return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", nargs="+", type=int, help="Optional subset for a labelled pilot; default uses completed folds.")
    args = parser.parse_args()
    directory = (ROOT / args.directory).resolve()
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    seed_everything(42, 4)
    results, predictions, auxiliary_available = [], {}, {}
    candidates = {(path.parent.parent.name, path.parent.name): path
                  for path in directory.glob("*/fold_*/metrics.json")}
    if args.fold is not None:
        if not args.fold or len(set(args.fold)) != len(args.fold) or not set(args.fold).issubset(range(manifest.n_splits)):
            raise ValueError("Invalid scenario fold selection.")
        candidates = {key: path for key, path in candidates.items() if int(key[1].split("_")[-1]) in args.fold}
    identities = {}
    for _, metrics_path in sorted(candidates.items()):
        path = metrics_path.parent / "best.pt"
        model, processor, checkpoint = load_cnn_checkpoint(path)
        fold, name = checkpoint["fold"], model.model_name
        assert_preprocessor_matches(processor.state, manifest, fold)
        identity = checkpoint["data_metadata"]["identity"]
        if fold in identities and identity != identities[fold]:
            raise ValueError("Routed Model A/B must use the same fold-fitted inputs/cache identity.")
        identities[fold] = identity
        config = TrainingConfig(**checkpoint["training_config"])
        # Cache path is in run_config, and its identity must match the checkpoint.
        run = json.loads((metrics_path.parent.parent.parent / "run_config.json").read_text(encoding="utf-8"))
        cache = Path(run["cache_dir"]) / f"fold_{fold}"
        if json.loads((cache / "metadata.json").read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Checkpoint/cache identity mismatch.")
        with np.load(cache / "data.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        validate_arrays(arrays, manifest, processor.max_length, model.tof_regions)
        indices = np.flatnonzero(manifest.table["fold"].to_numpy() == fold)
        present = arrays["thm_valid"][indices].any(axis=(1, 2)) | arrays["tof_sensor_present"][indices].any(axis=(1, 2))
        for scenario in ("observed", "imu_only", "aux_dropout50"):
            dataset = ScenarioDataset(arrays, indices, scenario)
            metrics, probabilities = evaluate_model(model, DataLoader(dataset, batch_size=64),
                CMIHierarchicalLoss(config), torch.device("cpu"))
            results.append({"model": name, "fold": fold, "scenario": scenario,
                "additional_auxiliary_drop_count": int(dataset.drop.sum()), **metrics})
            frame = pd.DataFrame({"sequence_id": arrays["sequence_id"][indices],
                "predicted_gesture": np.asarray(ALL_GESTURES)[probabilities.argmax(1)],
                "folds_sha256": manifest.fingerprint})
            frame = pd.concat([frame, pd.DataFrame(probabilities, columns=PROBABILITY_COLUMNS)], axis=1)
            predictions.setdefault((name, scenario), {})[fold] = frame
            auxiliary_available[(fold, scenario)] = dict(zip(arrays["sequence_id"][indices], present & ~dataset.drop))
            print(f"{name} fold {fold} {scenario}: {metrics['score']:.6f}", flush=True)
    # Fixed availability rule; no validation labels, confidence threshold or
    # per-fold weight search participates in choosing A versus B.
    if ("imu", "observed") in predictions and ("multisensor", "observed") in predictions:
        for scenario in ("observed", "imu_only", "aux_dropout50"):
            shared = set(predictions[("imu", scenario)]) & set(predictions[("multisensor", scenario)])
            for fold in sorted(shared):
                a = predictions[("imu", scenario)][fold].set_index("sequence_id")
                b = predictions[("multisensor", scenario)][fold].set_index("sequence_id").loc[a.index]
                use_b = np.array([auxiliary_available[(fold, scenario)][sid] for sid in a.index])
                combined = a.copy()
                combined.loc[use_b] = b.loc[use_b]
                combined = combined.reset_index()
                truth = manifest.table.set_index("sequence_id").loc[combined["sequence_id"], "gesture"]
                metrics = cmi_metrics(truth, combined["predicted_gesture"])
                predictions.setdefault(("routed", scenario), {})[fold] = combined
                results.append({"model": "routed", "fold": fold, "scenario": scenario,
                    "used_model_a": int((~use_b).sum()), "used_model_b": int(use_b.sum()), **metrics})
                print(f"routed fold {fold} {scenario}: {metrics['score']:.6f}", flush=True)
    evaluations = {}
    for (name, scenario), frames in predictions.items():
        if set(frames) == set(range(manifest.n_splits)):
            evaluations[f"{name}/{scenario}"] = evaluate_oof_frames(manifest, frames,
                directory / "scenarios" / args.name / name / scenario, experiment_name=f"{args.name}_{name}_{scenario}")
    output = ROOT / "experiments/results" / f"{args.name}_scenarios.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"folds_sha256": manifest.fingerprint, "results": results,
        "five_fold_evaluation": evaluations,
        "source_runs": {"primary": str(args.directory)},
        "scenario_note": "aux_dropout50 deterministically removes THM+ToF for roughly half the same validation sequences; natural missingness is retained; this is a stress test, not the hidden test distribution"},
        ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
