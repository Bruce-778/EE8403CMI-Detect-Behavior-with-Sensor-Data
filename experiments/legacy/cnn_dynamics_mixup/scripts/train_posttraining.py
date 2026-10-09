"""Run matched fine-tuning/control and multisensor-to-IMU distillation attempts."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.cnn_data import prepare_frozen_cnn_fold
from cmi_project.cnn_training import (TrainingConfig, choose_device, seed_everything,
    reuse_completed_fold, load_cnn_checkpoint)
from cmi_project.cnn import CNNConfig
from cmi_project.preprocessing import SensorDropoutConfig
from cmi_project.evaluation import (PROBABILITY_COLUMNS, _checked_predictions,
                                   cmi_metrics, evaluate_oof_frames)
from cmi_project.posttraining import (DistillationConfig, checked_pair,
    teacher_training_targets, train_posttraining_fold)
from cmi_project.validation import load_fold_manifest
from run_winner_experiments import verify_training_data


def routed_metrics(arrays, manifest, fold, student_frame, source_run, *, prediction_sink=None, updated_model="imu"):
    """Reuse unchanged Model B; evaluate new A under identical missingness rules."""
    if updated_model not in ("imu", "multisensor"):
        raise ValueError("Unknown updated branch.")
    a_frame = student_frame if updated_model == "imu" else pd.read_csv(source_run / f"imu/fold_{fold}/predictions.csv")
    b_frame = student_frame if updated_model == "multisensor" else pd.read_csv(source_run / f"multisensor/fold_{fold}/predictions.csv")
    a = _checked_predictions(manifest, fold, a_frame,
                             require_fingerprint=True).set_index("sequence_id")
    b = _checked_predictions(manifest, fold,
        b_frame,
        require_fingerprint=True).set_index("sequence_id").loc[a.index]
    a = a.astype({key: np.float64 for key in PROBABILITY_COLUMNS})
    b = b.astype({key: np.float64 for key in PROBABILITY_COLUMNS})
    positions = pd.Series(np.arange(len(manifest.table)), index=arrays["sequence_id"]).loc[a.index].to_numpy()
    time = arrays["time_mask"][positions]
    available = ((arrays["thm_valid"][positions] & time[..., None]).any(axis=(1, 2))
                 | (arrays["tof_sensor_present"][positions] & time[..., None]).any(axis=(1, 2)))
    dropout = np.array([int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16) % 2 == 0 for sid in a.index])
    result = {}
    for scenario, use_b in (("observed", available), ("aux_dropout50", available & ~dropout),
                            ("imu_only", np.zeros_like(available))):
        frame = a.copy()
        frame.loc[use_b] = b.loc[use_b]
        result[scenario] = {**cmi_metrics(frame["gesture"], frame["predicted_gesture"]),
                            "used_model_a": int((~use_b).sum()), "used_model_b": int(use_b.sum())}
        if prediction_sink is not None:
            prediction_sink[scenario] = frame.reset_index()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining.json")
    parser.add_argument("--fold", type=int, nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse only completed arms with identical inputs/settings; never overwrite partial training.")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    folds = args.fold or settings["folds"]
    if len(set(folds)) != len(folds) or not set(folds).issubset(range(5)):
        raise ValueError("Choose distinct folds in 0..4.")
    source = (ROOT / (args.source_run or settings["source_run"])).resolve()
    cache = (ROOT / (args.cache_dir or settings["cache_dir"])).resolve()
    output = (ROOT / (args.output_dir or settings["output_dir"])).resolve()
    name = args.name or settings["name"]
    if Path(name).name != name or not name or any(c in name for c in "\\/:"):
        raise ValueError("Use a plain experiment name.")
    data_dir = (ROOT / settings["data_dir"]).resolve()
    for path in (cache, output):
        if path == data_dir or data_dir in path.parents or path == source or source in path.parents:
            raise ValueError("Post-training outputs must be separate from raw data/source weights.")
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError("Keep output/cache directories separate.")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Use a fresh post-training output directory.")
    result_root = ROOT / "experiments/results"
    if any((result_root / f"{name}_{arm}.json").exists() for arm in ("finetune", "distill", "comparison")) and not args.resume:
        raise ValueError("Preserve prior evidence: choose a fresh experiment name.")
    verify_training_data(data_dir)
    distillation = DistillationConfig(**settings["distillation"])
    if distillation.weight <= 0:
        raise ValueError("Paired experiment needs a positive distillation weight.")
    summaries, predictions = {arm: [] for arm in ("finetune", "distill")}, {arm: {} for arm in ("finetune", "distill")}
    routed_predictions = {arm: {scenario: {} for scenario in ("observed", "aux_dropout50", "imu_only")}
                          for arm in ("baseline", "finetune", "distill")}
    for fold in folds:
        student_path = source / f"imu/fold_{fold}/best.pt"
        teacher_path = source / f"multisensor/fold_{fold}/best.pt"
        _, teacher, processor, checkpoint = checked_pair(student_path, teacher_path, manifest, fold)
        training_values = {**checkpoint["training_config"], **settings["training_overrides"]}
        if args.device:
            training_values["device"] = args.device
        training = TrainingConfig(**training_values)
        device = choose_device(training.device)
        seed_everything(training.seed + fold, training.cpu_threads)
        identity = checkpoint["data_metadata"]["identity"]
        arrays, metadata = prepare_frozen_cnn_fold(data_dir, cache / f"fold_{fold}", manifest, fold,
            processor, tof_regions=identity["tof_regions"], input_clip=identity["input_clip"])
        targets, eligible = teacher_training_targets(teacher, arrays, manifest, fold,
            batch_size=training.batch_size, device=device)
        teacher_source = {"teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                          "teacher_fold": fold, "teacher_source": str(teacher_path)}
        target_directory = output / f"teacher_targets/fold_{fold}"
        target_directory.mkdir(parents=True, exist_ok=args.resume)
        train = manifest.table["fold"].to_numpy() != fold
        target_path = target_directory / "training_targets.npz"
        if target_path.exists():
            with np.load(target_path, allow_pickle=False) as saved:
                for key, expected in (("sequence_id", arrays["sequence_id"][train]),
                                      ("logits", targets[train]), ("eligible", eligible[train])):
                    np.testing.assert_array_equal(saved[key], expected)
        else:
            np.savez_compressed(target_path, sequence_id=arrays["sequence_id"][train],
                                logits=targets[train], eligible=eligible[train])
        for arm, weight in (("finetune", 0.0), ("distill", distillation.weight)):
            directory = output / arm
            directory.mkdir(parents=True, exist_ok=True)
            run_config = {"cache_dir": str(cache),
                "settings": settings, "folds": folds, "folds_sha256": manifest.fingerprint,
                "training": asdict(training)}
            config_path = directory / "run_config.json"
            if config_path.exists():
                previous = json.loads(config_path.read_text(encoding="utf-8"))
                # Older completed runs omit newly introduced default loss weights.
                previous["training"] = asdict(TrainingConfig(**previous["training"]))
                if previous != run_config:
                    raise ValueError("Completed run settings changed; use a fresh experiment name/directory.")
            if not config_path.exists():
                config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
            fold_directory = directory / f"imu/fold_{fold}"
            kd = DistillationConfig(distillation.temperature, weight)
            if args.resume and (fold_directory / "metrics.json").is_file():
                summary, frame = reuse_completed_fold(fold_directory, processor, manifest, fold, "imu",
                    CNNConfig.from_dict(checkpoint["model_metadata"]["config"]), training,
                    SensorDropoutConfig(**checkpoint["sensor_dropout"]), metadata, identity["tof_regions"])
                _, _, completed = load_cnn_checkpoint(fold_directory / "best.pt")
                provenance = summary["posttraining"]
                if (completed["posttraining"] != provenance or provenance["distillation"] != asdict(kd)
                        or provenance["student_sha256"] != hashlib.sha256(student_path.read_bytes()).hexdigest()
                        or any(provenance[key] != value for key, value in teacher_source.items())):
                    raise ValueError("Completed post-training teacher/student provenance changed.")
            else:
                summary, frame = train_posttraining_fold(arrays, processor, manifest, fold, student_path,
                    fold_directory, training=training, distillation=kd, targets=targets,
                    eligible=eligible, data_metadata=metadata, source_teacher=teacher_source)
            routed = {}
            summary["routed_with_unchanged_model_b"] = routed_metrics(arrays, manifest, fold, frame, source,
                                                                      prediction_sink=routed)
            for scenario, routed_frame in routed.items():
                routed_predictions[arm][scenario][fold] = routed_frame
            original_a = pd.read_csv(source / f"imu/fold_{fold}/predictions.csv")
            baseline_routed = {}
            summary["baseline_routed_with_unchanged_model_b"] = routed_metrics(arrays, manifest, fold, original_a, source,
                                                                               prediction_sink=baseline_routed)
            for scenario, routed_frame in baseline_routed.items():
                routed_predictions["baseline"][scenario][fold] = routed_frame
            summaries[arm].append(summary)
            predictions[arm][fold] = frame
            result_root.mkdir(parents=True, exist_ok=True)
            (result_root / f"{name}_{arm}.json").write_text(json.dumps({"folds": folds,
                "completed_folds": sorted(predictions[arm]), "results": summaries[arm]}, indent=2, allow_nan=False), encoding="utf-8")
            print(f"COMPLETED_ARM {arm} fold {fold}: {summary['validation']['score']:.6f}", flush=True)
        del arrays, targets, teacher
    comparison = [{"fold": a["fold"], "baseline": a["baseline"]["score"],
        "finetune": a["validation"]["score"], "distill": b["validation"]["score"],
        "kd_minus_finetune": b["validation"]["score"] - a["validation"]["score"],
        "kd_minus_baseline": b["validation"]["score"] - a["baseline"]["score"]}
        for a, b in zip(summaries["finetune"], summaries["distill"])]
    pd.DataFrame(comparison).to_csv(output / "fold_scores.csv", index=False)
    complete = set(folds) == set(range(5))
    evaluations = {arm: evaluate_oof_frames(manifest, frames, output / arm / "imu/evaluation", experiment_name=f"{name}_{arm}")
                   for arm, frames in predictions.items()} if complete else {}
    routed_evaluations = {arm: {scenario: evaluate_oof_frames(manifest, frames,
        output / "scenarios" / arm / scenario, experiment_name=f"{name}_{arm}_{scenario}")
        for scenario, frames in scenarios.items()} for arm, scenarios in routed_predictions.items()} if complete else {}
    result = {"folds_sha256": manifest.fingerprint, "folds": folds, "complete_five_fold": complete,
              "comparison": comparison, "five_fold_evaluation": evaluations,
              "routed_five_fold_evaluation": routed_evaluations,
              "scope": ("Development CV; parameters frozen after fold-0 screening; not online scores" if complete else
                        "Development CV; fixed fold-0 pilot is screening, not five-fold/online improvement"),
              "policy": "Same initialization, train/validation IDs, scalers, seed, augmentation and epoch budget; only KD weight differs"}
    (result_root / f"{name}_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
