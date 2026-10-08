"""Controlled warm-start phase / cross-subject representation experiments."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.cnn import CNNConfig
from cmi_project.cnn_data import prepare_frozen_cnn_fold
from cmi_project.cnn_training import TrainingConfig, load_cnn_checkpoint, reuse_completed_fold
from cmi_project.evaluation import evaluate_oof_frames
from cmi_project.posttraining import DistillationConfig, train_posttraining_fold
from cmi_project.preprocessing import SensorDropoutConfig
from cmi_project.representation import RepresentationConfig, training_phase_targets
from cmi_project.validation import assert_preprocessor_matches, load_fold_manifest
from run_winner_experiments import verify_training_data
from train_posttraining import routed_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/representation.json")
    parser.add_argument("--method", required=True, choices=("phase", "cross_subject_supcon",
                                                          "phase_adapter_control", "phase_adapter"))
    parser.add_argument("--fold", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/representation/pilot_v1"))
    parser.add_argument("--name", default="representation_pilot_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.fold or len(set(args.fold)) != len(args.fold) or not set(args.fold).issubset(range(5)):
        raise ValueError("Use distinct fixed folds in 0..4.")
    if Path(args.name).name != args.name or any(c in args.name for c in "\\/:"):
        raise ValueError("Use a plain experiment name.")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    base = json.loads((ROOT / settings["base_config"]).read_text(encoding="utf-8"))
    experiment = RepresentationConfig(**settings["methods"][args.method])
    source = (ROOT / base["source_run"]).resolve()
    data_dir, cache = (ROOT / base["data_dir"]).resolve(), (ROOT / base["cache_dir"]).resolve()
    output = (ROOT / args.output_dir / args.method).resolve()
    control_setting = settings.get("method_control_runs", {}).get(args.method, settings["control_run"])
    control = (ROOT / control_setting).resolve() if control_setting is not None else None
    for protected in (source, data_dir, cache, *([control] if control is not None else [])):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Use a separate experiment output directory.")
    result_path = ROOT / f"experiments/results/{args.name}_{args.method}.json"
    if (result_path.exists() or output.exists()) and not args.resume:
        raise ValueError("Preserve existing attempts; use a fresh output/name or --resume.")
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    verify_training_data(data_dir)
    locked = {"settings": settings, "base_settings": base, "method": asdict(experiment),
              "folds_sha256": manifest.fingerprint, "source_run": str(source)}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != locked:
        raise ValueError("Experiment settings changed; use a fresh attempt.")
    config_path.write_text(json.dumps(locked, indent=2), encoding="utf-8")
    frames, summaries, controls = {}, [], []
    routed_frames = {scenario: {} for scenario in ("observed", "aux_dropout50", "imu_only")}
    for fold in args.fold:
        path = source / f"imu/fold_{fold}/best.pt"
        model, processor, checkpoint = load_cnn_checkpoint(path)
        assert_preprocessor_matches(processor.state, manifest, fold)
        if (checkpoint["fold"] != fold or checkpoint["folds_sha256"] != manifest.fingerprint
                or model.model_name != "imu" or model.metadata()["architecture"] != "grouped_masked_se_cnn_v3"):
            raise ValueError("Use the original same-fold IMU checkpoint.")
        training = TrainingConfig(**{**checkpoint["training_config"], **base["training_overrides"],
                                     **settings.get("method_training_overrides", {}).get(args.method, {})})
        identity = checkpoint["data_metadata"]["identity"]
        arrays, metadata = prepare_frozen_cnn_fold(data_dir, cache / f"fold_{fold}", manifest, fold,
            processor, tof_regions=identity["tof_regions"], input_clip=identity["input_clip"])
        student_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        control_summary = None
        if control is not None:
            control_summary, _ = reuse_completed_fold(control / f"imu/fold_{fold}", processor,
                manifest, fold, "imu", CNNConfig.from_dict(checkpoint["model_metadata"]["config"]),
                training, SensorDropoutConfig(**checkpoint["sensor_dropout"]), metadata, model.tof_regions)
            provenance = control_summary["posttraining"]
            if provenance["student_sha256"] != student_hash:
                raise ValueError("Control must use identical starting weights.")
            if experiment.trainable_scope == "full":
                if provenance["method"] != "supervised_finetuning_control":
                    raise ValueError("Use the zero-KD supervised control.")
            elif RepresentationConfig(**provenance["representation"]) != RepresentationConfig(
                    **{**asdict(experiment), "weight": 0.0}):
                raise ValueError("Adapter control must differ only in phase auxiliary weight.")
        phase = training_phase_targets(data_dir / "train.csv", arrays, manifest, fold) if experiment.method == "phase" else None
        if phase is not None:
            train = manifest.table.fold.to_numpy() != fold
            phase_path = output / f"phase_targets_fold_{fold}.npz"
            if phase_path.exists():
                with np.load(phase_path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["sequence_id"], arrays["sequence_id"][train])
                    np.testing.assert_array_equal(saved["phase"], phase[train])
            else:
                np.savez_compressed(phase_path, sequence_id=arrays["sequence_id"][train], phase=phase[train])
        directory = output / f"imu/fold_{fold}"
        if args.resume and (directory / "metrics.json").exists():
            summary, frame = reuse_completed_fold(directory, processor, manifest, fold, "imu",
                model.config, training, SensorDropoutConfig(**checkpoint["sensor_dropout"]), metadata, model.tof_regions)
            _, _, completed = load_cnn_checkpoint(directory / "best.pt")
            provenance = summary["posttraining"]
            expected_phase_hash = hashlib.sha256(phase[train].tobytes()).hexdigest() if phase is not None else None
            if (completed["posttraining"] != provenance or provenance["student_sha256"] != student_hash
                    or provenance["representation"] != asdict(experiment)
                    or provenance["phase_targets_sha256"] != expected_phase_hash):
                raise ValueError("Completed representation provenance changed.")
        else:
            summary, frame = train_posttraining_fold(arrays, processor, manifest, fold, path, directory,
                training=training, distillation=DistillationConfig(weight=0), targets=None, eligible=None,
                data_metadata=metadata, source_teacher={}, representation=experiment, phase_targets=phase)
        summaries.append(summary)
        reference_score = control_summary["validation"]["score"] if control_summary else summary["baseline"]["score"]
        controls.append({"fold": fold, "score": reference_score,
                         "matched_training_control": control_summary is not None,
                         "best_epoch": control_summary["best_epoch"] if control_summary else 0,
                         "reused_from": str(control / f"imu/fold_{fold}") if control_summary else str(path),
                         "method_minus_reference": summary["validation"]["score"] - reference_score})
        frames[fold] = frame
        routed = {}
        summary["routed_with_unchanged_model_b"] = routed_metrics(arrays, manifest, fold, frame, source,
                                                                  prediction_sink=routed)
        for scenario, routed_frame in routed.items():
            routed_frames[scenario][fold] = routed_frame
        result = {"method": asdict(experiment), "folds_sha256": manifest.fingerprint,
                  "folds": args.fold, "completed_folds": sorted(frames), "results": summaries, "control": controls,
                  "scope": "Short warm-start development CV; not full retraining, independent holdout or online score",
                  "matched_control": ("Same source, scalers, folds, seed, augmentation, LR and maximum epoch budget; new heads restore CPU RNG"
                    if control is not None else "This is the adapter control arm; comparison here is only against pretrained epoch 0")}
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"COMPLETED {args.method} fold {fold}: {summary['validation']['score']:.6f}", flush=True)
        del arrays, phase, model
    result["complete_five_fold"] = set(frames) == set(range(5))
    if result["complete_five_fold"]:
        result["five_fold_evaluation"] = evaluate_oof_frames(manifest, frames, output / "imu/evaluation",
                                                           experiment_name=f"{args.name}_{args.method}")
        result["routed_five_fold_evaluation"] = {scenario: evaluate_oof_frames(manifest, predictions,
            output / "scenarios" / scenario, experiment_name=f"{args.name}_{args.method}_{scenario}")
            for scenario, predictions in routed_frames.items()}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"complete_five_fold": result["complete_five_fold"], "control": controls,
                      "five_fold_evaluation": result.get("five_fold_evaluation", {})}, indent=2), flush=True)


if __name__ == "__main__":
    main()
