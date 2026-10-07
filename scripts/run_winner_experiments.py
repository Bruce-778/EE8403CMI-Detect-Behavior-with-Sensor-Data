"""Screen three fixed designs on fold 0, then validate one fixed design on all folds."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = ("cnn_grouped_se", "cnn_grouped_mixup", "cnn_dynamics_mixup")


def verify_training_data(data_dir):
    expected = json.loads((ROOT / "configs/data_source_hashes.json").read_text())["files"]
    actual = {}
    for name, reference in expected.items():
        path = data_dir / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        if actual[name] != reference:
            raise ValueError(f"Training input differs from fixed local data: {name}.")
        print("VERIFIED TRAINING INPUT", name, actual[name]["sha256"], flush=True)
    result = {"status": "verified", "files": actual}
    (ROOT / "data_source_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def recover_frozen_pilots(source):
    if hashlib.sha256((source / "configs/folds.csv").read_bytes()).digest() != hashlib.sha256((ROOT / "configs/folds.csv").read_bytes()).digest():
        raise ValueError("Continuation source uses different fixed folds.")
    if (ROOT / "outputs").exists():
        raise ValueError("Continue in a fresh workspace, preserving source artifacts.")
    selection = json.loads((source / "selection.json").read_text())
    rows = selection["designs"]
    if (len(rows) != len(DESIGNS) or {row["name"] for row in rows} != set(DESIGNS)
            or selection["selection_fold"] != 0
            or any(row["fold"] != 0 or not math.isfinite(row["score"]) for row in rows)
            or selection["selected"] != max(rows, key=lambda row: row["score"])["name"]):
        raise ValueError("Invalid frozen pilot selection.")
    for row in rows:
        name = row["name"]
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        source_config = json.loads((source / "configs" / f"{name}.json").read_text())
        if source_config != config:
            raise ValueError(f"Frozen pilot configuration changed: {name}.")
        fold = source / config["output_dir"] / "imu/fold_0"
        metrics = json.loads((fold / "metrics.json").read_text())
        if (metrics["fold"] != 0 or metrics["model"] != "imu"
                or metrics["validation"]["score"] != row["score"]
                or any(metrics[key] != row[key] for key in ("best_epoch", "epochs_run"))):
            raise ValueError(f"Frozen pilot metrics disagree with selection: {name}.")
    # Preserve evidence separately. Cross-notebook mounts have new source timestamps;
    # rebuild caches and retrain the selected fold 0 rather than weaken cache guards.
    for row in rows:
        name = row["name"]
        shutil.copytree(source / "outputs/experiments" / name / "imu/fold_0",
                        ROOT / "outputs/pilot_artifacts" / name / "imu/fold_0")
        result = ROOT / "experiments/results" / f"{name}_gpu_pilot.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "experiments/results" / result.name, result)
    print("RECOVERED FROZEN PILOT SELECTION; rebuilding caches and all ten selected models", flush=True)
    return rows


def run(*arguments):
    subprocess.run([sys.executable, "-s", "-u", *map(str, arguments)], cwd=ROOT, check=True)


def execute(args):
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU experiment requires an available CUDA accelerator.")
    verify_training_data(args.data_dir.resolve())
    directory = ROOT / "runtime_configs"
    directory.mkdir(exist_ok=True)
    for name in DESIGNS:
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        config["data_dir"] = str(args.data_dir.resolve())
        config["training"]["device"] = args.device
        path = directory / f"{name}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.continue_from is not None:
        rows = recover_frozen_pilots(args.continue_from.resolve())
    else:
        rows = []
        for name in DESIGNS:
            path = directory / f"{name}.json"
            config = json.loads(path.read_text())
            run(ROOT / "scripts/train_cnn.py", "--config", path, "--model", "imu", "--fold", "0", "--resume")
            metrics = json.loads((ROOT / config["output_dir"] / "imu/fold_0/metrics.json").read_text())
            rows.append({"name": name, "fold": 0, "score": metrics["validation"]["score"],
                         "best_epoch": metrics["best_epoch"], "epochs_run": metrics["epochs_run"]})
            run(ROOT / "scripts/summarize_cnn_experiment.py", config["output_dir"], "--name", f"{name}_gpu_pilot")
    selected = max(rows, key=lambda row: row["score"])["name"]
    selection = {"selection_fold": 0, "designs": rows, "selected": selected,
        "reference_imu_equal_blend_fold0": 0.7463748066010997,
        "policy": "One fixed configuration chosen on fold 0, applied to A/B and all five folds; no per-fold config selection",
        "scope": "Development CV; fold 0 used for design selection; validation selects checkpoints"}
    (ROOT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("DESIGN SELECTION", json.dumps(selection), flush=True)
    path = directory / f"{selected}.json"
    run(ROOT / "scripts/train_cnn.py", "--config", path, "--model", "both", "--fold", "0", "1", "2", "3", "4", "--resume")
    run(ROOT / "scripts/summarize_cnn_experiment.py", f"outputs/experiments/{selected}", "--name", "cnn_winner_selected")
    run(ROOT / "scripts/evaluate_cnn_scenarios.py", f"outputs/experiments/{selected}", "--name", "cnn_winner_selected")
    print("COMPLETED FIVE FOLD EXPERIMENTS", flush=True)


def archive_results():
    archive = ROOT.parent / "winner_experiments.zip"
    files = [ROOT / "selection.json"] if (ROOT / "selection.json").is_file() else []
    if (ROOT / "data_source_check.json").is_file():
        files.append(ROOT / "data_source_check.json")
    files += list((ROOT / "runtime_configs").glob("*.json"))
    files += list((ROOT / "experiments/results").glob("*.json"))
    files += [path for path in (ROOT / "outputs/experiments").rglob("*") if path.is_file()]
    files += [path for path in (ROOT / "outputs/pilot_artifacts").rglob("*") if path.is_file()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(files)):
            bundle.write(path, path.relative_to(ROOT).as_posix())
    print("SAVED EXPERIMENT ARTIFACTS", archive, archive.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--continue-from", type=Path, help="Previous notebook output root; retain frozen pilot selection, rebuild caches.")
    args = parser.parse_args()
    try:
        execute(args)
    finally:
        # Keep completed attempts downloadable even if a later fold fails.
        archive_results()


if __name__ == "__main__":
    main()
