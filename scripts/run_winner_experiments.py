"""Screen three fixed designs on fold 0, then validate one fixed design on all folds."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    subprocess.run([sys.executable, "-s", "-u", *map(str, arguments)], cwd=ROOT, check=True)


def execute(args):
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU experiment requires an available CUDA accelerator.")
    if args.continue_from is not None:
        source = args.continue_from.resolve()
        if hashlib.sha256((source / "configs/folds.csv").read_bytes()).digest() != hashlib.sha256((ROOT / "configs/folds.csv").read_bytes()).digest():
            raise ValueError("Continuation source uses different fixed folds.")
        if (ROOT / "outputs").exists():
            raise ValueError("Continue in a fresh workspace, preserving source artifacts.")
        shutil.copytree(source / "outputs", ROOT / "outputs")
        print("Recovered pilot outputs and caches from", source, flush=True)
    directory = ROOT / "runtime_configs"
    directory.mkdir(exist_ok=True)
    designs = ("cnn_grouped_se", "cnn_grouped_mixup", "cnn_dynamics_mixup")
    rows = []
    for name in designs:
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        config["data_dir"] = str(args.data_dir.resolve())
        config["training"]["device"] = args.device
        path = directory / f"{name}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
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
    files += list((ROOT / "runtime_configs").glob("*.json"))
    files += list((ROOT / "experiments/results").glob("*.json"))
    files += [path for path in (ROOT / "outputs/experiments").rglob("*") if path.is_file()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(files)):
            bundle.write(path, path.relative_to(ROOT).as_posix())
    print("SAVED EXPERIMENT ARTIFACTS", archive, archive.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--continue-from", type=Path, help="Previous notebook output root; reuse checked completed pilots/caches.")
    args = parser.parse_args()
    try:
        execute(args)
    finally:
        # Keep completed attempts downloadable even if a later fold fails.
        archive_results()


if __name__ == "__main__":
    main()
