"""Train the frozen selected design on all five subject folds and export results."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SELECTED = "cnn_dynamics_mixup"


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
    (ROOT / "data_source_check.json").write_text(
        json.dumps({"status": "verified", "files": actual}, indent=2), encoding="utf-8")


def run(*arguments):
    subprocess.run([sys.executable, "-s", "-u", *map(str, arguments)], cwd=ROOT, check=True)


def execute(args):
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU experiment requires an available CUDA accelerator.")
    verify_training_data(args.data_dir.resolve())
    config = json.loads((ROOT / "configs" / f"{SELECTED}.json").read_text())
    config["data_dir"] = str(args.data_dir.resolve())
    config["training"]["device"] = args.device
    directory = ROOT / "runtime_configs"
    directory.mkdir(exist_ok=True)
    path = directory / f"{SELECTED}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    selection = {"selected": SELECTED,
        "policy": "Frozen selected configuration; same A/B design for all five fixed subject folds",
        "scope": "Development CV; the original design was selected on fold 0 and validation selects checkpoints"}
    (ROOT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    arguments = ["--config", path, "--model", "both", "--fold", "0", "1", "2", "3", "4"]
    if args.resume:
        arguments.append("--resume")
    run(ROOT / "scripts/train_cnn.py", *arguments)
    run(ROOT / "scripts/summarize_cnn_experiment.py", config["output_dir"], "--name", "cnn_winner_selected")
    run(ROOT / "scripts/evaluate_cnn_scenarios.py", config["output_dir"], "--name", "cnn_winner_selected")
    print("COMPLETED SELECTED FIVE FOLD EXPERIMENT", flush=True)


def archive_results():
    archive = ROOT.parent / "winner_experiments.zip"
    files = [ROOT / name for name in ("selection.json", "data_source_check.json") if (ROOT / name).is_file()]
    files += list((ROOT / "runtime_configs").glob(f"{SELECTED}.json"))
    files += [ROOT / "configs" / name for name in ("folds.csv", "folds.meta.json", "data_source_hashes.json", "cnn_dynamics_mixup.json", "preprocessing_dynamics.json")]
    files += [ROOT / "experiments/results" / name for name in ("cnn_winner_selected.json", "cnn_winner_selected_scenarios.json") if (ROOT / "experiments/results" / name).is_file()]
    files += [path for path in (ROOT / "outputs/experiments" / SELECTED).rglob("*") if path.is_file()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(files)):
            bundle.write(path, path.relative_to(ROOT).as_posix())
    print("SAVED EXPERIMENT ARTIFACTS", archive, archive.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Reuse completed folds only when source/cache/settings match.")
    args = parser.parse_args()
    try:
        execute(args)
    finally:
        # A partial archive preserves completed work; the importer rejects missing folds.
        archive_results()


if __name__ == "__main__":
    main()
