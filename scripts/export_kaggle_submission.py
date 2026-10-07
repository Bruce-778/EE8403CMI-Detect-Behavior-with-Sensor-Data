"""Export a private-upload-ready, offline CMI notebook and frozen CNN bundle."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.cnn_training import load_cnn_checkpoint
from cmi_project.evaluation import ALL_GESTURES
from cmi_project.validation import assert_preprocessor_matches, load_fold_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kaggle_submission"))
    args = parser.parse_args()
    output = ROOT / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh export directory to preserve existing uploads.")
    bundle = output / "bundle"
    bundle.mkdir(parents=True)
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    members = []
    for run in ("cnn_v1", "cnn_hierarchical"):
        for name in ("imu", "multisensor"):
            for fold in range(5):
                source = ROOT / "outputs/experiments" / run / name / f"fold_{fold}/best.pt"
                if not source.with_name("metrics.json").is_file():
                    raise ValueError(f"Incomplete source checkpoint: {source}")
                _, processor, checkpoint = load_cnn_checkpoint(source)
                assert_preprocessor_matches(processor.state, manifest, fold)
                if checkpoint["fold"] != fold or checkpoint["folds_sha256"] != manifest.fingerprint:
                    raise ValueError("Source fold fingerprint mismatch.")
                state = deepcopy(checkpoint["preprocessor"])
                # Inference does not need training IDs or local machine paths.
                state.pop("train_sequence_ids", None)
                state.pop("train_subjects", None)
                compact = {key: checkpoint[key] for key in
                           ("state_dict", "label_order", "fold", "folds_sha256", "model_metadata")}
                compact.update(version=1, preprocessor=state,
                    input_clip=checkpoint["data_metadata"]["identity"]["input_clip"])
                destination = bundle / "weights" / run / name / f"fold_{fold}.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(compact, destination)
                members.append({"path": destination.relative_to(bundle).as_posix(), "source": run,
                    "model": name, "fold": fold, "weight_within_branch": 0.5,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    code = bundle / "src/cmi_project"
    code.mkdir(parents=True)
    for source in (ROOT / "src/cmi_project").glob("*.py"):
        shutil.copy2(source, code / source.name)
    selected = json.loads((ROOT / "experiments/results/cnn_final_selected_scenarios.json").read_text(encoding="utf-8"))
    evaluation = selected["five_fold_evaluation"]["routed/observed"]
    if evaluation["folds_sha256"] != manifest.fingerprint or evaluation["oof_sequences"] != len(manifest.table):
        raise ValueError("Selected local results do not match the exported folds.")
    commit = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    metadata = {"version": 1, "label_order": list(ALL_GESTURES), "folds": list(range(5)),
        "folds_sha256": manifest.fingerprint, "members": members,
        "fold_probability_weights": [0.2] * 5,
        "routing": "THM valid or ToF hardware present => B; otherwise A",
        "source_git_commit": commit,
        "local_validation_score": {"mean": evaluation["fold_mean"]["score"], "std": evaluation["fold_std"]["score"]},
        "note": "Local validation score is not a Kaggle leaderboard score. No test fitting or label access."}
    (bundle / "bundle.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output / "cmi-cnn-selected.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stored:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                stored.write(path, path.relative_to(bundle).as_posix())
    code_cell = '''import os
import sys
import zipfile
from pathlib import Path

competition = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
archives = list(Path("/kaggle/input").rglob("cmi-cnn-selected.zip"))
manifests = list(Path("/kaggle/input").rglob("bundle.json"))
if archives:
    if len(archives) != 1:
        raise RuntimeError("Attach exactly one selected CNN bundle.")
    bundle = Path("/kaggle/working/cmi-cnn-selected")
    bundle.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as package:
        for entry in package.infolist():
            target = (bundle / entry.filename).resolve()
            if bundle.resolve() not in target.parents:
                raise RuntimeError("Unsafe archive member.")
        package.extractall(bundle)
else:
    if len(manifests) != 1:
        raise RuntimeError("Attach the selected CNN weights dataset before running.")
    bundle = manifests[0].parent
sys.path.insert(0, str(bundle / "src"))
sys.path.insert(0, str(competition))
from cmi_project.inference import RoutedCNNPredictor
import kaggle_evaluation.cmi_inference_server

predictor = RoutedCNNPredictor(bundle, device="cpu", cpu_threads=2)
def predict(sequence, demographics):
    return predictor.predict(sequence, demographics)

server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    server.serve()
else:
    server.run_local_gateway((str(competition / "test.csv"), str(competition / "test_demographics.csv")))
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "cells": [{"cell_type": "markdown", "id": "overview", "metadata": {}, "source":
            ["# CMI selected CNN ensemble\n", "Frozen five-fold IMU/multisensor CNNs; equal probabilities and input-availability routing.\n",
             "Attach the private weights dataset and competition data. Disable Internet before Save & Run All.\n"]},
            {"cell_type": "code", "id": "inference", "execution_count": None, "metadata": {},
             "outputs": [], "source": code_cell.splitlines(keepends=True)}]}
    (output / "cmi-cnn-selected.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (output / "inference_cell.py").write_text(code_cell, encoding="utf-8")
    print(f"Exported {len(members)} frozen checkpoints; upload archive {archive} ({archive.stat().st_size:,} bytes)")
    print(output / "cmi-cnn-selected.ipynb")


if __name__ == "__main__":
    main()
