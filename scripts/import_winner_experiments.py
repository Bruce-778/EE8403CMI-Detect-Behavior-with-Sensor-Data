"""Import GPU artifacts into a fresh directory and verify fixed-fold evidence."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.cnn_training import load_cnn_checkpoint
from cmi_project.evaluation import _checked_predictions, cmi_metrics
from cmi_project.validation import assert_preprocessor_matches, load_fold_manifest


def extract_archive(archive: Path, destination: Path):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Import into a fresh directory; preserve previous CPU/GPU runs.")
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            relative = PurePosixPath(entry.filename)
            target = destination.joinpath(*relative.parts).resolve()
            if (relative.is_absolute() or "\\" in entry.filename or ":" in entry.filename
                    or ".." in relative.parts or destination.resolve() not in target.parents):
                raise ValueError("Archive path escapes its destination.")
        destination.mkdir(parents=True, exist_ok=True)
        bundle.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    destination = (ROOT / args.output_dir).resolve()
    output_root = (ROOT / "outputs/kaggle_training/imported").resolve()
    if output_root not in destination.parents:
        raise ValueError("Use a new child directory of outputs/kaggle_training/imported.")
    extract_archive(archive, destination)
    data_check = json.loads((destination / "data_source_check.json").read_text(encoding="utf-8"))
    expected_data = json.loads((ROOT / "configs/data_source_hashes.json").read_text(encoding="utf-8"))
    if data_check.get("status") != "verified" or data_check.get("files") != expected_data["files"]:
        raise ValueError("GPU training inputs disagree with fixed local training data.")
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    selection = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
    design_names = {row["name"] for row in selection["designs"]}
    if (design_names != {"cnn_grouped_se", "cnn_grouped_mixup", "cnn_dynamics_mixup"}
            or selection["selection_fold"] != 0
            or selection["selected"] != max(selection["designs"], key=lambda row: row["score"])["name"]):
        raise ValueError("Unexpected experiment selection policy.")
    checked = []
    directory = destination / "outputs/experiments" / selection["selected"]
    for name in ("imu", "multisensor"):
        for fold in range(5):
            path = directory / name / f"fold_{fold}"
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            model, processor, checkpoint = load_cnn_checkpoint(path / "best.pt")
            assert_preprocessor_matches(processor.state, manifest, fold)
            if (checkpoint["fold"] != fold or checkpoint["folds_sha256"] != manifest.fingerprint
                    or model.model_name != name or metrics["fold"] != fold or metrics["model"] != name
                    or metrics["folds_sha256"] != manifest.fingerprint
                    or checkpoint["validation_metrics"] != metrics["validation"]):
                raise ValueError("Checkpoint/metrics disagree with fixed folds.")
            frame = _checked_predictions(manifest, fold, pd.read_csv(path / "predictions.csv"),
                require_fingerprint=True)
            actual = cmi_metrics(frame["gesture"], frame["predicted_gesture"])
            if any(not np.isclose(actual[key], metrics["validation"][key], rtol=0, atol=1e-12) for key in actual):
                raise ValueError("Saved probabilities/labels disagree with reported validation metrics.")
            checked.append({"model": name, "fold": fold, "sequences": len(frame), **actual})
            print(f"PASS {name} fold {fold}: {len(frame)} sequences, CMI={actual['score']:.6f}", flush=True)
    result = {"status": "verified", "source_url": args.source_url,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "training_inputs": data_check["files"],
        "folds_sha256": manifest.fingerprint, "selection": selection, "fold_checks": checked,
        "selected_run": directory.relative_to(ROOT).as_posix(),
        "note": "Original artifact metadata retained; CPU runs were not overwritten."}
    (destination / "import_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Verified run:", directory, flush=True)


if __name__ == "__main__":
    main()
