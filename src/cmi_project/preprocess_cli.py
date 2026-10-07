"""Subject-grouped fold preparation and bounded-memory NPZ export."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .dataset_analysis import PROJECT_DIR
from .preprocessing import (
    FoldPreprocessor, PreprocessingConfig, SensorDropoutConfig,
    demographics_lookup, iter_csv_sequences,
)
from .validation import assert_preprocessor_matches, load_fold_manifest, scan_sequence_index


def save_sample(sample: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # tof_input is rebuilt from distance and validity to avoid duplicate disk data.
    np.savez_compressed(path, **{key: value for key, value in sample.items() if key != "tof_input"})


def export_sequences(preprocessor: FoldPreprocessor, csv_path: Path, demographics: pd.DataFrame,
                     fold_dir: Path, *, chunksize: int, split_by_id: dict[str, str] | None = None,
                     split: str = "test") -> dict[str, int]:
    manifests: dict[str, list] = {split: []} if split_by_id is None else {s: [] for s in sorted(set(split_by_id.values()))}
    wanted = None if split_by_id is None else set(split_by_id)
    for sequence in iter_csv_sequences(csv_path, chunksize=chunksize, sequence_ids=wanted,
                                       require_gesture=split_by_id is not None):
        sequence_id = str(sequence["sequence_id"].iloc[0])
        target_split = split if split_by_id is None else split_by_id[sequence_id]
        sample = preprocessor.transform(sequence, demographics, split=target_split)
        records = manifests[target_split]
        relative = Path("samples") / target_split / f"{len(records):06d}.npz"
        save_sample(sample, fold_dir / relative)
        records.append({"sequence_id": sequence_id, "subject": sample["subject"], "label": int(sample["label"]),
                        "original_length": int(sample["original_length"]), "kept_length": int(sample["kept_length"]),
                        "path": relative.as_posix()})
        count = sum(len(rows) for rows in manifests.values())
        if count % 1000 == 0:
            print(f"Exported {count:,} sequences", flush=True)
    for name, records in manifests.items():
        pd.DataFrame(records, columns=["sequence_id", "subject", "label", "original_length", "kept_length", "path"]).to_csv(
            fold_dir / f"{name}_manifest.csv", index=False, encoding="utf-8-sig")
    return {name: len(records) for name, records in manifests.items()}


def prepare_folds(data_dir: Path, output_dir: Path, *, config: PreprocessingConfig | None = None,
                  dropout: SensorDropoutConfig | None = None, folds_path: Path | None = None,
                  folds: list[int] | None = None, chunksize: int = 25000,
                  fit_only: bool = False, include_test: bool = True) -> list[dict]:
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Output must be outside the raw data directory.")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    folds = [0] if folds is None else folds
    # Fresh output prevents old NPZs/manifests from being mixed with new scalers.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is non-empty; choose a fresh --output-dir.")
    if include_test and not fit_only:
        if (data_dir / "test.csv").is_file() != (data_dir / "test_demographics.csv").is_file():
            raise ValueError("test.csv and test_demographics.csv must be supplied together.")
    config, dropout = config or PreprocessingConfig(), dropout or SensorDropoutConfig()
    demographics = pd.read_csv(data_dir / "train_demographics.csv", dtype={"subject": "string"})
    hands = demographics_lookup(demographics)
    print("Scanning metadata and validating the saved subject folds...", flush=True)
    manifest = load_fold_manifest(folds_path or PROJECT_DIR / "configs/folds.csv",
                                  expected_index=scan_sequence_index(data_dir / "train.csv", chunksize))
    index = manifest.table
    if not folds or len(set(folds)) != len(folds) or any(f < 0 or f >= manifest.n_splits for f in folds):
        raise ValueError("Invalid requested fold numbers.")
    unknown = set(index["subject"]) - set(hands)
    if unknown:
        raise ValueError(f"Missing demographics for: {sorted(unknown)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_dir / "folds.csv", index=False, encoding="utf-8-sig")
    (output_dir / "folds.meta.json").write_text(json.dumps(manifest.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps({
        "data_dir": str(data_dir), "preprocessing": asdict(config), "sensor_dropout": asdict(dropout),
        "n_splits": manifest.n_splits, "seed": manifest.metadata["random_state"], "folds": folds, "chunksize": chunksize,
        "folds_path": str(manifest.path), "folds_sha256": manifest.fingerprint,
        "fit_only": fit_only, "include_test": include_test,
        "sources": {name: {"bytes": (data_dir / name).stat().st_size,
                            "mtime_ns": (data_dir / name).stat().st_mtime_ns}
                    for name in ("train.csv", "train_demographics.csv")},
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summaries = []
    for fold in folds:
        train_index, validation_index = index[index["fold"] != fold], index[index["fold"] == fold]
        train_ids = set(train_index["sequence_id"])
        print(f"Fitting fold {fold}: {len(train_ids):,} training sequences / {len(validation_index):,} validation sequences", flush=True)
        preprocessor = FoldPreprocessor(config).fit(
            iter_csv_sequences(data_dir / "train.csv", chunksize=chunksize, sequence_ids=train_ids, require_gesture=True), demographics)
        preprocessor.state["validation"] = {"fold": fold, "n_splits": manifest.n_splits,
            "folds_sha256": manifest.fingerprint, "dataset_sha256": manifest.metadata["dataset_sha256"]}
        assert_preprocessor_matches(preprocessor.state, manifest, fold)
        fold_dir = output_dir / f"fold_{fold}"
        preprocessor.save(fold_dir / "preprocessor.json")
        (fold_dir / "sensor_dropout.json").write_text(json.dumps(asdict(dropout), indent=2), encoding="utf-8")
        # Use the saved object for validation/test and cache export as inference will.
        preprocessor = FoldPreprocessor.load(fold_dir / "preprocessor.json")
        summary = {"fold": fold, "train_sequences": len(train_index), "validation_sequences": len(validation_index),
                   "folds_sha256": manifest.fingerprint,
                   "train_subjects": sorted(set(train_index["subject"])),
                   "validation_subjects": sorted(set(validation_index["subject"])),
                   "max_length": preprocessor.max_length, "angular_velocity_unit": preprocessor.state["angular_velocity_unit"],
                   "exports": {}}
        if not fit_only:
            assignments = {sid: "train" if sid in train_ids else "validation" for sid in index["sequence_id"]}
            summary["exports"].update(export_sequences(preprocessor, data_dir / "train.csv", demographics,
                                                       fold_dir, chunksize=chunksize, split_by_id=assignments))
            if include_test:
                test_path, test_demo = data_dir / "test.csv", data_dir / "test_demographics.csv"
                if test_path.is_file() != test_demo.is_file():
                    raise ValueError("test.csv and test_demographics.csv must be supplied together.")
                if test_path.is_file():
                    summary["exports"].update(export_sequences(preprocessor, test_path,
                        pd.read_csv(test_demo, dtype={"subject": "string"}), fold_dir, chunksize=chunksize))
        (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append(summary)
        print(f"Fold {fold} complete: max_length={preprocessor.max_length}; saved to {fold_dir}", flush=True)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="CMI Step 2: train-fold-only preprocessing, masks and NPZ caches.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "preprocessing.json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds-path", type=Path, help="Existing canonical folds CSV; never regenerate during preprocessing.")
    parser.add_argument("--fold", type=int, nargs="+", help="Fold numbers; default 0. Use --fold 0 1 2 3 4 for all.")
    parser.add_argument("--chunksize", type=int)
    parser.add_argument("--fit-only", action="store_true", help="Save fold parameters without exporting tensor caches.")
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        prepare_folds(
            args.data_dir or PROJECT_DIR / settings.get("data_dir", "data"),
            args.output_dir or PROJECT_DIR / settings.get("output_dir", "outputs/preprocessing"),
            config=PreprocessingConfig.from_dict(settings.get("preprocessing", {})),
            dropout=SensorDropoutConfig(**settings.get("sensor_dropout", {})),
            folds_path=args.folds_path or PROJECT_DIR / settings.get("folds_path", "configs/folds.csv"),
            folds=args.fold, chunksize=args.chunksize if args.chunksize is not None else int(settings.get("chunksize", 25000)),
            fit_only=args.fit_only, include_test=not args.no_test,
        )
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
