"""IMU statistics with LightGBM or XGBoost on the same fixed subject folds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

from .dataset_analysis import PROJECT_DIR
from .evaluation import ALL_GESTURES, PROBABILITY_COLUMNS, cmi_metrics, evaluate_oof_frames
from .preprocessing import (
    IMU_FEATURES, PreprocessingConfig, demographics_lookup, engineer_sequence, iter_csv_sequences,
)
from .validation import FoldManifest, INDEX_COLUMNS, canonical_index, table_digest
from .validation import load_fold_manifest

SIGNALS = [*IMU_FEATURES, "acc_magnitude", "angular_velocity_magnitude", "linear_acc_magnitude"]
STATISTICS = ("mean", "std", "min", "max", "range", "valid_ratio")
FEATURE_COLUMNS = [f"{signal}__{statistic}" for signal in SIGNALS for statistic in STATISTICS]
DEFAULT_MODEL_PARAMETERS = {
    "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
    "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "n_jobs": 4,
}
DEFAULT_XGBOOST_PARAMETERS = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
    "min_child_weight": 1.0, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "n_jobs": 4,
}


def sequence_statistics(sequence: pd.DataFrame, handedness: int,
                        config: PreprocessingConfig | None = None) -> dict:
    """Full unpadded sequence, physical units, valid readings only; no fitted statistics.

    Missing channels keep NaN summaries and zero valid_ratio. Standard deviation
    uses ddof=0, so one observed sample has std=0. IDs/labels are metadata only.
    """
    raw = engineer_sequence(sequence, handedness, config or PreprocessingConfig())
    values, valid = raw["imu"], raw["imu_valid"]
    magnitudes, magnitude_valid = [], []
    for start in (0, 9, 12):
        observed = valid[:, start:start + 3].all(axis=1)
        magnitude = np.linalg.norm(values[:, start:start + 3], axis=1)
        magnitudes.append(magnitude)
        magnitude_valid.append(observed & np.isfinite(magnitude))
    values = np.column_stack([values, *magnitudes])
    valid = np.column_stack([valid, *magnitude_valid])
    record = {"sequence_id": raw["sequence_id"], "subject": raw["subject"],
              "gesture": raw["gesture"], "length": len(sequence)}
    for i, signal in enumerate(SIGNALS):
        observed = values[valid[:, i], i]
        statistics = dict.fromkeys(STATISTICS[:-1], float("nan"))
        if len(observed):
            minimum, maximum = float(observed.min()), float(observed.max())
            statistics = {"mean": float(observed.mean()), "std": float(observed.std(ddof=0)),
                          "min": minimum, "max": maximum, "range": maximum - minimum}
        statistics["valid_ratio"] = float(valid[:, i].mean())
        record.update({f"{signal}__{key}": value for key, value in statistics.items()})
    return record


def extract_imu_features(csv_path: Path, demographics: pd.DataFrame,
                         config: PreprocessingConfig | None = None, *, chunksize: int = 25000) -> pd.DataFrame:
    """Read IMU columns only and aggregate each complete sequence exactly once."""
    hands = demographics_lookup(demographics)
    records = []
    for sequence in iter_csv_sequences(csv_path, chunksize=chunksize, require_gesture=True,
                                       sensor_modalities=("IMU",)):
        subject = str(sequence["subject"].iloc[0])
        if subject not in hands:
            raise ValueError(f"Missing demographics for subject {subject}.")
        records.append(sequence_statistics(sequence, hands[subject], config))
        if len(records) % 2000 == 0:
            print(f"Extracted IMU statistics for {len(records):,} sequences", flush=True)
    if not records:
        raise ValueError("Training CSV has no sequences.")
    return pd.DataFrame(records, columns=[*INDEX_COLUMNS, *FEATURE_COLUMNS]).sort_values("sequence_id").reset_index(drop=True)


def train_baseline(features: pd.DataFrame, manifest: FoldManifest, output_dir: Path,
                   *, model_parameters: dict | None = None, experiment_name: str = "lightgbm_imu",
                   model_name: str = "lightgbm", feature_metadata: dict | None = None) -> dict:
    """Train each held-out subject fold once; no validation fitting or early stopping.

    Histograms and missing-value decisions are learned inside each training fold.
    Fixed boosting rounds make this a reference run without validation tuning.
    """
    index = canonical_index(features)
    if table_digest(index, INDEX_COLUMNS) != manifest.metadata["dataset_sha256"]:
        raise ValueError("Baseline sequence metadata differs from the fixed folds.")
    if not set(FEATURE_COLUMNS).issubset(features.columns):
        raise ValueError("Missing IMU statistic feature columns.")
    if not set(index["gesture"]).issubset(ALL_GESTURES):
        raise ValueError("Unknown gesture in baseline data.")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is non-empty; choose a fresh --output-dir.")
    matrix = features.set_index("sequence_id").loc[manifest.table["sequence_id"], FEATURE_COLUMNS].astype(float)
    if np.isinf(matrix.to_numpy()).any():
        raise ValueError("IMU statistics contain infinity; use NaN for unavailable features.")
    if model_name == "lightgbm":
        import lightgbm as library
        classifier = library.LGBMClassifier
        parameters = {**DEFAULT_MODEL_PARAMETERS, **(model_parameters or {})}
        parameters.update(objective="multiclass", random_state=42, deterministic=True,
                          force_col_wise=True, verbosity=-1, zero_as_missing=False)
    elif model_name == "xgboost":
        import xgboost as library
        classifier = library.XGBClassifier
        parameters = {**DEFAULT_XGBOOST_PARAMETERS, **(model_parameters or {})}
        parameters.update(objective="multi:softprob", random_state=42, tree_method="hist",
                          device="cpu", eval_metric="mlogloss", verbosity=0)
    else:
        raise ValueError("model_name must be lightgbm or xgboost.")
    metadata = {"model": model_name, "modalities": ["IMU"],
                "model_parameters": parameters, "features": FEATURE_COLUMNS, "statistics_std_ddof": 0,
                "missing_summary": "NaN", "sequence_window": "full sequence, no padding or truncation",
                "feature_values_sha256": hashlib.sha256(pd.util.hash_pandas_object(matrix, index=True).to_numpy().tobytes()).hexdigest(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__, model_name: library.__version__}}
    if feature_metadata is not None:
        metadata["feature_config"] = feature_metadata
    predictions_by_fold = {}
    labels = np.asarray(ALL_GESTURES)
    label_ids = {label: i for i, label in enumerate(ALL_GESTURES)}
    for fold in range(manifest.n_splits):
        train, validation = manifest.split(fold)
        train_ids, validation_ids = train["sequence_id"], validation["sequence_id"]
        print(f"Training fold {fold}: {len(train):,} train / {len(validation):,} validation sequences", flush=True)
        model = classifier(**parameters)
        # XGBoost requires consecutive local class IDs even if a fold lacks a class.
        global_classes = np.sort(train["gesture"].map(label_ids).unique())
        local_ids = {int(global_id): local_id for local_id, global_id in enumerate(global_classes)}
        encoded_labels = train["gesture"].map(label_ids).map(local_ids).to_numpy()
        model.fit(matrix.loc[train_ids], encoded_labels)
        probabilities = np.zeros((len(validation), len(ALL_GESTURES)), dtype=float)
        probabilities[:, global_classes[model.classes_.astype(int)]] = model.predict_proba(matrix.loc[validation_ids])
        predicted = labels[probabilities.argmax(axis=1)]
        predictions = pd.DataFrame({"sequence_id": validation_ids.to_numpy(), "predicted_gesture": predicted})
        predictions = pd.concat([predictions, pd.DataFrame(probabilities, columns=PROBABILITY_COLUMNS)], axis=1)
        predictions["folds_sha256"] = manifest.fingerprint
        predictions_by_fold[fold] = predictions
        score = cmi_metrics(validation["gesture"], predicted)
        print(f"Fold {fold} official CMI score: {score['score']:.6f}", flush=True)
    return evaluate_oof_frames(manifest, predictions_by_fold, output_dir / "evaluation",
        experiment_name=experiment_name, save_oof=False, experiment_config=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4: IMU statistics + LightGBM/XGBoost on the fixed subject folds.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/baseline.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        # Fail before scanning raw data if the selected backend dependency is absent.
        model_name = settings.get("model", "lightgbm")
        if model_name not in ("lightgbm", "xgboost"):
            raise ValueError("model must be lightgbm or xgboost.")
        __import__(model_name)
        data_dir = (PROJECT_DIR / settings["data_dir"]).resolve()
        output_dir = (args.output_dir or PROJECT_DIR / settings["output_dir"]).resolve()
        if output_dir == data_dir or data_dir in output_dir.parents:
            raise ValueError("Baseline outputs must be outside raw data.")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("Output directory is non-empty; choose a fresh --output-dir.")
        manifest = load_fold_manifest(PROJECT_DIR / settings["folds_path"])
        preprocessing = json.loads((PROJECT_DIR / settings["preprocessing_config"]).read_text(encoding="utf-8"))
        physics = PreprocessingConfig.from_dict(preprocessing.get("preprocessing", {}))
        print("Extracting full-sequence IMU statistics...", flush=True)
        features = extract_imu_features(data_dir / "train.csv",
            pd.read_csv(data_dir / "train_demographics.csv", dtype={"subject": "string"}), physics,
            chunksize=settings.get("chunksize", 25000))
        feature_metadata = {"physics": asdict(physics),
            "sources": {name: {"bytes": (data_dir / name).stat().st_size,
                               "mtime_ns": (data_dir / name).stat().st_mtime_ns}
                        for name in ("train.csv", "train_demographics.csv")}}
        summary = train_baseline(features, manifest, output_dir, model_name=model_name,
            model_parameters=settings.get("model_parameters"),
            experiment_name=settings.get("experiment_name", f"{model_name}_imu"), feature_metadata=feature_metadata)
        print(f"Five-fold CMI score: {summary['fold_mean']['score']:.6f} +/- {summary['fold_std']['score']:.6f}")
        print(f"Complete OOF score: {summary['oof']['score']:.6f}; results: {output_dir}")
    except ImportError as error:
        print(f"Missing baseline dependency: {error}. Install requirements.txt.", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
