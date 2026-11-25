"""Fixed sequence-level validation splits shared by every model and ablation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import check_random_state

from .dataset_analysis import PROJECT_DIR, SENSOR_COLUMNS

INDEX_COLUMNS = ["sequence_id", "subject", "gesture", "length"]
FOLD_COLUMNS = [*INDEX_COLUMNS, "fold"]
SPLITTER_IMPLEMENTATION = "StratifiedGroupKFold_consistent_group_shuffle_v1"


class ConsistentStratifiedGroupKFold(StratifiedGroupKFold):
    """Keep group counts and IDs aligned when shuffling, including sklearn <1.8.

    See upstream issue #32478 / fix #32540. Randomly permuting GROUP CODES
    before the standard unshuffled greedy assignment is equivalent to applying
    the permutation to count rows AND its inverse to sample group IDs. It uses
    sklearn's existing stratification algorithm; no global monkey patch.
    """

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("subject groups are required.")
        _, group_codes = np.unique(groups, return_inverse=True)
        if self.shuffle:
            permutation = check_random_state(self.random_state).permutation(int(group_codes.max()) + 1)
            group_codes = np.argsort(permutation)[group_codes]
        splitter = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=False)
        for _, validation in splitter.split(X, y, groups=group_codes):
            yield validation


def scan_sequence_index(path: Path, chunksize: int = 25000, *, include_sensors: bool = False) -> pd.DataFrame:
    """Aggregate each sequence exactly once, regardless of its frame count."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    metadata = ["sequence_id", "subject", "gesture"]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if not set(metadata).issubset(header):
        raise ValueError("Training CSV requires sequence_id, subject and gesture.")
    sensors = [c for columns in SENSOR_COLUMNS.values() for c in columns if c in header] if include_sensors else []
    dtype = {c: "string" for c in metadata}
    dtype.update({c: "float32" for c in sensors})
    states = {}
    with pd.read_csv(path, usecols=metadata + sensors, dtype=dtype, chunksize=chunksize) as reader:
        for chunk in reader:
            if chunk[metadata].isna().any().any():
                raise ValueError("Null sequence_id/subject/gesture in training data.")
            diagnostics = chunk[metadata].copy()
            diagnostics["length"] = 1
            if include_sensors:
                acc = chunk.reindex(columns=SENSOR_COLUMNS["IMU"][:3]).to_numpy(dtype=float)
                q = chunk.reindex(columns=SENSOR_COLUMNS["IMU"][3:]).to_numpy(dtype=float)
                rotation_valid = np.isfinite(q).all(axis=1) & (np.linalg.norm(np.where(np.isfinite(q), q, 0), axis=1) > 1e-8)
                thm = chunk.reindex(columns=SENSOR_COLUMNS["THM"]).to_numpy(dtype=float)
                tof = chunk.reindex(columns=SENSOR_COLUMNS["ToF"]).to_numpy(dtype=float)
                valid_tof = np.isfinite(tof) & (tof >= 0)
                present_tof = np.isfinite(tof) & ((tof >= 0) | (tof == -1))
                diagnostics["rotation_valid_frames"] = rotation_valid.astype(int)
                diagnostics["imu_unavailable_cells"] = (~np.isfinite(acc)).sum(axis=1) + 4 * (~rotation_valid)
                diagnostics["thm_valid_cells"] = np.isfinite(thm).sum(axis=1)
                diagnostics["tof_valid_cells"] = valid_tof.sum(axis=1)
                diagnostics["tof_present_cells"] = present_tof.sum(axis=1)
                diagnostics["tof_nan_cells"] = np.isnan(tof).sum(axis=1)
                diagnostics["tof_minus1_cells"] = (tof == -1).sum(axis=1)
            grouped = diagnostics.groupby(metadata, sort=False).sum(numeric_only=True).reset_index()
            for record in grouped.to_dict("records"):
                sid = record["sequence_id"]
                if sid not in states:
                    states[sid] = {**{k: record[k] for k in metadata},
                                   **{k: 0 for k in record if k not in metadata}}
                state = states[sid]
                if state["subject"] != record["subject"] or state["gesture"] != record["gesture"]:
                    raise ValueError(f"Inconsistent subject/gesture in {sid}.")
                for key in record.keys() - set(metadata):
                    state[key] += int(record[key])
    if not states:
        raise ValueError("Training CSV has no sequences.")
    return pd.DataFrame(states.values()).sort_values("sequence_id").reset_index(drop=True)


def canonical_index(index: pd.DataFrame, *, with_fold: bool = False) -> pd.DataFrame:
    columns = FOLD_COLUMNS if with_fold else INDEX_COLUMNS
    if not set(columns).issubset(index.columns) or index.empty:
        raise ValueError(f"Index requires non-empty columns: {columns}")
    result = index[columns].copy()
    if result.isna().any().any():
        raise ValueError("Null values in sequence index.")
    for key in ("sequence_id", "subject", "gesture"):
        result[key] = result[key].astype(str)
        if result[key].str.strip().eq("").any():
            raise ValueError(f"Empty {key}.")
    if result["sequence_id"].duplicated().any():
        raise ValueError("Duplicate sequence_id in index.")
    for key in (["length", "fold"] if with_fold else ["length"]):
        numbers = pd.to_numeric(result[key], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numbers).all() or (numbers != np.floor(numbers)).any():
            raise ValueError(f"{key} must contain finite integers.")
        result[key] = numbers.astype(np.int64)
    if (result["length"] < 1).any():
        raise ValueError("Sequence lengths must be positive.")
    return result.sort_values("sequence_id").reset_index(drop=True)


def table_digest(table: pd.DataFrame, columns: list[str]) -> str:
    records = table.sort_values("sequence_id")[columns].to_dict("records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_folds(table: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    result = canonical_index(table, with_fold=True)
    if n_splits < 2 or set(result["fold"]) != set(range(n_splits)):
        raise ValueError("Missing/out-of-range fold assignments.")
    if result.groupby("subject")["fold"].nunique().max() != 1:
        raise ValueError("Subject leakage: one subject is assigned to multiple folds.")
    coverage = np.zeros(len(result), dtype=int)
    for fold in range(n_splits):
        val = result["fold"].to_numpy() == fold
        coverage += val
        if set(result.loc[val, "subject"]) & set(result.loc[~val, "subject"]):
            raise ValueError(f"Subject leakage in fold {fold}.")
    if not np.all(coverage == 1):
        raise ValueError("Every sequence must appear in validation exactly once.")
    return result


def make_subject_folds(index: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    result = canonical_index(index)
    if result["subject"].nunique() < n_splits:
        raise ValueError("Need at least n_splits subjects.")
    result["fold"] = -1
    splitter = ConsistentStratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(result, result["gesture"], groups=result["subject"])):
        result.loc[validation, "fold"] = fold
    return validate_folds(result, n_splits)


@dataclass
class FoldManifest:
    table: pd.DataFrame
    metadata: dict
    path: Path

    @property
    def fingerprint(self) -> str:
        return self.metadata["folds_sha256"]

    @property
    def n_splits(self) -> int:
        return self.metadata["n_splits"]

    def split(self, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if fold not in range(self.n_splits):
            raise ValueError("Invalid fold number.")
        return (self.table[self.table["fold"] != fold].copy(), self.table[self.table["fold"] == fold].copy())


def load_fold_manifest(path: Path, *, expected_index: pd.DataFrame | None = None) -> FoldManifest:
    path = Path(path).resolve()
    meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Missing fixed folds or metadata. Run scripts/create_folds.py first.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("version") != 1 or metadata.get("group") != "subject" or metadata.get("stratify") != "gesture":
        raise ValueError("Incompatible fixed validation protocol.")
    table = validate_folds(pd.read_csv(path, dtype={k: "string" for k in INDEX_COLUMNS[:3]}), metadata["n_splits"])
    if table_digest(table, FOLD_COLUMNS) != metadata["folds_sha256"]:
        raise ValueError("Fixed fold fingerprint mismatch; assignment or metadata was edited.")
    if table_digest(table, INDEX_COLUMNS) != metadata["dataset_sha256"]:
        raise ValueError("Fixed dataset fingerprint mismatch.")
    if expected_index is not None:
        expected = canonical_index(expected_index)
        if table_digest(expected, INDEX_COLUMNS) != metadata["dataset_sha256"]:
            raise ValueError("Current dataset differs from fixed folds (IDs/subjects/labels/lengths).")
    return FoldManifest(table, metadata, path)


def save_fixed_folds(index: pd.DataFrame, path: Path, *, n_splits: int = 5, seed: int = 42) -> FoldManifest:
    """Create once; existing manifests are validated and reused, never replaced."""
    path = Path(path).resolve()
    meta_path = path.with_suffix(".meta.json")
    if path.exists() or meta_path.exists():
        saved = load_fold_manifest(path, expected_index=index)
        if saved.n_splits != n_splits or saved.metadata["random_state"] != seed or saved.metadata["shuffle"] is not True:
            raise ValueError("Requested protocol differs from existing fixed folds; use a new version/path.")
        return saved
    table = make_subject_folds(index, n_splits, seed)
    metadata = {
        "version": 1, "splitter": "StratifiedGroupKFold", "implementation": SPLITTER_IMPLEMENTATION,
        "n_splits": n_splits, "shuffle": True, "random_state": seed,
        "group": "subject", "stratify": "gesture", "sample_unit": "sequence",
        "dataset_sha256": table_digest(table, INDEX_COLUMNS), "folds_sha256": table_digest(table, FOLD_COLUMNS),
        "sequences": len(table), "subjects": int(table["subject"].nunique()),
        "labels": sorted(table["gesture"].unique().tolist()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a canonical experiment split.
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        table.to_csv(file, index=False)
    with meta_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, allow_nan=False)
    return load_fold_manifest(path)


def assert_preprocessor_matches(preprocessor_state: dict, manifest: FoldManifest, fold: int) -> None:
    train, _ = manifest.split(fold)
    provenance = preprocessor_state.get("validation", {})
    if provenance.get("folds_sha256") != manifest.fingerprint or provenance.get("fold") != fold:
        raise ValueError("Preprocessor was not fitted with the requested fixed validation folds.")
    if set(preprocessor_state["train_sequence_ids"]) != set(train["sequence_id"]):
        raise ValueError("Preprocessor training sequences differ from fixed folds.")
    if set(preprocessor_state["train_subjects"]) != set(train["subject"]):
        raise ValueError("Preprocessor training subjects differ from fixed folds.")


def write_fold_diagnostics(manifest: FoldManifest, index: pd.DataFrame, output_dir: Path) -> dict:
    """Descriptive diagnostics only; no validation statistic is fitted into a model."""
    if table_digest(canonical_index(index), INDEX_COLUMNS) != manifest.metadata["dataset_sha256"]:
        raise ValueError("Diagnostic data differs from frozen folds.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = manifest.table.merge(index.drop(columns=["subject", "gesture", "length", "fold"], errors="ignore"),
                                 on="sequence_id", how="left", validate="one_to_one")
    labels = manifest.metadata["labels"]
    summaries, classes, sensors = [], [], []
    global_counts = table["gesture"].value_counts()
    for fold in range(manifest.n_splits):
        train, validation = table[table["fold"] != fold], table[table["fold"] == fold]
        summaries.append({"fold": fold, "train_sequences": len(train), "validation_sequences": len(validation),
                          "train_subjects": train["subject"].nunique(), "validation_subjects": validation["subject"].nunique(),
                          "subject_overlap": len(set(train["subject"]) & set(validation["subject"])),
                          "train_classes": train["gesture"].nunique(), "validation_classes": validation["gesture"].nunique()})
        for split, subset in (("train", train), ("validation", validation)):
            counts = subset["gesture"].value_counts()
            for gesture in labels:
                classes.append({"fold": fold, "split": split, "gesture": gesture,
                                "sequence_count": int(counts.get(gesture, 0)), "ratio": counts.get(gesture, 0) / len(subset),
                                "global_ratio": global_counts[gesture] / len(table)})
            if "rotation_valid_frames" in subset:
                frames = int(subset["length"].sum())
                sensors.append({"fold": fold, "split": split, "sequences": len(subset), "frames": frames,
                    "rotation_missing_sequence_ratio": float((subset["rotation_valid_frames"] == 0).mean()),
                    "rotation_unavailable_frame_ratio": 1 - subset["rotation_valid_frames"].sum() / frames,
                    "imu_unavailable_cell_ratio": subset["imu_unavailable_cells"].sum() / (frames * 7),
                    "thm_missing_sequence_ratio": float((subset["thm_valid_cells"] == 0).mean()),
                    "thm_unavailable_cell_ratio": 1 - subset["thm_valid_cells"].sum() / (frames * 5),
                    "tof_missing_sequence_ratio": float((subset["tof_present_cells"] == 0).mean()),
                    "tof_no_valid_distance_sequence_ratio": float((subset["tof_valid_cells"] == 0).mean()),
                    "tof_invalid_pixel_ratio": 1 - subset["tof_valid_cells"].sum() / (frames * 320),
                    "tof_nan_pixel_ratio": subset["tof_nan_cells"].sum() / (frames * 320),
                    "tof_minus1_pixel_ratio": subset["tof_minus1_cells"].sum() / (frames * 320)})
    tables = {"fold_summary": pd.DataFrame(summaries), "class_distribution": pd.DataFrame(classes),
              "sensor_missingness": pd.DataFrame(sensors), "sequence_sensor_availability": table,
              "subject_folds": table.groupby(["subject", "fold"]).size().reset_index(name="sequence_count")}
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    missing = tables["class_distribution"].query("sequence_count == 0")
    report = {"folds_sha256": manifest.fingerprint, "sequences": len(table), "subjects": int(table["subject"].nunique()),
              "classes": len(labels), "n_splits": manifest.n_splits,
              "subject_overlap_all_folds": 0, "every_sequence_validated_once": True,
              "missing_class_splits": missing[["fold", "split", "gesture"]].to_dict("records")}
    (output_dir / "checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CMI 验证划分</title>'
    html += '<style>body{font-family:system-ui;margin:32px;line-height:1.6}table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #ddd}.scroll{overflow:auto}</style>'
    html += f'<h1>Step 3：固定 subject 验证划分</h1><p>{len(table)} sequences / {report["subjects"]} subjects / {len(labels)} 类 / {manifest.n_splits} folds</p>'
    html += f'<p>subject 交集：0；每条 sequence 恰好验证一次。划分指纹：<code>{escape(manifest.fingerprint)}</code></p>'
    html += '<p>统计基于原始数据。ToF -1 为无反射响应，NaN 为缺失；整段无有效距离与整段传感器缺失分别统计。这里只检查划分，还没有模型分数。</p>'
    for name, title in (("fold_summary", "每折样本数量"), ("sensor_missingness", "各折原始传感器缺失情况")):
        html += f'<h2>{title}</h2><div class="scroll">{tables[name].to_html(index=False, float_format=lambda v: f"{v:.4f}")}</div>'
    val_classes = tables["class_distribution"].query("split == 'validation'").pivot(index="gesture", columns="fold", values="sequence_count")
    html += f'<h2>验证类别分布（sequence 数量）</h2><div class="scroll">{val_classes.to_html()}</div></html>'
    (output_dir / "validation_report.html").write_text(html, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3: create/reuse and audit fixed subject validation folds.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/validation.json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--folds-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("n_splits") != 5 or config.get("shuffle") is not True or config.get("random_state") != 42:
            raise ValueError("Project validation protocol requires 5 folds, shuffle=True, random_state=42.")
        data_dir = (args.data_dir or PROJECT_DIR / config["data_dir"]).resolve()
        path = (args.folds_path or PROJECT_DIR / config["folds_path"]).resolve()
        output_dir = (args.output_dir or PROJECT_DIR / config["output_dir"]).resolve()
        if data_dir == output_dir or data_dir in output_dir.parents or data_dir == path.parent or data_dir in path.parents:
            raise ValueError("Fixed folds and reports must be outside raw data.")
        print("Scanning sequence metadata and raw sensor availability...", flush=True)
        index = scan_sequence_index(data_dir / "train.csv", config.get("chunksize", 25000), include_sensors=True)
        manifest = save_fixed_folds(index, path, n_splits=5, seed=42)
        report = write_fold_diagnostics(manifest, index, output_dir)
        print(f"Fixed folds: {path}\nSHA-256: {manifest.fingerprint}\nChecks: {report['sequences']} sequences, zero subject overlap.")
        print(f"Report: {output_dir / 'validation_report.html'}")
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
