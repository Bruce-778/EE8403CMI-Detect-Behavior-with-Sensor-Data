"""Compact fold-fitted temporal arrays, shared by Models A and B."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from .evaluation import ALL_GESTURES
from .preprocessing import (FoldPreprocessor, PreprocessingConfig, SensorDropoutConfig,
                            iter_csv_sequences, tof_region_pool, rotation_dependent_imu_indices)
from .validation import FoldManifest, assert_preprocessor_matches

ARRAY_KEYS = ("imu", "imu_valid", "thm", "thm_valid", "thm_observed", "tof", "tof_valid",
              "tof_fraction", "tof_sensor_present", "time_mask")


def compact_sample(sample: dict, *, tof_regions: int = 2, input_clip: float | None = 8.0) -> dict:
    """Reduce ToF space only; preserve time, validity and sensor-presence semantics."""
    if input_clip is not None and (not np.isfinite(input_clip) or input_clip <= 0):
        raise ValueError("input_clip must be positive or null.")
    tof, tof_valid = tof_region_pool(sample["tof"], sample["tof_valid"], tof_regions)
    side = 8 // tof_regions
    shape = (*sample["tof_valid"].shape[:-2], tof_regions, side, tof_regions, side)
    fraction = sample["tof_valid"].reshape(shape).mean(axis=(-3, -1), dtype=np.float32)
    result = {key: np.asarray(sample[key]) for key in ARRAY_KEYS if key not in ("tof", "tof_valid", "tof_fraction")}
    result.update(tof=tof.reshape(len(tof), -1), tof_valid=tof_valid.reshape(len(tof), -1),
                  tof_fraction=fraction.reshape(len(tof), -1))
    for key in ("imu", "thm", "tof"):
        result[key] = np.asarray(result[key], dtype=np.float32)
        if input_clip is not None:
            result[key] = np.clip(result[key], -input_clip, input_clip)
    return result


def source_signature(data_dir: Path) -> dict:
    return {name: {"bytes": (data_dir / name).stat().st_size,
                   "mtime_ns": (data_dir / name).stat().st_mtime_ns}
            for name in ("train.csv", "train_demographics.csv")}


def validate_arrays(arrays: dict, manifest: FoldManifest, length: int, regions: int) -> None:
    count = len(manifest.table)
    if not set((*ARRAY_KEYS, "sequence_id", "subject", "label")).issubset(arrays):
        raise ValueError("Incomplete CNN tensor cache.")
    for key in ("sequence_id", "subject"):
        if not np.array_equal(arrays[key], manifest.table[key].to_numpy(dtype=str)):
            raise ValueError(f"CNN cache {key} disagrees with the fixed folds.")
    labels = manifest.table["gesture"].map({g: i for i, g in enumerate(ALL_GESTURES)}).to_numpy()
    if not np.array_equal(arrays["label"], labels):
        raise ValueError("CNN cache class order disagrees with ALL_GESTURES.")
    imu_channels = arrays["imu"].shape[-1]
    if imu_channels not in (15, 34):
        raise ValueError("Unsupported cached IMU feature count.")
    channels = {"imu": imu_channels, "imu_valid": imu_channels, "thm": 5, "thm_valid": 5, "thm_observed": 5,
                "tof": 5 * regions**2, "tof_valid": 5 * regions**2,
                "tof_fraction": 5 * regions**2, "tof_sensor_present": 5}
    for key in ARRAY_KEYS:
        shape = (count, length) if key == "time_mask" else (count, length, channels[key])
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid CNN cache array {key}.")
        expected_dtype = np.float32 if key in ("imu", "thm", "tof", "tof_fraction") else np.bool_
        if arrays[key].dtype != expected_dtype:
            raise ValueError(f"Invalid CNN cache dtype for {key}.")
    if not arrays["time_mask"].any(axis=1).all():
        raise ValueError("CNN cache contains an empty sequence.")


def prepare_cnn_fold(data_dir: Path, cache_dir: Path, manifest: FoldManifest, fold: int, *,
                     preprocessing: PreprocessingConfig | None = None, tof_regions: int = 2,
                     sequence_length: int | None = None, input_clip: float | None = 8.0,
                     chunksize: int = 25000) -> tuple[dict, FoldPreprocessor, dict]:
    """Fit scalers on training subjects, validate raw metadata, then reuse one cache.

    Changing input settings or source files requires a fresh cache directory.
    Padding/cropping uses the existing pipeline's left-padding / tail window.
    """
    data_dir, cache_dir = Path(data_dir).resolve(), Path(cache_dir).resolve()
    if cache_dir == data_dir or data_dir in cache_dir.parents:
        raise ValueError("CNN cache must be outside raw data.")
    if sequence_length is not None and (not isinstance(sequence_length, int) or sequence_length < 1):
        raise ValueError("sequence_length must be a positive integer or null.")
    if tof_regions not in (1, 2, 4, 8) or chunksize < 1:
        raise ValueError("Invalid tof_regions/chunksize.")
    if input_clip is not None and (not np.isfinite(input_clip) or input_clip <= 0):
        raise ValueError("input_clip must be positive or null.")
    preprocessing = preprocessing or PreprocessingConfig()
    train, _ = manifest.split(fold)
    if not set(manifest.table["gesture"]).issubset(ALL_GESTURES):
        raise ValueError("CNN training requires the CMI gesture ontology.")
    preprocessing_identity = asdict(preprocessing)
    # Preserve the identity of the existing basic-feature caches.
    if not preprocessing.imu_dynamics:
        preprocessing_identity.pop("imu_dynamics")
    identity = json.loads(json.dumps({
        "version": 1, "data_dir": str(data_dir), "sources": source_signature(data_dir),
        "folds_sha256": manifest.fingerprint, "fold": fold, "preprocessing": preprocessing_identity,
        "tof_regions": tof_regions, "sequence_length": sequence_length, "input_clip": input_clip,
        "label_order": list(ALL_GESTURES),
    }))
    archive, meta_path = cache_dir / "data.npz", cache_dir / "metadata.json"
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["identity"] != identity:
            raise ValueError("CNN cache settings/sources differ; choose a fresh --cache-dir.")
        processor = FoldPreprocessor.load(cache_dir / "preprocessor.json")
        assert_preprocessor_matches(processor.state, manifest, fold)
        with np.load(archive, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        validate_arrays(arrays, manifest, processor.max_length, tof_regions)
        print(f"Reused fold {fold} cache: {len(manifest.table):,} sequences, length={processor.max_length}", flush=True)
        return arrays, processor, metadata
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise ValueError("Incomplete/non-empty CNN cache; choose a fresh --cache-dir.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    demographics = pd.read_csv(data_dir / "train_demographics.csv", dtype={"subject": "string"})
    print(f"Fitting fold {fold} normalization on {len(train):,} training sequences...", flush=True)
    processor = FoldPreprocessor(preprocessing).fit(iter_csv_sequences(
        data_dir / "train.csv", chunksize=chunksize, sequence_ids=set(train["sequence_id"]),
        require_gesture=True), demographics)
    processor.state["validation"] = {"fold": fold, "n_splits": manifest.n_splits,
        "folds_sha256": manifest.fingerprint, "dataset_sha256": manifest.metadata["dataset_sha256"]}
    if sequence_length is not None:
        processor.state["max_length"] = sequence_length
        processor.state["cnn_sequence_length_override"] = sequence_length
    assert_preprocessor_matches(processor.state, manifest, fold)
    expected = manifest.table.set_index("sequence_id")
    positions = {sid: i for i, sid in enumerate(manifest.table["sequence_id"])}
    arrays = {key: manifest.table[key].to_numpy(dtype=str) for key in ("sequence_id", "subject")}
    arrays["label"] = manifest.table["gesture"].map({g: i for i, g in enumerate(ALL_GESTURES)}).to_numpy(dtype=np.int64)
    seen = set()
    print(f"Building compact fold {fold} arrays (length={processor.max_length}, ToF={tof_regions}x{tof_regions})...", flush=True)
    for sequence in iter_csv_sequences(data_dir / "train.csv", chunksize=chunksize, require_gesture=True):
        sid = str(sequence["sequence_id"].iloc[0])
        if sid not in positions or sid in seen:
            raise ValueError("Raw sequences differ from the fixed folds.")
        row = expected.loc[sid]
        if str(sequence["subject"].iloc[0]) != row["subject"] or str(sequence["gesture"].iloc[0]) != row["gesture"] or len(sequence) != row["length"]:
            raise ValueError(f"Raw metadata differs from fixed folds: {sid}.")
        split = "validation" if row["fold"] == fold else "train"
        sample = compact_sample(processor.transform(sequence, demographics, split),
                                tof_regions=tof_regions, input_clip=input_clip)
        if not seen:
            arrays.update({key: np.empty((len(manifest.table), *value.shape), dtype=value.dtype)
                           for key, value in sample.items()})
        for key in ARRAY_KEYS:
            arrays[key][positions[sid]] = sample[key]
        seen.add(sid)
        if len(seen) % 2000 == 0:
            print(f"Prepared {len(seen):,} / {len(manifest.table):,} sequences", flush=True)
    if seen != set(positions) or source_signature(data_dir) != identity["sources"]:
        raise ValueError("Raw data coverage changed during CNN cache preparation.")
    validate_arrays(arrays, manifest, processor.max_length, tof_regions)
    processor.save(cache_dir / "preprocessor.json")
    np.savez_compressed(archive, **arrays)
    lengths = manifest.table["length"].to_numpy()
    metadata = {"identity": identity, "max_length": processor.max_length,
                "train_sequences": len(train), "validation_sequences": len(manifest.table) - len(train),
                "cropped_sequences": int((lengths > processor.max_length).sum()),
                "array_bytes": sum(a.nbytes for a in arrays.values())}
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Saved compact cache to {cache_dir}", flush=True)
    return arrays, processor, metadata


class CNNTensorDataset(Dataset):
    """In-memory views; train-only augmentation never changes the shared cache."""

    def __init__(self, arrays: dict, indices, *, training: bool = False, seed: int = 42,
                 dropout: SensorDropoutConfig | None = None):
        self.arrays, self.indices = arrays, np.asarray(indices, dtype=np.int64)
        self.training, self.seed = training, seed
        self.dropout = dropout or SensorDropoutConfig()
        self._rng, self._worker_seed = None, None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        position = self.indices[index]
        sample = {key: torch.from_numpy(self.arrays[key][position].copy()) for key in ARRAY_KEYS}
        sample["label"] = torch.tensor(int(self.arrays["label"][position]), dtype=torch.long)
        if self.training:
            worker = get_worker_info()
            seed = self.seed if worker is None else worker.seed
            if self._worker_seed != seed:
                self._rng, self._worker_seed = np.random.default_rng(seed), seed
            cfg, rng = self.dropout, self._rng
            if rng.random() < cfg.rotation_probability:
                dependent = rotation_dependent_imu_indices(sample["imu"].shape[-1])
                sample["imu"][:, dependent] = 0
                sample["imu_valid"][:, dependent] = False
            thm_drop = rng.random(5) < cfg.channel_probability
            tof_drop = rng.random(5) < cfg.channel_probability
            if rng.random() < cfg.thm_probability:
                thm_drop[:] = True
            if rng.random() < cfg.tof_probability:
                tof_drop[:] = True
            for key in ("thm", "thm_valid", "thm_observed"):
                sample[key][:, thm_drop] = 0
            regions_squared = sample["tof"].shape[-1] // 5
            for key in ("tof", "tof_valid", "tof_fraction"):
                sample[key][:, np.repeat(tof_drop, regions_squared)] = 0
            sample["tof_sensor_present"][:, tof_drop] = False
        return sample
