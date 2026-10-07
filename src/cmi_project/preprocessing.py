"""Fold-fitted CMI preprocessing. No global imputation or test-fitted statistics.

Arrays use time first, except ``tof_input`` which is (sensor, map, time, h, w).
Quaternion convention: scalar first (wxyz), active device-to-world rotation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .dataset_analysis import SENSOR_COLUMNS


ACC_COLUMNS = SENSOR_COLUMNS["IMU"][:3]
ROT_COLUMNS = SENSOR_COLUMNS["IMU"][3:]
IMU_FEATURES = [
    "acc_x", "acc_y", "acc_z",
    "rot6d_c0_x", "rot6d_c0_y", "rot6d_c0_z",
    "rot6d_c1_x", "rot6d_c1_y", "rot6d_c1_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "linear_acc_x", "linear_acc_y", "linear_acc_z",
]
MODALITIES = ["rotation", "thm", "tof"]


@dataclass(frozen=True)
class PreprocessingConfig:
    length_quantile: float = 0.95
    # None means radians per counter step, never an assumed sampling rate.
    sample_period_seconds: float | None = None
    gravity: float = 9.80665
    quaternion_direction: str = "device_to_world"
    canonical_hand: int = 1
    sensor_permutation: tuple[int, ...] = (0, 1, 4, 3, 2)
    # Source grids may have different orientations. Rotate BEFORE reflection.
    tof_base_rot90: tuple[int, ...] = (0, 0, 0, 0, 0)
    tof_mirror: tuple[str, ...] = ("lr", "lr", "lr", "lr", "lr")
    tof_normalization: str = "standard"
    epsilon: float = 1e-8

    def __post_init__(self):
        if not 0 < self.length_quantile <= 1:
            raise ValueError("length_quantile must be in (0, 1].")
        if self.sample_period_seconds is not None and (
            not np.isfinite(self.sample_period_seconds) or self.sample_period_seconds <= 0
        ):
            raise ValueError("sample_period_seconds must be positive or null.")
        if self.canonical_hand not in (0, 1):
            raise ValueError("canonical_hand must be 0 or 1.")
        if self.quaternion_direction not in ("device_to_world", "world_to_device"):
            raise ValueError("Unknown quaternion_direction.")
        if sorted(self.sensor_permutation) != list(range(5)):
            raise ValueError("sensor_permutation must be a permutation of 0..4.")
        # The same operation must be its own inverse for either canonical hand.
        if any(self.sensor_permutation[self.sensor_permutation[i]] != i for i in range(5)):
            raise ValueError("sensor_permutation must be an involution.")
        if len(self.tof_base_rot90) != 5 or any(k not in (0, 1, 2, 3) for k in self.tof_base_rot90):
            raise ValueError("tof_base_rot90 needs five integers in 0..3.")
        if len(self.tof_mirror) != 5 or any(x not in ("lr", "ud", "both", "none") for x in self.tof_mirror):
            raise ValueError("tof_mirror needs five of: lr, ud, both, none.")
        if any(self.tof_mirror[i] != self.tof_mirror[self.sensor_permutation[i]] for i in range(5)):
            raise ValueError("Swapped sensors must use the same mirror operation.")
        if self.tof_normalization not in ("standard", "minmax"):
            raise ValueError("tof_normalization must be standard or minmax.")
        if not np.isfinite(self.gravity) or self.gravity <= 0 or not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("gravity and epsilon must be finite and positive.")

    @classmethod
    def from_dict(cls, values: dict) -> "PreprocessingConfig":
        values = dict(values)
        for key in ("sensor_permutation", "tof_base_rot90", "tof_mirror"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def demographics_lookup(demographics: pd.DataFrame) -> dict[str, int]:
    if not {"subject", "handedness"}.issubset(demographics.columns):
        raise ValueError("Demographics requires subject and handedness.")
    if demographics["subject"].isna().any() or demographics["subject"].astype(str).duplicated().any():
        raise ValueError("Demographics subjects must be non-null and unique.")
    hands = pd.to_numeric(demographics["handedness"], errors="raise")
    if not hands.isin([0, 1]).all():
        raise ValueError("Unknown handedness; expected 0=left, 1=right.")
    return dict(zip(demographics["subject"].astype(str), hands.astype(int)))


def sequence_metadata(sequence: pd.DataFrame) -> tuple[str, str, str | None]:
    if sequence.empty:
        raise ValueError("Empty sequence.")
    values = []
    for key in ("sequence_id", "subject"):
        if key not in sequence or sequence[key].isna().any() or sequence[key].astype(str).nunique() != 1:
            raise ValueError(f"Every sequence must have one non-null {key}.")
        values.append(str(sequence[key].iloc[0]))
    label = None
    if "gesture" in sequence:
        if sequence["gesture"].isna().any() or sequence["gesture"].astype(str).nunique() != 1:
            raise ValueError(f"Conflicting/missing gesture in {values[0]}.")
        label = str(sequence["gesture"].iloc[0])
    return values[0], values[1], label


def sensor_values(sequence: pd.DataFrame, columns: list[str]) -> np.ndarray:
    # Optional absent sensor columns are genuinely missing, never measured zero.
    return sequence.reindex(columns=columns).to_numpy(dtype=np.float64, na_value=np.nan)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(axis=-1, keepdims=True)
    xyz = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate([w, xyz], axis=-1)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Active rotation matrix for normalized, scalar-first quaternions."""
    w, x, y, z = q.T
    return np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ], axis=-1).reshape(-1, 3, 3)


def transform_grids(grids: np.ndarray, mirrored: bool, config: PreprocessingConfig) -> np.ndarray:
    """Apply identical source rotation, channel order and reflection to ANY map."""
    out = np.stack([
        np.rot90(grids[:, s], k=config.tof_base_rot90[s], axes=(-2, -1)) for s in range(5)
    ], axis=1)
    if mirrored:
        out = out[:, config.sensor_permutation].copy()
        for sensor, mode in enumerate(config.tof_mirror):
            if mode in ("lr", "both"):
                out[:, sensor] = out[:, sensor, :, ::-1]
            if mode in ("ud", "both"):
                out[:, sensor] = out[:, sensor, ::-1, :]
    return out.copy()


def engineer_sequence(sequence: pd.DataFrame, handedness: int,
                      config: PreprocessingConfig) -> dict:
    """Compute physical features BEFORE scaling and BEFORE cropping/padding.

    Reflection S=diag(-1,1,1) acts as a'=S a and R'=S R S. This is
    q'=(w,x,-y,-z), and angular velocity (an axial vector) becomes (wx,-wy,-wz).
    Missing rotation is never interpolated; velocity needs adjacent valid poses.
    """
    sequence_id, subject, label = sequence_metadata(sequence)
    if handedness not in (0, 1):
        raise ValueError("handedness must be 0 or 1.")
    missing = set(ACC_COLUMNS + ["sequence_counter"]) - set(sequence.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    counters = pd.to_numeric(sequence["sequence_counter"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(counters).all() or (counters < 0).any() or (counters != np.floor(counters)).any():
        raise ValueError("sequence_counter must contain finite, non-negative integers.")
    order = np.argsort(counters, kind="stable")
    sequence = sequence.iloc[order]
    counters = counters[order]
    if (np.diff(counters) <= 0).any():
        raise ValueError(f"Duplicate sequence_counter in {sequence_id}.")
    mirrored = handedness != config.canonical_hand
    acc = sensor_values(sequence, ACC_COLUMNS)
    if mirrored:
        acc[:, 0] *= -1
    acc_valid = np.isfinite(acc)
    q = sensor_values(sequence, ROT_COLUMNS)
    norm = np.linalg.norm(np.where(np.isfinite(q), q, 0), axis=1)
    rotation_valid = np.isfinite(q).all(axis=1) & np.isfinite(norm) & (norm > config.epsilon)
    q[~rotation_valid] = [1, 0, 0, 0]  # Numerical placeholder; masks stay FALSE.
    q /= np.where(rotation_valid, norm, 1)[:, None]
    if config.quaternion_direction == "world_to_device":
        q[:, 1:] *= -1
    if mirrored:
        q[:, 2:] *= -1
    rotation = quaternion_to_matrix(q)
    # Columns are concatenated as c0_xyz, c1_xyz, not interleaved row-major.
    rot6d = rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    gravity_device = rotation[:, 2, :] * config.gravity  # R.T @ [0,0,g].
    linear_acc = acc - gravity_device
    velocity = np.zeros_like(acc)
    velocity_valid = np.zeros(len(sequence), dtype=bool)
    if len(sequence) > 1:
        previous_inverse = q[:-1].copy()
        previous_inverse[:, 1:] *= -1
        relative = quaternion_multiply(previous_inverse, q[1:])
        relative[relative[:, 0] < 0] *= -1  # q and -q: same shortest rotation.
        xyz_norm = np.linalg.norm(relative[:, 1:], axis=1)
        angles = 2 * np.arctan2(xyz_norm, np.clip(relative[:, 0], 0, 1))
        factors = np.divide(angles, xyz_norm, out=np.full_like(angles, 2), where=xyz_norm > config.epsilon)
        dt = np.diff(counters)
        if config.sample_period_seconds is not None:
            dt = dt * config.sample_period_seconds
        velocity[1:] = relative[:, 1:] * factors[:, None] / dt[:, None]
        velocity_valid[1:] = rotation_valid[:-1] & rotation_valid[1:] & (np.diff(counters) == 1)
    imu = np.concatenate([acc, rot6d, velocity, linear_acc], axis=1)
    imu_valid = np.concatenate([
        acc_valid, np.repeat(rotation_valid[:, None], 6, axis=1),
        np.repeat(velocity_valid[:, None], 3, axis=1), acc_valid & rotation_valid[:, None],
    ], axis=1) & np.isfinite(imu)
    imu = np.where(imu_valid, imu, 0)

    thm = sensor_values(sequence, SENSOR_COLUMNS["THM"])
    if mirrored:
        thm = thm[:, config.sensor_permutation]
    thm_observed = np.isfinite(thm)
    # Fill only this full sequence; an all-missing channel remains missing.
    thm = pd.DataFrame(np.where(thm_observed, thm, np.nan)).ffill().bfill().to_numpy()
    thm_valid = np.isfinite(thm)
    thm = np.where(thm_valid, thm, 0)

    raw_tof = sensor_values(sequence, SENSOR_COLUMNS["ToF"]).reshape(-1, 5, 8, 8)
    # Generate every diagnostic mask BEFORE replacing any raw value.
    tof_valid = np.isfinite(raw_tof) & (raw_tof >= 0)
    tof_nan = np.isnan(raw_tof)
    tof_minus1 = raw_tof == -1
    # -1 means no response from a present sensor, not absent hardware.
    tof_present = np.isfinite(raw_tof) & ((raw_tof >= 0) | tof_minus1)
    tof = transform_grids(np.where(tof_valid, raw_tof, 0), mirrored, config)
    tof_valid = transform_grids(tof_valid, mirrored, config)
    tof_nan = transform_grids(tof_nan, mirrored, config)
    tof_minus1 = transform_grids(tof_minus1, mirrored, config)
    tof_present = transform_grids(tof_present, mirrored, config).any(axis=(-1, -2))
    return {
        "sequence_id": sequence_id, "subject": subject, "gesture": label,
        "imu": imu, "imu_valid": imu_valid, "rotation_valid": rotation_valid,
        "thm": thm, "thm_valid": thm_valid, "thm_observed": thm_observed,
        "tof": tof, "tof_valid": tof_valid, "tof_nan": tof_nan,
        "tof_minus1": tof_minus1, "tof_sensor_present": tof_present,
        "sequence_counter": counters.astype(np.int64),
    }


class ChannelStatistics:
    """Merge batch moments in float64, using only explicitly observed values."""

    def __init__(self, channels: int):
        self.count = np.zeros(channels, dtype=np.int64)
        self.mean = np.zeros(channels)
        self.m2 = np.zeros(channels)
        self.minimum = np.full(channels, np.inf)
        self.maximum = np.full(channels, -np.inf)

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, len(self.count))
        valid = np.asarray(valid, dtype=bool).reshape(values.shape) & np.isfinite(values)
        count = valid.sum(axis=0)
        clean = np.where(valid, values, 0)
        mean = clean.sum(axis=0) / np.maximum(count, 1)
        m2 = np.square(np.where(valid, values - mean, 0)).sum(axis=0)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta**2 * self.count * count / np.maximum(total, 1)
        self.mean += delta * count / np.maximum(total, 1)
        self.count = total
        self.minimum = np.minimum(self.minimum, np.where(valid, values, np.inf).min(axis=0))
        self.maximum = np.maximum(self.maximum, np.where(valid, values, -np.inf).max(axis=0))

    def finish(self, epsilon: float, method: str = "standard") -> dict:
        seen = self.count > 0
        minimum, maximum = np.where(seen, self.minimum, 0), np.where(seen, self.maximum, 0)
        offset = self.mean if method == "standard" else minimum
        scale = np.sqrt(self.m2 / np.maximum(self.count, 1)) if method == "standard" else maximum - minimum
        scale = np.where(scale > epsilon, scale, 1)
        return {"method": method, "count": self.count.tolist(), "offset": offset.tolist(),
                "scale": scale.tolist(), "minimum": minimum.tolist(), "maximum": maximum.tolist()}


def normalize(values: np.ndarray, valid: np.ndarray, statistics: dict) -> tuple[np.ndarray, np.ndarray]:
    # Channels never observed in this TRAIN fold cannot be normalized from test.
    valid = valid & (np.asarray(statistics["count"]) > 0)
    out = np.where(valid, (values - statistics["offset"]) / statistics["scale"], 0)
    return out.astype(np.float32), valid


def left_pad_tail(values: np.ndarray, length: int, fill=0) -> np.ndarray:
    result = np.full((length, *values.shape[1:]), fill, dtype=values.dtype)
    kept = values[-length:]
    result[-len(kept):] = kept
    return result


def refresh_input_availability(sample: dict) -> None:
    sample["modality_available"] = np.stack([
        sample["imu_valid"][:, 3:9].any(axis=1), sample["thm_valid"].any(axis=1),
        sample["tof_sensor_present"].any(axis=1),
    ], axis=1) & sample["time_mask"][:, None]
    sample["sequence_available"] = sample["modality_available"].any(axis=0)
    sample["thm_channel_available"] = sample["thm_valid"].any(axis=0)
    sample["tof_channel_available"] = sample["tof_sensor_present"].any(axis=0)
    # Each sensor supplies two channels to Conv3d: distance and validity.
    sample["tof_input"] = np.stack([sample["tof"], sample["tof_valid"].astype(np.float32)], axis=2).transpose(1, 2, 0, 3, 4).copy()


class FoldPreprocessor:
    VERSION = 1

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.state: dict | None = None

    def fit(self, sequences: Iterable[pd.DataFrame], demographics: pd.DataFrame) -> "FoldPreprocessor":
        hands = demographics_lookup(demographics)
        imu_stats, thm_stats, tof_stats = ChannelStatistics(15), ChannelStatistics(5), ChannelStatistics(5)
        lengths, subjects, sequence_ids, labels = [], set(), set(), set()
        for sequence in sequences:
            sequence_id, subject, label = sequence_metadata(sequence)
            if sequence_id in sequence_ids:
                raise ValueError(f"Duplicate fitting sequence: {sequence_id}")
            if subject not in hands:
                raise ValueError(f"Missing demographics for {subject}.")
            if label is None:
                raise ValueError("fit requires training gesture labels.")
            raw = engineer_sequence(sequence, hands[subject], self.config)
            imu_stats.update(raw["imu"], raw["imu_valid"])
            # Imputed THM values are useful inputs, but DO NOT count as observations.
            thm_stats.update(raw["thm"], raw["thm_observed"])
            tof_stats.update(raw["tof"].transpose(0, 2, 3, 1), raw["tof_valid"].transpose(0, 2, 3, 1))
            lengths.append(len(sequence))
            subjects.add(subject)
            sequence_ids.add(sequence_id)
            labels.add(label)
        if not lengths:
            raise ValueError("No training sequences to fit.")
        self.state = {
            "version": self.VERSION, "config": asdict(self.config), "imu_features": IMU_FEATURES,
            "modality_order": MODALITIES, "max_length": int(np.ceil(np.quantile(lengths, self.config.length_quantile))),
            "angular_velocity_unit": "rad/s" if self.config.sample_period_seconds is not None else "rad/counter_step",
            "train_subjects": sorted(subjects), "train_sequence_ids": sorted(sequence_ids),
            "labels": sorted(labels), "train_sequences": len(lengths), "train_frames": sum(lengths),
            "statistics": {"imu": imu_stats.finish(self.config.epsilon), "thm": thm_stats.finish(self.config.epsilon),
                           "tof": tof_stats.finish(self.config.epsilon, self.config.tof_normalization)},
        }
        return self

    @property
    def max_length(self) -> int:
        if self.state is None:
            raise RuntimeError("Fit or load this preprocessor first.")
        return self.state["max_length"]

    def transform(self, sequence: pd.DataFrame, demographics: pd.DataFrame,
                  split: str = "test") -> dict:
        length = self.max_length
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test.")
        sequence_id, subject, label = sequence_metadata(sequence)
        if split == "validation" and subject in self.state["train_subjects"]:
            raise ValueError(f"Subject leakage: validation subject {subject} appeared in fit.")
        if split == "train" and sequence_id not in self.state["train_sequence_ids"]:
            raise ValueError(f"Training sequence {sequence_id} did not appear in fit.")
        hands = demographics_lookup(demographics)
        if subject not in hands:
            raise ValueError(f"Missing demographics for {subject}.")
        raw = engineer_sequence(sequence, hands[subject], self.config)
        statistics = self.state["statistics"]
        raw["imu"], raw["imu_valid"] = normalize(raw["imu"], raw["imu_valid"], statistics["imu"])
        raw["thm"], raw["thm_valid"] = normalize(raw["thm"], raw["thm_valid"], statistics["thm"])
        tof, tof_valid = normalize(raw["tof"].transpose(0, 2, 3, 1), raw["tof_valid"].transpose(0, 2, 3, 1), statistics["tof"])
        raw["tof"], raw["tof_valid"] = tof.transpose(0, 3, 1, 2), tof_valid.transpose(0, 3, 1, 2)
        target = -1
        if label is not None:
            if label not in self.state["labels"]:
                raise ValueError(f"Label {label!r} was never observed in the training fold.")
            target = self.state["labels"].index(label)
        sample = {
            "sequence_id": sequence_id, "subject": subject, "label": np.int64(target),
            "original_length": np.int64(len(sequence)), "kept_length": np.int64(min(len(sequence), length)),
            "source_sequence_available": np.asarray([
                raw["rotation_valid"].any(), raw["thm_observed"].any(), raw["tof_sensor_present"].any(),
            ], dtype=bool),
            "source_thm_channel_available": raw["thm_observed"].any(axis=0),
            "source_tof_channel_available": raw["tof_sensor_present"].any(axis=0),
            "time_mask": np.arange(length) >= length - min(len(sequence), length),
        }
        for key in ("imu", "imu_valid", "rotation_valid", "thm", "thm_valid", "thm_observed",
                    "tof", "tof_valid", "tof_nan", "tof_minus1", "tof_sensor_present", "sequence_counter"):
            sample[key] = left_pad_tail(raw[key], length, -1 if key == "sequence_counter" else 0)
        refresh_input_availability(sample)
        return sample

    def save(self, path: Path) -> None:
        self.max_length
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FoldPreprocessor":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != cls.VERSION or state.get("imu_features") != IMU_FEATURES:
            raise ValueError("Incompatible preprocessing parameters/schema.")
        if not isinstance(state.get("max_length"), int) or state["max_length"] < 1:
            raise ValueError("Invalid saved max_length.")
        for modality, channels in (("imu", 15), ("thm", 5), ("tof", 5)):
            stats = state["statistics"][modality]
            for key in ("count", "offset", "scale"):
                if len(stats[key]) != channels or not np.isfinite(stats[key]).all():
                    raise ValueError(f"Invalid saved {modality} {key}.")
            if (np.asarray(stats["scale"]) <= 0).any() or (np.asarray(stats["count"]) < 0).any():
                raise ValueError(f"Invalid saved {modality} statistics.")
        result = cls(PreprocessingConfig.from_dict(state["config"]))
        result.state = state
        return result


@dataclass(frozen=True)
class SensorDropoutConfig:
    rotation_probability: float = 0.05
    thm_probability: float = 0.10
    tof_probability: float = 0.10
    channel_probability: float = 0.03

    def __post_init__(self):
        if any(not 0 <= p <= 1 for p in asdict(self).values()):
            raise ValueError("Dropout probabilities must be in [0,1].")


def sensor_dropout(sample: dict, rng: np.random.Generator, *, training: bool,
                   config: SensorDropoutConfig | None = None) -> dict:
    """Drop normalized inputs on each training access, never fit/cache statistics.

    Source availability and NaN/-1/observed masks retain raw provenance.
    Effective value/validity/presence/availability maps change together.
    """
    out = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in sample.items()}
    if not training:
        return out
    config = config or SensorDropoutConfig()
    drop_rot = rng.random() < config.rotation_probability
    drop_thm = rng.random(5) < config.channel_probability
    drop_tof = rng.random(5) < config.channel_probability
    if rng.random() < config.thm_probability:
        drop_thm[:] = True
    if rng.random() < config.tof_probability:
        drop_tof[:] = True
    if drop_rot:
        out["imu"][:, 3:] = 0
        out["imu_valid"][:, 3:] = False
        out["rotation_valid"][:] = False
    out["thm"][:, drop_thm] = 0
    out["thm_valid"][:, drop_thm] = False
    out["tof"][:, drop_tof] = 0
    out["tof_valid"][:, drop_tof] = False
    out["tof_sensor_present"][:, drop_tof] = False
    refresh_input_availability(out)
    return out


def tof_region_pool(distance: np.ndarray, validity: np.ndarray, regions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Mean of VALID pixels in a regions x regions partition; empty regions -> 0.

    Input (...,5,8,8); output (...,5,regions,regions) and a matching boolean mask.
    regions=1 is a per-frame/per-sensor mean baseline. Works before/after scaling.
    """
    if regions not in (1, 2, 4, 8) or distance.shape != validity.shape or distance.shape[-2:] != (8, 8):
        raise ValueError("Expected matching 8x8 maps and regions in {1,2,4,8}.")
    side = 8 // regions
    shape = (*distance.shape[:-2], regions, side, regions, side)
    valid = validity.astype(bool).reshape(shape)
    count = valid.sum(axis=(-3, -1))
    total = np.where(valid, distance.reshape(shape), 0).sum(axis=(-3, -1))
    return (total / np.maximum(count, 1)).astype(np.float32), count > 0


def iter_csv_sequences(path: Path, *, chunksize: int = 25000,
                       sequence_ids: set[str] | None = None, require_gesture: bool = False,
                       sensor_modalities: tuple[str, ...] | None = None) -> Iterator[pd.DataFrame]:
    """Read contiguous sequences, retaining a final partial sequence across chunks.

    Fails on reappearing/non-contiguous IDs instead of silently splitting samples.
    Rows within a sequence are sorted by counter during feature engineering.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    modalities = tuple(SENSOR_COLUMNS) if sensor_modalities is None else sensor_modalities
    if not modalities or set(modalities) - set(SENSOR_COLUMNS) or len(set(modalities)) != len(modalities):
        raise ValueError("sensor_modalities must contain unique known sensor names.")
    selected_sensors = [column for modality in modalities for column in SENSOR_COLUMNS[modality]]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["sequence_id", "sequence_counter", "subject", *ACC_COLUMNS]
    if require_gesture:
        required.append("gesture")
    absent = set(required) - set(header)
    if absent:
        raise ValueError(f"CSV missing required columns: {sorted(absent)}")
    metadata = [c for c in ("sequence_id", "sequence_counter", "subject", "gesture") if c in header]
    columns = metadata + [c for c in selected_sensors if c in header]
    dtype = {c: "float32" for c in selected_sensors if c in header}
    dtype.update({c: "string" for c in metadata if c != "sequence_counter"})
    dtype["sequence_counter"] = "float64"
    seen: set[str] = set()
    pending = None
    with pd.read_csv(path, usecols=columns, dtype=dtype, chunksize=chunksize) as reader:
        for chunk in reader:
            if chunk["sequence_id"].isna().any():
                raise ValueError("CSV contains null sequence_id.")
            if pending is not None:
                chunk = pd.concat([pending, chunk], ignore_index=True)
            ids = chunk["sequence_id"].to_numpy(dtype=str)
            boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
            for start, end in zip(boundaries[:-2], boundaries[1:-1]):
                sequence_id = ids[start]
                if sequence_id in seen:
                    raise ValueError(f"Non-contiguous sequence_id: {sequence_id}")
                seen.add(sequence_id)
                if sequence_ids is None or sequence_id in sequence_ids:
                    yield chunk.iloc[start:end].copy()
            pending = chunk.iloc[boundaries[-2]:].copy()
            if str(pending["sequence_id"].iloc[0]) in seen:
                raise ValueError(f"Non-contiguous sequence_id: {pending['sequence_id'].iloc[0]}")
    if pending is not None:
        sequence_id = str(pending["sequence_id"].iloc[0])
        seen.add(sequence_id)
        if sequence_ids is None or sequence_id in sequence_ids:
            yield pending
    if sequence_ids is not None and (sequence_ids - seen):
        raise ValueError(f"Requested sequence IDs absent from CSV: {sorted(sequence_ids - seen)[:5]}")
