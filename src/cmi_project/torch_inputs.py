"""Cached sensor input adapter and shared padding-aware temporal operations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocessing import SensorDropoutConfig, refresh_input_availability, sensor_dropout


class CMIFoldDataset(Dataset):
    """Load one cached sample at a time; apply sensor dropout ONLY to train."""

    def __init__(self, manifest_path: Path, *, training: bool = False,
                 dropout_config: SensorDropoutConfig | None = None):
        self.manifest_path = Path(manifest_path).resolve()
        if training and self.manifest_path.name != "train_manifest.csv":
            raise ValueError("Sensor dropout is only permitted for train_manifest.csv.")
        self.training = training
        self.records = pd.read_csv(self.manifest_path).to_dict("records")
        config_path = self.manifest_path.parent / "sensor_dropout.json"
        if dropout_config is None and config_path.is_file():
            dropout_config = SensorDropoutConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        self.dropout_config = dropout_config or SensorDropoutConfig()
        self._rng = None
        self._seed = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = (self.manifest_path.parent / self.records[index]["path"]).resolve()
        if self.manifest_path.parent not in path.parents:
            raise ValueError("Sample path escapes the fold directory.")
        with np.load(path, allow_pickle=False) as stored:
            sample = {key: stored[key].copy() for key in stored.files}
        refresh_input_availability(sample)
        seed = torch.initial_seed()  # DataLoader gives each worker its own seed.
        if self._rng is None or self._seed != seed:
            self._rng, self._seed = np.random.default_rng(seed), seed
        sample = sensor_dropout(sample, self._rng, training=self.training, config=self.dropout_config)
        result = {
            "sequence_id": str(sample["sequence_id"]), "subject": str(sample["subject"]),
            "label": torch.as_tensor(sample["label"], dtype=torch.long),
        }
        # Raw provenance remains in NPZ; only effective masks enter the model.
        for key in ("imu", "imu_valid", "thm", "thm_valid", "tof_input", "tof_sensor_present",
                    "time_mask", "modality_available", "sequence_available"):
            result[key] = torch.from_numpy(np.ascontiguousarray(sample[key]))
        result["thm_observed"] = torch.from_numpy(sample["thm_observed"] & sample["thm_valid"])
        return result


def downsample_time_mask(mask: torch.Tensor, kernel_size: int, stride: int = 1,
                         padding: int = 0, dilation: int = 1) -> torch.Tensor:
    """Conv output is available iff its receptive field includes a valid input.

    Uses exactly the same temporal convolution geometry, including dilation.
    Accepts (batch,time). Output positions with only padding stay unavailable.
    """
    if mask.ndim != 2 or min(kernel_size, stride, dilation) < 1 or padding < 0:
        raise ValueError("Expected a (batch,time) mask and valid convolution geometry.")
    weights = torch.ones(1, 1, kernel_size, device=mask.device, dtype=torch.float32)
    return F.conv1d(mask[:, None].float(), weights, stride=stride, padding=padding, dilation=dilation)[:, 0] > 0


def masked_mean_time(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of (batch,time,features), excluding padding; all-missing -> zeros."""
    clean = torch.where(mask[..., None], values, torch.zeros_like(values))
    return clean.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
