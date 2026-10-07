"""Optional PyTorch input adapter and padding-aware model building blocks.

The preprocessor itself has no PyTorch dependency. This module supplies an
example ToF branch, not a trained classifier or a full training pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
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


class MaskedAttentionPool(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.score = nn.Linear(features, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        clean = torch.where(mask[..., None], values, torch.zeros_like(values))
        logits = self.score(clean).squeeze(-1).masked_fill(~mask, -torch.inf)
        # Softmax(all -inf) is NaN. Use a safe row, then force its weights to zero.
        logits = torch.where(mask.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=1).masked_fill(~mask, 0)
        return (clean * weights[..., None]).sum(dim=1)


class MaskedSelfAttention(nn.Module):
    """key_padding_mask excludes padded keys; re-mask outputs of padded queries."""

    def __init__(self, features: int, heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(features, heads, batch_first=True)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        clean = torch.where(mask[..., None], values, torch.zeros_like(values))
        safe_mask = mask.clone()
        safe_mask[~mask.any(dim=1), 0] = True
        out, _ = self.attention(clean, clean, clean, key_padding_mask=~safe_mask, need_weights=False)
        return torch.where(mask[..., None], out, torch.zeros_like(out))


class TokenChannelNorm3d(nn.Module):
    """Normalize channels at EACH (time,h,w); padding never enters another token.

    BatchNorm3d/GroupNorm aggregate over time or space, so ordinary versions
    would let padding change normalization statistics.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class ToF3DEncoder(nn.Module):
    """Shared sensor-specific spatiotemporal CNN using distance AND validity.

    Input: (B,5,2,T,8,8), time_mask (B,T), sensor_present (B,T,5).
    Output: temporal tokens (B,T',5*C), time_mask (B,T'), and
    modality_mask (B,T') for ToF-only attention/pooling.
    """

    def __init__(self, channels: tuple[int, ...] = (16, 32)):
        super().__init__()
        if not channels or any(c <= 0 for c in channels):
            raise ValueError("channels must be positive.")
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        previous = 2
        for channels_out in channels:
            self.convolutions.append(nn.Conv3d(previous, channels_out, 3, stride=2, padding=1))
            self.normalizations.append(TokenChannelNorm3d(channels_out))
            previous = channels_out
        self.output_features = 5 * previous

    def forward(self, tof_input: torch.Tensor, time_mask: torch.Tensor,
                sensor_present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tof_input.ndim != 6 or tof_input.shape[1:3] != (5, 2) or tof_input.shape[-2:] != (8, 8):
            raise ValueError("tof_input must have shape (B,5,2,T,8,8).")
        batch, sensors, maps, time, height, width = tof_input.shape
        if time_mask.shape != (batch, time) or sensor_present.shape != (batch, time, sensors):
            raise ValueError("Mismatched time/sensor masks.")
        values = tof_input.reshape(batch * sensors, maps, time, height, width)
        sensor_mask = (sensor_present & time_mask[..., None]).permute(0, 2, 1).reshape(batch * sensors, time)
        output_time_mask = time_mask
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            values = torch.where(sensor_mask[:, None, :, None, None], values, torch.zeros_like(values))
            values = convolution(values)
            sensor_mask = downsample_time_mask(sensor_mask, 3, 2, 1)
            output_time_mask = downsample_time_mask(output_time_mask, 3, 2, 1)
            values = F.gelu(normalization(values))
            values = torch.where(sensor_mask[:, None, :, None, None], values, torch.zeros_like(values))
        # Validity is an input map: spatial convolution learns patterns of no response.
        # Temporal aggregation is left to a masked pool, rather than an unmasked mean.
        tokens = values.mean(dim=(-1, -2)).transpose(1, 2)
        tokens = tokens.reshape(batch, sensors, tokens.shape[1], tokens.shape[2]).permute(0, 2, 1, 3).flatten(2)
        modality_mask = sensor_mask.reshape(batch, sensors, -1).any(dim=1) & output_time_mask
        return tokens, output_time_mask, modality_mask
