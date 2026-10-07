"""V1 temporal CNNs: IMU only, or three independently encoded modalities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .torch_inputs import downsample_time_mask, masked_mean_time


@dataclass(frozen=True)
class CNNConfig:
    imu_channels: tuple[int, ...] = (32, 64, 96)
    auxiliary_channels: tuple[int, ...] = (16, 32, 48)
    kernel_size: int = 5
    hidden_features: int = 128
    dropout: float = 0.2
    pooling: str = "mean_max"
    stage_kernel_sizes: tuple[int, ...] = ()

    def __post_init__(self):
        for widths in (self.imu_channels, self.auxiliary_channels):
            if not widths or any(not isinstance(c, int) or c < 2 for c in widths):
                raise ValueError("CNN channel widths must be integers >= 2.")
        if self.kernel_size < 1 or self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.hidden_features < 2 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid hidden_features/dropout.")
        if self.pooling not in ("mean_max", "attention_max"):
            raise ValueError("pooling must be mean_max or attention_max.")
        if self.stage_kernel_sizes and (len(self.stage_kernel_sizes) != len(self.imu_channels)
                or len(self.stage_kernel_sizes) != len(self.auxiliary_channels)
                or any(not isinstance(k, int) or k < 1 or k % 2 != 1 for k in self.stage_kernel_sizes)):
            raise ValueError("stage_kernel_sizes needs one positive odd kernel per encoder stage.")

    @classmethod
    def from_dict(cls, values: dict) -> "CNNConfig":
        values = dict(values)
        for key in ("imu_channels", "auxiliary_channels", "stage_kernel_sizes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


class TokenChannelNorm1d(nn.Module):
    """LayerNorm across channels at each time step, excluding other tokens."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class MaskedResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: int, stride: int, dropout: float):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv1 = nn.Conv1d(inputs, outputs, kernel_size, stride, kernel_size // 2, bias=False)
        self.conv2 = nn.Conv1d(outputs, outputs, 3, padding=1, bias=False)
        self.norm1, self.norm2 = TokenChannelNorm1d(outputs), TokenChannelNorm1d(outputs)
        self.skip = nn.Conv1d(inputs, outputs, 1, stride=stride, bias=False)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.where(mask[:, None], values, 0)
        residual = self.skip(values)
        out = self.conv1(values)
        first_mask = downsample_time_mask(mask, self.kernel_size, self.stride, self.kernel_size // 2)
        out = self.dropout(F.gelu(self.norm1(out)))
        out = torch.where(first_mask[:, None], out, 0)
        out = self.conv2(out)
        final_mask = downsample_time_mask(first_mask, 3, padding=1)
        out = F.gelu(self.norm2(out) + residual)
        return torch.where(final_mask[:, None], out, 0), final_mask


class TemporalCNNEncoder(nn.Module):
    def __init__(self, inputs: int, channels: tuple[int, ...], kernel_size: int = 5,
                 dropout: float = 0.2, pooling: str = "mean_max", stage_kernel_sizes: tuple[int, ...] = ()):
        super().__init__()
        blocks = []
        for i, output in enumerate(channels):
            kernel = stage_kernel_sizes[i] if stage_kernel_sizes else kernel_size
            blocks.append(MaskedResidualBlock(inputs, output, kernel, 1 if i == 0 else 2, dropout))
            inputs = output
        self.blocks = nn.ModuleList(blocks)
        self.output_features = 2 * channels[-1]
        self.attention = (nn.Sequential(nn.Conv1d(channels[-1], max(4, channels[-1] // 4), 1),
            nn.Tanh(), nn.Conv1d(max(4, channels[-1] // 4), 1, 1)) if pooling == "attention_max" else None)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = values.transpose(1, 2)
        for block in self.blocks:
            out, mask = block(out, mask)
        if self.attention is None:
            mean = masked_mean_time(out.transpose(1, 2), mask)
        else:
            scores = self.attention(out).squeeze(1).masked_fill(~mask, -torch.inf)
            # Empty modalities must have zero weights, finite gradients, and
            # zero embeddings. Softmax of an all-negative-infinity row is NaN.
            scores = torch.where(mask.any(dim=1, keepdim=True), scores, 0)
            weights = scores.softmax(dim=1) * mask
            mean = (out * weights[:, None]).sum(dim=2)
        maximum = out.masked_fill(~mask[:, None], -torch.inf).amax(dim=2)
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, 0)
        return torch.cat([mean, maximum], dim=1)


class CMI1DCNN(nn.Module):
    """18-class Model A (imu) or Model B (multisensor), with explicit masks.

    ToF uses masked spatial region means + pixel-valid fractions + hardware
    presence, then ordinary temporal Conv1d. No Conv2d/Conv3d is used here.
    """

    def __init__(self, model_name: str = "imu", *, tof_regions: int = 2,
                 config: CNNConfig | None = None, num_classes: int = 18):
        super().__init__()
        if model_name not in ("imu", "multisensor") or tof_regions not in (1, 2, 4, 8):
            raise ValueError("Expected model imu/multisensor and tof_regions in {1,2,4,8}.")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        self.model_name, self.tof_regions = model_name, tof_regions
        self.config = config or CNNConfig()
        self.imu_encoder = TemporalCNNEncoder(30, self.config.imu_channels,
            self.config.kernel_size, self.config.dropout, self.config.pooling, self.config.stage_kernel_sizes)
        features = self.imu_encoder.output_features + 1
        if model_name == "multisensor":
            self.thm_encoder = TemporalCNNEncoder(15, self.config.auxiliary_channels,
                self.config.kernel_size, self.config.dropout, self.config.pooling, self.config.stage_kernel_sizes)
            self.tof_encoder = TemporalCNNEncoder(10 * tof_regions**2 + 5, self.config.auxiliary_channels,
                self.config.kernel_size, self.config.dropout, self.config.pooling, self.config.stage_kernel_sizes)
            features += self.thm_encoder.output_features + self.tof_encoder.output_features + 2
        self.classifier = nn.Sequential(
            nn.Linear(features, self.config.hidden_features), nn.LayerNorm(self.config.hidden_features),
            nn.GELU(), nn.Dropout(self.config.dropout), nn.Linear(self.config.hidden_features, num_classes),
        )

    def metadata(self) -> dict:
        architecture = ("masked_residual_1d_cnn_v2" if self.config.pooling != "mean_max"
                        or self.config.stage_kernel_sizes else "masked_residual_1d_cnn_v1")
        return {"model": self.model_name, "architecture": architecture,
                "config": asdict(self.config), "tof_regions": self.tof_regions,
                "normalization": "train-fold standardization + per-token channel LayerNorm",
                "parameters": sum(p.numel() for p in self.parameters())}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        time = batch["time_mask"].bool()
        imu_valid = batch["imu_valid"].bool() & time[..., None]
        imu = torch.cat([torch.where(imu_valid, batch["imu"], 0), imu_valid.float()], dim=-1)
        imu_mask = imu_valid.any(dim=-1)
        parts = [self.imu_encoder(imu, imu_mask)]
        availability = [imu_mask.any(dim=1)]
        if self.model_name == "multisensor":
            thm_valid = batch["thm_valid"].bool() & time[..., None]
            thm = torch.cat([torch.where(thm_valid, batch["thm"], 0), thm_valid.float(),
                             (batch["thm_observed"].bool() & thm_valid).float()], dim=-1)
            thm_mask = thm_valid.any(dim=-1)
            presence = batch["tof_sensor_present"].bool() & time[..., None]
            region_presence = presence.repeat_interleave(self.tof_regions**2, dim=-1)
            tof_valid = batch["tof_valid"].bool() & region_presence
            tof = torch.cat([torch.where(tof_valid, batch["tof"], 0),
                             torch.where(region_presence, batch["tof_fraction"], 0), presence.float()], dim=-1)
            tof_mask = presence.any(dim=-1)
            parts.extend([self.thm_encoder(thm, thm_mask), self.tof_encoder(tof, tof_mask)])
            availability.extend([thm_mask.any(dim=1), tof_mask.any(dim=1)])
        parts.append(torch.stack(availability, dim=1).float())
        return self.classifier(torch.cat(parts, dim=1))
