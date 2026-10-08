"""Selected grouped, masked SE CNN: IMU-only and multisensor branches."""

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
    encoder_style: str = "grouped"
    normalization: str = "masked_batch"
    squeeze_excitation: bool = True
    stem_channels: tuple[int, ...] = (16, 32)
    imu_feature_count: int = 34

    def __post_init__(self):
        for widths in (self.imu_channels, self.auxiliary_channels):
            if not widths or any(not isinstance(c, int) or c < 2 for c in widths):
                raise ValueError("CNN channel widths must be integers >= 2.")
        if self.kernel_size < 1 or self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.hidden_features < 2 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid hidden_features/dropout.")
        # Retain serialized fields so the scored checkpoints load unchanged.
        if (self.pooling != "mean_max" or self.encoder_style != "grouped"
                or self.normalization != "masked_batch" or not self.squeeze_excitation
                or self.stage_kernel_sizes):
            raise ValueError("Use grouped masked SE CNN with mean/max pooling and a shared kernel.")
        if self.imu_feature_count not in (15, 34):
            raise ValueError("imu_feature_count must be 15 or 34.")
        if not self.stem_channels or any(not isinstance(c, int) or c < 2 for c in self.stem_channels):
            raise ValueError("stem_channels must contain integers >= 2.")

    @classmethod
    def from_dict(cls, values: dict) -> "CNNConfig":
        values = dict(values)
        for key in ("imu_channels", "auxiliary_channels", "stage_kernel_sizes", "stem_channels"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


class MaskedBatchNorm1d(nn.Module):
    """Training moments use real tokens only; evaluation uses frozen moments."""

    def __init__(self, channels: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask[:, None]
        clean = torch.where(valid, values, 0)
        count = mask.sum().to(values.dtype)
        if self.training:
            has_statistics = count > 1
            denominator = count.clamp_min(1)
            mean = clean.sum((0, 2)) / denominator
            centered = torch.where(valid, clean - mean[None, :, None], 0)
            variance = centered.square().sum((0, 2)) / denominator
            with torch.no_grad():
                updated_mean = self.running_mean.lerp(mean.detach(), self.momentum)
                updated_var = self.running_var.lerp(variance.detach() * count / (count - 1).clamp_min(1), self.momentum)
                self.running_mean.copy_(torch.where(has_statistics, updated_mean, self.running_mean))
                self.running_var.copy_(torch.where(has_statistics, updated_var, self.running_var))
            # Keeping this decision on-device avoids a CUDA synchronization for
            # every norm layer while preserving empty/singleton batch semantics.
            mean = torch.where(has_statistics, mean, self.running_mean)
            variance = torch.where(has_statistics, variance, self.running_var)
        else:
            mean, variance = self.running_mean, self.running_var
        out = (clean - mean[None, :, None]) * torch.rsqrt(variance[None, :, None] + self.eps)
        return torch.where(valid, out * self.weight[None, :, None] + self.bias[None, :, None], 0)


class MaskedSqueezeExcitation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, max(4, channels // 8)), nn.GELU(),
                                  nn.Linear(max(4, channels // 8), channels), nn.Sigmoid())

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = masked_mean_time(values.transpose(1, 2), mask)
        return values * self.gate(pooled)[:, :, None]


class MaskedResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: int, stride: int, dropout: float):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv1 = nn.Conv1d(inputs, outputs, kernel_size, stride, kernel_size // 2, bias=False)
        self.conv2 = nn.Conv1d(outputs, outputs, 3, padding=1, bias=False)
        self.norm1, self.norm2 = MaskedBatchNorm1d(outputs), MaskedBatchNorm1d(outputs)
        self.skip = nn.Conv1d(inputs, outputs, 1, stride=stride, bias=False)
        self.dropout = nn.Dropout1d(dropout)
        self.se = MaskedSqueezeExcitation(outputs)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.where(mask[:, None], values, 0)
        residual = self.skip(values)
        out = self.conv1(values)
        first_mask = downsample_time_mask(mask, self.kernel_size, self.stride, self.kernel_size // 2)
        out = self.dropout(F.gelu(self.norm1(out, first_mask)))
        out = torch.where(first_mask[:, None], out, 0)
        out = self.conv2(out)
        final_mask = downsample_time_mask(first_mask, 3, padding=1)
        out = self.norm2(out, final_mask)
        out = self.se(out, final_mask)
        out = F.gelu(out + residual)
        return torch.where(final_mask[:, None], out, 0), final_mask


class TemporalCNNEncoder(nn.Module):
    def __init__(self, inputs: int, channels: tuple[int, ...], kernel_size: int = 5,
                 dropout: float = 0.2, *, downsample: bool = True):
        super().__init__()
        blocks = []
        for i, output in enumerate(channels):
            blocks.append(MaskedResidualBlock(inputs, output, kernel_size,
                1 if i == 0 or not downsample else 2, dropout))
            inputs = output
        self.blocks = nn.ModuleList(blocks)
        self.output_features = 2 * channels[-1]

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, mask = self.encode_sequence(values, mask)
        return self.pool_sequence(out, mask)

    def encode_sequence(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = values.transpose(1, 2)
        for block in self.blocks:
            out, mask = block(out, mask)
        return out, mask

    def pool_sequence(self, out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = masked_mean_time(out.transpose(1, 2), mask)
        maximum = out.masked_fill(~mask[:, None], -torch.inf).amax(dim=2)
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, 0)
        return torch.cat([mean, maximum], dim=1)


class GroupedIMUEncoder(nn.Module):
    """Independent stems for acceleration, pose, angular velocity and linear acc."""

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.feature_count = config.imu_feature_count
        self.groups = [list(range(0, 3)), list(range(3, 9)), list(range(9, 12)), list(range(12, 15))]
        if self.feature_count == 34:
            self.groups[0] += list(range(15, 21)) + list(range(28, 31))
            self.groups[2] += [21]
            self.groups[3] += list(range(22, 28)) + list(range(31, 34))
        self.stems = nn.ModuleList([TemporalCNNEncoder(2 * len(indices), config.stem_channels,
            config.kernel_size, config.dropout, downsample=False)
            for indices in self.groups])
        self.fusion = TemporalCNNEncoder(4 * config.stem_channels[-1], config.imu_channels,
            config.kernel_size, config.dropout)
        self.output_features = self.fusion.output_features

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, available = self.encode_sequence(values, mask)
        return self.fusion.pool_sequence(out, available)

    def encode_sequence(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts, masks = [], []
        for stem, group in zip(self.stems, self.groups):
            valid_indices = [self.feature_count + i for i in group]
            valid = values[..., valid_indices].bool() & mask[..., None]
            indices = group + valid_indices
            out, available = stem.encode_sequence(values[..., indices], valid.any(-1))
            parts.append(out)
            masks.append(available)
        merged_mask = torch.stack(masks).any(0)
        return self.fusion.encode_sequence(torch.cat(parts, dim=1).transpose(1, 2), merged_mask)


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
        self.imu_encoder = GroupedIMUEncoder(self.config)
        features = self.imu_encoder.output_features + 1
        if model_name == "multisensor":
            self.thm_encoder = TemporalCNNEncoder(15, self.config.auxiliary_channels,
                self.config.kernel_size, self.config.dropout)
            self.tof_encoder = TemporalCNNEncoder(10 * tof_regions**2 + 5, self.config.auxiliary_channels,
                self.config.kernel_size, self.config.dropout)
            features += self.thm_encoder.output_features + self.tof_encoder.output_features + 2
        self.classifier = nn.Sequential(
            nn.Linear(features, self.config.hidden_features), nn.LayerNorm(self.config.hidden_features),
            nn.GELU(), nn.Dropout(self.config.dropout), nn.Linear(self.config.hidden_features, num_classes),
        )

    def metadata(self) -> dict:
        architecture = "grouped_masked_se_cnn_v3"
        return {"model": self.model_name, "architecture": architecture,
                "config": asdict(self.config), "tof_regions": self.tof_regions,
                "normalization": f"train-fold standardization + {self.config.normalization}",
                "parameters": sum(p.numel() for p in self.parameters())}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if batch["imu"].shape[-1] != self.config.imu_feature_count:
            raise ValueError("IMU input features disagree with model configuration.")
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
