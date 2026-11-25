"""Train-only phase supervision and cross-subject contrastive CNN experiments."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .cnn import CMI1DCNN
from .cnn_data import CNNTensorDataset
from .preprocessing import left_pad_tail


@dataclass(frozen=True)
class RepresentationConfig:
    method: str = "phase"
    weight: float = 0.1
    temperature: float = 0.1
    trainable_scope: str = "full"

    def __post_init__(self):
        if self.method not in ("phase", "cross_subject_supcon", "official_metric"):
            raise ValueError("Unknown representation method.")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Auxiliary weight must be finite and nonnegative.")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Contrastive temperature must be finite and positive.")
        if self.trainable_scope not in ("full", "phase_heads") or (self.trainable_scope == "phase_heads" and self.method != "phase"):
            raise ValueError("Frozen-head adaptation requires the phase method.")
        if self.method == "official_metric" and self.weight:
            raise ValueError("Official-metric fine-tuning changes supervised weights, not an auxiliary loss.")


class RepresentationIMUCNN(CMI1DCNN):
    """Phase probabilities come from sensors; annotations NEVER enter forward."""

    def __init__(self, model_name="imu", *, phase_enabled=False, **kwargs):
        if model_name != "imu":
            raise ValueError("Representation pilot changes only Model A.")
        super().__init__(model_name, **kwargs)
        self.phase_enabled = phase_enabled
        if phase_enabled:
            channels = self.config.imu_channels[-1]
            self.phase_head = nn.Conv1d(channels, 3, 1)
            self.phase_attention = nn.Conv1d(channels, 3, 1)
            self.phase_adapter = nn.Linear(3 * channels, 2 * channels)
            nn.init.zeros_(self.phase_adapter.weight)
            nn.init.zeros_(self.phase_adapter.bias)

    @classmethod
    def from_starting_model(cls, model, phase_enabled):
        # New-head initialization must not change the shared training dropout RNG.
        state = torch.random.get_rng_state()
        augmented = cls(tof_regions=model.tof_regions, config=model.config,
                        phase_enabled=phase_enabled)
        torch.random.set_rng_state(state)
        missing, unexpected = augmented.load_state_dict(model.state_dict(), strict=False)
        expected = {f"{head}.{part}" for head in ("phase_head", "phase_attention", "phase_adapter")
                    for part in ("weight", "bias")} if phase_enabled else set()
        if set(missing) != expected or unexpected:
            raise ValueError("Starting architecture differs from the phase experiment.")
        return augmented.to(next(model.parameters()).device)

    def metadata(self):
        result = super().metadata()
        if self.phase_enabled:
            result["architecture"] = "phase_residual_grouped_cnn_v1"
        return result

    def forward_outputs(self, batch):
        if batch["imu"].shape[-1] != self.config.imu_feature_count:
            raise ValueError("IMU features disagree with model configuration.")
        valid = batch["imu_valid"].bool() & batch["time_mask"][..., None].bool()
        inputs = torch.cat([torch.where(valid, batch["imu"], 0), valid.float()], -1)
        sequence, mask = self.imu_encoder.encode_sequence(inputs, valid.any(-1))
        embedding = self.imu_encoder.fusion.pool_sequence(sequence, mask)
        phase_logits = None
        if self.phase_enabled:
            phase_logits = self.phase_head(sequence)
            scores = self.phase_attention(sequence) + phase_logits.log_softmax(1)
            # An entirely unavailable sequence must have zero attention, never NaN.
            scores = scores.masked_fill(~mask[:, None], -1e4)
            weights = scores.softmax(-1) * mask[:, None]
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            pooled = torch.einsum("bpt,bct->bpc", weights, sequence).flatten(1)
            correction = self.phase_adapter(pooled) * mask.any(1, keepdim=True)
            embedding = embedding + correction
        features = torch.cat([embedding, valid.any((1, 2))[:, None].float()], 1)
        return self.classifier(features), embedding, phase_logits, mask

    def forward(self, batch):
        return self.forward_outputs(batch)[0]

    def phase_loss(self, logits, phase, input_mask, output_mask):
        # Soft targets count annotated input paths through the main convolution
        # branches. Residual shortcuts have shorter support; no oracle segmentation
        # is fed to attention. Stems share identical temporal geometry.
        if phase.shape != input_mask.shape:
            raise ValueError("Invalid training phase shape.")
        valid = (phase >= 0) & input_mask
        if (phase[valid] > 2).any():
            raise ValueError("Invalid training phase labels.")
        counts = F.one_hot(phase.clamp_min(0), 3).transpose(1, 2).float() * valid[:, None]
        blocks = [*self.imu_encoder.stems[0].blocks, *self.imu_encoder.fusion.blocks]
        for block in blocks:
            for conv in (block.conv1, block.conv2):
                counts = F.conv1d(counts, counts.new_ones(3, 1, conv.kernel_size[0]),
                                 stride=conv.stride, padding=conv.padding, groups=3)
        total = counts.sum(1)
        if counts.shape != logits.shape or output_mask.shape != total.shape:
            raise ValueError("Phase supervision and encoder geometry disagree.")
        usable = (total > 0) & output_mask
        target = counts / total[:, None].clamp_min(1)
        losses = -(target * logits.log_softmax(1)).sum(1)
        return (losses * usable).sum() / usable.sum().clamp_min(1)


def cross_subject_contrastive_loss(embedding, labels, subjects, temperature):
    """Same gesture, different subjects are positives; different gestures negatives.

    Same-subject/same-gesture pairs and self-pairs are excluded. Anchors without
    positives contribute no loss. Subject IDs are only training loss metadata.
    """
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Contrastive temperature must be finite and positive.")
    z = F.normalize(embedding, dim=1)
    same_label = labels[:, None] == labels[None]
    positive = same_label & (subjects[:, None] != subjects[None])
    denominator_mask = (~same_label) | positive
    usable = positive.any(1)
    if not usable.any():
        return embedding.sum() * 0
    scores = z @ z.T / temperature
    log_denominator = scores.masked_fill(~denominator_mask, -torch.inf)[usable].logsumexp(1)
    positive_scores = (scores[usable] * positive[usable]).sum(1) / positive[usable].sum(1)
    return (log_denominator - positive_scores).mean()


PHASE_LABELS = {"Relaxes and moves hand to target location": 0,
                "Moves hand to target location": 0,
                "Hand at target location": 1, "Performs gesture": 2}


def training_phase_targets(path, arrays, manifest, fold, *, chunksize=25000):
    """Read narrow annotations, aligned by counter; leave ALL held-out rows -100."""
    train = manifest.table[manifest.table.fold != fold].set_index("sequence_id")
    positions = {str(sid): i for i, sid in enumerate(arrays["sequence_id"])}
    target = np.full(arrays["time_mask"].shape, -100, dtype=np.int64)
    pieces = {}
    columns = ["sequence_id", "sequence_counter", "behavior", "subject", "gesture"]
    with pd.read_csv(path, usecols=columns, chunksize=chunksize) as reader:
        for chunk in reader:
            selected = chunk[chunk.sequence_id.isin(train.index)]
            for sid, frame in selected.groupby("sequence_id", sort=False):
                pieces.setdefault(str(sid), []).append(frame)
    for sid, record in train.iterrows():
        if sid not in pieces:
            raise ValueError(f"Missing phase annotations: {sid}")
        frame = pd.concat(pieces.pop(sid)).sort_values("sequence_counter", kind="stable")
        counter = frame.sequence_counter.to_numpy(dtype=float)
        if (len(frame) != record.length or not np.isfinite(counter).all()
                or len(np.unique(counter)) != len(frame)
                or set(frame.subject) != {record.subject} or set(frame.gesture) != {record.gesture}):
            raise ValueError(f"Phase annotations disagree with fixed sequence: {sid}")
        labels = frame.behavior.map(PHASE_LABELS)
        if labels.isna().any():
            raise ValueError(f"Unknown or missing training behavior in {sid}")
        position = positions[sid]
        target[position] = left_pad_tail(labels.to_numpy(dtype=np.int64), target.shape[1], -100)
        if not np.array_equal(target[position] >= 0, arrays["time_mask"][position]):
            raise ValueError("Phase tail/padding alignment differs from sensor inputs.")
    return target


class RepresentationDataset(Dataset):
    def __init__(self, arrays, indices, manifest, fold, *, seed, dropout, phase=None):
        self.base = CNNTensorDataset(arrays, indices, training=True, seed=seed, dropout=dropout)
        validation = manifest.table.fold.to_numpy() == fold
        if validation[self.base.indices].any():
            raise ValueError("Validation cannot enter representation training.")
        if phase is not None:
            if phase.shape != arrays["time_mask"].shape or (phase[validation] != -100).any():
                raise ValueError("Phase targets must exclude all validation sequences.")
            if not np.array_equal(phase[self.base.indices] >= 0, arrays["time_mask"][self.base.indices]):
                raise ValueError("Missing/misaligned training phase annotations.")
        self.phase = phase
        self.subjects = pd.factorize(arrays["subject"])[0]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        position = self.base.indices[index]
        sample["training_subject"] = torch.tensor(int(self.subjects[position]))
        if self.phase is not None:
            sample["training_phase"] = torch.from_numpy(self.phase[position].copy())
        return sample
