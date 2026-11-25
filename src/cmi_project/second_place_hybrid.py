"""Project-owned additions around the unmodified, pinned second-place models."""
from pathlib import Path

import torch
from torch import nn

from .evaluation import ALL_GESTURES
from .second_place import load_reference


def load_project_reference(root):
    root = Path(root)
    return load_reference(root / 'src/cmi_project/vendor/second_place',
                          root / 'configs/second_place_source.json')


class JointMetricLoss(nn.Module):
    """Joint soft CE plus our official nine-class/binary auxiliary objectives.

    The joint ontology comes only from training IDs. Soft targets retain the
    original phase-wise Mixup; phase CE remains in the training runner. No extra
    label smoothing, checkpoint selection or inference decision rule changes.
    """
    def __init__(self, joint_labels, macro_weight=.5, binary_weight=.1):
        super().__init__()
        if not joint_labels or any(label[1] not in ALL_GESTURES for label in joint_labels):
            raise ValueError('Training-only joint gesture ontology required.')
        indices = torch.tensor([ALL_GESTURES.index(label[1]) for label in joint_labels])
        groups = indices.clamp_max(8)
        if set(groups.tolist()) != set(range(9)):
            raise ValueError('All official metric groups must occur in the training ontology.')
        self.register_buffer('groups', groups)
        self.macro_weight, self.binary_weight = float(macro_weight), float(binary_weight)
        if min(self.macro_weight, self.binary_weight) < 0:
            raise ValueError('Nonnegative auxiliary weights required.')

    def forward(self, logits, soft_targets):
        if logits.shape != soft_targets.shape or logits.shape[1] != len(self.groups):
            raise ValueError('Joint target/logit axes differ.')
        joint = -(soft_targets * logits.log_softmax(1)).sum(1).mean()
        if not (self.macro_weight or self.binary_weight):
            return joint
        grouped_logits = torch.stack([torch.logsumexp(logits[:, self.groups == group], 1)
                                      for group in range(9)], 1)
        grouped_targets = torch.stack([soft_targets[:, self.groups == group].sum(1)
                                       for group in range(9)], 1)
        macro = -(grouped_targets * grouped_logits.log_softmax(1)).sum(1).mean()
        odds = torch.logsumexp(grouped_logits[:, :8], 1) - grouped_logits[:, 8]
        binary = nn.functional.binary_cross_entropy_with_logits(odds, grouped_targets[:, :8].sum(1))
        return joint + self.macro_weight * macro + self.binary_weight * binary
