import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from cmi_project.evaluation import ALL_GESTURES
from cmi_project.second_place_hybrid import JointMetricLoss


class HybridLossTests(unittest.TestCase):
    def test_joint_axis_permutation_and_mixup(self):
        labels = [('orientation', gesture, 'move') for gesture in ALL_GESTURES]
        labels += [('other_orientation', ALL_GESTURES[0], 'move')]
        torch.manual_seed(14)
        logits = torch.randn(4, len(labels), requires_grad=True)
        targets = torch.rand_like(logits); targets /= targets.sum(1, keepdim=True)
        loss = JointMetricLoss(labels)(logits, targets)
        permutation = torch.randperm(len(labels))
        other = JointMetricLoss([labels[i] for i in permutation])(logits[:, permutation], targets[:, permutation])
        torch.testing.assert_close(loss, other)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        original = -(targets * logits.log_softmax(1)).sum(1).mean()
        torch.testing.assert_close(JointMetricLoss(labels, 0, 0)(logits, targets), original)


if __name__ == '__main__':
    unittest.main()
