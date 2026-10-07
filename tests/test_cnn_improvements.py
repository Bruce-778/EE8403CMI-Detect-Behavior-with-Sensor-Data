"""Checks for experimental changes whose mask/loss semantics affect scores."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.cnn_training import CMIHierarchicalLoss, TrainingConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_cnn_scenarios import AUXILIARY_KEYS, ScenarioDataset, FixedProbabilityEnsemble
from test_cnn import toy_batch
from cmi_project.cnn import CMI1DCNN, CNNConfig, TemporalCNNEncoder
import numpy as np


class HierarchicalLossTests(unittest.TestCase):
    def test_zero_weights_exactly_preserve_cross_entropy(self):
        logits = torch.randn(4, 18)
        target = torch.tensor([0, 7, 8, 17])
        config = TrainingConfig()
        torch.testing.assert_close(CMIHierarchicalLoss(config)(logits, target),
            torch.nn.functional.cross_entropy(logits, target, label_smoothing=config.label_smoothing))

    def test_grouped_loss_uses_sum_of_probabilities_and_backpropagates(self):
        logits = torch.randn(4, 18, requires_grad=True)
        target = torch.tensor([0, 7, 8, 17])
        config = TrainingConfig(label_smoothing=0, macro_loss_weight=0.5, binary_loss_weight=0.1)
        probabilities = logits.softmax(1)
        grouped = torch.cat([probabilities[:, :8], probabilities[:, 8:].sum(1, keepdim=True)], 1)
        macro = -grouped[torch.arange(4), target.clamp_max(8)].log().mean()
        binary = torch.nn.functional.binary_cross_entropy(probabilities[:, :8].sum(1), (target < 8).float())
        expected = torch.nn.functional.cross_entropy(logits, target) + 0.5 * macro + 0.1 * binary
        actual = CMIHierarchicalLoss(config)(logits, target)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_extreme_logits_are_finite_and_negative_weights_rejected(self):
        logits = torch.tensor([[1000.] + [-1000.] * 17], requires_grad=True)
        criterion = CMIHierarchicalLoss(TrainingConfig(macro_loss_weight=0.5, binary_loss_weight=0.1))
        loss = criterion(logits, torch.tensor([17]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            TrainingConfig(macro_loss_weight=-1)


class ScenarioTests(unittest.TestCase):
    def test_equal_probability_ensemble_preserves_imu_only_invariance(self):
        config = CNNConfig(imu_channels=(4, 8), auxiliary_channels=(4, 8), hidden_features=8)
        a, b = (CMI1DCNN("imu", config=config).eval() for _ in range(2))
        ensemble = FixedProbabilityEnsemble(a, b).eval()
        batch = toy_batch()
        expected = (a(batch).softmax(1) + b(batch).softmax(1)) / 2
        torch.testing.assert_close(ensemble(batch).softmax(1), expected)
        for key in AUXILIARY_KEYS:
            batch[key].zero_()
        torch.testing.assert_close(ensemble(batch).softmax(1), expected)

    def test_masking_is_label_independent_and_does_not_mutate_arrays(self):
        arrays = {key: value.numpy() for key, value in toy_batch().items()}
        arrays.update(sequence_id=np.array(["Q0", "Q1", "Q2"]), label=np.array([0, 8, 17]))
        original = {key: value.copy() for key, value in arrays.items()}
        scenario = ScenarioDataset(arrays, [0, 1, 2], "imu_only")
        for key in AUXILIARY_KEYS:
            self.assertFalse(scenario[0][key].any())
        self.assertTrue(scenario[0]["imu_valid"].any())
        a = ScenarioDataset(arrays, [0, 1, 2], "aux_dropout50").drop
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], original[key])
        arrays["label"][:] = 3
        b = ScenarioDataset(arrays, [0, 1, 2], "aux_dropout50").drop
        np.testing.assert_array_equal(a, b)


class AttentionCNNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_empty_attention_branch_is_zero_and_gradients_are_finite(self):
        encoder = TemporalCNNEncoder(5, (4, 8), pooling="attention_max", stage_kernel_sizes=(5, 9)).eval()
        values = torch.full((2, 12, 5), float("nan"))
        mask = torch.zeros(2, 12, dtype=torch.bool)
        output = encoder(values, mask)
        torch.testing.assert_close(output, torch.zeros_like(output))
        output.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None))

    def test_attention_ignores_padding_and_checkpoint_config_roundtrips(self):
        batch = toy_batch()
        config = CNNConfig(imu_channels=(4, 8), auxiliary_channels=(4, 8),
            hidden_features=8, pooling="attention_max", stage_kernel_sizes=(5, 9))
        model = CMI1DCNN("multisensor", config=config).eval()
        corrupt = {key: value.clone() for key, value in batch.items()}
        for key in ("imu", "thm", "tof", "tof_fraction"):
            corrupt[key][~batch["time_mask"]] = float("nan")
        torch.testing.assert_close(model(batch), model(corrupt))
        restored = CMI1DCNN("multisensor", config=CNNConfig.from_dict(model.metadata()["config"])).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(batch), restored(batch))
        with self.assertRaises(ValueError):
            CNNConfig(stage_kernel_sizes=(4, 9, 13))


if __name__ == "__main__":
    unittest.main()
