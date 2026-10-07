"""Mask and checkpoint invariants for the winner-inspired grouped CNN."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.cnn import CMI1DCNN, CNNConfig, MaskedBatchNorm1d
from cmi_project.cnn_training import mixup_sensor_batch, CMIHierarchicalLoss, TrainingConfig
from test_cnn import toy_batch


class GroupedCNNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_batch_norm_moments_exclude_padding_and_empty_rows(self):
        first, second = MaskedBatchNorm1d(3), MaskedBatchNorm1d(3)
        second.load_state_dict(first.state_dict())
        values = torch.randn(2, 3, 5)
        mask = torch.ones(2, 5, dtype=torch.bool)
        expected = first(values, mask)
        padded = torch.full((3, 3, 9), float("nan"))
        padded[:2, :, -5:] = values
        padded_mask = torch.zeros(3, 9, dtype=torch.bool)
        padded_mask[:2, -5:] = True
        actual = second(padded, padded_mask)
        torch.testing.assert_close(actual[:2, :, -5:], expected)
        torch.testing.assert_close(first.running_mean, second.running_mean)
        torch.testing.assert_close(first.running_var, second.running_var)
        self.assertFalse(actual[~padded_mask[:, None].expand_as(actual)].any())
        before = second.running_mean.clone()
        second(padded, torch.zeros_like(padded_mask)).sum().backward()
        torch.testing.assert_close(second.running_mean, before)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in second.parameters()))

    def test_grouped_encoders_ignore_missing_values_and_roundtrip(self):
        config = CNNConfig(imu_channels=(4, 8), auxiliary_channels=(4, 8),
            stem_channels=(4,), hidden_features=8, encoder_style="grouped",
            normalization="masked_batch", squeeze_excitation=True)
        batch = toy_batch()
        batch["imu_valid"][1, :, 3:] = False
        corrupt = {key: value.clone() for key, value in batch.items()}
        corrupt["imu"][~batch["imu_valid"]] = float("nan")
        for name in ("imu", "multisensor"):
            model = CMI1DCNN(name, config=config).eval()
            expected = model(batch)
            torch.testing.assert_close(model(corrupt), expected)
            model(corrupt).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            restored = CMI1DCNN(name, config=CNNConfig.from_dict(model.metadata()["config"])).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(restored(batch), expected)

    def test_mixup_masks_corrupt_padding_and_keeps_weighted_targets(self):
        batch = toy_batch()
        batch["label"] = torch.tensor([0, 8, 17])
        batch["imu"][~batch["imu_valid"]] = float("nan")
        original = batch["imu"].clone()
        permutation = torch.tensor([1, 2, 0])
        mixed = mixup_sensor_batch(batch, permutation, 0.7)
        self.assertTrue(torch.isfinite(mixed["imu"]).all())
        torch.testing.assert_close(mixed["time_mask"], batch["time_mask"] | batch["time_mask"][permutation])
        torch.testing.assert_close(batch["imu"], original, equal_nan=True)
        torch.testing.assert_close(mixed["label"], batch["label"])
        endpoint = mixup_sensor_batch(batch, permutation, 0.0)
        torch.testing.assert_close(endpoint["imu_valid"], batch["imu_valid"][permutation])
        logits = torch.randn(3, 18, requires_grad=True)
        criterion = CMIHierarchicalLoss(TrainingConfig(macro_loss_weight=0.5, binary_loss_weight=0.1))
        loss = 0.7 * criterion(logits, batch["label"]) + 0.3 * criterion(logits, batch["label"][permutation])
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_gru_pooling_excludes_padding_and_empty_modalities(self):
        from cmi_project.cnn import TemporalCNNEncoder
        encoder = TemporalCNNEncoder(5, (4, 8), pooling="gru_mean").eval()
        values = torch.full((2, 12, 5), float("nan"))
        mask = torch.zeros(2, 12, dtype=torch.bool)
        output = encoder(values, mask)
        torch.testing.assert_close(output, torch.zeros_like(output))
        output.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None))
        tokens = torch.randn(2, 8, 7)
        gapped = torch.tensor([[False, True, False, True, True, False, False],
                              [False, False, True, True, True, True, False]])
        pooled = encoder.pool_sequence(tokens, gapped)
        for i in range(2):
            compact = tokens[i:i+1, :, gapped[i]]
            expected = encoder.pool_sequence(compact, torch.ones(1, compact.shape[-1], dtype=torch.bool))
            torch.testing.assert_close(pooled[i:i+1], expected)
        config = CNNConfig(imu_channels=(4, 8), auxiliary_channels=(4, 8), stem_channels=(4,),
            hidden_features=8, encoder_style="grouped", pooling="gru_mean", normalization="masked_batch")
        model = CMI1DCNN("multisensor", config=config).eval()
        batch = toy_batch()
        changed = {key: value.clone() for key, value in batch.items()}
        changed["imu"][~batch["imu_valid"]] = float("nan")
        torch.testing.assert_close(model(batch), model(changed))
        restored = CMI1DCNN("multisensor", config=CNNConfig.from_dict(model.metadata()["config"])).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(batch), restored(batch))


if __name__ == "__main__":
    unittest.main()
