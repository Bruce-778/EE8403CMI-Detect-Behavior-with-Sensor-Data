"""Mask behavior under corrupt padding and fully unavailable modality inputs."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    from cmi_project.torch_inputs import (
        CMIFoldDataset,
        downsample_time_mask, masked_mean_time,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional torch dependency is not installed.")
class TorchInputTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_pool_ignores_nan_padding_and_empty_rows(self):
        mask = torch.tensor([[False, False, True, True], [False] * 4])
        values = torch.ones(2, 4, 8)
        corrupt = torch.where(mask[..., None], values, torch.full_like(values, float("nan")))
        mean = masked_mean_time(corrupt, mask)
        torch.testing.assert_close(mean[0], torch.ones(8))
        torch.testing.assert_close(mean[1], torch.zeros(8))

    def test_downsample_mask_matches_known_receptive_fields(self):
        mask = torch.tensor([[False, False, False, True, True, True], [False] * 6])
        actual = downsample_time_mask(mask, 3, 2, 1)
        torch.testing.assert_close(actual, torch.tensor([[False, True, True], [False] * 3]))


    def test_dataset_rebuilds_maps_and_applies_dropout_only_on_training_reads(self):
        import pandas as pd
        from test_preprocessing import fixture, demo
        from cmi_project.preprocess_cli import save_sample
        from cmi_project.preprocessing import FoldPreprocessor

        sequence = fixture()
        processor = FoldPreprocessor().fit([sequence], demo())
        sample = processor.transform(sequence, demo(), "train")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_sample(sample, root / "samples" / "0.npz")
            pd.DataFrame([{"path": "samples/0.npz"}]).to_csv(root / "train_manifest.csv", index=False)
            (root / "sensor_dropout.json").write_text(json.dumps({
                "rotation_probability": 1, "thm_probability": 1,
                "tof_probability": 1, "channel_probability": 0,
            }), encoding="utf-8")
            training = CMIFoldDataset(root / "train_manifest.csv", training=True)[0]
            evaluation = CMIFoldDataset(root / "train_manifest.csv", training=False)[0]
            self.assertFalse(training["tof_input"].any())
            self.assertFalse(training["modality_available"].any())
            self.assertFalse(training["thm_observed"].any())
            self.assertTrue(evaluation["thm_observed"].all())
            torch.testing.assert_close(evaluation["tof_input"], torch.from_numpy(sample["tof_input"]))
            self.assertEqual(evaluation["sequence_id"], "A")
            pd.DataFrame([{"path": "samples/0.npz"}]).to_csv(root / "validation_manifest.csv", index=False)
            with self.assertRaises(ValueError):
                CMIFoldDataset(root / "validation_manifest.csv", training=True)


if __name__ == "__main__":
    unittest.main()
