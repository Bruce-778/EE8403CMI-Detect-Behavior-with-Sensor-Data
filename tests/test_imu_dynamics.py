"""Physical differences, rotation missingness and train-only dynamic scalers."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.preprocessing import (PreprocessingConfig, FoldPreprocessor, engineer_sequence,
    DYNAMIC_IMU_FEATURES, SensorDropoutConfig, sensor_dropout)
from cmi_project.cnn_data import compact_sample, CNNTensorDataset
from cmi_project.cnn import CNNConfig, CMI1DCNN
from test_preprocessing import fixture


class DynamicIMUTests(unittest.TestCase):
    def setUp(self):
        self.config = PreprocessingConfig(imu_dynamics=True)
        self.demographics = pd.DataFrame({"subject": ["S1", "S2"], "handedness": [1, 1]})

    def test_differences_do_not_cross_sequence_or_counter_gaps(self):
        frame = fixture(12)
        frame["acc_x"] = np.arange(12, dtype=float)
        raw = engineer_sequence(frame, 1, self.config)
        self.assertEqual(raw["imu"].shape, (12, 34))
        np.testing.assert_allclose(raw["imu"][1:, 16], 1)
        self.assertFalse(raw["imu_valid"][0, 16:21].any())
        frame.loc[6:, "sequence_counter"] += 2
        gap = engineer_sequence(frame, 1, self.config)
        self.assertFalse(gap["imu_valid"][6, 16:21].any())
        isolated = engineer_sequence(frame.iloc[6:], 1, self.config)
        self.assertFalse(isolated["imu_valid"][0, 16:21].any())

    def test_missing_rotation_preserves_acceleration_dynamics(self):
        frame = fixture(12)
        frame["acc_x"] = np.arange(12, dtype=float)
        frame[["rot_w", "rot_x", "rot_y", "rot_z"]] = np.nan
        raw = engineer_sequence(frame, 1, self.config)
        self.assertTrue(raw["imu_valid"][1:, 15:21].all())
        self.assertFalse(raw["imu_valid"][:, 21:28].any())
        self.assertFalse(raw["imu_valid"][:, 31:34].any())
        processor = FoldPreprocessor(self.config).fit([frame], self.demographics)
        sample = processor.transform(frame, self.demographics, "train")
        dropped = sensor_dropout(sample, np.random.default_rng(42), training=True, config=SensorDropoutConfig(
            rotation_probability=1, thm_probability=0, tof_probability=0, channel_probability=0))
        np.testing.assert_array_equal(dropped["imu_valid"][:, 15:21], sample["imu_valid"][:, 15:21])
        compact = compact_sample(sample)
        arrays = {key: value[None] for key, value in compact.items()}
        arrays["label"] = np.array([0])
        from_dataset = CNNTensorDataset(arrays, [0], training=True, dropout=SensorDropoutConfig(
            rotation_probability=1, thm_probability=0, tof_probability=0, channel_probability=0))[0]
        np.testing.assert_array_equal(from_dataset["imu_valid"].numpy(), dropped["imu_valid"])

    def test_saved_dynamic_scaler_and_model_use_consistent_features(self):
        torch.set_num_threads(1)
        frame = fixture(12)
        frame["acc_x"] = np.arange(12, dtype=float)
        processor = FoldPreprocessor(self.config).fit([frame], self.demographics)
        self.assertEqual(processor.state["imu_features"], DYNAMIC_IMU_FEATURES)
        held_out = frame.copy()
        held_out["subject"] = "S2"
        held_out["acc_x"] = 1e6
        before = processor.state["statistics"]["imu"]["offset"].copy()
        processor.transform(held_out, self.demographics, "validation")
        self.assertEqual(before, processor.state["statistics"]["imu"]["offset"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preprocessor.json"
            processor.save(path)
            restored = FoldPreprocessor.load(path)
            self.assertEqual(json.loads(json.dumps(processor.state)), restored.state)
        sample = compact_sample(restored.transform(frame, self.demographics, "train"))
        batch = {key: torch.from_numpy(value[None]) for key, value in sample.items()}
        config = CNNConfig(imu_feature_count=34, imu_channels=(4, 8), auxiliary_channels=(4, 8),
            stem_channels=(4,), hidden_features=8, encoder_style="grouped", normalization="masked_batch",
            squeeze_excitation=True)
        model = CMI1DCNN("imu", config=config).eval()
        self.assertTrue(torch.isfinite(model(batch)).all())


if __name__ == "__main__":
    unittest.main()
