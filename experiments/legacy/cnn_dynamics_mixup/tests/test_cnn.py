"""Padding, missing modalities, fold isolation and real CNN training checks."""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmi_project.cnn import CMI1DCNN, CNNConfig, TemporalCNNEncoder
from cmi_project.cnn_data import CNNTensorDataset, compact_sample, prepare_cnn_fold
from cmi_project.cnn_training import EarlyStopping, TrainingConfig, load_cnn_checkpoint, train_cnn_fold, reuse_completed_fold
from cmi_project.evaluation import ALL_GESTURES, PROBABILITY_COLUMNS, evaluate_oof_frames
from cmi_project.preprocessing import FoldPreprocessor, SensorDropoutConfig
from cmi_project.validation import save_fixed_folds, scan_sequence_index
from test_preprocessing import fixture


def toy_batch():
    batch, length, regions = 3, 12, 2
    time = torch.tensor([[False] * 6 + [True] * 6, [True] * 12, [False] * 12])
    return {
        "imu": torch.randn(batch, length, 15), "imu_valid": time[..., None].expand(-1, -1, 15).clone(),
        "thm": torch.randn(batch, length, 5), "thm_valid": time[..., None].expand(-1, -1, 5).clone(),
        "thm_observed": time[..., None].expand(-1, -1, 5).clone(),
        "tof": torch.randn(batch, length, 5 * regions**2),
        "tof_valid": time[..., None].expand(-1, -1, 5 * regions**2).clone(),
        "tof_fraction": torch.rand(batch, length, 5 * regions**2),
        "tof_sensor_present": time[..., None].expand(-1, -1, 5).clone(), "time_mask": time,
    }


def tiny_config():
    return CNNConfig(imu_feature_count=15, imu_channels=(4, 8), auxiliary_channels=(4, 8), hidden_features=8, dropout=0.1)


class CNNModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_padding_corruption_and_missing_channels_do_not_change_logits(self):
        batch = toy_batch()
        batch["imu_valid"][1, :, 3:] = False
        batch["thm_valid"][1] = False
        batch["tof_sensor_present"][1] = False
        corrupt = {key: value.clone() for key, value in batch.items()}
        for key, valid in (("imu", batch["imu_valid"]), ("thm", batch["thm_valid"]),
                           ("tof", batch["tof_valid"] & batch["tof_sensor_present"].repeat_interleave(4, -1)),
                           ("tof_fraction", batch["tof_sensor_present"].repeat_interleave(4, -1))):
            corrupt[key][~valid] = float("nan")
        for name in ("imu", "multisensor"):
            model = CMI1DCNN(name, config=tiny_config()).eval()
            a, b = model(batch), model(corrupt)
            self.assertEqual(a.shape, (3, 18))
            torch.testing.assert_close(a, b)
            self.assertTrue(torch.isfinite(b).all())
            b.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_absent_encoder_is_zero_and_model_a_uses_only_imu(self):
        encoder = TemporalCNNEncoder(5, (4, 8)).eval()
        encoded = encoder(torch.full((2, 12, 5), float("nan")), torch.zeros(2, 12, dtype=torch.bool))
        torch.testing.assert_close(encoded, torch.zeros_like(encoded))
        batch = toy_batch()
        model = CMI1DCNN("imu", config=tiny_config()).eval()
        restricted = {key: batch[key] for key in ("imu", "imu_valid", "time_mask")}
        torch.testing.assert_close(model(batch), model(restricted))

    def test_region_means_ignore_invalid_pixels_and_distinguish_no_response(self):
        raw = fixture(4)
        from cmi_project.dataset_analysis import SENSOR_COLUMNS
        raw[SENSOR_COLUMNS["ToF"]] = -1.0
        raw.loc[0, "tof_1_v0"] = 4.0
        processor = FoldPreprocessor().fit([raw], pd.DataFrame({"subject": ["S1"], "handedness": [1]}))
        sample = processor.transform(raw, pd.DataFrame({"subject": ["S1"], "handedness": [1]}), "train")
        # Put a known normalized reading in one valid pixel. All invalid pixels
        # are corrupt; region means must still use just that observed pixel.
        sample["tof"][:] = float("nan")
        sample["tof"][0, 0, 0, 0] = 7.0
        compact = compact_sample(sample, input_clip=None)
        self.assertEqual(compact["tof"][0, 0], 7.0)
        self.assertEqual(compact["tof_fraction"][0, 0], 1 / 16)
        self.assertTrue(compact["tof_sensor_present"][1].all())
        self.assertFalse(compact["tof_valid"][1].any())
        self.assertTrue(np.isfinite(compact["tof"]).all())
        sample["tof_sensor_present"][:] = False
        self.assertFalse(compact_sample(sample)["tof_sensor_present"].any())

    def test_early_stopping_keeps_actual_best_and_ignores_tiny_patience_gains(self):
        stopper = EarlyStopping(2, 0.01)
        self.assertEqual(stopper.update(0.5, 1), (True, False))
        self.assertEqual(stopper.update(0.505, 2), (True, False))
        self.assertEqual(stopper.update(0.504, 3), (False, True))
        self.assertEqual(stopper.best_epoch, 2)
        with self.assertRaises(ValueError):
            stopper.update(float("nan"), 4)


class CNNPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        labels = (ALL_GESTURES[0], ALL_GESTURES[1], ALL_GESTURES[-1])
        sequences = []
        for subject in range(10):
            for i, label in enumerate(labels):
                frame = fixture(5 + i, f"Q{subject:02}_{i}", f"S{subject:02}", label)
                frame["acc_x"] = float(i * 2 + subject / 10)
                sequences.append(frame)
        pd.concat(sequences).to_csv(self.data / "train.csv", index=False)
        self.demographics = pd.DataFrame({"subject": [f"S{i:02}" for i in range(10)], "handedness": [1] * 10})
        self.demographics.to_csv(self.data / "train_demographics.csv", index=False)
        self.manifest = save_fixed_folds(scan_sequence_index(self.data / "train.csv", 7), self.root / "folds.csv")

    def prepare(self, fold=0):
        with redirect_stdout(io.StringIO()):
            return prepare_cnn_fold(self.data, self.root / "cache" / f"fold_{fold}", self.manifest, fold, chunksize=11)

    def test_train_only_scaling_cache_reuse_and_global_class_order(self):
        train, validation = self.manifest.split(0)
        frame = pd.read_csv(self.data / "train.csv")
        frame.loc[frame["subject"].isin(validation["subject"]), "acc_x"] = 1e8
        frame.to_csv(self.data / "train.csv", index=False)
        arrays, processor, metadata = self.prepare()
        self.assertLess(processor.state["statistics"]["imu"]["offset"][0], 10)
        self.assertEqual(set(processor.state["train_subjects"]), set(train["subject"]))
        expected = self.manifest.table["gesture"].map({g: i for i, g in enumerate(ALL_GESTURES)}).to_numpy()
        np.testing.assert_array_equal(arrays["label"], expected)
        self.assertNotEqual(processor.state["labels"], list(ALL_GESTURES))
        again, _, repeated = self.prepare()
        np.testing.assert_array_equal(arrays["imu"], again["imu"])
        self.assertEqual(metadata, repeated)
        with self.assertRaisesRegex(ValueError, "settings/sources"):
            prepare_cnn_fold(self.data, self.root / "cache/fold_0", self.manifest, 0, sequence_length=4)

    def test_augmentation_is_train_only_reproducible_and_cache_is_unchanged(self):
        arrays, _, _ = self.prepare()
        indices = np.flatnonzero(self.manifest.table["fold"].to_numpy() != 0)
        cfg = SensorDropoutConfig(rotation_probability=1, thm_probability=1, tof_probability=1, channel_probability=0)
        original = {key: value.copy() for key, value in arrays.items()}
        train = CNNTensorDataset(arrays, indices, training=True, dropout=cfg)[0]
        self.assertFalse(train["imu_valid"][:, 3:].any())
        self.assertFalse(train["thm_valid"].any())
        self.assertFalse(train["thm_observed"].any())
        self.assertFalse(train["tof_sensor_present"].any())
        self.assertFalse(train["tof_fraction"].any())
        validation = CNNTensorDataset(arrays, indices)[0]
        self.assertTrue(validation["thm_valid"].any())
        self.assertTrue(validation["tof_sensor_present"].any())
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], original[key])
        a, b = (CNNTensorDataset(arrays, indices, training=True, seed=17) for _ in range(2))
        for _ in range(3):
            x, y = a[0], b[0]
            for key in x:
                torch.testing.assert_close(x[key], y[key])

    def test_training_checkpoint_roundtrip_early_stop_and_reproducibility(self):
        arrays, processor, metadata = self.prepare()
        training = TrainingConfig(epochs=4, batch_size=8, cpu_threads=1, early_stopping_patience=1, min_delta=1.0)
        results = []
        with redirect_stdout(io.StringIO()):
            for run in range(2):
                folder = self.root / f"run_{run}"
                summary, predictions = train_cnn_fold(arrays, processor, self.manifest, 0, folder,
                    model_name="multisensor", model_config=tiny_config(), training=training,
                    data_metadata=metadata, save_plots=False)
                self.assertTrue(summary["stopped_early"])
                self.assertEqual(summary["epochs_run"], 2)
                self.assertTrue((folder / "history.csv").is_file())
                model, restored, checkpoint = load_cnn_checkpoint(folder / "best.pt")
                self.assertEqual(restored.state, processor.state)
                self.assertEqual(checkpoint["best_epoch"], summary["best_epoch"])
                self.assertEqual(model.model_name, "multisensor")
                reused, repeated = reuse_completed_fold(folder, processor, self.manifest, 0,
                    "multisensor", tiny_config(), training, SensorDropoutConfig(), metadata, 2)
                self.assertEqual(reused, json.loads(json.dumps(summary)))
                cached_processor = FoldPreprocessor.load(folder / "preprocessor.json")
                cached_reused, _ = reuse_completed_fold(folder, cached_processor, self.manifest, 0,
                    "multisensor", tiny_config(), training, SensorDropoutConfig(), metadata, 2)
                self.assertEqual(cached_reused, reused)
                cached_processor.state["statistics"]["imu"]["offset"][0] += 0.001
                with self.assertRaisesRegex(ValueError, "settings/data differ"):
                    reuse_completed_fold(folder, cached_processor, self.manifest, 0, "multisensor", tiny_config(),
                        training, SensorDropoutConfig(), metadata, 2)
                np.testing.assert_allclose(repeated[PROBABILITY_COLUMNS], predictions[PROBABILITY_COLUMNS], atol=1e-7)
                with self.assertRaisesRegex(ValueError, "settings/data differ"):
                    reuse_completed_fold(folder, processor, self.manifest, 0, "multisensor", tiny_config(),
                        replace(training, learning_rate=0.002), SensorDropoutConfig(), metadata, 2)
                np.testing.assert_allclose(predictions[PROBABILITY_COLUMNS].sum(axis=1), 1, atol=1e-6)
                results.append(predictions)
        np.testing.assert_array_equal(results[0][PROBABILITY_COLUMNS], results[1][PROBABILITY_COLUMNS])

    def test_five_fold_predictions_cover_subjects_and_use_shared_oof_evaluator(self):
        predictions = {}
        training = TrainingConfig(epochs=1, batch_size=16, cpu_threads=1)
        with redirect_stdout(io.StringIO()):
            for fold in range(5):
                arrays, processor, metadata = self.prepare(fold)
                _, predictions[fold] = train_cnn_fold(arrays, processor, self.manifest, fold,
                    self.root / "runs" / f"fold_{fold}", model_config=tiny_config(), training=training,
                    data_metadata=metadata, save_plots=False)
            summary = evaluate_oof_frames(self.manifest, predictions, self.root / "evaluation", experiment_name="test_cnn")
        self.assertEqual(summary["oof_sequences"], len(self.manifest.table))
        self.assertEqual(summary["n_splits"], 5)
        self.assertTrue((self.root / "evaluation/oof_predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
