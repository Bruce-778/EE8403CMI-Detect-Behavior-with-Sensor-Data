"""Gradient direction, teacher isolation, frozen inputs and real post-training checks."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.cnn_data import prepare_frozen_cnn_fold
from cmi_project.cnn_training import TrainingConfig, train_cnn_fold
from cmi_project.posttraining import (DistillationConfig, DistillationDataset, checked_pair,
    distillation_loss, teacher_training_targets, train_posttraining_fold)
from cmi_project.preprocessing import SensorDropoutConfig
from cmi_project.evaluation import cmi_metrics, PROBABILITY_COLUMNS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_posttraining import routed_metrics
import test_cnn as cnn_fixture


class DistillationLossTests(unittest.TestCase):
    def test_kl_direction_temperature_scaling_and_detached_teacher(self):
        student = torch.tensor([[2., -1., 0.], [0., 1., -1.]], requires_grad=True)
        teacher = torch.tensor([[0., 2., -1.], [1., 0., -1.]], requires_grad=True)
        temperature = 2.
        q = (teacher.detach() / temperature).softmax(-1)
        log_p = (student / temperature).log_softmax(-1)
        expected = (q * (q.log() - log_p)).sum() / 2 * temperature ** 2
        actual = distillation_loss(student, teacher, torch.ones(2, dtype=torch.bool), temperature)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_unavailable_teacher_rows_have_zero_gradient_and_extremes_are_finite(self):
        student = torch.tensor([[1000., -1000.], [1., 2.]], requires_grad=True)
        teacher = torch.tensor([[-1000., 1000.], [float("nan"), float("nan")]])
        loss = distillation_loss(student, teacher, torch.tensor([True, False]), 2.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        torch.testing.assert_close(student.grad[1], torch.zeros(2))
        self.assertTrue(torch.isfinite(student.grad).all())
        zero = distillation_loss(student.detach(), torch.full_like(teacher, float("nan")),
                                 torch.zeros(2, dtype=torch.bool), 2.)
        self.assertEqual(float(zero), 0)
        for values in ({"temperature": 0}, {"weight": -1}, {"temperature": float("nan")}):
            with self.assertRaises(ValueError):
                DistillationConfig(**values)


class PosttrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = cnn_fixture.CNNPipelineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.arrays, self.processor, self.metadata = self.fixture.prepare()
        self.manifest = self.fixture.manifest
        self.root = self.fixture.root
        for name in ("imu", "multisensor"):
            train_cnn_fold(self.arrays, self.processor, self.manifest, 0, self.root / name,
                model_name=name, model_config=cnn_fixture.tiny_config(), training=TrainingConfig(epochs=1, cpu_threads=1),
                data_metadata=self.metadata, save_plots=False)
        self.student_path, self.teacher_path = self.root / "imu/best.pt", self.root / "multisensor/best.pt"

    def test_teacher_targets_exclude_validation_and_freeze_all_parameters(self):
        _, teacher, _, _ = checked_pair(self.student_path, self.teacher_path, self.manifest, 0)
        original = deepcopy(teacher.state_dict())
        targets, eligible = teacher_training_targets(teacher, self.arrays, self.manifest, 0, batch_size=8)
        validation = self.manifest.table["fold"].to_numpy() == 0
        self.assertTrue(np.isnan(targets[validation]).all())
        self.assertFalse(eligible[validation].any())
        self.assertTrue(np.isfinite(targets[~validation]).all())
        for key, value in original.items():
            torch.testing.assert_close(value, teacher.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        with self.assertRaisesRegex(ValueError, "validation"):
            DistillationDataset(self.arrays, np.flatnonzero(validation), targets, eligible,
                                seed=42, dropout=SensorDropoutConfig())
        leaked = torch.load(self.teacher_path, weights_only=True)
        leaked["preprocessor"]["train_subjects"].append("validation-subject")
        path = self.root / "leaked_teacher.pt"
        torch.save(leaked, path)
        with self.assertRaisesRegex(ValueError, "subjects"):
            checked_pair(self.student_path, path, self.manifest, 0)
        with self.assertRaises(ValueError):
            checked_pair(self.student_path, self.teacher_path, self.manifest, 1)

    def test_frozen_cache_never_refits_and_rejects_changed_scalers(self):
        cache = self.root / "frozen"
        first, metadata = prepare_frozen_cnn_fold(self.fixture.data, cache, self.manifest, 0, self.processor,
                                                 chunksize=11)
        second, repeated = prepare_frozen_cnn_fold(self.fixture.data, cache, self.manifest, 0, self.processor)
        self.assertEqual(metadata, repeated)
        np.testing.assert_array_equal(first["imu"], second["imu"])
        np.testing.assert_array_equal(first["imu"], self.arrays["imu"])
        changed = deepcopy(self.processor)
        changed.state["statistics"]["imu"]["offset"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_frozen_cnn_fold(self.fixture.data, cache, self.manifest, 0, changed)

    def test_real_paired_training_preserves_sources_and_can_keep_epoch_zero(self):
        _, teacher, processor, _ = checked_pair(self.student_path, self.teacher_path, self.manifest, 0)
        targets, eligible = teacher_training_targets(teacher, self.arrays, self.manifest, 0)
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.student_path, self.teacher_path)]
        for weight in (0., 0.5):
            result, predictions = train_posttraining_fold(self.arrays, processor, self.manifest, 0,
                self.student_path, self.root / f"post_{weight}",
                training=TrainingConfig(epochs=1, batch_size=8, cpu_threads=1, min_delta=1.,
                                       learning_rate=0.0001),
                distillation=DistillationConfig(weight=weight), targets=targets, eligible=eligible,
                data_metadata=self.metadata, source_teacher={"teacher_fold": 0}, save_plots=False)
            self.assertGreaterEqual(result["validation"]["score"], result["baseline"]["score"])
            self.assertEqual(len(predictions), int((self.manifest.table["fold"] == 0).sum()))
            self.assertEqual(result["posttraining"]["teacher_validation_target_sequences"], 0)
            self.assertIn(result["best_epoch"], (0, 1))
        after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.student_path, self.teacher_path)]
        self.assertEqual(before, after)

    def test_routed_report_checks_fingerprints_and_preserves_missing_sensor_rules(self):
        a = pd.read_csv(self.root / "imu/predictions.csv")
        b = pd.read_csv(self.root / "multisensor/predictions.csv")
        expected_a = cmi_metrics(a["gesture"], a["predicted_gesture"])
        expected_b = cmi_metrics(b["gesture"], b["predicted_gesture"])
        arrays = deepcopy(self.arrays)
        arrays["thm_valid"][:] = False
        arrays["tof_sensor_present"][:] = False
        with patch("train_posttraining.pd.read_csv", return_value=b):
            saved = {}
            result = routed_metrics(arrays, self.manifest, 0, a, self.root, prediction_sink=saved)
            self.assertEqual(result["observed"]["score"], expected_a["score"])
            self.assertEqual(result["observed"]["used_model_b"], 0)
            for scenario, frame in saved.items():
                self.assertEqual(set(frame.sequence_id), set(a.sequence_id))
                self.assertEqual(cmi_metrics(frame.gesture, frame.predicted_gesture)["score"], result[scenario]["score"])
            arrays["thm_valid"][:] = arrays["time_mask"][..., None]
            result = routed_metrics(arrays, self.manifest, 0, a, self.root)
            self.assertEqual(result["observed"]["score"], expected_b["score"])
            self.assertEqual(result["observed"]["used_model_a"], 0)
            self.assertEqual(result["imu_only"]["score"], expected_a["score"])
            removed = sum(int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16) % 2 == 0
                          for sid in a["sequence_id"])
            self.assertEqual(result["aux_dropout50"]["used_model_a"], removed)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                routed_metrics(arrays, self.manifest, 0, a.drop(columns="folds_sha256"), self.root)
        with patch("train_posttraining.pd.read_csv", return_value=a):
            frames = {}
            result = routed_metrics(arrays, self.manifest, 0, b, self.root,
                                    updated_model="multisensor", prediction_sink=frames)
            self.assertEqual(result["observed"]["score"], expected_b["score"])
            np.testing.assert_allclose(frames["observed"].set_index("sequence_id").loc[b.sequence_id, PROBABILITY_COLUMNS],
                                       b[PROBABILITY_COLUMNS], rtol=0, atol=0)
            np.testing.assert_allclose(frames["imu_only"].set_index("sequence_id").loc[a.sequence_id, PROBABILITY_COLUMNS],
                                       a[PROBABILITY_COLUMNS], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
