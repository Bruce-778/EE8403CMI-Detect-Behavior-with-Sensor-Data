"""Phase alignment, no-oracle inference, contrastive gradients and real training."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.cnn import CMI1DCNN
from cmi_project.cnn_training import TrainingConfig, train_cnn_fold, load_cnn_checkpoint, CMIHierarchicalLoss
from cmi_project.posttraining import DistillationConfig, train_posttraining_fold
from cmi_project.preprocessing import SensorDropoutConfig
from cmi_project.representation import (RepresentationConfig, RepresentationDataset, RepresentationIMUCNN,
                                       cross_subject_contrastive_loss, training_phase_targets)
import test_cnn as cnn_fixture


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_initial_phase_model_matches_source_and_preserves_dropout_rng(self):
        base = CMI1DCNN(config=cnn_fixture.tiny_config())
        rng = torch.random.get_rng_state()
        model = RepresentationIMUCNN.from_starting_model(base, True)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        batch = cnn_fixture.toy_batch()
        torch.random.set_rng_state(rng)
        expected = base(batch)
        torch.random.set_rng_state(rng)
        actual = model(batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        base.eval(); model.eval()
        for key in base.state_dict():
            torch.testing.assert_close(base.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)

    def test_phase_forward_ignores_truth_and_padding_loss_geometry_is_finite(self):
        model = RepresentationIMUCNN(config=cnn_fixture.tiny_config(), phase_enabled=True).eval()
        batch = cnn_fixture.toy_batch()
        original = model(batch)
        poisoned = deepcopy(batch)
        poisoned.update(training_phase=torch.full_like(batch["time_mask"], 999),
                        training_subject=torch.tensor([999, 888, 777]), label=torch.tensor([17, 16, 15]))
        poisoned["imu"][~batch["time_mask"]] = float("nan")
        torch.testing.assert_close(original, model(poisoned), rtol=0, atol=0)
        logits, _, phase_logits, mask = model.forward_outputs(batch)
        phase = torch.arange(batch["time_mask"].shape[1])[None].expand(3, -1) % 3
        phase = torch.where(batch["time_mask"], phase, -100)
        loss = model.phase_loss(phase_logits, phase, batch["time_mask"], mask)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.phase_head.weight.grad).all())
        self.assertGreater(float(model.phase_head.weight.grad.abs().sum()), 0)
        empty = model.phase_loss(phase_logits, torch.full_like(phase, -100),
                                torch.zeros_like(batch["time_mask"]), mask)
        self.assertEqual(float(empty.detach()), 0)

    def test_cross_subject_loss_excludes_self_and_within_subject_positive_pairs(self):
        embedding = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
        labels, subjects = torch.tensor([0, 0, 1]), torch.tensor([1, 2, 1])
        z = torch.nn.functional.normalize(embedding, dim=1)
        scores = z @ z.T / .1
        expected = ((torch.logsumexp(scores[0, 1:], 0) - scores[0, 1]) +
                    (torch.logsumexp(scores[1, [0, 2]], 0) - scores[1, 0])) / 2
        loss = cross_subject_contrastive_loss(embedding, labels, subjects, .1)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(embedding.grad).all())
        zero = cross_subject_contrastive_loss(embedding, labels, torch.ones(3), .1)
        self.assertEqual(float(zero.detach()), 0)
        single = cross_subject_contrastive_loss(embedding[:1], labels[:1], subjects[:1], .1)
        self.assertEqual(float(single.detach()), 0)

    def test_metric_objective_reduces_only_subtype_penalty(self):
        logits = torch.randn(2, 18, requires_grad=True)
        labels_a, labels_b = torch.tensor([1, 9]), torch.tensor([1, 17])
        old = CMIHierarchicalLoss(TrainingConfig(macro_loss_weight=.5, binary_loss_weight=.1))
        new = CMIHierarchicalLoss(TrainingConfig(eighteen_loss_weight=.25, macro_loss_weight=.5, binary_loss_weight=.1))
        torch.testing.assert_close(new(logits, labels_a) - new(logits, labels_b),
                                   .25 * (old(logits, labels_a) - old(logits, labels_b)))
        collapsed = CMIHierarchicalLoss(TrainingConfig(eighteen_loss_weight=0, macro_loss_weight=.5, binary_loss_weight=.1))
        torch.testing.assert_close(collapsed(logits, labels_a), collapsed(logits, labels_b))
        new(logits, labels_a).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class RepresentationPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = cnn_fixture.CNNPipelineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.arrays, self.processor, self.metadata = self.fixture.prepare()
        self.manifest, self.root = self.fixture.manifest, self.fixture.root

    def test_annotation_alignment_tail_padding_and_validation_exclusion(self):
        raw = pd.read_csv(self.fixture.data / "train.csv")
        names = np.array(["Moves hand to target location", "Hand at target location", "Performs gesture"])
        raw["behavior"] = names[raw.sequence_counter.astype(int) % 3]
        raw = raw.iloc[::-1]  # sorting must recover counter order across chunks
        path = self.root / "annotations.csv"
        raw.to_csv(path, index=False)
        phase = training_phase_targets(path, self.arrays, self.manifest, 0, chunksize=7)
        train = self.manifest.table.fold.to_numpy() != 0
        self.assertTrue((phase[~train] == -100).all())
        for position in np.flatnonzero(train):
            frame = raw[raw.sequence_id == self.arrays["sequence_id"][position]].sort_values("sequence_counter")
            length = self.processor.max_length
            labels = frame.sequence_counter.to_numpy(dtype=int)[-length:] % 3
            np.testing.assert_array_equal(phase[position, -len(labels):], labels)
            self.assertTrue((phase[position, :-len(labels)] == -100).all())
        dataset = RepresentationDataset(self.arrays, np.flatnonzero(train), self.manifest, 0,
            phase=phase, seed=42, dropout=SensorDropoutConfig())
        self.assertIn("training_phase", dataset[0])
        with self.assertRaisesRegex(ValueError, "Validation"):
            RepresentationDataset(self.arrays, np.flatnonzero(~train), self.manifest, 0,
                phase=phase, seed=42, dropout=SensorDropoutConfig())
        phase[~train] = 0
        with self.assertRaisesRegex(ValueError, "validation"):
            RepresentationDataset(self.arrays, np.flatnonzero(train), self.manifest, 0,
                phase=phase, seed=42, dropout=SensorDropoutConfig())

    def test_real_training_both_methods_preserves_sources_and_round_trips(self):
        source = self.root / "source"
        train_cnn_fold(self.arrays, self.processor, self.manifest, 0, source, model_name="imu",
            model_config=cnn_fixture.tiny_config(), training=TrainingConfig(epochs=1, cpu_threads=1),
            data_metadata=self.metadata, save_plots=False)
        path = source / "best.pt"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        train = self.manifest.table.fold.to_numpy() != 0
        phase = np.full(self.arrays["time_mask"].shape, -100, dtype=np.int64)
        phase[train] = np.where(self.arrays["time_mask"][train], 2, -100)
        attempts = [("phase", "phase", "full", .1),
                    ("cross_subject_supcon", "cross_subject_supcon", "full", .1),
                    ("frozen_control", "phase", "phase_heads", 0.),
                    ("frozen_phase", "phase", "phase_heads", .1)]
        for name, method, scope, weight in attempts:
            result, prediction = train_posttraining_fold(self.arrays, self.processor, self.manifest,
                0, path, self.root / name, training=TrainingConfig(epochs=1, batch_size=8,
                cpu_threads=1, learning_rate=.0001), distillation=DistillationConfig(weight=0),
                targets=None, eligible=None, data_metadata=self.metadata, source_teacher={},
                representation=RepresentationConfig(method=method, trainable_scope=scope, weight=weight),
                phase_targets=phase if method == "phase" else None, save_plots=False)
            self.assertGreaterEqual(result["validation"]["score"], result["baseline"]["score"])
            restored, _, saved = load_cnn_checkpoint(self.root / name / "best.pt")
            self.assertEqual(restored.metadata(), saved["model_metadata"])
            self.assertEqual(result["posttraining"]["validation_annotation_sequences"], 0)
            self.assertEqual(len(prediction), int((~train).sum()))
            if scope == "phase_heads":
                original = torch.load(path, weights_only=True)["state_dict"]
                for key, value in original.items():
                    torch.testing.assert_close(value, saved["state_dict"][key], rtol=0, atol=0)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_multisensor_metric_finetuning_uses_unchanged_architecture(self):
        source = self.root / "multisensor"
        train_cnn_fold(self.arrays, self.processor, self.manifest, 0, source, model_name="multisensor",
            model_config=cnn_fixture.tiny_config(), training=TrainingConfig(epochs=1, cpu_threads=1),
            data_metadata=self.metadata, save_plots=False)
        path = source / "best.pt"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        for weight in (1., .25):
            result, _ = train_posttraining_fold(self.arrays, self.processor, self.manifest, 0, path,
                self.root / f"metric_{weight}", training=TrainingConfig(epochs=1, batch_size=8,
                cpu_threads=1, learning_rate=.0001, eighteen_loss_weight=weight),
                distillation=DistillationConfig(weight=0), targets=None, eligible=None,
                data_metadata=self.metadata, source_teacher={}, model_name="multisensor",
                representation=RepresentationConfig(method="official_metric", weight=0), save_plots=False)
            self.assertEqual(result["model"], "multisensor")
            self.assertEqual(result["model_metadata"]["architecture"], "grouped_masked_se_cnn_v3")
            self.assertEqual(result["posttraining"]["training_annotation_sequences"], 0)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
