import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmi_project.second_place import (ARCHITECTURES, ReferenceDataset, build_cache, collapse_probabilities,
                                       fit_joint_labels, load_reference)
from cmi_project.evaluation import ALL_GESTURES
from cmi_project.second_place_evaluation import CausalJointAssignment, check_arm, read_aligned_logits


class SecondPlaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT / 'src/cmi_project/vendor/second_place'
        if not folder.exists():
            raise unittest.SkipTest('Pinned vendored upstream checkout is required')
        torch.set_num_threads(2)
        cls.ref = load_reference(folder, ROOT / 'configs/second_place_source.json')

    def test_train_only_joint_ontology_and_marginal_probabilities(self):
        metadata = pd.DataFrame({'sequence_id': ['train1', 'train2', 'val'],
            'orientation': ['up', 'down', 'SECRET'], 'gesture': [ALL_GESTURES[0], ALL_GESTURES[0], 'SECRET'],
            'initial_behavior': ['move', 'move', 'SECRET']})
        labels = fit_joint_labels(metadata, ['train1', 'train2'])
        self.assertEqual(len(labels), 2)
        self.assertFalse(any('SECRET' in key for key in labels))
        result = collapse_probabilities(np.array([[.3, .7]]), labels)
        self.assertEqual(result[0, 0], 1.)
        self.assertEqual(result.sum(), 1.)

    def test_anonymous_logit_index_preserves_sequence_column(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from train_second_place import prediction_frame
        metadata = pd.DataFrame({'sequence_id': ['b', 'a'], 'subject': ['p', 'q'],
            'gesture': [ALL_GESTURES[0], ALL_GESTURES[1]]})
        ids = pd.Index(['a', 'b'])  # np.load IDs and v1 fallback have no index name
        aligned = metadata.set_index('sequence_id').loc[ids].reset_index(names='sequence_id')
        labels = [('up', ALL_GESTURES[0], 'move'), ('up', ALL_GESTURES[1], 'move')]
        frame = prediction_frame(aligned, np.array([[0., 1.], [1., 0.]]), labels, 'fixed')
        self.assertEqual(frame.sequence_id.tolist(), ['a', 'b'])
        self.assertEqual(frame.predicted_gesture.tolist(), [ALL_GESTURES[1], ALL_GESTURES[0]])

    def test_all_six_upstream_architectures_forward_backward_and_annotation_isolation(self):
        for arch, names in ARCHITECTURES.items():
            for full, name in enumerate(names):
                with self.subTest(arch=arch, full=full):
                    model = getattr(self.ref.models, name)(input_size=15, n_classes=18)
                    x = torch.randn(2, 8, 335 if full else 15)
                    lengths = torch.tensor([8, 5])
                    model.train()
                    outputs = model(x, lengths, None)
                    outputs['gesture_logits'].square().mean().backward()
                    self.assertTrue(torch.isfinite(outputs['gesture_logits']).all())
                    model.eval()
                    with torch.no_grad():
                        a = model(x, lengths, torch.zeros(2, 8, dtype=torch.long))['gesture_logits']
                        b = model(x, lengths, torch.full((2, 8), 2, dtype=torch.long))['gesture_logits']
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_dropout_does_not_modify_cache_and_validation_cannot_read_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            features = np.ones((4, 335), dtype=np.float32)
            np.save(path / 'features.npy', features)
            np.save(path / 'phases.npy', np.array([0, 0, 1, 2]))
            np.save(path / 'offsets.npy', [0, 4])
            pd.DataFrame({'sequence_id': ['a'], 'subject': ['s'], 'orientation': ['up'],
                          'gesture': [ALL_GESTURES[0]], 'initial_behavior': ['move']}).to_csv(path / 'metadata.csv', index=False)
            train = ReferenceDataset(path, [0], [('up', ALL_GESTURES[0], 'move')], use_tof=True, rotation_zero=True, training=True)
            state = random.getstate()
            try:
                random.seed(31)  # first draw < .1
                x, _, _, _ = train[0]
                self.assertEqual(x[:, 15:].sum(), 0)
                self.assertEqual(x[:, 3:15].sum(), 0)
            finally:
                random.setstate(state)
            np.testing.assert_array_equal(np.load(path / 'features.npy'), features)
            val = ReferenceDataset(path, [0], [], use_tof=True, training=False)
            val.metadata[['orientation', 'gesture', 'initial_behavior']] = 'POISON'
            x, _, target, phase = val[0]
            self.assertEqual(target.item(), -1)
            self.assertTrue((phase == -1).all())
            self.assertEqual(x.sum(), features.sum())
            train.close()
            val.close()

    def test_real_training_checkpoint_and_held_out_prediction_pipeline(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from train_second_place import train_one
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            cache = path / 'cache'
            cache.mkdir()
            np.save(cache / 'features.npy', np.random.default_rng(1).normal(size=(48, 335)).astype(np.float32))
            np.save(cache / 'phases.npy', np.tile([0, 0, 1, 1, 2, 2, 2, 2], 6))
            np.save(cache / 'offsets.npy', np.arange(7) * 8)
            metadata = pd.DataFrame({'sequence_id': [f's{i}' for i in range(6)],
                'subject': [f'p{i}' for i in range(6)], 'gesture': [ALL_GESTURES[i % 2] for i in range(6)],
                'orientation': ['up'] * 6, 'initial_behavior': ['move'] * 6, 'fold': [1, 1, 1, 1, 0, 0]})
            metadata = metadata.iloc[[0, 1, 2, 3, 5, 4]].reset_index(drop=True)
            metadata.to_csv(cache / 'metadata.csv', index=False)
            manifest = SimpleNamespace(fingerprint='fixed', path=path / 'folds.csv',
                split=lambda fold: (metadata[metadata.fold != fold], metadata[metadata.fold == fold]))
            args = SimpleNamespace(output=path / 'run', device='cpu', batch_size=2, epochs=1, folds=[0])
            labels = fit_joint_labels(metadata, metadata.sequence_id[:4])
            out = train_one(args, 0, 'base', 'imu', cache, self.ref, manifest, metadata, labels)
            predictions = pd.read_csv(out / 'predictions.csv')
            self.assertEqual(set(predictions.sequence_id), {'s4', 's5'})
            checkpoint = torch.load(out / 'last.pt', weights_only=True, map_location='cpu')
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertFalse(set(checkpoint['provenance']['train_subjects']) & set(checkpoint['provenance']['validation_subjects']))
            self.assertEqual(json.loads((out / 'metrics.json').read_text())['validation_sequences'], 2)
            report, ids, logits, _ = check_arm(out, manifest, metadata, self.ref.provenance, epochs=1)
            self.assertEqual(ids.tolist(), ['s5', 's4'])  # logits are not sorted prediction CSV rows!
            self.assertEqual(report['sequences'], 2)
            (out / 'joint_sequence_ids.npy').unlink()
            legacy_ids, legacy_logits = read_aligned_logits(out, metadata, 0)
            self.assertEqual(legacy_ids.tolist(), ids.tolist())
            np.testing.assert_array_equal(legacy_logits, logits)
            provenance = json.loads((out / 'provenance.json').read_text())
            provenance['train_subjects'].append('p5')
            (out / 'provenance.json').write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'train_subjects'):
                check_arm(out, manifest, metadata, self.ref.provenance, epochs=1)

    def test_causal_joint_history_matches_upstream_prefixes_and_preserves_returned_decisions(self):
        import ast
        from scipy.optimize import linear_sum_assignment
        tree = ast.parse((ROOT / 'src/cmi_project/vendor/second_place/test.py').read_text(encoding='utf-8'))
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'solve_capacity1_with_hungarian')
        namespace = {'np': np, 'linear_sum_assignment': linear_sum_assignment}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<verified reference decoder>', 'exec'), namespace)
        scores = np.random.default_rng(5).normal(size=(8, 8))
        decoder = CausalJointAssignment(8)
        for i, vector in enumerate(scores):
            self.assertEqual(decoder.predict_one('subject', vector), namespace[node.name](scores[:i + 1]))
        self.assertEqual(decoder.predict_one('subject', np.arange(8)), 7)
        self.assertEqual(decoder.overflow, 1)
        self.assertEqual(decoder.predict_one('other', np.arange(8)), 7)
        causal = CausalJointAssignment(2)
        first = causal.predict_one('s', [.9, .8])
        second = causal.predict_one('s', [100, 0])
        self.assertEqual((first, second), (0, 0))  # latent past assignment changes, returned past label stays fixed
        with self.assertRaises(ValueError):
            causal.predict_one('s', [np.nan, 0])

    def test_raw_cache_uses_upstream_physics_and_keeps_metadata_outside_features(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            row = {'sequence_id': 's', 'subject': 'p', 'orientation': 'up',
                   'gesture': ALL_GESTURES[0], 'behavior': 'Moves hand to target location',
                   'acc_x': 0., 'acc_y': 0., 'acc_z': 9.81,
                   'rot_x': 0., 'rot_y': 0., 'rot_z': 0., 'rot_w': 1.}
            row.update({f'tof_{i}_v{j}': 123. for i in range(1, 6) for j in range(64)})
            frame = pd.DataFrame([{**row, 'sequence_counter': i} for i in range(4)])
            frame.loc[3, 'behavior'] = 'Performs gesture'
            frame.to_csv(path / 'train.csv', index=False)
            pd.DataFrame({'subject': ['p'], 'handedness': [1]}).to_csv(path / 'train_demographics.csv', index=False)
            manifest = SimpleNamespace(fingerprint='fixed', table=pd.DataFrame({
                'sequence_id': ['s'], 'subject': ['p'], 'gesture': [ALL_GESTURES[0]], 'fold': [0]}))
            build_cache(path, path / 'cache', manifest, self.ref)
            result = np.load(path / 'cache/features.npy')
            self.assertEqual(result.shape, (4, 335))
            np.testing.assert_allclose(result[:, 12:15], 0, atol=1e-6)
            np.testing.assert_allclose(result[:, 15:], 123.)
            np.testing.assert_array_equal(np.load(path / 'cache/phases.npy'), [0, 0, 0, 2])


if __name__ == '__main__':
    unittest.main()
