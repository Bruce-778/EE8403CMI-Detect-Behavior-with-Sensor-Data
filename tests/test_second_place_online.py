"""Causal prefix, immutable base weights and subject-isolated pseudo updates."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cmi_project.second_place_online import SubjectOnlineAdapter, VARIANTS


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(3)
        self.dropout = torch.nn.Dropout(.3)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x, lengths, phase):
        assert phase is None
        return {'gesture_logits': self.head(self.dropout(self.norm(x[:, :, :3].mean(1))))}


class OnlineCausalityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.models = {v: TinyModel().eval() for v in VARIANTS}
        self.inputs = [np.random.default_rng(i).normal(size=(4, 335)).astype(np.float32) for i in range(35)]

    def test_other_subject_interleaving_cannot_change_prefix_or_update(self):
        isolated = SubjectOnlineAdapter(copy.deepcopy(self.models), 2, seed=42)
        interleaved = SubjectOnlineAdapter(copy.deepcopy(self.models), 2, seed=42)
        before = {v: copy.deepcopy(m.state_dict()) for v, m in interleaved.baseline.items()}
        returned = []
        for i, features in enumerate(self.inputs):
            expected, expected_logits = isolated.predict_one(f'a{i}', 'A', features)
            actual, actual_logits = interleaved.predict_one(f'a{i}', 'A', features)
            returned.append(actual)
            self.assertEqual(actual, expected)
            np.testing.assert_array_equal(actual_logits, expected_logits)
            # B updates its own models before A reaches its first threshold.
            for j in range(2):
                interleaved.predict_one(f'b{i}_{j}', 'B', features * 10)
        self.assertEqual(len(returned), 35)
        for variant, model in interleaved.baseline.items():
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[variant][name], value))
        a_traces = [x for x in interleaved.trace if x['subject'] == 'A']
        self.assertEqual(len(a_traces), 4)
        for trace in a_traces:
            self.assertEqual(trace['arrival_count'], 32)
            self.assertEqual(trace['sequence_ids'], [f'a{i}' for i in range(32)])
            self.assertTrue(all(sid.startswith('a') for sid in trace['sequence_ids']))
        # Train-mode BN changed A's state but the shared frozen state is intact.
        self.assertFalse(torch.equal(interleaved.subjects['A']['models']['all'].norm.running_mean,
                                     interleaved.baseline['all'].norm.running_mean))

    def test_prefix_invariant_to_future_and_fresh_state(self):
        left = SubjectOnlineAdapter(copy.deepcopy(self.models), 2, seed=142)
        right = SubjectOnlineAdapter(copy.deepcopy(self.models), 2, seed=142)
        prefix_left, prefix_right = [], []
        for i in range(33):
            prefix_left.append(left.predict_one(str(i), 'A', self.inputs[i])[0])
            prefix_right.append(right.predict_one(str(i), 'A', self.inputs[i])[0])
        left.predict_one('future', 'A', self.inputs[34])
        right.predict_one('different_future', 'A', self.inputs[34] * -100)
        self.assertEqual(prefix_left, prefix_right)
        fresh = SubjectOnlineAdapter(copy.deepcopy(self.models), 2, seed=142)
        self.assertEqual(fresh.predict_one('0', 'A', self.inputs[0])[0], prefix_left[0])
        self.assertEqual(fresh.trace, [])
        with self.assertRaises(ValueError):
            fresh.predict_one('0', 'A', self.inputs[0])


if __name__ == '__main__':
    unittest.main()
