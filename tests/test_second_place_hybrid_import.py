"""Reject misattributed training changes and crossed per-fold class axes."""
import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_second_place_hybrid import validate_hybrid


class HybridImportGuardTests(unittest.TestCase):
    def setUp(self):
        self.original = {'architecture': 'base', 'variant': 'imu', 'fold': 2, 'seed': 44,
            'joint_labels': [['o1', 'g1', 'b1'], ['o2', 'g2', 'b2']],
            'train_sequence_ids': ['train'], 'train_subjects': ['train_subject'],
            'validation_sequence_ids': ['val'], 'validation_subjects': ['val_subject'],
            'source_commit': 'fixed', 'source_files': {'model.py': 'fixed'}, 'folds_sha256': 'fixed'}
        self.new = {**self.original, 'hybrid_experiment': 'base_imu_official_group_loss_v1',
            'epochs': 50, 'batch_size': 32, 'optimizer': 'Adam lr=.001 wd=.0001',
            'scheduler': '10% step warmup, cosine', 'mixup_alpha': .5, 'phase_loss_weight': 1.,
            'gradient_clip_norm': 1., 'auxiliary_loss': {'macro9_weight': .5, 'binary_weight': .1,
                'targets': 'training joint soft Mixup targets collapsed by gesture; no extra label smoothing'}}

    def test_protocol_drift_rejected(self):
        validate_hybrid(self.new, self.original)
        for key, value in [('epochs', 49), ('phase_loss_weight', 0), ('mixup_alpha', .4),
                           ('auxiliary_loss', {'macro9_weight': .5, 'binary_weight': .2}),
                           ('hybrid_experiment', None)]:
            candidate = {**self.new, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_hybrid(candidate, self.original)

    def test_axis_train_identity_and_seed_drift_rejected(self):
        for key, value in [('joint_labels', list(reversed(self.new['joint_labels']))),
                           ('train_sequence_ids', ['val']), ('validation_subjects', ['train_subject']),
                           ('seed', 42), ('variant', 'all')]:
            candidate = copy.deepcopy(self.new)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_hybrid(candidate, self.original)


if __name__ == '__main__':
    unittest.main()
