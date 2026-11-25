"""Reject source drift or duplicate folds before creating merged artifacts."""
import copy
import importlib.util
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('reference_merge', ROOT / 'scripts/merge_second_place_batches.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MergeGuardTests(unittest.TestCase):
    def setUp(self):
        self.records = [{'status': 'verified', 'architectures': ['base'], 'folds': [fold],
                         'arms': [{'architecture': 'base', 'variant': variant, 'fold': fold}
                                  for variant in ('all', 'all_rot', 'imu', 'imu_rot')],
                         'source_commit': 'fixed', 'folds_sha256': 'fixed',
                         'training_inputs': {'train': 'fixed'}, 'metadata_sha256': 'fixed',
                         'orders': [42, 142, 242]} for fold in range(5)]

    def test_all_unique_folds_required(self):
        module.validate_records(self.records)
        duplicate = copy.deepcopy(self.records)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, 'each fixed fold'):
            module.validate_records(duplicate)
        with self.assertRaises(ValueError):
            module.validate_records(self.records[:-1])

    def test_source_drift_and_missing_branch_rejected(self):
        for key in ('source_commit', 'folds_sha256', 'training_inputs', 'metadata_sha256', 'orders'):
            drift = copy.deepcopy(self.records)
            drift[3][key] = 'different'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs'):
                module.validate_records(drift)
        missing = copy.deepcopy(self.records)
        missing[2]['arms'].pop()
        with self.assertRaisesRegex(ValueError, 'four unique branches'):
            module.validate_records(missing)


if __name__ == '__main__':
    unittest.main()
