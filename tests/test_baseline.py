"""Missing-reading statistics, IMU isolation, and actual held-out model training."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmi_project.baseline import FEATURE_COLUMNS, extract_imu_features, sequence_statistics, train_baseline
from cmi_project.evaluation import ALL_GESTURES
from cmi_project.validation import save_fixed_folds
from test_preprocessing import fixture


class BaselineFeatureTests(unittest.TestCase):
    def test_statistics_exclude_missing_values_and_keep_physical_zero(self):
        sequence = fixture(4)
        sequence["acc_x"] = [0, 3, np.nan, 6]
        sequence["acc_y"] = 0
        sequence["acc_z"] = 0
        sequence[["rot_w", "rot_x", "rot_y", "rot_z"]] = np.nan
        features = sequence_statistics(sequence, 1)
        self.assertEqual(features["acc_x__mean"], 3)
        self.assertAlmostEqual(features["acc_x__std"], np.sqrt(6))
        self.assertEqual(features["acc_x__range"], 6)
        self.assertEqual(features["acc_x__valid_ratio"], .75)
        self.assertEqual(features["acc_magnitude__mean"], 3)
        self.assertEqual(features["acc_magnitude__valid_ratio"], .75)
        self.assertTrue(np.isnan(features["linear_acc_x__mean"]))
        self.assertTrue(np.isnan(features["rot6d_c0_x__std"]))
        self.assertEqual(features["angular_velocity_magnitude__valid_ratio"], 0)
        self.assertEqual(len(FEATURE_COLUMNS), 108)
        self.assertFalse(set(FEATURE_COLUMNS) & {"sequence_id", "subject", "gesture", "length"})

    def test_coordinate_conversion_and_single_sample_statistics(self):
        right = fixture(1)
        right["acc_x"] = 2
        left = right.copy()
        left["acc_x"] = -2
        a, b = sequence_statistics(right, 1), sequence_statistics(left, 0)
        np.testing.assert_allclose([a[c] for c in FEATURE_COLUMNS], [b[c] for c in FEATURE_COLUMNS], equal_nan=True)
        self.assertEqual(a["acc_x__std"], 0)
        self.assertEqual(a["acc_x__range"], 0)
        self.assertEqual(a["acc_x__valid_ratio"], 1)
        self.assertTrue(np.isnan(a["angular_velocity_x__mean"]))

    def test_reader_ignores_other_modalities_and_handles_chunk_boundaries(self):
        a, b = fixture(4), fixture(3, "B", "S2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            data = pd.concat([a, b], ignore_index=True)
            data["thm_1"] = "not a number"
            data["tof_1_v0"] = "not a number"
            data.to_csv(path, index=False)
            demographics = pd.DataFrame({"subject": ["S1", "S2"], "handedness": [1, 0]})
            features = extract_imu_features(path, demographics, chunksize=2)
            self.assertEqual(features["sequence_id"].tolist(), ["A", "B"])
            self.assertEqual(features["length"].tolist(), [4, 3])
            self.assertEqual(features.shape, (2, 112))
            data.loc[6, "gesture"] = "different"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                extract_imu_features(path, demographics, chunksize=2)


class BaselineTrainingChecks:
    model_name = "lightgbm"
    gestures = ALL_GESTURES

    def test_five_models_use_exact_folds_and_write_only_two_files(self):
        library = __import__(self.model_name)
        classifier = getattr(library, "LGBMClassifier" if self.model_name == "lightgbm" else "XGBClassifier")
        rows = []
        for subject in range(10):
            for label_index, gesture in enumerate(self.gestures):
                sequence = fixture(3, f"Q{subject:02}_{label_index:02}", f"S{subject:02}", gesture)
                sequence["acc_x"] = label_index + np.arange(3)
                rows.append(sequence_statistics(sequence, 1))
        features = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = save_fixed_folds(features, root / "folds.csv")
            fold_bytes = (root / "folds.csv").read_bytes()
            fit_calls, predict_calls = [], []
            original_fit, original_predict = classifier.fit, classifier.predict_proba

            def checked_fit(model, X, y, **kwargs):
                fold = len(fit_calls)
                train, validation = manifest.split(fold)
                self.assertEqual(set(X.index), set(train["sequence_id"]))
                self.assertFalse(set(X.index) & set(validation["sequence_id"]))
                self.assertEqual(X.columns.tolist(), FEATURE_COLUMNS)
                self.assertNotIn("eval_set", kwargs)
                fit_calls.append(fold)
                return original_fit(model, X, y, **kwargs)

            def checked_predict(model, X, **kwargs):
                _, validation = manifest.split(len(predict_calls))
                self.assertEqual(set(X.index), set(validation["sequence_id"]))
                probabilities = original_predict(model, X, **kwargs)
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                predict_calls.append(len(predict_calls))
                return probabilities

            parameters = {"n_estimators": 2, "n_jobs": 1}
            parameters.update({"num_leaves": 3, "min_child_samples": 1} if self.model_name == "lightgbm" else {"max_depth": 2})
            with patch.object(classifier, "fit", checked_fit), patch.object(classifier, "predict_proba", checked_predict), redirect_stdout(io.StringIO()):
                result = train_baseline(features.sample(frac=1, random_state=9), manifest, root / "run",
                    model_parameters=parameters, model_name=self.model_name, experiment_name=f"{self.model_name}_fixture")
            self.assertEqual(fit_calls, list(range(5)))
            self.assertEqual(predict_calls, list(range(5)))
            self.assertEqual(result["oof_sequences"], len(features))
            self.assertEqual((root / "folds.csv").read_bytes(), fold_bytes)
            actual_files = sorted(p.relative_to(root / "run").as_posix() for p in (root / "run").rglob("*") if p.is_file())
            self.assertEqual(actual_files, ["evaluation/fold_scores.csv", "evaluation/metrics.json"])
            summary = json.loads((root / "run/evaluation/metrics.json").read_text())
            self.assertEqual(summary["experiment_config"]["model"], self.model_name)
            self.assertEqual(summary["folds_sha256"], manifest.fingerprint)
            scores = pd.read_csv(root / "run/evaluation/fold_scores.csv")
            self.assertEqual(scores["fold"].tolist(), list(range(5)))
            self.assertEqual(scores["validation_sequences"].sum(), len(features))
            self.assertAlmostEqual(summary["fold_mean"]["score"], scores["score"].mean())
            changed = features.copy()
            changed.loc[0, "gesture"] = "different"
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                train_baseline(changed, manifest, root / "bad", model_name=self.model_name)
            with self.assertRaisesRegex(ValueError, "non-empty"):
                train_baseline(features, manifest, root / "run", model_name=self.model_name)


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "Install requirements.txt to test LightGBM.")
class BaselineTrainingTests(BaselineTrainingChecks, unittest.TestCase):
    pass


@unittest.skipUnless(importlib.util.find_spec("xgboost"), "Install requirements.txt to test XGBoost.")
class XGBoostTrainingTests(BaselineTrainingChecks, unittest.TestCase):
    model_name = "xgboost"
    # Non-consecutive global class IDs must be mapped correctly inside each fold.
    gestures = (ALL_GESTURES[0], ALL_GESTURES[4], ALL_GESTURES[17])


if __name__ == "__main__":
    unittest.main()
