"""Fixed group splits, drift detection, raw availability and OOF correctness."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.evaluation import (
    ALL_GESTURES, NON_TARGET_GESTURES, PROBABILITY_COLUMNS,
    cmi_metrics, evaluate_oof, write_fold_predictions,
)
from cmi_project.validation import (
    ConsistentStratifiedGroupKFold, assert_preprocessor_matches, load_fold_manifest,
    make_subject_folds, save_fixed_folds, scan_sequence_index, validate_folds, write_fold_diagnostics,
)


def sequence_index(subjects=10, labels=("G1", "G2")):
    return pd.DataFrame([{"sequence_id": f"Q{s:02}_{g}", "subject": f"S{s:02}",
                          "gesture": label, "length": 2 + s}
                         for s in range(subjects) for g, label in enumerate(labels)])


class ValidationTests(unittest.TestCase):
    def test_subject_isolation_single_validation_coverage_and_sequence_weights(self):
        index = sequence_index()
        folds = make_subject_folds(index)
        seen = []
        for fold in range(5):
            validation, train = folds[folds["fold"] == fold], folds[folds["fold"] != fold]
            self.assertFalse(set(train["subject"]) & set(validation["subject"]))
            self.assertEqual(validation["gesture"].value_counts().to_dict(), {"G1": 2, "G2": 2})
            seen.extend(validation["sequence_id"])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(index["sequence_id"]))
        index.loc[0, "length"] = 100000  # Long sequences do not gain splitting weight.
        changed = make_subject_folds(index.sample(frac=1, random_state=17))
        np.testing.assert_array_equal(changed["fold"], folds["fold"])

    def test_upstream_shuffle_regression_keeps_group_counts_aligned(self):
        y = np.array([0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1])
        groups = np.array([1,1,2,2,3,3,3,4,5,5,5,5,6,6,7,7,8,8])
        for seed in range(8):
            splitter = ConsistentStratifiedGroupKFold(3, shuffle=True, random_state=seed)
            for train, validation in splitter.split(np.zeros((len(y), 1)), y, groups):
                self.assertFalse(set(groups[train]) & set(groups[validation]))
                self.assertAlmostEqual(y[validation].mean(), 1 / 3)

    def test_reuse_is_immutable_and_dataset_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            index = sequence_index()
            manifest = save_fixed_folds(index, path)
            content, metadata = path.read_bytes(), path.with_suffix(".meta.json").read_bytes()
            again = save_fixed_folds(index.sample(frac=1, random_state=3), path)
            self.assertEqual(again.fingerprint, manifest.fingerprint)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.with_suffix(".meta.json").read_bytes(), metadata)
            with self.assertRaisesRegex(ValueError, "protocol"):
                save_fixed_folds(index, path, seed=17)
            for key, value in (("subject", "different"), ("gesture", "different"), ("length", 99), ("sequence_id", "different")):
                changed = index.copy()
                changed.loc[0, key] = value
                with self.assertRaisesRegex(ValueError, "dataset differs"):
                    load_fold_manifest(path, expected_index=changed)
            with self.assertRaises(ValueError):
                load_fold_manifest(path, expected_index=index.iloc[:-1])

    def test_duplicate_ids_leakage_and_fold_tampering_fail_closed(self):
        index = sequence_index()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            make_subject_folds(pd.concat([index, index.iloc[:1]]))
        folded = make_subject_folds(index)
        subject = folded.loc[0, "subject"]
        same_subject = folded.index[folded["subject"] == subject]
        folded.loc[same_subject[0], "fold"] = (folded.loc[same_subject[0], "fold"] + 1) % 5
        with self.assertRaisesRegex(ValueError, "Subject leakage"):
            validate_folds(folded, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            manifest = save_fixed_folds(index, path)
            altered = manifest.table.copy()
            a, b = altered["subject"].iloc[0], altered.loc[altered["fold"] != altered["fold"].iloc[0], "subject"].iloc[0]
            fold_a, fold_b = altered.loc[altered["subject"] == a, "fold"].iloc[0], altered.loc[altered["subject"] == b, "fold"].iloc[0]
            altered.loc[altered["subject"] == a, "fold"] = fold_b
            altered.loc[altered["subject"] == b, "fold"] = fold_a
            altered.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_fold_manifest(path)

    def test_sensor_diagnostics_separate_no_response_from_missing(self):
        from test_preprocessing import fixture
        from cmi_project.dataset_analysis import SENSOR_COLUMNS
        a, b = fixture(2), fixture(3, "B", "S2")
        a.loc[0, "rot_w"] = np.nan
        a[SENSOR_COLUMNS["ToF"]] = -1
        b[SENSOR_COLUMNS["ToF"]] = np.nan
        b[SENSOR_COLUMNS["THM"]] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.csv"
            pd.concat([a, b]).to_csv(source, index=False)
            index = scan_sequence_index(source, chunksize=1, include_sensors=True).set_index("sequence_id")
            self.assertEqual(index.loc["A", "rotation_valid_frames"], 1)
            self.assertEqual(index.loc["A", "tof_present_cells"], 640)
            self.assertEqual(index.loc["A", "tof_valid_cells"], 0)
            self.assertEqual(index.loc["B", "tof_present_cells"], 0)
            self.assertEqual(index.loc["B", "tof_nan_cells"], 960)
            self.assertEqual(index.loc["B", "thm_valid_cells"], 0)
            manifest = save_fixed_folds(index.reset_index(), root / "folds.csv", n_splits=2)
            report = write_fold_diagnostics(manifest, index.reset_index(), root / "diagnostics")
            self.assertTrue(report["every_sequence_validated_once"])
            self.assertTrue((root / "diagnostics/validation_report.html").is_file())

    def test_preprocessor_provenance_requires_exact_training_members(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_fixed_folds(sequence_index(), Path(directory) / "folds.csv")
            train, _ = manifest.split(0)
            state = {"validation": {"fold": 0, "folds_sha256": manifest.fingerprint},
                     "train_sequence_ids": train["sequence_id"].tolist(), "train_subjects": train["subject"].unique().tolist()}
            assert_preprocessor_matches(state, manifest, 0)
            state["train_sequence_ids"].pop()
            with self.assertRaisesRegex(ValueError, "training sequences"):
                assert_preprocessor_matches(state, manifest, 0)


class OOFTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = save_fixed_folds(sequence_index(labels=ALL_GESTURES), self.root / "folds.csv")
        self.predictions = self.root / "predictions"
        for fold in range(5):
            _, val = self.manifest.split(fold)
            predictions = val[["sequence_id"]].copy()
            predictions["predicted_gesture"] = val["gesture"]
            write_fold_predictions(self.manifest, fold, predictions.iloc[::-1], self.predictions / f"fold_{fold}.csv")

    def tearDown(self):
        self.temporary.cleanup()

    def test_competition_metric_collapses_non_target_subtypes(self):
        predictions = list(ALL_GESTURES)
        predictions[8:] = list(NON_TARGET_GESTURES[1:] + NON_TARGET_GESTURES[:1])
        self.assertEqual(cmi_metrics(ALL_GESTURES, predictions), {"score": 1, "binary_f1": 1, "macro_f1_9class": 1})
        wrong = [NON_TARGET_GESTURES[0]] * 18
        score = cmi_metrics(ALL_GESTURES, wrong)
        self.assertEqual(score["binary_f1"], 0)
        self.assertAlmostEqual(score["macro_f1_9class"], (20 / 28) / 9)

    def test_full_oof_export_mean_std_and_metadata(self):
        result = evaluate_oof(self.manifest, self.predictions, self.root / "results", experiment_name="test_fixture")
        self.assertEqual(result["fold_mean"]["score"], 1)
        self.assertEqual(result["fold_std"]["score"], 0)
        self.assertEqual(result["std_ddof"], 1)
        self.assertEqual(result["oof_sequences"], 180)
        self.assertEqual(result["folds_sha256"], self.manifest.fingerprint)
        scores = pd.read_csv(self.root / "results/fold_scores.csv")
        self.assertEqual(scores["fold"].tolist(), list(range(5)))
        oof = pd.read_csv(self.root / "results/oof_predictions.csv")
        self.assertEqual(oof["sequence_id"].nunique(), 180)

    def test_incomplete_duplicate_wrong_fold_or_wrong_hash_oof_rejected(self):
        path = self.predictions / "fold_0.csv"
        original = pd.read_csv(path)
        variants = [original.iloc[:-1], pd.concat([original, original.iloc[:1]])]
        for key, value in (("fold", 1), ("folds_sha256", "wrong"), ("predicted_gesture", "unknown"), ("subject", "wrong")):
            wrong = original.copy()
            wrong.loc[0, key] = value
            variants.append(wrong)
        for variant in variants:
            variant.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                evaluate_oof(self.manifest, self.predictions, self.root / "results", experiment_name="bad_fixture")
        original.drop(columns="folds_sha256").to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            evaluate_oof(self.manifest, self.predictions, self.root / "results", experiment_name="bad_fixture")

    def test_named_probabilities_are_validated_and_retained(self):
        for fold in range(5):
            _, val = self.manifest.split(fold)
            frame = val[["sequence_id"]].reset_index(drop=True)
            frame["predicted_gesture"] = val["gesture"].to_numpy()
            probs = np.zeros((len(frame), 18))
            for i, label in enumerate(frame["predicted_gesture"]):
                probs[i, ALL_GESTURES.index(label)] = 1
            frame = pd.concat([frame, pd.DataFrame(probs, columns=PROBABILITY_COLUMNS)], axis=1)
            write_fold_predictions(self.manifest, fold, frame, self.predictions / f"fold_{fold}.csv")
        evaluate_oof(self.manifest, self.predictions, self.root / "results", experiment_name="prob_fixture")
        actual = pd.read_csv(self.root / "results/oof_predictions.csv")
        self.assertTrue(set(PROBABILITY_COLUMNS).issubset(actual.columns))
        invalid = pd.read_csv(self.predictions / "fold_0.csv")
        invalid.loc[0, PROBABILITY_COLUMNS] = .5
        with self.assertRaisesRegex(ValueError, "probabilities"):
            write_fold_predictions(self.manifest, 0, invalid, self.predictions / "bad.csv")


if __name__ == "__main__":
    unittest.main()
