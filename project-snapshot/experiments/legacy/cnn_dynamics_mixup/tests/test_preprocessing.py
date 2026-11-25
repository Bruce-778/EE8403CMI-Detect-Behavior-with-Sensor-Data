"""Independent physical fixtures and fold-leakage/export regression checks."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmi_project.dataset_analysis import ALL_SENSOR_COLUMNS, SENSOR_COLUMNS
from cmi_project.preprocess_cli import prepare_folds
from cmi_project.validation import save_fixed_folds, scan_sequence_index
from cmi_project.preprocessing import (
    FoldPreprocessor, PreprocessingConfig, SensorDropoutConfig, engineer_sequence,
    iter_csv_sequences, sensor_dropout, tof_region_pool,
)


def fixture(length=4, sequence_id="A", subject="S1", label="G1"):
    values = {column: np.ones(length) for column in ALL_SENSOR_COLUMNS}
    values.update({"acc_x": np.zeros(length), "acc_y": np.zeros(length),
                   "acc_z": np.full(length, 9.80665), "rot_w": np.ones(length),
                   "rot_x": np.zeros(length), "rot_y": np.zeros(length), "rot_z": np.zeros(length)})
    values.update({"sequence_id": [sequence_id] * length, "subject": [subject] * length,
                   "sequence_counter": np.arange(length)})
    if label is not None:
        values["gesture"] = [label] * length
    return pd.DataFrame(values)


def demo():
    return pd.DataFrame({"subject": ["S1", "S2", "S3", "S4"], "handedness": [1, 0, 1, 1]})


class PreprocessingTests(unittest.TestCase):
    def test_stationary_and_known_tilt_remove_gravity(self):
        config = PreprocessingConfig()
        sequence = fixture()
        raw = engineer_sequence(sequence, 1, config)
        np.testing.assert_allclose(raw["imu"][:, 12:], 0, atol=1e-10)
        np.testing.assert_allclose(raw["imu"][:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (4, 1)))
        # +90 degrees about x: device gravity is [0,g,0].
        sequence[["rot_w", "rot_x", "rot_y", "rot_z"]] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
        sequence[["acc_x", "acc_y", "acc_z"]] = [0, config.gravity, 0]
        tilted = engineer_sequence(sequence, 1, config)
        np.testing.assert_allclose(tilted["imu"][:, 12:], 0, atol=1e-10)
        inverse = sequence.copy()
        inverse["rot_x"] *= -1
        inverted = engineer_sequence(inverse, 1, PreprocessingConfig(quaternion_direction="world_to_device"))
        np.testing.assert_allclose(inverted["imu"], tilted["imu"], atol=1e-10)

    def test_known_angular_velocity_and_antipodal_quaternions(self):
        sequence = fixture()
        angles = np.arange(4) * .1
        sequence["rot_w"], sequence["rot_z"] = np.cos(angles / 2), np.sin(angles / 2)
        config = PreprocessingConfig(sample_period_seconds=.02)
        raw = engineer_sequence(sequence, 1, config)
        np.testing.assert_allclose(raw["imu"][1:, 9:12], np.tile([0, 0, 5], (3, 1)), atol=1e-10)
        self.assertFalse(raw["imu_valid"][0, 9:12].any())
        sequence.loc[[1, 3], ["rot_w", "rot_x", "rot_y", "rot_z"]] *= -1
        antipodal = engineer_sequence(sequence, 1, config)
        np.testing.assert_allclose(antipodal["imu"], raw["imu"], atol=1e-10)
        mirrored = engineer_sequence(sequence, 0, config)
        np.testing.assert_allclose(mirrored["imu"][1:, 11], -5, atol=1e-10)

    def test_missing_rotation_does_not_become_identity_or_cross_gap(self):
        sequence = fixture(6)
        sequence.loc[1, "rot_w"] = np.nan
        sequence.loc[3, ["rot_w", "rot_x", "rot_y", "rot_z"]] = 0
        sequence.loc[5, "sequence_counter"] = 8
        raw = engineer_sequence(sequence, 1, PreprocessingConfig())
        self.assertTrue(raw["imu_valid"][:, :3].all())
        self.assertFalse(raw["imu_valid"][[1, 3], 3:].any())
        self.assertFalse(raw["imu_valid"][:, 9:12].any())
        np.testing.assert_array_equal(raw["imu"][[1, 3], 3:], 0)

    def test_reflection_preserves_physics_and_permutates_every_map(self):
        sequence = fixture(2)
        sequence[["acc_x", "acc_y", "acc_z"]] = [2, 3, 4]
        sequence[["rot_w", "rot_x", "rot_y", "rot_z"]] = [.5, .5, .5, .5]
        for sensor in range(1, 6):
            sequence[f"thm_{sensor}"] = sensor * 10
            sequence[[f"tof_{sensor}_v{p}" for p in range(64)]] = np.arange(64) + sensor * 100
        sequence.loc[0, "tof_3_v0"] = np.nan
        sequence.loc[0, "tof_3_v1"] = -1
        sequence.loc[0, "thm_3"] = np.nan
        right = engineer_sequence(sequence, 1, PreprocessingConfig())
        left = engineer_sequence(sequence, 0, PreprocessingConfig())
        np.testing.assert_allclose(left["imu"][:, :3], right["imu"][:, :3] * [-1, 1, 1])
        np.testing.assert_allclose(left["imu"][:, 12:], right["imu"][:, 12:] * [-1, 1, 1])
        np.testing.assert_allclose(left["imu"][:, 3:9], right["imu"][:, 3:9] * [1, -1, -1, -1, 1, 1])
        np.testing.assert_array_equal(left["thm"], right["thm"][:, [0, 1, 4, 3, 2]])
        np.testing.assert_array_equal(left["thm_observed"], right["thm_observed"][:, [0, 1, 4, 3, 2]])
        for key in ("tof", "tof_valid", "tof_nan", "tof_minus1"):
            np.testing.assert_array_equal(left[key], right[key][:, [0, 1, 4, 3, 2], :, ::-1])
        self.assertTrue(left["tof_nan"][0, 4, 0, 7])
        self.assertTrue(left["tof_minus1"][0, 4, 0, 6])

    def test_thm_fill_is_local_and_statistics_exclude_imputed_values(self):
        a, b = fixture(), fixture(sequence_id="B")
        a["thm_1"] = [np.nan, 10, np.nan, 30]
        a["thm_2"] = np.nan
        b["thm_1"] = [100, np.nan, np.nan, np.nan]
        b["thm_2"] = np.nan
        preprocessor = FoldPreprocessor().fit([a, b], demo())
        stats = preprocessor.state["statistics"]["thm"]
        self.assertEqual(stats["count"][:2], [3, 0])
        self.assertAlmostEqual(stats["offset"][0], 140 / 3)
        raw = engineer_sequence(a, 1, PreprocessingConfig())
        np.testing.assert_array_equal(raw["thm"][:, 0], [10, 10, 10, 30])
        np.testing.assert_array_equal(raw["thm_observed"][:, 0], [False, True, False, True])
        self.assertFalse(raw["thm_valid"][:, 1].any())
        transformed = preprocessor.transform(a, demo(), "train")
        np.testing.assert_array_equal(transformed["thm"][:, 1], 0)
        self.assertTrue(transformed["thm_valid"][:, 0].all())

    def test_tof_observed_zero_invalid_values_and_normalization(self):
        sequence = fixture(2)
        sequence[SENSOR_COLUMNS["ToF"]] = np.nan
        sequence.loc[0, ["tof_1_v0", "tof_1_v1", "tof_1_v2"]] = [0, -1, 10]
        sequence.loc[1, ["tof_1_v0", "tof_1_v1"]] = [20, np.inf]
        preprocessor = FoldPreprocessor().fit([sequence], demo())
        stats = preprocessor.state["statistics"]["tof"]
        self.assertEqual(stats["count"], [3, 0, 0, 0, 0])
        self.assertEqual(stats["offset"][0], 10)
        self.assertAlmostEqual(stats["scale"][0], np.sqrt(200 / 3))
        sample = preprocessor.transform(sequence, demo(), "train")
        self.assertTrue(sample["tof_valid"][0, 0, 0, 0])
        self.assertLess(sample["tof"][0, 0, 0, 0], 0)
        self.assertTrue(sample["tof_minus1"][0, 0, 0, 1])
        self.assertTrue(sample["tof_nan"][0, 0, 0, 3])
        np.testing.assert_array_equal(sample["tof"][~sample["tof_valid"]], 0)
        np.testing.assert_array_equal(sample["tof_input"][:, 1].transpose(1, 0, 2, 3), sample["tof_valid"])
        minmax = FoldPreprocessor(PreprocessingConfig(tof_normalization="minmax")).fit([sequence], demo())
        mini = minmax.transform(sequence, demo(), "train")
        self.assertEqual(mini["tof"][0, 0, 0, 0], 0)
        self.assertEqual(mini["tof"][1, 0, 0, 0], 1)

    def test_no_response_is_distinct_from_absent_tof_hardware(self):
        sequence = fixture()
        sequence[SENSOR_COLUMNS["ToF"]] = -1
        sequence[SENSOR_COLUMNS["THM"]] = np.nan
        preprocessor = FoldPreprocessor().fit([sequence], demo())
        sample = preprocessor.transform(sequence, demo(), "train")
        self.assertFalse(sample["tof_valid"].any())
        self.assertTrue(sample["sequence_available"][2])
        self.assertFalse(sample["sequence_available"][1])
        absent = sequence.copy()
        absent[SENSOR_COLUMNS["ToF"]] = np.nan
        missing = preprocessor.transform(absent, demo())
        self.assertFalse(missing["sequence_available"][2])
        np.testing.assert_array_equal(missing["tof_input"], 0)

    def test_fold_fit_is_unchanged_by_extreme_validation_values_and_length(self):
        train = fixture()
        train["acc_x"] = [0, 2, 4, 6]
        preprocessor = FoldPreprocessor().fit([train], demo())
        saved = json.dumps(preprocessor.state, sort_keys=True)
        validation = fixture(50, "VAL", "S3")
        validation["acc_x"] = 1000
        sample = preprocessor.transform(validation, demo(), "validation")
        self.assertEqual(preprocessor.max_length, 4)
        self.assertEqual(json.dumps(preprocessor.state, sort_keys=True), saved)
        np.testing.assert_array_equal(sample["sequence_counter"], [46, 47, 48, 49])
        np.testing.assert_allclose(sample["imu"][:, 0], (1000 - 3) / np.sqrt(5), rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "Subject leakage"):
            preprocessor.transform(train, demo(), "validation")

    def test_padding_masks_do_not_confuse_measured_zero_with_missing(self):
        preprocessor = FoldPreprocessor(PreprocessingConfig(length_quantile=1)).fit([fixture(6)], demo())
        sample = preprocessor.transform(fixture(2, "TEST", "S3", None), demo())
        np.testing.assert_array_equal(sample["time_mask"], [False] * 4 + [True] * 2)
        self.assertTrue(sample["imu_valid"][-2:, :3].all())
        self.assertFalse(sample["imu_valid"][:4].any())
        self.assertFalse(sample["modality_available"][:4].any())
        np.testing.assert_array_equal(sample["imu"][:4], 0)
        np.testing.assert_array_equal(sample["tof_input"][:, :, :4], 0)
        self.assertEqual(sample["label"], -1)

    def test_saved_parameters_and_dropout_are_consistent(self):
        sequence = fixture()
        preprocessor = FoldPreprocessor().fit([sequence], demo())
        expected = preprocessor.transform(sequence, demo(), "train")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.json"
            preprocessor.save(path)
            actual = FoldPreprocessor.load(path).transform(sequence, demo(), "train")
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        forced = SensorDropoutConfig(1, 1, 1, 0)
        dropped = sensor_dropout(actual, np.random.default_rng(42), training=True, config=forced)
        np.testing.assert_array_equal(dropped["imu"][:, :3], actual["imu"][:, :3])
        self.assertFalse(dropped["imu_valid"][:, 3:].any())
        self.assertFalse(dropped["modality_available"].any())
        self.assertFalse(dropped["tof_input"].any())
        self.assertFalse(dropped["thm"].any())
        np.testing.assert_array_equal(dropped["source_sequence_available"], actual["source_sequence_available"])
        unchanged = sensor_dropout(actual, np.random.default_rng(42), training=False, config=forced)
        for key in actual:
            np.testing.assert_array_equal(unchanged[key], actual[key])
        self.assertTrue(actual["imu_valid"][:, 3:9].all())  # No mutation.

    def test_region_pool_excludes_invalid_pixels_and_handles_empty_regions(self):
        distance = np.full((1, 5, 8, 8), np.nan)
        valid = np.zeros_like(distance, dtype=bool)
        distance[0, 0, 0, :2] = [2, 6]
        valid[0, 0, 0, :2] = True
        pooled, available = tof_region_pool(distance, valid, regions=2)
        self.assertEqual(pooled[0, 0, 0, 0], 4)
        self.assertEqual(int(available.sum()), 1)
        self.assertTrue(np.isfinite(pooled).all())
        whole, _ = tof_region_pool(distance, valid, regions=1)
        self.assertEqual(whole[0, 0, 0, 0], 4)

    def test_invalid_metadata_is_rejected_and_rows_are_ordered(self):
        sequence = fixture()
        ordered = engineer_sequence(sequence.iloc[[3, 1, 0, 2]], 1, PreprocessingConfig())
        np.testing.assert_array_equal(ordered["sequence_counter"], [0, 1, 2, 3])
        bad = sequence.copy()
        bad.loc[1, "sequence_counter"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            engineer_sequence(bad, 1, PreprocessingConfig())
        bad = sequence.copy()
        bad.loc[1, "subject"] = "S2"
        with self.assertRaises(ValueError):
            FoldPreprocessor().fit([bad], demo())
        demographics = demo()
        demographics.loc[0, "handedness"] = np.nan
        with self.assertRaises(ValueError):
            FoldPreprocessor().fit([sequence], demographics)


class CSVAndFoldTests(unittest.TestCase):
    def test_chunk_boundaries_and_noncontiguous_ids(self):
        sequences = [fixture(3), fixture(2, "B", "S2")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.concat(sequences).to_csv(path, index=False)
            for chunksize in (1, 2, 4, 10):
                read = list(iter_csv_sequences(path, chunksize=chunksize))
                self.assertEqual([len(x) for x in read], [3, 2])
            selected = list(iter_csv_sequences(path, chunksize=1, sequence_ids={"B"}))
            self.assertEqual(len(selected), 1)
            with self.assertRaisesRegex(ValueError, "absent"):
                list(iter_csv_sequences(path, sequence_ids={"missing"}))
            pd.concat([sequences[0].iloc[:1], sequences[1], sequences[0].iloc[1:]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Non-contiguous"):
                list(iter_csv_sequences(path, chunksize=2))

    def test_end_to_end_split_fit_export_with_unlabeled_test(self):
        sequences = [fixture(3 + i, f"Q{i}_{label}", f"S{i+1}", label)
                     for i in range(4) for label in ("G1", "G2")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            pd.concat(sequences).to_csv(data / "train.csv", index=False)
            demo().to_csv(data / "train_demographics.csv", index=False)
            fixture(12, "TEST", "S3", None).to_csv(data / "test.csv", index=False)
            demo().to_csv(data / "test_demographics.csv", index=False)
            folds_path = root / "fixed_folds.csv"
            save_fixed_folds(scan_sequence_index(data / "train.csv"), folds_path, n_splits=2)
            with redirect_stdout(io.StringIO()):
                summaries = prepare_folds(data, root / "output", folds_path=folds_path, chunksize=3, folds=[0, 1])
            index = pd.read_csv(root / "output" / "folds.csv")
            self.assertEqual(index.groupby("subject")["fold"].nunique().max(), 1)
            for summary in summaries:
                self.assertFalse(set(summary["train_subjects"]) & set(summary["validation_subjects"]))
                fold_dir = root / "output" / f"fold_{summary['fold']}"
                preprocessor = FoldPreprocessor.load(fold_dir / "preprocessor.json")
                train_lengths = index[index["fold"] != summary["fold"]]["length"]
                self.assertEqual(preprocessor.max_length, int(np.ceil(np.quantile(train_lengths, .95))))
                self.assertEqual(summary["exports"]["train"] + summary["exports"]["validation"], 8)
                test_manifest = pd.read_csv(fold_dir / "test_manifest.csv")
                self.assertEqual(test_manifest["label"].iloc[0], -1)
                with np.load(fold_dir / test_manifest["path"].iloc[0], allow_pickle=False) as sample:
                    self.assertEqual(sample["tof"].shape, (preprocessor.max_length, 5, 8, 8))
                    self.assertTrue(np.isfinite(sample["imu"]).all())
            with self.assertRaisesRegex(ValueError, "non-empty"):
                with redirect_stdout(io.StringIO()):
                    prepare_folds(data, root / "output", folds_path=folds_path, fit_only=True)


if __name__ == "__main__":
    unittest.main()
