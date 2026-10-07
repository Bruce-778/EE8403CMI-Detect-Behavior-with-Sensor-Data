"""Small fixtures with independently known counts; no full-dataset tests."""

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
from cmi_project.dataset_analysis import ALL_SENSOR_COLUMNS, SENSOR_COLUMNS, analyze


class DatasetAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        # A appears in three separated rows; B and C each appear twice.
        self.train = pd.DataFrame({
            "sequence_id": ["A", "B", "C", "A", "B", "C", "A"],
            "subject": ["S1", "S2", "S2", "S1", "S2", "S2", "S1"],
            "gesture": ["G1", "G2", "G1", "G1", "G2", "G1", "G1"],
            "sequence_type": ["Target"] * 7,
            **{column: np.ones(7) for column in ALL_SENSOR_COLUMNS},
        })
        self.train.loc[0, SENSOR_COLUMNS["ToF"]] = -1
        self.train.loc[[1, 4], SENSOR_COLUMNS["ToF"]] = np.nan
        self.train.loc[2, "tof_1_v0"] = np.nan
        self.train.loc[2, "tof_1_v1"] = -1
        self.train.loc[3, "acc_x"] = -1  # An IMU negative value is valid.
        self.train.loc[4, "rot_w"] = np.nan
        self.train.loc[5, SENSOR_COLUMNS["THM"]] = np.nan
        self.demographics = pd.DataFrame({"subject": ["S1", "S2"], "age": [24, 35]})

    def tearDown(self):
        self.temporary.cleanup()

    def run_analysis(self, chunksize=2, output="output"):
        self.train.to_csv(self.data / "train.csv", index=False)
        self.demographics.to_csv(self.data / "train_demographics.csv", index=False)
        with redirect_stdout(io.StringIO()):
            summary = analyze(self.data, self.root / output, chunksize, expected_gestures=2, plots=False)
        return summary

    def table(self, name, output="output"):
        return pd.read_csv(self.root / output / "tables" / f"{name}.csv")

    def test_known_sequence_and_missingness_counts(self):
        summary = self.run_analysis()
        self.assertEqual((summary["train_rows"], summary["sequence_count"], summary["subject_count"],
                          summary["gesture_count"]), (7, 3, 2, 2))
        self.assertTrue(summary["all_integrity_checks_passed"])
        seq = self.table("sequence_summary").set_index("sequence_id")
        self.assertEqual(seq["length"].to_dict(), {"A": 3, "B": 2, "C": 2})
        gestures = self.table("gesture_distribution").set_index("gesture")
        self.assertEqual(gestures["sequence_count"].to_dict(), {"G1": 2, "G2": 1})
        self.assertEqual(gestures["row_count"].to_dict(), {"G1": 5, "G2": 2})
        sensors = self.table("sensor_missingness").set_index("modality")
        self.assertEqual(int(sensors.loc["ToF", "nan_cells"]), 641)
        self.assertEqual(int(sensors.loc["ToF", "minus1_cells"]), 321)
        self.assertAlmostEqual(sensors.loc["ToF", "unavailable_ratio"], 962 / 2240)
        self.assertAlmostEqual(sensors.loc["ToF", "minus1_ratio_of_non_nan"], 321 / 1599)
        self.assertEqual(int(sensors.loc["ToF", "rows_all_unavailable"]), 3)
        self.assertEqual(int(sensors.loc["ToF", "sequences_fully_unavailable"]), 1)
        self.assertEqual(int(sensors.loc["IMU", "nan_cells"]), 1)
        self.assertEqual(int(sensors.loc["IMU", "minus1_cells"]), 0)
        self.assertEqual(int(sensors.loc["THM", "nan_cells"]), 5)
        self.assertTrue(seq.loc["B", "ToF_fully_unavailable"])
        self.assertFalse(seq.loc["A", "ToF_fully_unavailable"])
        parsed = json.loads((self.root / "output" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(parsed["sequence_count"], 3)

    def test_chunk_boundary_invariance(self):
        self.run_analysis(chunksize=2, output="small")
        self.run_analysis(chunksize=5, output="large")
        for path in (self.root / "small" / "tables").glob("*.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(self.root / "large" / "tables" / path.name))

    def test_conflicting_and_missing_labels_are_not_assigned(self):
        self.train.loc[3, "gesture"] = "G2"
        summary = self.run_analysis()
        self.assertFalse(summary["all_integrity_checks_passed"])
        self.assertEqual(summary["invalid_label_sequences"], 1)
        seq = self.table("sequence_summary").set_index("sequence_id")
        self.assertTrue(pd.isna(seq.loc["A", "gesture"]))
        self.assertEqual(json.loads(seq.loc["A", "observed_gestures"]), ["G1", "G2"])
        self.train.loc[3, "gesture"] = "G1"
        self.train.loc[6, "gesture"] = None
        summary = self.run_analysis(output="missing_label")
        self.assertEqual(summary["invalid_label_sequences"], 1)
        self.assertTrue(pd.isna(self.table("sequence_summary", "missing_label").set_index("sequence_id").loc["A", "gesture"]))

    def test_missing_and_duplicate_demographics_keys(self):
        self.demographics = pd.DataFrame({"subject": ["S1", "S1"], "age": [24, 25]})
        summary = self.run_analysis()
        self.assertFalse(summary["all_integrity_checks_passed"])
        self.assertEqual(summary["missing_demographics_subjects"], 1)
        failed = set(self.table("integrity_issues")["check"])
        self.assertIn("demographics_subject_unique", failed)
        self.assertIn("all_train_subjects_have_demographics", failed)

    def test_missing_sequence_or_subject_is_reported(self):
        self.train.loc[6, "sequence_id"] = None
        self.train.loc[1, "subject"] = None
        summary = self.run_analysis()
        self.assertFalse(summary["all_integrity_checks_passed"])
        failed = set(self.table("integrity_issues")["check"])
        self.assertIn("sequence_id_not_null", failed)
        self.assertIn("sequence_rows_reconcile", failed)
        self.assertIn("one_non_null_subject_per_sequence", failed)
        self.assertEqual(summary["train_rows"], 7)

    def test_schema_and_output_location_validation(self):
        self.run_analysis()
        with self.assertRaises(ValueError):
            analyze(self.data, self.data / "outputs", plots=False)
        self.train.drop(columns="rot_w").to_csv(self.data / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "rot_w"):
            analyze(self.data, self.root / "bad_schema", plots=False)


if __name__ == "__main__":
    unittest.main()
