import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_history_postprocessing import CausalHistoryAssignment, decode_fold, training_capacities


class HistoryPostprocessingTests(unittest.TestCase):
    def test_capacity_fit_ignores_validation_labels(self):
        table = pd.DataFrame({"sequence_id": list("abcde"), "subject": ["train"] * 3 + ["val"] * 2,
                              "gesture": ["a", "a", "b", "a", "b"], "fold": [1, 1, 1, 0, 0]})
        np.testing.assert_array_equal(training_capacities(table, 0, ["a", "b"]), [2, 1])
        table.loc[table.fold == 0, "gesture"] = "poisoned_unknown_label"
        np.testing.assert_array_equal(training_capacities(table, 0, ["a", "b"]), [2, 1])
        table.loc[table.fold == 0, "subject"] = "train"
        with self.assertRaises(ValueError):
            training_capacities(table, 0, ["a", "b"])

    def test_causal_assignment_and_overflow_preserve_past_returns(self):
        decoder = CausalHistoryAssignment([1, 1])
        first = decoder.predict_one("s", [.9, .1])
        second = decoder.predict_one("s", [.8, .2])
        self.assertEqual((first, second), (0, 1))
        self.assertEqual(decoder.predict_one("s", [.95, .05]), 0)
        self.assertEqual(decoder.overflow, 1)
        self.assertEqual(decoder.predict_one("another", [.95, .05]), 0)

    def test_input_copy_and_prefix_independence(self):
        p = np.array([.9, .1])
        a, b = CausalHistoryAssignment([1, 1]), CausalHistoryAssignment([1, 1])
        self.assertEqual(a.predict_one("s", p), b.predict_one("s", p.copy()))
        p[:] = [.1, .9]
        self.assertEqual(a.predict_one("s", [.8, .2]), b.predict_one("s", [.8, .2]))
        np.testing.assert_allclose(a.history["s"][0], [.9, .1])

    def test_decode_ignores_true_labels_and_annotations(self):
        frame = pd.DataFrame({"sequence_id": ["z", "a", "m"], "subject": ["s"] * 3,
                              "p_a": [.9, .8, .7], "p_b": [.1, .2, .3],
                              "folds_sha256": ["fixed"] * 3,
                              "gesture": ["a", "a", "b"], "orientation": ["up"] * 3,
                              "behavior": ["gesture"] * 3})
        actual, _ = decode_fold(frame, [2, 1], 42, labels=["a", "b"], probability_columns=["p_a", "p_b"])
        frame[["gesture", "orientation", "behavior"]] = "poison"
        other, _ = decode_fold(frame, [2, 1], 42, labels=["a", "b"], probability_columns=["p_a", "p_b"])
        pd.testing.assert_frame_equal(actual, other)

    def test_invalid_inputs(self):
        for counts in ([0, 1], [1.5, 2.0], [], [[1, 2]]):
            with self.assertRaises(ValueError):
                CausalHistoryAssignment(counts)
        for probability in ([0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], [1]):
            with self.assertRaises(ValueError):
                CausalHistoryAssignment([1, 1]).predict_one("s", probability)


if __name__ == "__main__":
    unittest.main()
