"""Check exported inference against held-out OOF and public test examples."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmi_project.evaluation import ALL_GESTURES, PROBABILITY_COLUMNS
from cmi_project.inference import RoutedCNNPredictor
from cmi_project.preprocessing import iter_csv_sequences
from cmi_project.validation import load_fold_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("outputs/kaggle_submission"))
    args = parser.parse_args()
    directory = ROOT / args.directory
    predictor = RoutedCNNPredictor(directory / "bundle")
    manifest = load_fold_manifest(ROOT / "configs/folds.csv")
    if predictor.folds_sha256 != manifest.fingerprint:
        raise ValueError("Submission differs from the fixed local folds.")
    expected = pd.read_csv(ROOT / "outputs/experiments/cnn_hierarchical/scenarios/cnn_final_selected/routed/observed/oof_predictions.csv",
                           dtype={"sequence_id": str}).set_index("sequence_id")
    demographics = pd.read_csv(ROOT / "data/train_demographics.csv", dtype={"subject": str})
    checked = set()
    records = []
    for sequence in iter_csv_sequences(ROOT / "data/train.csv", chunksize=25000, require_gesture=True):
        sid = str(sequence.sequence_id.iloc[0])
        fold = int(expected.loc[sid, "fold"])
        if fold in checked:
            continue
        probability = predictor.predict_proba(sequence, demographics, fold=fold)
        reference = expected.loc[sid, PROBABILITY_COLUMNS].to_numpy(dtype=float)
        np.testing.assert_allclose(probability, reference, rtol=2e-5, atol=2e-6)
        assert ALL_GESTURES[int(probability.argmax())] == expected.loc[sid, "predicted_gesture"]
        poisoned = sequence.copy()
        poisoned["gesture"] = "not-a-real-gesture"
        np.testing.assert_array_equal(probability, predictor.predict_proba(poisoned, demographics, fold=fold))
        records.append({"fold": fold, "sequence_id": sid, "max_probability_error": float(abs(probability-reference).max())})
        checked.add(fold)
        print(f"PASS held-out fold {fold}: {sid}; label ignored", flush=True)
        if len(checked) == 5:
            break
    assert checked == set(range(5))
    test = pd.read_csv(ROOT / "data/test.csv", dtype={"subject": str, "sequence_id": str})
    demo = pd.read_csv(ROOT / "data/test_demographics.csv", dtype={"subject": str})
    public_predictions = []
    for sid, sequence in test.groupby("sequence_id", sort=False):
        prediction = predictor.predict(sequence, demo)
        assert prediction in ALL_GESTURES and isinstance(prediction, str)
        missing = sequence.copy()
        auxiliary = [column for column in missing if column.startswith(("thm_", "tof_"))]
        missing[auxiliary] = np.nan
        imu_prediction = predictor.predict(missing, demo)
        assert imu_prediction in ALL_GESTURES
        public_predictions.append({"sequence_id": sid, "prediction": prediction, "forced_imu_prediction": imu_prediction})
        print(f"PASS public test {sid}: {prediction}; all-aux-missing => {imu_prediction}", flush=True)
    summary = {"status": "passed", "folds_sha256": manifest.fingerprint,
        "heldout_checks": records, "public_test_examples": public_predictions,
        "note": "Public examples have no labels; this is inference verification, not a test score."}
    (directory / "local_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
