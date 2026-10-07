import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("winner_pipeline", Path(__file__).resolve().parents[1] / "scripts/run_winner_experiments.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class WinnerContinuationTests(unittest.TestCase):
    def test_training_data_hash_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            data = root / "data"
            data.mkdir()
            content = b"same training data\n"
            (data / "train.csv").write_bytes(content)
            expected = {"files": {"train.csv": {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}
            (root / "configs/data_source_hashes.json").write_text(json.dumps(expected))
            with patch.object(pipeline, "ROOT", root):
                pipeline.verify_training_data(data)
                self.assertEqual(json.loads((root / "data_source_check.json").read_text())["files"], expected["files"])
                (data / "train.csv").write_bytes(b"different bytes\n")
                with self.assertRaisesRegex(ValueError, "differs"):
                    pipeline.verify_training_data(data)

    def test_frozen_selection_preserves_pilots_without_active_cache_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = Path(directory) / "new", Path(directory) / "old"
            for workspace in (root, source):
                (workspace / "configs").mkdir(parents=True)
                (workspace / "configs/folds.csv").write_text("immutable folds")
            rows = []
            for i, name in enumerate(pipeline.DESIGNS):
                config = {"output_dir": f"outputs/experiments/{name}", "seed": 42}
                for workspace in (root, source):
                    (workspace / "configs" / f"{name}.json").write_text(json.dumps(config))
                row = {"name": name, "fold": 0, "score": 0.7 + i * 0.01, "best_epoch": 1, "epochs_run": 2}
                rows.append(row)
                fold = source / config["output_dir"] / "imu/fold_0"
                fold.mkdir(parents=True)
                (fold / "metrics.json").write_text(json.dumps({"fold": 0, "model": "imu", "validation": {"score": row["score"]}, "best_epoch": 1, "epochs_run": 2}))
                (fold / "best.pt").write_bytes(b"saved pilot evidence")
                summary = source / "experiments/results" / f"{name}_gpu_pilot.json"
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text(json.dumps(row))
            selection = {"designs": rows, "selected": rows[-1]["name"], "selection_fold": 0}
            (source / "selection.json").write_text(json.dumps(selection))
            (source / "outputs/cnn_cache").mkdir()
            with patch.object(pipeline, "ROOT", root):
                bad = dict(selection, selected=rows[0]["name"])
                (source / "selection.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "selection"):
                    pipeline.recover_frozen_pilots(source)
                self.assertFalse((root / "outputs").exists())
                (source / "selection.json").write_text(json.dumps(selection))
                self.assertEqual(pipeline.recover_frozen_pilots(source), rows)
                self.assertFalse((root / "outputs/cnn_cache").exists())
                self.assertFalse((root / "outputs/experiments").exists())
                self.assertEqual((root / "outputs/pilot_artifacts" / rows[-1]["name"] / "imu/fold_0/best.pt").read_bytes(), b"saved pilot evidence")
                self.assertTrue((source / "outputs/cnn_cache").exists())


if __name__ == "__main__":
    unittest.main()
