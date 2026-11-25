import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location("winner_pipeline", Path(__file__).resolve().parents[1] / "scripts/run_winner_experiments.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class WinnerPipelineTests(unittest.TestCase):
    def test_exported_training_cell_runs_without_old_pilot_state(self):
        exporter_spec = importlib.util.spec_from_file_location(
            "winner_exporter", pipeline.ROOT / "scripts/export_kaggle_training.py")
        exporter = importlib.util.module_from_spec(exporter_spec)
        exporter_spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            for name in ("folds.csv", "folds.meta.json", "data_source_hashes.json",
                         "preprocessing_dynamics.json", "cnn_dynamics_mixup.json"):
                (root / "configs" / name).write_bytes((pipeline.ROOT / "configs" / name).read_bytes())
            with patch.object(exporter, "ROOT", root), patch("sys.argv", ["export_kaggle_training.py"]):
                exporter.main()
            notebook = json.loads((root / "outputs/kaggle_training/cmi-winner-training.ipynb").read_text())
            code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
            for source in code:
                compile(source, "exported training notebook", "exec")
            with patch("subprocess.run") as run:
                exec(code[-1], {"project": root, "data_dir": root / "data",
                               "sys": SimpleNamespace(executable="python"),
                               "subprocess": SimpleNamespace(run=run)})
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[2], str(root / "scripts/run_winner_experiments.py"))
            self.assertNotIn("--continue-from", command)

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


if __name__ == "__main__":
    unittest.main()
