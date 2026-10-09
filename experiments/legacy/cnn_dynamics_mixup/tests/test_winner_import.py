import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("winner_import",
    Path(__file__).resolve().parents[1] / "scripts/import_winner_experiments.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WinnerImportTests(unittest.TestCase):
    def test_rejects_archive_escape_and_preserves_completed_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, destination = root / "artifacts.zip", root / "imported"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.json", "{}")
            with self.assertRaisesRegex(ValueError, "escapes"):
                module.extract_archive(archive, destination)
            self.assertFalse(destination.exists())
            self.assertFalse((root / "outside.json").exists())
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("selection.json", "{}")
            module.extract_archive(archive, destination)
            preserved = (destination / "selection.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "fresh directory"):
                module.extract_archive(archive, destination)
            self.assertEqual((destination / "selection.json").read_bytes(), preserved)


if __name__ == "__main__":
    unittest.main()
