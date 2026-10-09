"""Shared artifact boundaries remain safe after archiving the old model runner."""
import sys
from pathlib import Path
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cmi_project.artifact_io import extract_archive


class ArtifactBoundaryTests(unittest.TestCase):
    def test_escape_rejected_before_any_extraction(self):
        for filename in ('../escape', '/absolute', 'C:/escape', '..\\escape'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / 'artifact.zip'
                with zipfile.ZipFile(archive, 'w') as bundle:
                    bundle.writestr('good/file', 'first')
                    # Windows ZipInfo normalizes backslashes when writing. Patch
                    # both archive headers to test an actual malformed input ZIP.
                    written_name = filename.replace(chr(92), '/')
                    bundle.writestr(written_name, 'bad')
                if chr(92) in filename:
                    archive.write_bytes(archive.read_bytes().replace(
                        written_name.encode(), filename.encode()))
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    extract_archive(archive, root / 'imported')
                self.assertFalse((root / 'imported').exists())

    def test_valid_artifact_import_preserves_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / 'artifact.zip', 'w') as bundle:
                bundle.writestr('fold/provenance.json', b'{"seed":42}')
            extract_archive(root / 'artifact.zip', root / 'imported')
            path = root / 'imported/fold/provenance.json'
            self.assertEqual(path.read_bytes(), b'{"seed":42}')
            with self.assertRaisesRegex(ValueError, 'fresh'):
                extract_archive(root / 'artifact.zip', root / 'imported')
            self.assertEqual(path.read_bytes(), b'{"seed":42}')
