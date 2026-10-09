"""Compatibility import for shared integrity checks; old model entrypoint archived."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmi_project.artifact_io import extract_archive

if __name__ == '__main__':
    raise SystemExit('Old CNN entrypoint archived. Use scripts/main_experiment.py; see experiments/legacy/cnn_dynamics_mixup/README.md for the historical snapshot.')
