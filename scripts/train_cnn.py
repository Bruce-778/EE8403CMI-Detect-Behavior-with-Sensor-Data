"""Run the main v1 1D CNN experiments from the project root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmi_project.cnn_training import main

if __name__ == "__main__":
    raise SystemExit(main())
