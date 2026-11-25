"""Run Step 2 without installing the local package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmi_project.preprocess_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
