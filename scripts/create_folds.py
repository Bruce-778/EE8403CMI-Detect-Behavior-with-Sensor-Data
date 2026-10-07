"""Create the project's canonical validation folds, or audit the saved split."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.validation import main

if __name__ == "__main__":
    raise SystemExit(main())
