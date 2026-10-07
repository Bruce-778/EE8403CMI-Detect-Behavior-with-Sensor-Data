"""Evaluate held-out predictions exported by any model/ablation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
