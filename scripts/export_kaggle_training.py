"""Create an offline GPU training notebook containing our source and immutable folds."""

import base64
import argparse
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-pilots", action="store_true", help="Attach the previous training notebook's output as an input.")
    args = parser.parse_args()
    stream = io.BytesIO()
    files = [*sorted((ROOT / "src/cmi_project").glob("*.py")),
             *sorted((ROOT / "scripts").glob("*.py")),
             ROOT / "configs/folds.csv", ROOT / "configs/folds.meta.json"]
    files += [ROOT / "configs" / name for name in ("preprocessing.json", "preprocessing_dynamics.json",
        "cnn_grouped_se.json", "cnn_grouped_mixup.json", "cnn_dynamics_mixup.json")]
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    encoded = base64.b64encode(stream.getvalue()).decode()
    setup = f'''import base64, io, zipfile, sys, subprocess, hashlib, json
from pathlib import Path
project = Path("/kaggle/working/cmi_worktree")
project.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode("{encoded}"))) as bundle:
    bundle.extractall(project)
server = next(Path("/kaggle/input").rglob("cmi_inference_server.py"))
data_dir = server.parent.parent
import torch
print("CUDA available:", torch.cuda.is_available(), "torch", torch.__version__, flush=True)
print("Training sequences from:", data_dir, flush=True)
print("Fixed folds file sha256:", hashlib.sha256((project / "configs/folds.csv").read_bytes()).hexdigest(), flush=True)
'''
    if args.continue_pilots:
        setup += '''pilot_sources = [p.parent for p in Path("/kaggle/input").rglob("selection.json")
    if (p.parent / "outputs/experiments/cnn_dynamics_mixup/imu/fold_0/best.pt").is_file()]
if len(pilot_sources) != 1:
    raise RuntimeError("Attach the previous CMI Winner Inspired CNN Training notebook output.")
pilot_source = pilot_sources[0]
print("Frozen pilot source:", pilot_source, flush=True)
'''
    training = '''command = [sys.executable, "-u", str(project / "scripts/run_winner_experiments.py"),
    "--data-dir", str(data_dir), "--device", "cuda"]
'''
    if args.continue_pilots:
        training += 'command += ["--continue-from", str(pilot_source)]\n'
    training += 'subprocess.run(command, cwd=project, check=True)\n'
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Winner-inspired CNN experiments\n",
                "Three fixed fold-0 pilots; one configuration reused for all five subject folds and both sensor models.\n",
                "Only training-fold statistics; no hidden test fitting or leaderboard tuning. Source written for this project.\n"]},
            *[{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
               "source": source.splitlines(keepends=True)} for source in (setup, training)]
        ]}
    folder = ROOT / "outputs/kaggle_training"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("cmi-winner-training-continue.ipynb" if args.continue_pilots else "cmi-winner-training.ipynb")
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(path, "bytes", path.stat().st_size)


if __name__ == "__main__":
    main()
