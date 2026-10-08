"""Export a private, offline Kaggle GPU reproduction notebook, without launching it."""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', nargs='+', type=int, default=[0])
    parser.add_argument('--architectures', nargs='+', choices=['base', 'simple', 'deep'], default=['base'])
    parser.add_argument('--run-name', default='pilot_v1')
    parser.add_argument('--notebook-name', default='cmi-second-place-training.ipynb')
    args = parser.parse_args()
    if set(args.folds) - set(range(5)) or len(set(args.folds)) != len(args.folds):
        raise ValueError('Use unique fixed folds 0..4.')
    if (Path(args.run_name).name != args.run_name or any(c in args.run_name for c in '\\/:')
            or Path(args.notebook_name).name != args.notebook_name or not args.notebook_name.endswith('.ipynb')):
        raise ValueError('Use plain run and notebook file names.')
    stream = io.BytesIO()
    files = [*sorted((ROOT / 'src/cmi_project').glob('*.py')),
             ROOT / 'scripts/train_second_place.py', ROOT / 'scripts/run_winner_experiments.py',
             *[ROOT / 'configs' / name for name in ['folds.csv', 'folds.meta.json', 'second_place_source.json', 'data_source_hashes.json']]]
    source = json.loads((ROOT / 'configs/second_place_source.json').read_text())
    files += [ROOT / 'outputs/reference_code/second_place' / name for name in source['files']]
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    encoded = base64.b64encode(stream.getvalue()).decode()
    setup = f'''import base64, io, zipfile, sys, subprocess, hashlib, json
from pathlib import Path
project = Path('/kaggle/working/cmi_second_place')
project.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(base64.b64decode('{encoded}'))) as bundle:
    bundle.extractall(project)
sys.path.insert(0, str(project / 'src'))
from cmi_project.second_place import load_reference
reference = load_reference(project / 'outputs/reference_code/second_place', project / 'configs/second_place_source.json')
server = next(Path('/kaggle/input').rglob('cmi_inference_server.py'))
data_dir = server.parent.parent
import torch
assert torch.cuda.is_available(), 'GPU required for the 50-epoch reproduction'
print('PINNED UPSTREAM', reference.provenance['commit'], flush=True)
print('CUDA', torch.cuda.get_device_name(), 'torch', torch.__version__, flush=True)
print('FIXED FOLDS', hashlib.sha256((project / 'configs/folds.csv').read_bytes()).hexdigest(), flush=True)
print('TRAINING DATA DIRECTORY', data_dir, flush=True)
'''
    training = f'''command = [sys.executable, '-s', '-u', str(project / 'scripts/train_second_place.py'),
    '--data-dir', str(data_dir), '--output', str(project / 'outputs/second_place' / {args.run_name!r}),
    '--device', 'cuda', '--folds', *{[str(f) for f in args.folds]!r},
    '--architectures', *{args.architectures!r}, '--epochs', '50', '--batch-size', '32']
subprocess.run(command, cwd=project, check=True)
print('Completed requested folds; no competition submission.', flush=True)
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Second-place reference method on fixed subject folds\n',
            f'Pinned official repository {source["commit"]}. Four missing-sensor variants, original architectures and 50 epochs.\n',
            f'Folds {args.folds}; architectures {args.architectures}. A partial fold run is screening, not full CV or leaderboard reproduction.\n',
            'Train-only joint ontology; original fixed 200-frame tail/right padding/no learned scaler.\n',
            'Cached arrays are copied before dropout; subject correction is applied once. No pseudo-label updates or online submission.\n']},
            *[{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
               'source': text.splitlines(keepends=True)} for text in [setup, training]]]}
    out = ROOT / 'outputs/second_place/notebooks'
    out.mkdir(parents=True, exist_ok=True)
    path = out / args.notebook_name
    path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    print(path, path.stat().st_size)


if __name__ == '__main__':
    main()
