"""Primary experiment: second-place base with IMU metric-group auxiliary losses."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/main_experiment.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_selection():
    selected = json.loads(CONFIG.read_text(encoding='utf-8'))
    record_path = ROOT / selected['result_record']
    if sha(record_path) != selected['result_sha256']:
        raise ValueError('Selected audited result bytes changed.')
    result = json.loads(record_path.read_text(encoding='utf-8'))
    if (result['status'] != 'verified' or result['folds'] != list(range(5))
            or result['source_commit'] != selected['source_commit']
            or len(result['arms']) != 20):
        raise ValueError('Primary experiment requires the complete audited hybrid.')
    return selected, result


def verify_selection(selected, result):
    imported = ROOT / selected['imported_directory']
    for item in result['hybrid_merge_provenance']['copied_files']:
        if sha(imported / item['path']) != item['sha256']:
            raise ValueError(f"Selected artifact bytes changed: {item['path']}")
    for item in result['completed_oof_coverage']:
        path = imported / 'comparison' / item['scenario'] / item['method'] / 'evaluation' / 'oof_predictions.csv'
        if item['sequences'] != 8151 or sha(path) != item['sha256']:
            raise ValueError(f'Selected OOF bytes changed: {path}')
    print('Verified selected arm artifacts and all 15 complete OOF files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action')
    status = actions.add_parser('status', help='Show the selected audited results; never starts training.')
    status.add_argument('--verify', action='store_true', help='Verify all selected copied artifact and OOF bytes.')
    train = actions.add_parser('train', help='Explicitly retrain all ten IMU arms; original ToF reuse is separate.')
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    train.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    export = actions.add_parser('export', help='Export a fresh offline training notebook; does not launch it.')
    export.add_argument('--run-name', required=True)
    export.add_argument('--notebook-name', required=True)
    args = parser.parse_args()
    selected, result = load_selection()
    if args.action in (None, 'status'):
        if getattr(args, 'verify', False):
            verify_selection(selected, result)
        print(selected['name'], '(fixed development CV; online score not measured)')
        for scenario in ('observed', 'aux_dropout50', 'imu_only'):
            values = result['scenarios'][scenario]['new']
            print(f"{scenario}: {values['fold_mean']['score']:.6f} +/- {values['fold_std']['score']:.6f}")
        return
    if args.action == 'train':
        command = ['train_second_place_hybrid.py', '--output', str(args.output),
                   '--data-dir', str(args.data_dir), '--device', args.device]
    else:
        destination = ROOT / 'outputs/second_place/notebooks' / args.notebook_name
        if destination.exists():
            raise ValueError('Use a fresh notebook name; preserve previous exports.')
        command = ['export_second_place_hybrid.py', '--run-name', args.run_name,
                   '--notebook-name', args.notebook_name]
    subprocess.run([sys.executable, '-s', '-u', str(ROOT / 'scripts' / command[0]),
                    *command[1:]], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
