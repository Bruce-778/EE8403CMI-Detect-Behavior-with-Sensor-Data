"""Audit ten new IMU arms, reuse verified ToF arms, and compare complete OOF.

This importer never changes either source experiment or interprets partial
training as a five-fold result. It uses the existing strict reference auditor
after constructing a new package with explicit routed logit IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

from evaluate_second_place import ROOT, dump
from import_winner_experiments import extract_archive
from train_second_place import VARIANTS, prediction_frame
from cmi_project.evaluation import cmi_metrics, write_fold_predictions, _checked_predictions
from cmi_project.second_place_evaluation import check_arm, causal_decode, read_aligned_logits
from cmi_project.validation import load_fold_manifest


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_hybrid(provenance, original):
    expected = {'hybrid_experiment': 'base_imu_official_group_loss_v1',
                'architecture': 'base', 'epochs': 50, 'batch_size': 32,
                'optimizer': 'Adam lr=.001 wd=.0001',
                'scheduler': '10% step warmup, cosine', 'mixup_alpha': .5,
                'phase_loss_weight': 1., 'gradient_clip_norm': 1.,
                'auxiliary_loss': {'macro9_weight': .5, 'binary_weight': .1,
                    'targets': 'training joint soft Mixup targets collapsed by gesture; no extra label smoothing'}}
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError('Hybrid loss or unchanged training settings differ from the registered trial.')
    if provenance.get('variant') not in ('imu', 'imu_rot'):
        raise ValueError('Only the two new IMU variants are permitted.')
    for key in ('joint_labels', 'train_sequence_ids', 'train_subjects',
                'validation_sequence_ids', 'validation_subjects', 'fold',
                'variant', 'architecture', 'source_commit', 'source_files',
                'folds_sha256', 'seed'):
        if provenance.get(key) != original.get(key):
            raise ValueError(f'Hybrid/original per-fold {key} differs.')
    if provenance.get('seed') != 42 + provenance['fold']:
        raise ValueError('Original per-fold seed differs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--original-import', type=Path,
        default=ROOT / 'outputs/kaggle_training/imported/second_place_base_five_fold_merged_v1')
    parser.add_argument('--original-record', type=Path,
        default=ROOT / 'experiments/results/second_place_base_five_fold_merged_v1.json')
    parser.add_argument('--launch-record', type=Path,
        default=ROOT / 'experiments/results/second_place_hybrid_training_v1.json')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source-url', required=True)
    args = parser.parse_args()
    work = (ROOT / args.work_dir).resolve()
    destination = (ROOT / args.output_dir).resolve()
    if ((ROOT / 'outputs/second_place').resolve() not in work.parents or work.exists()
            or (ROOT / 'outputs/kaggle_training/imported').resolve() not in destination.parents
            or destination.exists() or Path(args.name).name != args.name
            or any(c in args.name for c in '\\/:')
            or (ROOT / f'experiments/results/{args.name}.json').exists()):
        raise ValueError('Preserve prior evidence and use fresh experiment directories/name.')
    launch, original = read(args.launch_record), read(args.original_record)
    if (args.source_url != launch['source_url'] or original['status'] != 'verified'
            or original['folds'] != list(range(5)) or original['architectures'] != ['base']):
        raise ValueError('Use the exact registered hybrid version and full verified base control.')
    # Verify immutable source bytes against the original complete merge manifest.
    original_root = args.original_import.resolve()
    manifest_path = original_root / 'merge_provenance.json'
    original_provenance = read(manifest_path)
    for item in original_provenance['copied_files']:
        path = (original_root / item['path']).resolve()
        if original_root not in path.parents or sha(path) != item['sha256']:
            raise ValueError('Original copied artifact bytes changed.')
    work.mkdir(parents=True)
    extracted = work / 'new_imu_original_download'
    extract_archive(args.archive.resolve(), extracted)
    arms = sorted(extracted.glob('outputs/second_place/*/base/*/fold_*/provenance.json'))
    expected = {(v, f) for v in ('imu', 'imu_rot') for f in range(5)}
    if len(arms) != 10 or not arms or any(p.parents[3] != arms[0].parents[3] for p in arms):
        raise ValueError('The new archive must contain all ten IMU arms in one run.')
    new_run = arms[0].parents[3]
    old_runs = sorted(original_root.glob('outputs/second_place/*/cache/metadata.csv'))
    if len(old_runs) != 1:
        raise ValueError('Original complete base root is ambiguous.')
    old_run = old_runs[0].parent.parent
    metadata_path = new_run / 'cache/metadata.csv'
    if sha(metadata_path) != original['metadata_sha256']:
        raise ValueError('New original-row metadata differs from complete base control.')
    for name in ('folds.csv', 'folds.meta.json', 'second_place_source.json', 'data_source_hashes.json'):
        if sha(extracted / 'configs' / name) != sha(ROOT / 'configs' / name):
            raise ValueError('New fixed input/source config bytes differ.')
    if read(extracted / 'data_source_check.json')['files'] != original['training_inputs']:
        raise ValueError('New raw training input bytes differ.')
    manifest = load_fold_manifest(ROOT / 'configs/folds.csv')
    metadata = pd.read_csv(metadata_path)
    source = read(ROOT / 'configs/second_place_source.json')
    import torch
    torch.set_num_threads(2)
    audited, new_audit = {}, []
    seen = set()
    for path in arms:
        p = read(path)
        key = (p['variant'], p['fold'])
        if key not in expected or key in seen or path.parent != new_run / 'base' / key[0] / f'fold_{key[1]}':
            raise ValueError('Duplicate, misplaced, or unexpected IMU arm.')
        seen.add(key)
        old_path = old_run / 'base' / key[0] / f'fold_{key[1]}'
        validate_hybrid(p, read(old_path / 'provenance.json'))
        metrics, ids, logits, labels = check_arm(path.parent, manifest, metadata, source)
        audited[key] = (ids, logits, labels)
        new_audit.append(metrics)
    if seen != expected:
        raise ValueError('Missing IMU fold/variant.')
    # Audit original models again, including controls, before copying ToF arms.
    for variant in VARIANTS:
        for fold in range(5):
            metrics, ids, logits, labels = check_arm(old_run / 'base' / variant / f'fold_{fold}', manifest, metadata, source)
            if (variant, fold) not in audited:
                audited[(variant, fold)] = (ids, logits, labels)
    dump(work / 'new_imu_audit.json', {'status': 'verified', 'arms': new_audit,
        'source_url': args.source_url, 'archive_sha256': sha(args.archive)})
    package = work / 'package'
    combined_run = package / 'outputs/second_place/hybrid_imu_group_loss_merged_v1'
    combined_run.mkdir(parents=True)
    shutil.copytree(extracted / 'configs', package / 'configs')
    shutil.copy2(extracted / 'data_source_check.json', package / 'data_source_check.json')
    shutil.copytree(new_run / 'cache', combined_run / 'cache')
    copied = []
    for variant in VARIANTS:
        selected_run = new_run if variant in ('imu', 'imu_rot') else old_run
        for fold in range(5):
            directory = selected_run / 'base' / variant / f'fold_{fold}'
            target = combined_run / directory.relative_to(selected_run)
            shutil.copytree(directory, target)
            for path in directory.rglob('*'):
                if path.is_file():
                    copied_path = target / path.relative_to(directory)
                    digest = sha(path)
                    if sha(copied_path) != digest:
                        raise ValueError('Copied arm evidence changed.')
                    copied.append({'source': 'new_imu' if variant in ('imu', 'imu_rot') else 'original_tof',
                        'fold': fold, 'variant': variant, 'path': copied_path.relative_to(package).as_posix(), 'sha256': digest})
    for fold in range(5):
        ids, first, labels = audited[('imu', fold)]
        meta = metadata.set_index('sequence_id').loc[ids].reset_index(names='sequence_id')
        for scenario in ('observed', 'aux_dropout50', 'imu_only'):
            drop = np.array([scenario == 'imu_only' or (scenario == 'aux_dropout50' and
                int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16) % 2 == 0) for sid in ids])
            full, rot = ~meta.tof_missing.to_numpy() & ~drop, meta.rotation_missing.to_numpy()
            logits = np.zeros_like(first)
            for variant, (use_full, missing_rot) in VARIANTS.items():
                arm_ids, arm_logits, arm_labels = audited[(variant, fold)]
                if arm_labels != labels or set(arm_ids) != set(ids):
                    raise ValueError('Cross-arm joint axis or validation coverage differs.')
                mask = (full == use_full) & (rot == missing_rot)
                logits[mask] = arm_logits[arm_ids.get_indexer(ids)][mask]
            frame = prediction_frame(meta, logits, labels, manifest.fingerprint)
            directory = combined_run / 'routed' / scenario / f'fold_{fold}'
            write_fold_predictions(manifest, fold, frame, directory / 'predictions.csv')
            np.save(directory / 'joint_logits.npy', logits)
            np.save(directory / 'joint_sequence_ids.npy', ids.to_numpy(dtype=str))
            dump(directory / 'metrics.json', cmi_metrics(meta.gesture, frame.predicted_gesture))
    provenance = {'protocol': 'Only IMU metric-group training losses changed; unchanged original ToF arms reused by fold. No online pseudo-label updates.',
        'new_imu': {'source_url': args.source_url, 'archive_sha256': sha(args.archive), 'archive_bytes': args.archive.stat().st_size,
                    'launch_record_sha256': sha(args.launch_record), 'code_sha256': launch['code_sha256']},
        'original_base': {'record_sha256': sha(args.original_record), 'merge_manifest_sha256': sha(manifest_path),
                          'source_batches': original_provenance['source_batches']}, 'copied_files': copied}
    dump(package / 'merge_provenance.json', provenance)
    archive = work / 'hybrid_merged_experiments.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in package.rglob('*'):
            if path.is_file():
                bundle.write(path, path.relative_to(package).as_posix())
    audit_name = args.name + '_routing_audit'
    subprocess.run([sys.executable, '-s', str(ROOT / 'scripts/evaluate_second_place.py'),
        '--archive', str(archive), '--output-dir', str(destination), '--name', audit_name,
        '--source-url', args.source_url], cwd=ROOT, check=True)
    record_path = ROOT / f'experiments/results/{args.name}.json'
    result = read(ROOT / f'experiments/results/{audit_name}.json')
    result['scope'] = 'Full fixed-five-fold development IMU auxiliary-loss ablation with unchanged original ToF arms'
    result['hybrid_merge_provenance'] = provenance
    result['matched_second_place_base'] = {}
    for scenario, values in result['scenarios'].items():
        control = original['scenarios'][scenario]
        # Replay unchanged control routing/history rather than trusting scores
        # copied from a JSON. Original routed logits already matched arm checks
        # at the complete import, and their exact bytes were verified above.
        replay = {method: [] for method in ('new', 'history_42', 'history_142', 'history_242')}
        for fold in range(5):
            directory = old_run / 'routed' / scenario / f'fold_{fold}'
            saved = _checked_predictions(manifest, fold, pd.read_csv(directory / 'predictions.csv'), require_fingerprint=True)
            replay['new'].append(cmi_metrics(saved.gesture, saved.predicted_gesture))
            ids, logits = read_aligned_logits(directory, metadata, fold)
            labels = audited[('imu', fold)][2]
            subjects = metadata.set_index('sequence_id').subject
            for seed in (42, 142, 242):
                decoded, overflow = causal_decode(ids, logits, subjects, labels, manifest.fingerprint, seed + fold)
                checked = _checked_predictions(manifest, fold, decoded, require_fingerprint=True)
                replay[f'history_{seed}'].append(cmi_metrics(checked.gesture, checked.predicted_gesture))
        for method, scores in replay.items():
            for key in scores[0]:
                if (not np.isclose(np.mean([s[key] for s in scores]), control[method]['fold_mean'][key], atol=1e-12, rtol=0)
                        or not np.isclose(np.std([s[key] for s in scores], ddof=1), control[method]['fold_std'][key], atol=1e-12, rtol=0)):
                    raise ValueError('Matched base control replay differs from verified mean/sampleSD.')
        comparison = {}
        for method in ('new', 'history_42', 'history_142', 'history_242'):
            comparison[method] = {'base_fold_mean': control[method]['fold_mean'],
                'base_fold_std': control[method]['fold_std'],
                'hybrid_fold_mean': values[method]['fold_mean'], 'hybrid_fold_std': values[method]['fold_std'],
                'delta_fold_mean': {k: values[method]['fold_mean'][k] - control[method]['fold_mean'][k]
                                    for k in values[method]['fold_mean']}}
        result['matched_second_place_base'][scenario] = comparison
    dump(record_path, result)
    dump(destination / 'import_check.json', result)
    print('Hybrid comparison verified:', record_path, flush=True)


if __name__ == '__main__':
    main()
