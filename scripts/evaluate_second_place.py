"""Import and audit reference results, then compare identical held-out samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmi_project.evaluation import (PROBABILITY_COLUMNS, _checked_predictions, cmi_metrics, evaluate_oof_frames,
                                   write_fold_predictions)
from cmi_project.second_place_evaluation import check_arm, causal_decode, read_aligned_logits
from cmi_project.validation import load_fold_manifest
from import_winner_experiments import extract_archive
from train_second_place import VARIANTS, prediction_frame


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--metadata', type=Path, help='v1 ONLY: raw-order cache/metadata.csv downloaded from the same notebook version')
    args = parser.parse_args()
    if Path(args.name).name != args.name or any(c in args.name for c in '\\/:'):
        raise ValueError('Use a plain experiment name.')
    destination = (ROOT / args.output_dir).resolve()
    if (ROOT / 'outputs/kaggle_training/imported').resolve() not in destination.parents:
        raise ValueError('Use a fresh child of outputs/kaggle_training/imported.')
    record = ROOT / f'experiments/results/{args.name}.json'
    if record.exists():
        raise ValueError('Preserve prior experiment records.')
    extract_archive(args.archive.resolve(), destination)
    manifest = load_fold_manifest(ROOT / 'configs/folds.csv')
    imported_manifest = load_fold_manifest(destination / 'configs/folds.csv')
    source = json.loads((ROOT / 'configs/second_place_source.json').read_text())
    expected_data = json.loads((ROOT / 'configs/data_source_hashes.json').read_text())
    actual_data = json.loads((destination / 'data_source_check.json').read_text())
    if (manifest.fingerprint != imported_manifest.fingerprint or actual_data.get('status') != 'verified'
            or actual_data.get('files') != expected_data['files']
            or json.loads((destination / 'configs/second_place_source.json').read_text()) != source):
        raise ValueError('Training inputs/fixed folds/source code differ from frozen local evidence.')
    arms = sorted(destination.glob('outputs/second_place/*/*/*/fold_*/provenance.json'))
    if not arms:
        raise ValueError('No reference training artifacts found.')
    run = arms[0].parents[3]
    if any(p.parents[3] != run for p in arms):
        raise ValueError('Archive must contain exactly one experiment root.')
    metadata_path = run / 'cache/metadata.csv'
    if not metadata_path.exists():
        if args.metadata is None:
            raise ValueError('v1 ZIP excludes raw-order metadata. Download cache/metadata.csv from the same exact notebook version; do not guess sorted logit row order.')
        metadata_path = args.metadata.resolve()
    metadata = pd.read_csv(metadata_path)
    if not metadata.sequence_id.is_unique or set(metadata.sequence_id) != set(manifest.table.sequence_id):
        raise ValueError('Cache metadata does not cover the fixed sequence index.')
    aligned = metadata.set_index('sequence_id').loc[manifest.table.sequence_id]
    if any(not np.array_equal(aligned[k], manifest.table[k]) for k in ('subject', 'gesture', 'fold')):
        raise ValueError('Cache metadata disagrees with fixed folds.')
    result = {'status': 'verified', 'source_url': args.source_url,
        'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'source_commit': source['commit'], 'folds_sha256': manifest.fingerprint,
        'training_inputs': actual_data['files'], 'arms': [], 'scenarios': {},
        'orders': [42, 142, 242], 'online_score': None,
        'protocol': 'Independent sequence baseline; causal joint history reported separately. No pseudo-label weight updates.'}
    audited = {}
    for path in arms:
        metrics, ids, logits, labels = check_arm(path.parent, manifest, metadata, source)
        key = (metrics['architecture'], metrics['variant'], metrics['fold'])
        if key in audited or path.parent != run / key[0] / key[1] / f'fold_{key[2]}':
            raise ValueError('Duplicate or misplaced arm evidence.')
        audited[key] = (ids, logits, labels)
        result['arms'].append(metrics)
    folds = sorted({k[2] for k in audited})
    architectures = sorted({k[0] for k in audited})
    if any((a, v, f) not in audited for a in architectures for v in VARIANTS for f in folds):
        raise ValueError('Requested folds must finish all four sensor variants before routed comparison.')
    result.update(folds=folds, architectures=architectures,
        scope='full fixed five-fold development OOF' if folds == list(range(5)) else 'partial held-out folds; not five-fold CV')
    for scenario in ('observed', 'aux_dropout50', 'imu_only'):
        baseline, candidate, history, rows = {}, {}, {seed: {} for seed in result['orders']}, []
        for fold in folds:
            ids, first, labels = audited[(architectures[0], 'imu', fold)]
            meta = metadata.set_index('sequence_id').loc[ids].reset_index()
            drop = np.array([scenario == 'imu_only' or (scenario == 'aux_dropout50' and
                int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16) % 2 == 0) for sid in ids])
            full = ~meta.tof_missing.to_numpy() & ~drop
            rot = meta.rotation_missing.to_numpy()
            logit_list = []
            for arch in architectures:
                combined = np.zeros_like(first)
                for variant, (use_full, missing_rot) in VARIANTS.items():
                    arm_ids, arm_logits, arm_labels = audited[(arch, variant, fold)]
                    if arm_labels != labels:
                        raise ValueError('Ensemble joint ontology mismatch.')
                    mask = (full == use_full) & (rot == missing_rot)
                    combined[mask] = arm_logits[arm_ids.get_indexer(ids)][mask]
                logit_list.append(combined)
            logits = np.mean(logit_list, axis=0)
            saved_dir = run / 'routed' / scenario / f'fold_{fold}'
            saved_ids, saved_logits = read_aligned_logits(saved_dir, metadata, fold)
            if not np.allclose(logits, saved_logits[saved_ids.get_indexer(ids)], atol=1e-6, rtol=0):
                raise ValueError('Saved routing differs from the audited model branches.')
            frame = prediction_frame(meta, logits, labels, manifest.fingerprint)
            candidate[fold] = _checked_predictions(manifest, fold, frame, require_fingerprint=True)
            saved = _checked_predictions(manifest, fold, pd.read_csv(saved_dir / 'predictions.csv'), require_fingerprint=True)
            if (not np.array_equal(candidate[fold].predicted_gesture, saved.predicted_gesture)
                    or not np.allclose(candidate[fold][PROBABILITY_COLUMNS], saved[PROBABILITY_COLUMNS], atol=1e-6, rtol=0)):
                raise ValueError('Saved routed predictions differ from reconstructed decisions.')
            old = pd.read_csv(ROOT / f'outputs/winner_comparison/candidate/routed/{scenario}/oof_predictions.csv')
            baseline[fold] = _checked_predictions(manifest, fold, old[old.fold == fold], require_fingerprint=True)
            old_metrics = cmi_metrics(baseline[fold].gesture, baseline[fold].predicted_gesture)
            new_metrics = cmi_metrics(candidate[fold].gesture, candidate[fold].predicted_gesture)
            saved_metrics = json.loads((saved_dir / 'metrics.json').read_text(encoding='utf-8'))
            if any(not np.isclose(new_metrics[k], saved_metrics[k], atol=1e-12, rtol=0) for k in new_metrics):
                raise ValueError('Reported routed metric mismatch.')
            rows.append({'fold': fold, 'sequences': len(frame), 'old': old_metrics, 'new': new_metrics,
                         'delta': new_metrics['score'] - old_metrics['score']})
            write_fold_predictions(manifest, fold, frame, destination / f'comparison/{scenario}/fold_{fold}/predictions.csv')
            for seed in result['orders']:
                decoded, overflow = causal_decode(ids, logits, meta.set_index('sequence_id').subject,
                                                  labels, manifest.fingerprint, seed + fold)
                history[seed][fold] = decoded
                path = destination / f'comparison/{scenario}/seed_{seed}/fold_{fold}'
                write_fold_predictions(manifest, fold, decoded, path / 'predictions.csv')
                decoded[['sequence_id', 'arrival_position']].to_csv(path / 'arrival_order.csv', index=False)
                rows[-1].setdefault('history', {})[str(seed)] = {'overflow_fallbacks': overflow,
                    **cmi_metrics(meta.gesture, decoded.predicted_gesture)}
            print(scenario, fold, 'new', new_metrics['score'], 'delta', rows[-1]['delta'], flush=True)
        result['scenarios'][scenario] = {'folds': rows}
        if folds == list(range(5)):
            for label, frames in [('old', baseline), ('new', candidate),
                                  *[(f'history_{seed}', history[seed]) for seed in result['orders']]]:
                result['scenarios'][scenario][label] = evaluate_oof_frames(manifest, frames,
                    destination / f'comparison/{scenario}/{label}/evaluation', experiment_name=f'{args.name}_{scenario}_{label}')
    dump(destination / 'import_check.json', result)
    dump(record, result)
    print('Verified compact record:', record, flush=True)


if __name__ == '__main__':
    main()
