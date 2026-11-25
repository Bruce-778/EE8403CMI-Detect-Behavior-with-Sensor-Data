"""Real-input CPU timing/numerical pilot; not a scored CV experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmi_project.second_place import load_reference
from cmi_project.second_place_evaluation import read_aligned_logits
from cmi_project.second_place_online import SubjectOnlineAdapter, VARIANTS
from cmi_project.validation import load_fold_manifest
from merge_second_place_batches import sha


def load_models(reference, imported, fold):
    run = imported / 'outputs/second_place/base_five_fold_merged'
    file_checks = {x['path']: x['sha256'] for x in json.loads((imported / 'merge_provenance.json').read_text())['copied_files']}
    models, labels, checkpoint_hashes = {}, None, {}
    manifest = load_fold_manifest(ROOT / 'configs/folds.csv')
    train, validation = manifest.split(fold)
    for variant in VARIANTS:
        path = run / 'base' / variant / f'fold_{fold}' / 'last.pt'
        digest = sha(path)
        if digest != file_checks[path.relative_to(imported).as_posix()]:
            raise ValueError('Verified original weight bytes changed.')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        p = checkpoint['provenance']
        if (p['fold'] != fold or p['seed'] != 42 + fold or checkpoint['epoch'] != 50
                or p['train_sequence_ids'] != sorted(train.sequence_id)
                or p['validation_sequence_ids'] != sorted(validation.sequence_id)
                or p['source_commit'] != reference.provenance['commit']):
            raise ValueError('Actual fixed-fold training provenance changed.')
        if labels is not None and labels != p['joint_labels']:
            raise ValueError('Joint axes differ within the fold.')
        labels = p['joint_labels']
        model = getattr(reference.models, 'ALLModel' if variant.startswith('all') else 'IMUModel')(15, len(labels))
        model.load_state_dict(checkpoint['model_state'], strict=True)
        models[variant] = model.eval()
        checkpoint_hashes[variant] = digest
    return models, labels, checkpoint_hashes


def verified_cache(cache):
    check = json.loads((cache / 'cache_check.json').read_text())
    if check['status'] != 'verified':
        raise ValueError('Unverified local cache.')
    for filename in ('features.npy', 'offsets.npy', 'metadata.csv', 'provenance.json'):
        if sha(cache / filename) != check['files'][filename]['sha256']:
            raise ValueError('Local cache bytes changed.')
    # Read only input routing/identity fields into the predictor-side reader.
    meta = pd.read_csv(cache / 'metadata.csv', usecols=['sequence_id', 'subject', 'fold', 'rotation_missing', 'tof_missing'])
    return meta, np.load(cache / 'features.npy', mmap_mode='r'), np.load(cache / 'offsets.npy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--imported', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    output = ROOT / f'experiments/results/{args.name}.json'
    if Path(args.name).name != args.name or output.exists():
        raise ValueError('Use a fresh plain experiment name.')
    result = {'status': 'started', 'scope': 'CPU feasibility pilot only; no CV score', 'fold': 0,
        'orders': [42], 'scenario': 'observed', 'runtime': {'torch': str(torch.__version__), 'threads': args.threads},
        'protocol': 'Only same-subject arrived sensors/pseudo decisions. Fresh Adam5e-5 at32; BN train. No validation gesture/orientation/phase in inference/update. Subject-isolated model/buffer/RNG differs from upstream global buffers.'}
    started = time.perf_counter()
    try:
        torch.set_num_threads(args.threads)
        reference = load_reference(ROOT / 'outputs/reference_code/second_place', ROOT / 'configs/second_place_source.json')
        models, labels, hashes = load_models(reference, args.imported, 0)
        meta, features, offsets = verified_cache(args.cache)
        manifest = load_fold_manifest(ROOT / 'configs/folds.csv')
        expected = manifest.table.set_index('sequence_id')
        aligned = expected.loc[meta.sequence_id]
        if (not meta.sequence_id.is_unique or set(meta.sequence_id) != set(expected.index)
                or not np.array_equal(meta.subject, aligned.subject) or not np.array_equal(meta.fold, aligned.fold)):
            raise ValueError('Local feature row identities disagree with fixed folds.')
        fold_ids = sorted(meta.loc[meta.fold == 0, 'sequence_id'])
        order = np.asarray(fold_ids)[np.random.default_rng(42).permutation(len(fold_ids))]
        subject = sorted(meta.loc[meta.fold == 0, 'subject'].unique())[0]
        indexed = meta.reset_index(names='cache_index').set_index('sequence_id')
        sample_ids = [sid for sid in order if indexed.loc[sid, 'subject'] == subject][:40]
        if len(sample_ids) < 40:
            raise ValueError('Predetermined pilot subject has fewer than40 arrivals.')
        result.update(subject=subject, sequence_ids=sample_ids, checkpoint_sha256=hashes,
                      source_commit=reference.provenance['commit'])
        cloud_run = args.imported / 'outputs/second_place/base_five_fold_merged'
        # Cloud metadata order is retained and logit IDs are checked by reader.
        cloud_meta = pd.read_csv(cloud_run / 'cache/metadata.csv')
        cloud = {variant: read_aligned_logits(cloud_run / 'base' / variant / 'fold_0', cloud_meta, 0) for variant in VARIANTS}
        adapter = SubjectOnlineAdapter(models, len(labels), seed=42)
        drift, decisions = [], []
        replay_start = time.perf_counter()
        for position, sid in enumerate(sample_ids):
            row = indexed.loc[sid]
            a, b = offsets[int(row.cache_index):int(row.cache_index) + 2]
            value = np.array(features[a:b], copy=True)
            decision, logits = adapter.predict_one(sid, subject, value)
            decisions.append(int(decision))
            variant = ('imu' if row.tof_missing else 'all') + ('_rot' if row.rotation_missing else '')
            ids, saved_logits = cloud[variant]
            if position < 32:
                original = saved_logits[ids.get_loc(sid)]
                drift.append({'sequence_id': sid, 'max_abs_logit_difference': float(np.abs(logits - original).max()),
                              'same_independent_joint_argmax': bool(logits.argmax() == original.argmax())})
        result.update(status='completed', seconds=time.perf_counter() - started,
            replay_seconds=time.perf_counter() - replay_start, returned_joint_decisions=decisions,
            prediction_timings=adapter.prediction_seconds, updates=adapter.trace,
            pre_update_cpu_vs_cloud=drift, frozen_weight_files_unchanged=all(sha(cloud_run/'base'/v/'fold_0/last.pt') == hashes[v] for v in VARIANTS))
        by_variant = {v: [x['seconds'] for x in adapter.prediction_seconds if x['variant'] == v] for v in VARIANTS}
        result['timing_summary'] = {'prediction_seconds_median': {v: float(np.median(values)) for v, values in by_variant.items() if values},
            'update_seconds': {x['variant']: x['seconds'] for x in adapter.trace},
            'max_pre_update_logit_drift': max(x['max_abs_logit_difference'] for x in drift),
            'pre_update_argmax_agreement': sum(x['same_independent_joint_argmax'] for x in drift) / len(drift)}
        print(json.dumps(result['timing_summary'], indent=2), flush=True)
        features._mmap.close()
    except Exception as error:
        result.update(status='failed', seconds=time.perf_counter() - started, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
