"""Matched CPU single-sequence controls and subject-isolated pseudo-label OOF.

All five held-out folds, three fixed scenarios/orders and both pseudo-only and
history+pseudo modes. Original weights are immutable; no Kaggle training runs.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from pilot_second_place_online import ROOT, load_models, verified_cache
from cmi_project.second_place import load_reference
from cmi_project.second_place_evaluation import causal_decode
from cmi_project.second_place_online import SubjectOnlineAdapter
from cmi_project.evaluation import cmi_metrics, evaluate_oof_frames, write_fold_predictions
from cmi_project.validation import load_fold_manifest
from merge_second_place_batches import sha
from train_second_place import prediction_frame

SCENARIOS = ('observed', 'aux_dropout50', 'imu_only')
ORDERS = (42, 142, 242)


def incoming(cache_meta, features, offsets, sid, scenario):
    row = cache_meta.loc[sid]
    index = int(row.cache_index)
    a, b = offsets[index:index + 2]
    value = np.array(features[a:b], copy=True)
    if (bool((value[:, 3:15] == 0).all()) != bool(row.rotation_missing)
            or bool((value[:, 15:] == 0).all()) != bool(row.tof_missing)):
        raise ValueError('Routing flags differ from arrived sensors.')
    if scenario == 'imu_only' or (scenario == 'aux_dropout50' and int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % 2 == 0):
        value[:, 15:] = 0
    return str(row.subject), value


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--imported', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    compact_path = ROOT / f'experiments/results/{args.name}.json'
    if ((ROOT / 'outputs/second_place').resolve() not in out.parents or out.exists()
            or Path(args.name).name != args.name or compact_path.exists()):
        raise ValueError('Use a fresh output directory and plain experiment name.')
    out.mkdir(parents=True)
    started = time.perf_counter()
    result = {'status': 'running', 'scope': 'full fixed five-fold CPU online OOF requested; incomplete until90 adaptation jobs finish',
        'orders': list(ORDERS), 'scenarios': list(SCENARIOS), 'jobs': [], 'controls': [],
        'runtime': {'torch': str(torch.__version__), 'device': 'cpu', 'threads': args.threads},
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/cmi_project/second_place_online.py']},
        'protocol': 'Source test.py single actual-length inference. Same-subject arrived-only model/buffer/RNG state;32-sample fresh Adam5e-5 and trainBN. Subject isolation is an explicit upstream adaptation. No annotation inputs or future rows; immutable returned decisions; fresh fold/scenario/order/mode state.',
        'checkpoint_hashes': {}, 'online_competition_score': None}
    current = None
    try:
        torch.set_num_threads(args.threads)
        reference = load_reference(ROOT / 'outputs/reference_code/second_place', ROOT / 'configs/second_place_source.json')
        manifest = load_fold_manifest(ROOT / 'configs/folds.csv')
        result['folds_sha256'] = manifest.fingerprint
        result['source_commit'] = reference.provenance['commit']
        meta, features, offsets = verified_cache(args.cache)
        expected = manifest.table.set_index('sequence_id').loc[meta.sequence_id]
        if (not meta.sequence_id.is_unique or set(meta.sequence_id) != set(manifest.table.sequence_id)
                or not np.array_equal(meta.subject, expected.subject) or not np.array_equal(meta.fold, expected.fold)):
            raise ValueError('Arrived input identities differ from fixed folds.')
        cache_meta = meta.reset_index(names='cache_index').set_index('sequence_id')
        result['cache_check_sha256'] = sha(args.cache / 'cache_check.json')
        frames = {scenario: {key: {} for key in ['independent', *[f'history_{s}' for s in ORDERS],
                  *[f'{mode}_{s}' for mode in ('pseudo_only', 'history_pseudo') for s in ORDERS]]} for scenario in SCENARIOS}
        for fold in range(5):
            models, labels, hashes = load_models(reference, args.imported, fold)
            result['checkpoint_hashes'][str(fold)] = hashes
            ids = pd.Index(sorted(meta.loc[meta.fold == fold, 'sequence_id']))
            # Control frames use same actual-length CPU forward pass as adaptation.
            for scenario in SCENARIOS:
                current = {'fold': fold, 'scenario': scenario, 'mode': 'CPU_control'}
                controller = SubjectOnlineAdapter(models, len(labels), seed=42 + fold, use_history=False, update_enabled=False)
                logits = []
                for i, sid in enumerate(ids):
                    subject, value = incoming(cache_meta, features, offsets, sid, scenario)
                    _, vector = controller.predict_one(sid, subject, value)
                    logits.append(vector)
                logits = np.stack(logits)
                frame = prediction_frame(pd.DataFrame({'sequence_id': ids}), logits, labels, manifest.fingerprint)
                frames[scenario]['independent'][fold] = frame
                write_fold_predictions(manifest, fold, frame, out / scenario / 'independent' / f'fold_{fold}/predictions.csv')
                np.save(out / scenario / 'independent' / f'fold_{fold}/joint_logits.npy', logits)
                np.save(out / scenario / 'independent' / f'fold_{fold}/joint_sequence_ids.npy', ids.to_numpy(dtype=str))
                for seed in ORDERS:
                    decoded, overflow = causal_decode(ids, logits, cache_meta.subject, labels, manifest.fingerprint, seed + fold)
                    frames[scenario][f'history_{seed}'][fold] = decoded
                    destination = out / scenario / f'history_{seed}/fold_{fold}'
                    write_fold_predictions(manifest, fold, decoded, destination / 'predictions.csv')
                    decoded[['sequence_id', 'arrival_position']].to_csv(destination / 'arrival_order.csv', index=False)
                result['controls'].append({**current, 'sequences': len(ids), 'seconds': sum(x['seconds'] for x in controller.prediction_seconds)})
                print('CONTROL', fold, scenario, 'complete', len(ids), flush=True)
                del controller
                for seed in ORDERS:
                    order = ids.to_numpy()[np.random.default_rng(seed + fold).permutation(len(ids))]
                    for mode in ('pseudo_only', 'history_pseudo'):
                        current = {'fold': fold, 'scenario': scenario, 'order': seed, 'mode': mode}
                        dump(out / 'current_job.json', {**current, 'status': 'running', 'arrival': 0, 'completed_jobs': len(result['jobs'])})
                        adapter = SubjectOnlineAdapter(models, len(labels), seed=(seed + fold) * 10 + SCENARIOS.index(scenario), use_history=mode == 'history_pseudo')
                        decisions, arrival_positions = {}, {}
                        job_start = time.perf_counter()
                        for position, sid in enumerate(order):
                            subject, value = incoming(cache_meta, features, offsets, sid, scenario)
                            choice, _ = adapter.predict_one(sid, subject, value)
                            decisions[sid] = labels[choice][1]
                            arrival_positions[sid] = position
                            if (position + 1) % 100 == 0:
                                dump(out / 'current_job.json', {**current, 'status': 'running', 'arrival': position + 1,
                                    'sequences': len(ids), 'updates': len(adapter.trace), 'completed_jobs': len(result['jobs']), 'seconds': time.perf_counter() - job_start})
                        # Verify every recorded training input belonged to this
                        # subject and arrived at/before the update trigger.
                        for trace in adapter.trace:
                            if any(cache_meta.loc[sid, 'subject'] != trace['subject'] or arrival_positions[sid] > arrival_positions[trace['trigger_sequence_id']] for sid in trace['sequence_ids']):
                                raise ValueError('Cross-subject or future data in update.')
                        frame = pd.DataFrame({'sequence_id': ids, 'predicted_gesture': [decisions[sid] for sid in ids],
                            'folds_sha256': manifest.fingerprint})
                        frames[scenario][f'{mode}_{seed}'][fold] = frame
                        destination = out / scenario / f'{mode}_{seed}/fold_{fold}'
                        write_fold_predictions(manifest, fold, frame, destination / 'predictions.csv')
                        dump(destination / 'update_trace.json', adapter.trace)
                        pd.DataFrame({'sequence_id': list(order), 'arrival_position': range(len(order))}).to_csv(destination / 'arrival_order.csv', index=False)
                        joined = frame.set_index('sequence_id').loc[ids]
                        truth = manifest.table.set_index('sequence_id').loc[ids, 'gesture']
                        job = {**current, 'sequences': len(ids), 'updates': len(adapter.trace),
                            'overflow_fallbacks': sum(s['decoder'].overflow for s in adapter.subjects.values()),
                            'seconds': time.perf_counter() - job_start, **cmi_metrics(truth, joined.predicted_gesture),
                            'predictions_sha256': sha(destination / 'predictions.csv'), 'update_trace_sha256': sha(destination / 'update_trace.json')}
                        result['jobs'].append(job)
                        dump(out / 'progress.json', result)
                        print('COMPLETE', current, 'score', job['score'], 'seconds', job['seconds'], flush=True)
                        del adapter
                        gc.collect()
            del models
            gc.collect()
        result['evaluation'] = {scenario: {key: evaluate_oof_frames(manifest, by_fold, out / scenario / key / 'evaluation', experiment_name=f'{args.name}_{scenario}_{key}') for key, by_fold in value.items()} for scenario, value in frames.items()}
        result.update(status='verified', scope='full fixed five-fold8151 matched CPU single-sequence OOF; base architecture; not author original tenfold ensemble', seconds=time.perf_counter() - started)
        features._mmap.close()
    except Exception as error:
        result.update(status='failed', current_job=current, seconds=time.perf_counter() - started, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        dump(out / 'progress.json', result)
        dump(compact_path, result)


if __name__ == '__main__':
    main()
