"""Fixed-weight complementarity screening; no fitted weights or model training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmi_project.evaluation import ALL_GESTURES, PROBABILITY_COLUMNS, _checked_predictions, evaluate_oof_frames
from cmi_project.second_place import collapse_probabilities
from cmi_project.second_place_evaluation import causal_decode
from cmi_project.validation import load_fold_manifest

WEIGHTS = (.1, .2, .5)
ORDERS = (42, 142, 242)
SCENARIOS = ('observed', 'aux_dropout50', 'imu_only')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    output = ROOT / 'outputs/second_place/hybrid' / args.name
    record_path = ROOT / 'experiments/results' / f'{args.name}.json'
    if Path(args.name).name != args.name or output.exists() or record_path.exists():
        raise ValueError('Fresh plain experiment name required.')
    manifest = load_fold_manifest(ROOT / 'configs/folds.csv')
    base = ROOT / 'outputs/kaggle_training/imported/second_place_base_five_fold_merged_v1/outputs/second_place/base_five_fold_merged'
    original_record = json.loads((ROOT / 'experiments/results/second_place_base_five_fold_merged_v1.json').read_text())
    if original_record['status'] != 'verified':
        raise ValueError('Verified base inputs required.')
    result = {'status': 'running', 'scope': 'fixed-five-fold development complementarity screening; weights fixed in code, no optimal selection',
        'own_weights': WEIGHTS, 'orders': ORDERS, 'folds_sha256': manifest.fingerprint,
        'source_sha256': {}, 'scenarios': {},
        'protocol': 'Same sequence IDs and missing scenarios. Joint lift preserves second-place conditional orientation/initial-behavior distribution within each gesture. Marginal decision control reported separately. Truth used only for final metric/diagnostic, never fusion or history. All preset weights/orders reported; no per-fold selection.',
        'training_change': False, 'online_competition_score': None}
    output.mkdir(parents=True)
    try:
        for scenario in SCENARIOS:
            own_path = ROOT / f'outputs/winner_comparison/candidate/routed/{scenario}/oof_predictions.csv'
            own = pd.read_csv(own_path)
            result['source_sha256'][str(own_path.relative_to(ROOT))] = sha(own_path)
            frames, complementarity = {}, []
            for fold in range(5):
                directory = base / f'routed/{scenario}/fold_{fold}'
                base_frame = _checked_predictions(manifest, fold, pd.read_csv(directory / 'predictions.csv'), require_fingerprint=True)
                own_frame = _checked_predictions(manifest, fold, own[own.fold == fold], require_fingerprint=True).set_index('sequence_id').loc[base_frame.sequence_id]
                ids = pd.Index(np.load(directory / 'joint_sequence_ids.npy', allow_pickle=False))
                logits = np.load(directory / 'joint_logits.npy', allow_pickle=False)
                if not ids.is_unique or set(ids) != set(base_frame.sequence_id):
                    raise ValueError('Joint logit IDs differ from paired samples.')
                logits = logits[ids.get_indexer(base_frame.sequence_id)]
                labels = json.loads((base / f'base/imu/fold_{fold}/provenance.json').read_text())['joint_labels']
                q = np.exp(logits - logits.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
                p = collapse_probabilities(q, labels)
                if not np.allclose(p, base_frame[PROBABILITY_COLUMNS], atol=1e-6, rtol=0):
                    raise ValueError('Base probabilities/logit class axis differs.')
                if not np.array_equal([labels[i][1] for i in q.argmax(1)], base_frame.predicted_gesture):
                    raise ValueError('Base joint decisions differ.')
                for path in [directory / 'predictions.csv', directory / 'joint_sequence_ids.npy', directory / 'joint_logits.npy', base / f'base/imu/fold_{fold}/provenance.json']:
                    result['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
                frames.setdefault('base_joint', {})[fold] = base_frame
                frames.setdefault('ours', {})[fold] = own_frame.reset_index()
                own_p = own_frame[PROBABILITY_COLUMNS].to_numpy()
                axes = np.array([ALL_GESTURES.index(label[1]) for label in labels])
                def make_frame(probability, choices):
                    frame = base_frame[['sequence_id', 'folds_sha256']].copy()
                    frame['predicted_gesture'] = choices
                    for i, name in enumerate(PROBABILITY_COLUMNS): frame[name] = probability[:, i]
                    return frame
                frames.setdefault('base_marginal_control', {})[fold] = make_frame(p, np.asarray(ALL_GESTURES)[p.argmax(1)])
                for weight in WEIGHTS:
                    blend = (1 - weight) * p + weight * own_p
                    frames.setdefault(f'marginal_ours_{weight}', {})[fold] = make_frame(blend, np.asarray(ALL_GESTURES)[blend.argmax(1)])
                    lifted = q * blend[:, axes] / np.maximum(p[:, axes], 1e-300)
                    lifted /= lifted.sum(1, keepdims=True)
                    frames.setdefault(f'joint_ours_{weight}', {})[fold] = make_frame(collapse_probabilities(lifted, labels), [labels[i][1] for i in lifted.argmax(1)])
                    for seed in ORDERS:
                        decoded, overflow = causal_decode(pd.Index(base_frame.sequence_id), np.log(np.maximum(lifted, 1e-300)), manifest.table.set_index('sequence_id').subject, labels, manifest.fingerprint, seed + fold)
                        if overflow: raise ValueError('Unexpected history overflow.')
                        frames.setdefault(f'joint_history_ours_{weight}_order_{seed}', {})[fold] = decoded
                truth = base_frame.gesture.to_numpy()
                old_correct = own_frame.predicted_gesture.to_numpy() == truth
                new_correct = base_frame.predicted_gesture.to_numpy() == truth
                complementarity.append({'fold': fold, 'sequences': len(truth), 'ours_correct_base_wrong': int((old_correct & ~new_correct).sum()),
                    'base_correct_ours_wrong': int((new_correct & ~old_correct).sum()), 'both_wrong': int((~new_correct & ~old_correct).sum())})
            summaries = {name: evaluate_oof_frames(manifest, folds, output / scenario / name, experiment_name=f'{args.name}_{scenario}_{name}') for name, folds in frames.items()}
            if abs(summaries['base_joint']['fold_mean']['score'] - original_record['scenarios'][scenario]['new']['fold_mean']['score']) > 1e-12:
                raise ValueError('Base input scores differ from audited source record.')
            result['scenarios'][scenario] = {'evaluations': summaries, 'complementarity': complementarity,
                'delta_from_base_joint': {name: value['fold_mean']['score'] - summaries['base_joint']['fold_mean']['score'] for name, value in summaries.items()}}
            print(scenario, {name: round(value['fold_mean']['score'], 6) for name, value in summaries.items()}, flush=True)
        result['status'] = 'verified'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        record_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
