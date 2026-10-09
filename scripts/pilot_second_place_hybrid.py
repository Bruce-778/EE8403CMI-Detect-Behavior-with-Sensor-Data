"""Actual training-fold batch checks for the metric-loss hybrid, not a CV score."""
from pathlib import Path
import hashlib
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmi_project.second_place import ReferenceDataset, fit_joint_labels
from cmi_project.second_place_hybrid import JointMetricLoss, load_project_reference
from cmi_project.validation import load_fold_manifest


def main():
    path = ROOT / 'experiments/results/second_place_hybrid_training_batch_v1.json'
    if path.exists(): raise ValueError('Preserve the existing pilot record.')
    result = {'status': 'running', 'scope': 'two actual train-fold batches; no validation/CV score', 'branches': []}
    torch.set_num_threads(2)
    try:
        reference = load_project_reference(ROOT)
        manifest = load_fold_manifest(ROOT / 'configs/folds.csv')
        cache = ROOT / 'outputs/second_place/online_cache_cpu_v1'
        meta = pd.read_csv(cache / 'metadata.csv')
        train, validation = manifest.split(0)
        indices = np.flatnonzero(meta.sequence_id.isin(train.sequence_id))
        if set(meta.iloc[indices].subject) & set(validation.subject): raise ValueError('Subject leakage.')
        labels = fit_joint_labels(meta, train.sequence_id)
        result.update(source_commit=reference.provenance['commit'], folds_sha256=manifest.fingerprint,
            runtime={'torch': str(torch.__version__), 'device': 'cpu', 'threads': 2},
            train_ids=sorted(train.sequence_id), validation_ids=sorted(validation.sequence_id))
        for variant in ['imu', 'imu_rot']:
            torch.manual_seed(42); np.random.seed(42); random.seed(42)
            dataset = ReferenceDataset(cache, indices, labels, use_tof=False, rotation_zero=variant == 'imu_rot', training=True)
            mixup = reference.MixupDataset(dataset, alpha=.5, num_classes=len(labels))
            x, lengths, targets, phase = reference.collate_fn([mixup[i] for i in range(32)])
            model = reference.models.IMUModel(input_size=15, n_classes=len(labels)).train()
            original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            started = time.perf_counter()
            output = model(x, lengths, None)
            criterion = JointMetricLoss(labels)
            joint_only = JointMetricLoss(labels, 0, 0)(output['gesture_logits'], targets)
            hybrid = criterion(output['gesture_logits'], targets)
            phase_logits = output['phase_logits'].reshape(-1, 3); mask = phase.reshape(-1) >= 0
            phase_loss = torch.nn.functional.cross_entropy(phase_logits[mask], phase.reshape(-1)[mask])
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
            optimizer.zero_grad(); loss = hybrid + phase_loss; loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite actual hybrid batch.')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            result['branches'].append({'variant': variant, 'batch': 32, 'max_length': int(lengths.max()),
                'joint_classes': len(labels), 'original_joint_loss': float(joint_only.detach()),
                'hybrid_classification_loss': float(hybrid.detach()), 'unchanged_phase_loss': float(phase_loss.detach()),
                'seconds': time.perf_counter() - started,
                'parameters_changed': any(not torch.equal(original_state[k], v) for k,v in model.state_dict().items()),
                'gradients_finite': True})
            dataset.close()
        result['status'] = 'completed'
    except Exception as error:
        result.update(status='failed', error=repr(error)); raise
    finally:
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in {'train_ids','validation_ids'}}, indent=2), flush=True)


if __name__ == '__main__': main()
