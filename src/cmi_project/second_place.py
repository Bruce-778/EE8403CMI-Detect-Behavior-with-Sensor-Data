"""Pinned upstream models and training-only adapters for a controlled reproduction.

Third-party sources remain outside tracked code. No Lightning, logger, inference
server or upstream top-level training/test entrypoint executes on import.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .evaluation import ALL_GESTURES

PHASES = {'Moves hand to target location': 0,
          'Relaxes and moves hand to target location': 0,
          'Hand at target location': 1, 'Performs gesture': 2}
SPECIAL_SUBJECTS = {'SUBJ_019262', 'SUBJ_045235'}
ARCHITECTURES = {'base': ('IMUModel', 'ALLModel'),
                 'simple': ('IMUSimpleModel', 'ALLSimpleModel'),
                 'deep': ('IMUDeepModel', 'ALLDeepModel')}


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_reference(folder: Path, source_manifest: Path):
    folder = Path(folder)
    provenance = json.loads(Path(source_manifest).read_text(encoding='utf-8'))
    for name, expected in provenance['files'].items():
        if source_hash(folder / name) != expected:
            raise ValueError(f'Pinned upstream source mismatch: {name}')

    def module(name, file):
        spec = importlib.util.spec_from_file_location(name, folder / file)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    # model.py uses an absolute import; do not leave that alias in our runtime.
    prior = sys.modules.get('masked_batchnorm')
    try:
        sys.modules['masked_batchnorm'] = module('_cmi_reference_bn', 'masked_batchnorm.py')
        models = module('_cmi_reference_models', 'model.py')
    finally:
        if prior is None:
            sys.modules.pop('masked_batchnorm', None)
        else:
            sys.modules['masked_batchnorm'] = prior
    utils = module('_cmi_reference_utils', 'utils.py')
    wanted = {'make_feature_from_np', 'MixupDataset', 'collate_fn'}
    tree = ast.parse((folder / 'train.py').read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    if {node.name for node in selected} != wanted:
        raise ValueError('Pinned upstream entrypoints are incomplete.')
    namespace = {'np': np, 'torch': torch, 'random': random,
                 'GestureDataset': Dataset, 'Dataset': Dataset,
                 'quaternion_to_6d_rotation': utils.quaternion_to_6d_rotation,
                 'calculate_angular_velocity_from_quat': utils.calculate_angular_velocity_from_quat,
                 'remove_gravity_from_acc': utils.remove_gravity_from_acc}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(folder / 'train.py'), 'exec'), namespace)
    return SimpleNamespace(models=models, provenance=provenance, **{name: namespace[name] for name in wanted})


def fit_joint_labels(metadata: pd.DataFrame, train_ids):
    """Fit output ontology only from this fold's actual training IDs."""
    if not metadata.sequence_id.is_unique:
        raise ValueError('Duplicate sequence metadata.')
    indexed = metadata.set_index('sequence_id')
    train = indexed.loc[list(train_ids)]
    if train[['orientation', 'gesture', 'initial_behavior']].isna().any().any():
        raise ValueError('Missing training joint label.')
    labels = sorted(set(map(tuple, train[['orientation', 'gesture', 'initial_behavior']].to_numpy())))
    return labels


def collapse_probabilities(joint_probability, labels):
    out = np.zeros((len(joint_probability), len(ALL_GESTURES)), dtype=np.float64)
    for index, (_, gesture, _) in enumerate(labels):
        out[:, ALL_GESTURES.index(gesture)] += joint_probability[:, index]
    return out


def build_cache(data_dir, cache, manifest, reference):
    """Stream raw sequences; compute upstream features BEFORE tail cropping."""
    cache = Path(cache)
    if cache.exists():
        raise ValueError('Use a fresh reference cache; never silently reuse changed inputs.')
    demo = pd.read_csv(Path(data_dir) / 'train_demographics.csv').set_index('subject')
    if not demo.index.is_unique:
        raise ValueError('Duplicate demographics subject.')
    raw_columns = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
    tof_columns = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
    columns = ['sequence_id', 'subject', 'sequence_counter', 'orientation', 'gesture', 'behavior', *raw_columns, *tof_columns]
    expected = manifest.table.set_index('sequence_id')
    samples, phases, metadata, seen = [], [], [], set()
    pending = None

    def consume(seq):
        sid = str(seq.sequence_id.iloc[0])
        subject = str(seq.subject.iloc[0])
        if sid in seen or sid not in expected.index:
            raise ValueError('Unexpected/non-contiguous sequence ID.')
        seen.add(sid)
        if seq.subject.nunique() != 1 or seq.gesture.nunique() != 1 or seq.orientation.nunique() != 1:
            raise ValueError('Conflicting sequence annotation.')
        row = expected.loc[sid]
        if subject != row.subject or str(seq.gesture.iloc[0]) != row.gesture:
            raise ValueError('Raw sequence differs from fixed folds.')
        seq = seq.sort_values('sequence_counter')
        behavior = seq.behavior.to_list()
        initial = next((b for b in behavior if b in list(PHASES)[:2]), None)
        if initial is None:
            raise ValueError(f'No initial phase in {sid}.')
        data = np.column_stack([seq[raw_columns].to_numpy(dtype=float),
                                np.full(len(seq), demo.loc[subject, 'handedness'])])
        tof = seq[tof_columns].to_numpy(dtype=float)
        tof[~np.isfinite(tof) | (tof == -1)] = 0
        imu, tof = reference.make_feature_from_np(data.copy(), tof.copy())
        if subject in SPECIAL_SUBJECTS:
            imu[:, [0, 1, *range(3, 14)]] *= -1
            tof[:] = 0
        x = np.concatenate([imu, tof], axis=1).astype(np.float32)[-200:]
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite reference features in {sid}.')
        samples.append(x)
        phases.append(np.asarray([PHASES.get(b, -1) for b in behavior], dtype=np.int64)[-200:])
        metadata.append({'sequence_id': sid, 'subject': subject, 'gesture': str(seq.gesture.iloc[0]),
                         'orientation': str(seq.orientation.iloc[0]), 'initial_behavior': initial,
                         'fold': int(row.fold), 'rotation_missing': bool((x[:, 3:15] == 0).all()),
                         'tof_missing': bool((x[:, 15:] == 0).all())})
        if len(samples) % 1000 == 0:
            print(f'Reference cache: {len(samples)}/{len(expected)} sequences', flush=True)

    with pd.read_csv(Path(data_dir) / 'train.csv', usecols=columns, chunksize=25000) as reader:
        for chunk in reader:
            if pending is not None:
                chunk = pd.concat([pending, chunk], ignore_index=True)
            ids = chunk.sequence_id.to_numpy()
            boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
            for a, b in zip(boundaries[:-2], boundaries[1:-1]):
                consume(chunk.iloc[a:b])
            pending = chunk.iloc[boundaries[-2]:].copy()
    if pending is not None:
        consume(pending)
    if seen != set(expected.index):
        raise ValueError('Reference cache does not cover the entire fixed manifest.')
    lengths = np.asarray([len(x) for x in samples], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    cache.mkdir(parents=True)
    np.save(cache / 'features.npy', np.concatenate(samples))
    np.save(cache / 'phases.npy', np.concatenate(phases))
    np.save(cache / 'offsets.npy', offsets)
    pd.DataFrame(metadata).to_csv(cache / 'metadata.csv', index=False)
    (cache / 'provenance.json').write_text(json.dumps({'folds_sha256': manifest.fingerprint,
        'source_commit': reference.provenance['commit'], 'source_files': reference.provenance['files'],
        'protocol': 'raw upstream 15D IMU and 320D ToF, no learned scaler, fixed published length 200; right padding',
        'sequences': len(samples)}, indent=2), encoding='utf-8')


class ReferenceDataset(Dataset):
    def __init__(self, cache, indices, labels, *, use_tof, rotation_zero=False, training=False):
        self.features = np.load(Path(cache) / 'features.npy', mmap_mode='r')
        self.phases = np.load(Path(cache) / 'phases.npy', mmap_mode='r')
        self.offsets = np.load(Path(cache) / 'offsets.npy')
        self.metadata = pd.read_csv(Path(cache) / 'metadata.csv')
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = {tuple(label): i for i, label in enumerate(labels)}
        self.use_tof, self.rotation_zero, self.training = use_tof, rotation_zero, training

    def __len__(self):
        return len(self.indices)

    def close(self):
        for array in (self.features, self.phases):
            array._mmap.close()

    def __getitem__(self, index):
        index = self.indices[index]
        a, b = self.offsets[index:index + 2]
        # Copy BEFORE any dropout/zeroing; avoid upstream cache mutation bug.
        x = np.array(self.features[a:b, :335 if self.use_tof else 15], copy=True)
        if self.rotation_zero:
            x[:, 3:15] = 0
        if self.use_tof and self.training and random.random() < .1:
            x[:, 15:] = 0
        if self.training:
            row = self.metadata.iloc[index]
            target = self.labels[(row.orientation, row.gesture, row.initial_behavior)]
            phase = np.array(self.phases[a:b], copy=True)
        else:
            # Prediction does not consume any held-out annotation.
            target, phase = -1, np.full(b - a, -1, dtype=np.int64)
        return torch.from_numpy(x), torch.tensor(b - a), torch.tensor(target), torch.from_numpy(phase)
