"""Subject-isolated, strictly arriving-data pseudo-label adaptation.

Adaptation of pinned test.py: keep its routing, 32-sample buffers, fresh Adam
5e-5 step and train-mode BatchNorm. Its global cross-subject buffers/weights are
replaced by independent subject state to satisfy this experiment's protocol.
The public interface accepts only sensors and identifiers, never annotations.
"""
import copy
import hashlib
import time

import numpy as np
import torch

from .second_place_evaluation import CausalJointAssignment

VARIANTS = ('imu', 'imu_rot', 'all', 'all_rot')


class SubjectOnlineAdapter:
    def __init__(self, baseline_models, classes, *, seed, use_history=True):
        if set(baseline_models) != set(VARIANTS):
            raise ValueError('All four original branches required.')
        self.baseline = baseline_models
        self.classes, self.seed, self.use_history = classes, int(seed), use_history
        self.subjects, self.seen, self.trace = {}, set(), []
        self.prediction_seconds = []
        for model in self.baseline.values():
            model.eval()

    def _state(self, subject):
        if subject not in self.subjects:
            self.subjects[subject] = {'models': {}, 'buffers': {v: [] for v in VARIANTS},
                'decoder': CausalJointAssignment(self.classes), 'arrived': [], 'updates': {v: 0 for v in VARIANTS}}
        return self.subjects[subject]

    def predict_one(self, sequence_id, subject, features):
        sid, subject = str(sequence_id), str(subject)
        x = np.asarray(features, dtype=np.float32)
        if sid in self.seen or x.ndim != 2 or x.shape[1] != 335 or not 1 <= len(x) <= 200 or not np.isfinite(x).all():
            raise ValueError('Unique arrived ID and finite 200-frame source features required.')
        self.seen.add(sid)
        state = self._state(subject)
        state['arrived'].append(sid)
        no_tof, no_rot = bool((x[:, 15:] == 0).all()), bool((x[:, 3:15] == 0).all())
        variant = ('imu' if no_tof else 'all') + ('_rot' if no_rot else '')
        model = state['models'].get(variant, self.baseline[variant])
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(torch.from_numpy(x[:, :15] if no_tof else x).unsqueeze(0),
                           torch.tensor([len(x)]), None)['gesture_logits'][0].numpy().copy()
        self.prediction_seconds.append({'variant': variant, 'seconds': time.perf_counter() - start})
        if logits.shape != (self.classes,) or not np.isfinite(logits).all():
            raise ValueError('Invalid online joint output.')
        decision = (state['decoder'].predict_one(subject, logits) if self.use_history else int(logits.argmax()))
        # The chosen class is fixed before the update and is never revised.
        imu = x[:, :15].copy()
        zero_rot = imu.copy()
        zero_rot[:, 3:] = 0
        eligible = {'imu_rot': zero_rot}
        if not no_rot:
            eligible['imu'] = imu
        if not no_tof:
            full_rot = x.copy()
            full_rot[:, 3:15] = 0
            eligible['all_rot'] = full_rot
            if not no_rot:
                eligible['all'] = x.copy()
        for branch, value in eligible.items():
            state['buffers'][branch].append((sid, value, decision))
        for branch in VARIANTS:
            if len(state['buffers'][branch]) >= 32:
                self._update(subject, state, branch)
        return decision, logits

    def _update(self, subject, state, variant):
        buffer = state['buffers'][variant]
        if len(buffer) != 32 or not set(item[0] for item in buffer).issubset(state['arrived']):
            raise ValueError('Updates require 32 same-subject already-arrived samples.')
        if variant not in state['models']:
            state['models'][variant] = copy.deepcopy(self.baseline[variant])
        model = state['models'][variant]
        lengths = torch.tensor([len(x) for _, x, _ in buffer], dtype=torch.long)
        padded = torch.zeros(32, int(lengths.max()), buffer[0][1].shape[1])
        for i, (_, value, _) in enumerate(buffer):
            padded[i, :len(value)] = torch.from_numpy(value)
        targets = torch.tensor([label for _, _, label in buffer], dtype=torch.long)
        # RNG isolation makes other subjects' interleaving unable to affect this
        # subject's dropout or updates. Seeds depend only on fixed protocol IDs.
        rng_key = f'{self.seed}|{subject}|{variant}|{state["updates"][variant]}'
        rng_seed = int(hashlib.sha256(rng_key.encode()).hexdigest()[:15], 16)
        start = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(rng_seed)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(padded, lengths, None)['gesture_logits'], targets)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite pseudo-label loss.')
            loss.backward()
            optimizer.step()
            model.eval()
        self.trace.append({'subject': subject, 'variant': variant,
            'trigger_sequence_id': state['arrived'][-1], 'arrival_count': len(state['arrived']),
            'sequence_ids': [sid for sid, _, _ in buffer], 'pseudo_joint_labels': targets.tolist(),
            'update_number': state['updates'][variant], 'seed': rng_seed,
            'seconds': time.perf_counter() - start, 'loss': float(loss.detach())})
        state['updates'][variant] += 1
        buffer.clear()
