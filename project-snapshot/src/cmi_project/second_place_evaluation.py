"""Strict result alignment and causal joint-class history for reference trials."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .evaluation import PROBABILITY_COLUMNS, _checked_predictions, cmi_metrics
from .second_place import collapse_probabilities, fit_joint_labels


class CausalJointAssignment:
    """Upstream capacity-one Hungarian decision, returning only the latest row.

    The original asserts when a subject exceeds the joint ontology size. We
    preserve coverage with explicit baseline fallbacks for unsupported prefixes.
    No future rows, true orientation, phase, or gesture enters this predictor.
    """
    def __init__(self, classes):
        if not isinstance(classes, int) or classes < 1:
            raise ValueError('Positive joint-class count required.')
        self.classes, self.history, self.overflow = classes, {}, 0

    def predict_one(self, subject, logits):
        vector = np.asarray(logits, dtype=np.float64)
        if vector.shape != (self.classes,) or not np.isfinite(vector).all():
            raise ValueError('Invalid joint logits.')
        arrived = self.history.setdefault(str(subject), [])
        if len(arrived) >= self.classes:
            self.overflow += 1
            return int(vector.argmax())
        arrived.append(vector.copy())
        scores = np.stack(arrived)
        row, column = linear_sum_assignment(scores.max() - scores)
        chosen = column[row == len(arrived) - 1]
        if len(chosen) != 1:
            raise ValueError('Latest row was not assigned.')
        return int(chosen[0])


def read_aligned_logits(directory, metadata, fold):
    """Recover v1 raw-order rows from its cache metadata, never guess CSV order."""
    path = Path(directory)
    logits = np.load(path / 'joint_logits.npy', allow_pickle=False)
    id_path = path / 'joint_sequence_ids.npy'
    ids = (np.load(id_path, allow_pickle=False).astype(str) if id_path.exists()
           else metadata.loc[metadata.fold == fold, 'sequence_id'].to_numpy(dtype=str))
    expected = metadata.loc[metadata.fold == fold, 'sequence_id'].astype(str)
    if (logits.ndim != 2 or len(logits) != len(ids) or len(set(ids)) != len(ids)
            or set(ids) != set(expected) or not np.isfinite(logits).all()):
        raise ValueError('Joint-logit sequence alignment/coverage is invalid.')
    # A future archive with explicit IDs still has to agree with its metadata.
    return pd.Index(ids), logits


def check_arm(directory, manifest, metadata, source, *, epochs=50):
    path = Path(directory)
    provenance = json.loads((path / 'provenance.json').read_text(encoding='utf-8'))
    fold = provenance['fold']
    train, validation = manifest.split(fold)
    for key, expected in [('train_sequence_ids', train.sequence_id), ('train_subjects', set(train.subject)),
                          ('validation_sequence_ids', validation.sequence_id),
                          ('validation_subjects', set(validation.subject))]:
        if provenance.get(key) != sorted(expected):
            raise ValueError(f'Actual training/validation {key} differs from fixed folds.')
    if (provenance.get('folds_sha256') != manifest.fingerprint or provenance.get('source_commit') != source['commit']
            or provenance.get('source_files') != source['files'] or provenance.get('epochs') != epochs
            or provenance.get('checkpoint_selection') != 'last epoch as upstream test.py; no early stopping'):
        raise ValueError('Source, fold, training budget, or checkpoint rule mismatch.')
    labels = [tuple(x) for x in provenance['joint_labels']]
    if labels != fit_joint_labels(metadata, train.sequence_id):
        raise ValueError('Joint ontology differs from training-only labels.')
    ids, logits = read_aligned_logits(path, metadata, fold)
    if logits.shape[1] != len(labels):
        raise ValueError('Joint output dimension differs from trained ontology.')
    if 'validation_logit_sequence_ids' in provenance and provenance['validation_logit_sequence_ids'] != ids.tolist():
        raise ValueError('Provenance/logit row order mismatch.')
    frame = _checked_predictions(manifest, fold, pd.read_csv(path / 'predictions.csv'), require_fingerprint=True)
    index = ids.get_indexer(frame.sequence_id)
    aligned = logits[index]
    prediction = np.asarray([labels[i][1] for i in aligned.argmax(axis=1)])
    shifted = aligned - aligned.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    if (not np.array_equal(prediction, frame.predicted_gesture)
            or not np.allclose(collapse_probabilities(probability, labels), frame[PROBABILITY_COLUMNS], atol=1e-6, rtol=0)):
        raise ValueError('Saved predictions disagree with joint logits.')
    metrics = cmi_metrics(frame.gesture, frame.predicted_gesture)
    saved = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
    if saved.get('epoch') != epochs or any(not np.isclose(metrics[k], saved[k], atol=1e-12, rtol=0) for k in metrics):
        raise ValueError('Reported final epoch/metric mismatch.')
    history = pd.read_csv(path / 'history.csv')
    if (history.epoch.tolist() != list(range(1, epochs + 1))
            or not np.isfinite(history[['loss', 'seconds', 'lr', *metrics]]).all().all()
            or any(not np.isclose(history.iloc[-1][k], metrics[k], atol=1e-12, rtol=0) for k in metrics)):
        raise ValueError('Incomplete training history.')
    # Check CPU-loadable saved weights and checkpoint identity, not just CSV claims.
    import torch
    checkpoint = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
    normalized = json.loads(json.dumps(checkpoint['provenance']))
    if checkpoint['epoch'] != epochs or normalized != provenance:
        raise ValueError('Checkpoint disagrees with audited training provenance.')
    if not checkpoint['model_state'] or any(not torch.isfinite(x).all() for x in checkpoint['model_state'].values()):
        raise ValueError('Invalid saved model weights.')
    return {'fold': fold, 'architecture': provenance['architecture'], 'variant': provenance['variant'],
            'sequences': len(frame), 'joint_classes': len(labels), **metrics}, ids, logits, labels


def causal_decode(ids, logits, subjects, labels, fingerprint, seed):
    """Fixed order depends on IDs and seed only; all returned decisions stay fixed."""
    ids = pd.Index(ids.astype(str))
    sorted_indices = np.argsort(ids.to_numpy())
    order = sorted_indices[np.random.default_rng(seed).permutation(len(ids))]
    decoder = CausalJointAssignment(len(labels))
    decisions, positions = np.empty(len(ids), dtype=int), np.empty(len(ids), dtype=int)
    for position, index in enumerate(order):
        decisions[index] = decoder.predict_one(subjects.loc[ids[index]], logits[index])
        positions[index] = position
    frame = pd.DataFrame({'sequence_id': ids, 'predicted_gesture': [labels[i][1] for i in decisions],
                          'folds_sha256': fingerprint, 'arrival_position': positions})
    return frame, decoder.overflow
