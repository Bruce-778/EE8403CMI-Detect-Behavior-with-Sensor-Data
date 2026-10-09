"""Five-fold IMU auxiliary-loss ablation on project-owned second-place code.

Only IMU/IMU-rotation-missing branches are retrained. Original ToF branches are
reused during the separate local routed comparison. All 50 epochs are retained.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmi_project.second_place import (ARCHITECTURES, ReferenceDataset, build_cache,
    collapse_probabilities, fit_joint_labels, load_reference)
from cmi_project.evaluation import (ALL_GESTURES, PROBABILITY_COLUMNS, cmi_metrics,
    evaluate_oof_frames, write_fold_predictions)
from cmi_project.validation import load_fold_manifest
from run_winner_experiments import verify_training_data
from cmi_project.second_place_hybrid import JointMetricLoss

VARIANTS = {'imu': (False, False), 'imu_rot': (False, True),
            'all': (True, False), 'all_rot': (True, True)}


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def schedule_factor(step, total):
    warmup = int(total * .1)
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0., .5 * (1 + math.cos(math.pi * progress)))


def make_predictions(model, dataset, reference, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=reference.collate_fn)
    model.eval()
    logits = []
    with torch.inference_mode():
        for x, lengths, _, _ in loader:
            # No true phase/gesture/orientation enters the model.
            out = model(x.to(device), lengths.to(device), None)['gesture_logits']
            logits.append(out.cpu().numpy())
    result = np.concatenate(logits)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite reference prediction.')
    return result


def prediction_frame(metadata, logits, labels, fingerprint):
    probability = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    collapsed = collapse_probabilities(probability, labels)
    # Upstream inference chooses argmax JOINT class then maps to gesture.
    # Marginal 18D probabilities are exported for analysis, not used to change it.
    frame = pd.DataFrame({'sequence_id': metadata.sequence_id.to_numpy(),
        'predicted_gesture': [labels[i][1] for i in logits.argmax(axis=1)],
        'folds_sha256': fingerprint})
    for i, column in enumerate(PROBABILITY_COLUMNS):
        frame[column] = collapsed[:, i]
    return frame


def train_one(args, fold, architecture, variant, cache, reference, manifest, metadata, labels):
    train, val = manifest.split(fold)
    train_indices = np.flatnonzero(metadata.sequence_id.isin(train.sequence_id))
    val_indices = np.flatnonzero(metadata.sequence_id.isin(val.sequence_id))
    use_tof, rotation_zero = VARIANTS[variant]
    torch.manual_seed(42 + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + fold)
    np.random.seed(42 + fold)
    random.seed(42 + fold)
    model_type = ARCHITECTURES[architecture][int(use_tof)]
    model = getattr(reference.models, model_type)(input_size=15, n_classes=len(labels)).to(args.device)
    criterion = JointMetricLoss(labels, macro_weight=.5, binary_weight=.1).to(args.device)
    output = args.output / architecture / variant / f'fold_{fold}'
    if output.exists():
        raise ValueError(f'Completed/partial arm exists; preserve it: {output}')
    output.mkdir(parents=True)
    source_ds = ReferenceDataset(cache, train_indices, labels, use_tof=use_tof,
                                rotation_zero=rotation_zero, training=True)
    training_ds = reference.MixupDataset(source_ds, alpha=.5, num_classes=len(labels))
    validation_ds = ReferenceDataset(cache, val_indices, labels, use_tof=use_tof,
                                    rotation_zero=rotation_zero, training=False)
    loader = DataLoader(training_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, collate_fn=reference.collate_fn)
    if not len(loader):
        raise ValueError('Insufficient training sequences for the published batch size.')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    total = len(loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_factor(step, total))
    history = []
    provenance = {'source_commit': reference.provenance['commit'], 'source_files': reference.provenance['files'],
        'folds_sha256': manifest.fingerprint, 'fold': fold, 'architecture': architecture, 'variant': variant,
        'train_sequence_ids': sorted(train.sequence_id), 'train_subjects': sorted(set(train.subject)),
        'validation_sequence_ids': sorted(val.sequence_id), 'validation_subjects': sorted(set(val.subject)),
        'joint_labels': labels, 'seed': 42 + fold, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'validation_logit_sequence_ids': metadata.iloc[val_indices].sequence_id.tolist(),
        'runtime': {'torch': str(torch.__version__), 'numpy': str(np.__version__), 'device': args.device},
        'optimizer': 'Adam lr=.001 wd=.0001', 'scheduler': '10% step warmup, cosine',
        'mixup_alpha': .5, 'phase_loss_weight': 1., 'gradient_clip_norm': 1.,
        'checkpoint_selection': 'last epoch as upstream test.py; no early stopping',
        'adaptations': ['fixed project folds', 'train-only joint ontology', 'copy before augmentation',
                        'special-subject correction applied once', 'plain PyTorch training, no external logger'],
        'prediction_rule': 'argmax joint logits -> gesture; marginal gesture probabilities are diagnostic only'}
    provenance['hybrid_experiment'] = 'base_imu_official_group_loss_v1'
    provenance['auxiliary_loss'] = {'macro9_weight': .5, 'binary_weight': .1,
        'targets': 'training joint soft Mixup targets collapsed by gesture; no extra label smoothing'}
    provenance['adaptations'].append('Project hybrid: official9 CE and binary BCE added to original joint CE; phase CE unchanged')
    dump(output / 'provenance.json', provenance)
    print(f'SECOND PLACE {architecture}/{variant} fold {fold}: {len(train_indices)} train/{len(val_indices)} val, {len(labels)} classes, {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        losses = []
        for x, lengths, target, phases in loader:
            x, lengths, target, phases = [v.to(args.device) for v in (x, lengths, target, phases)]
            optimizer.zero_grad(set_to_none=True)
            out = model(x, lengths, None)
            classification = criterion(out['gesture_logits'], target)
            phase_logits = out['phase_logits'].reshape(-1, 3)
            phase = phases.reshape(-1)
            valid_phase = phase >= 0
            phase_loss = torch.nn.functional.cross_entropy(phase_logits[valid_phase], phase[valid_phase]) if valid_phase.any() else phase_logits.sum() * 0
            loss = classification + phase_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; stop and preserve evidence.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        logits = make_predictions(model, validation_ds, reference, args.device, args.batch_size)
        frame = prediction_frame(metadata.iloc[val_indices], logits, labels, manifest.fingerprint)
        metrics = cmi_metrics(metadata.iloc[val_indices].gesture, frame.predicted_gesture)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'seconds': time.perf_counter() - start,
                        'lr': optimizer.param_groups[0]['lr'], **metrics})
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(f'{architecture}/{variant} fold {fold} epoch {epoch:02d}/{args.epochs}: CMI={metrics["score"]:.6f}, loss={history[-1]["loss"]:.4f}, {history[-1]["seconds"]:.1f}s', flush=True)
        torch.save({'model_state': model.state_dict(), 'provenance': provenance, 'epoch': epoch}, output / 'last.pt')
        np.save(output / 'joint_logits.npy', logits)
        np.save(output / 'joint_sequence_ids.npy', metadata.iloc[val_indices].sequence_id.to_numpy(dtype=str))
        write_fold_predictions(manifest, fold, frame, output / 'predictions.csv')
        dump(output / 'metrics.json', {'scope': 'held-out single fold' if len(args.folds) < 5 else 'individual held-out fold',
                                      'epoch': epoch, 'validation_sequences': len(val_indices), **metrics})
    source_ds.close()
    validation_ds.close()
    return output


def routed_evaluation(args, reference, manifest, metadata, labels, fold):
    val_indices = np.flatnonzero(metadata.fold == fold)
    rows = metadata.iloc[val_indices].reset_index(drop=True)
    for scenario in ['observed', 'aux_dropout50', 'imu_only']:
        drop = np.array([scenario == 'imu_only' or (scenario == 'aux_dropout50' and
            int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16) % 2 == 0) for sid in rows.sequence_id])
        use_full = ~rows.tof_missing.to_numpy() & ~drop
        rotation_missing = rows.rotation_missing.to_numpy()
        combined = []
        for arch in args.architectures:
            arrays = {v: np.load(args.output / arch / v / f'fold_{fold}/joint_logits.npy') for v in VARIANTS}
            logits = arrays['imu'].copy()
            for variant, (full, rot) in VARIANTS.items():
                select = (use_full == full) & (rotation_missing == rot)
                logits[select] = arrays[variant][select]
            combined.append(logits)
        logits = np.mean(combined, axis=0)
        frame = prediction_frame(rows, logits, labels, manifest.fingerprint)
        directory = args.output / 'routed' / scenario / f'fold_{fold}'
        write_fold_predictions(manifest, fold, frame, directory / 'predictions.csv')
        np.save(directory / 'joint_logits.npy', logits)
        np.save(directory / 'joint_sequence_ids.npy', rows.sequence_id.to_numpy(dtype=str))
        metrics = cmi_metrics(rows.gesture, frame.predicted_gesture)
        dump(directory / 'metrics.json', metrics)
        print(f'ROUTED fold {fold} {scenario}: {metrics["score"]:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--reference', type=Path, default=ROOT / 'src/cmi_project/vendor/second_place')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/second_place/hybrid_imu_group_loss_v1')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--architectures', nargs='+', choices=['base'], default=['base'])
    parser.add_argument('--variants', nargs='+', choices=['imu', 'imu_rot'], default=['imu', 'imu_rot'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.epochs != 50 or args.batch_size != 32 or set(args.folds) != set(range(5)) or set(args.variants) != {'imu', 'imu_rot'}:
        raise ValueError('This registered ablation requires all five folds, both IMU branches, 50 epochs and batch32.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; do not pretend CPU is the GPU reproduction.')
    if len(set(args.folds)) != len(args.folds) or set(args.folds) - set(range(5)) or args.epochs < 1 or args.batch_size < 2:
        raise ValueError('Invalid fold/budget selection.')
    args.output = args.output.resolve()
    if (args.output.exists() and any(args.output.iterdir())) or ROOT.resolve() not in args.output.parents or (ROOT / 'data').resolve() in args.output.parents:
        raise ValueError('Use a fresh experiment output inside project, outside raw data.')
    torch.set_num_threads(4)
    verify_training_data(args.data_dir.resolve())
    reference = load_reference(args.reference, ROOT / 'configs/second_place_source.json')
    manifest = load_fold_manifest(ROOT / 'configs/folds.csv')
    cache = args.output / 'cache'
    args.output.mkdir(parents=True)
    try:
        build_cache(args.data_dir, cache, manifest, reference)
        metadata = pd.read_csv(cache / 'metadata.csv')
        for fold in args.folds:
            train, _ = manifest.split(fold)
            labels = fit_joint_labels(metadata, train.sequence_id)
            for arch in args.architectures:
                for variant in args.variants:
                    train_one(args, fold, arch, variant, cache, reference, manifest, metadata, labels)
            if set(args.variants) == set(VARIANTS):
                routed_evaluation(args, reference, manifest, metadata, labels, fold)
        if set(args.folds) == set(range(5)) and set(args.variants) == set(VARIANTS):
            for scenario in ['observed', 'aux_dropout50', 'imu_only']:
                frames = {fold: pd.read_csv(args.output / 'routed' / scenario / f'fold_{fold}/predictions.csv') for fold in range(5)}
                evaluate_oof_frames(manifest, frames, args.output / 'routed' / scenario / 'evaluation', experiment_name=f'second_place_{scenario}')
    finally:
        # Exclude multi-GB training cache; preserve partial artifacts on failure.
        archive = ROOT.parent / 'second_place_hybrid_experiments.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in args.output.rglob('*'):
                if path.is_file() and (cache not in path.parents or path.name in {'metadata.csv', 'provenance.json'}):
                    bundle.write(path, path.relative_to(ROOT).as_posix())
            for name in ['folds.csv', 'folds.meta.json', 'second_place_source.json', 'data_source_hashes.json']:
                bundle.write(ROOT / 'configs' / name, f'configs/{name}')
            if (ROOT / 'data_source_check.json').exists():
                bundle.write(ROOT / 'data_source_check.json', 'data_source_check.json')
        print('SAVED', archive, archive.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
