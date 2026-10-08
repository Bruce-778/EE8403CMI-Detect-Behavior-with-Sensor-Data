"""Paired IMU fine-tuning and privileged multisensor knowledge distillation."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cnn_data import ARRAY_KEYS, CNNTensorDataset, validate_arrays
from .cnn_training import (CMIHierarchicalLoss, EarlyStopping, TrainingConfig,
                           choose_device, evaluate_model, load_cnn_checkpoint, seed_everything)
from .evaluation import ALL_GESTURES, PROBABILITY_COLUMNS, write_fold_predictions, _checked_predictions
from .preprocessing import SensorDropoutConfig, preprocessor_states_equal
from .validation import assert_preprocessor_matches


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float = 2.0
    weight: float = 0.5

    def __post_init__(self):
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Distillation temperature must be positive and finite.")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Distillation weight must be finite and nonnegative.")


def distillation_loss(student_logits, teacher_logits, eligible, temperature):
    """T² KL(teacher || student), averaged over the entire training batch."""
    eligible = eligible.bool()
    if teacher_logits.shape != student_logits.shape or eligible.shape != student_logits.shape[:1]:
        raise ValueError("Teacher/student logits or eligibility shapes disagree.")
    # Unavailable teacher rows have no contribution, even if their logits are NaN.
    clean = torch.where(eligible[:, None], teacher_logits.detach(), 0)
    target = (clean / temperature).softmax(-1)
    log_student = (student_logits / temperature).log_softmax(-1)
    per_sample = nn.functional.kl_div(log_student, target, reduction="none").sum(-1)
    return torch.where(eligible, per_sample, 0).mean() * temperature ** 2


def checked_pair(student_path, teacher_path, manifest, fold):
    student, processor, checkpoint = load_cnn_checkpoint(student_path)
    teacher, teacher_processor, teacher_checkpoint = load_cnn_checkpoint(teacher_path)
    for model, saved_processor, saved, expected in (
            (student, processor, checkpoint, "imu"),
            (teacher, teacher_processor, teacher_checkpoint, "multisensor")):
        assert_preprocessor_matches(saved_processor.state, manifest, fold)
        if saved["fold"] != fold or saved["folds_sha256"] != manifest.fingerprint or model.model_name != expected:
            raise ValueError("Use same-fold IMU student and multisensor teacher.")
    if (not preprocessor_states_equal(processor.state, teacher_processor.state)
            or student.tof_regions != teacher.tof_regions
            or checkpoint["data_metadata"]["identity"] != teacher_checkpoint["data_metadata"]["identity"]):
        raise ValueError("Teacher/student must have identical fold-fitted input parameters.")
    teacher.eval().requires_grad_(False)
    return student, teacher, processor, checkpoint


def teacher_training_targets(teacher, arrays, manifest, fold, *, batch_size=64, device="cpu"):
    """Only training sequence IDs receive teacher targets; validation is excluded."""
    indices = np.flatnonzero(manifest.table["fold"].to_numpy() != fold)
    loader = DataLoader(CNNTensorDataset(arrays, indices), batch_size=batch_size, shuffle=False)
    teacher = teacher.to(device).eval().requires_grad_(False)
    parts = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: batch[key].to(device) for key in ARRAY_KEYS}
            logits = teacher(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite teacher training targets.")
            parts.append(logits.cpu().numpy())
    targets = np.full((len(manifest.table), len(ALL_GESTURES)), np.nan, dtype=np.float32)
    targets[indices] = np.concatenate(parts)
    time_mask = arrays["time_mask"]
    eligible = ((arrays["thm_valid"] & time_mask[..., None]).any(axis=(1, 2))
                | (arrays["tof_sensor_present"] & time_mask[..., None]).any(axis=(1, 2)))
    eligible[manifest.table["fold"].to_numpy() == fold] = False
    return targets, eligible


class DistillationDataset(Dataset):
    def __init__(self, arrays, indices, targets, eligible, *, seed, dropout):
        self.base = CNNTensorDataset(arrays, indices, training=True, seed=seed, dropout=dropout)
        self.targets, self.eligible = targets, eligible
        if not np.isfinite(targets[self.base.indices]).all():
            raise ValueError("Training targets missing: validation rows cannot enter distillation.")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        position = self.base.indices[index]
        sample["teacher_logits"] = torch.from_numpy(self.targets[position].copy())
        sample["teacher_eligible"] = torch.tensor(bool(self.eligible[position]))
        return sample


def verify_starting_predictions(student, arrays, manifest, fold, source, device, criterion, batch_size):
    indices = np.flatnonzero(manifest.table["fold"].to_numpy() == fold)
    loader = DataLoader(CNNTensorDataset(arrays, indices), batch_size=batch_size)
    metrics, probability = evaluate_model(student.to(device), loader, criterion, device)
    original = _checked_predictions(manifest, fold, pd.read_csv(Path(source).with_name("predictions.csv")),
                                    require_fingerprint=True).set_index("sequence_id")
    expected = original.loc[arrays["sequence_id"][indices], PROBABILITY_COLUMNS].to_numpy()
    np.testing.assert_allclose(probability, expected, rtol=2e-5, atol=2e-6)
    if not np.array_equal(probability.argmax(1), expected.argmax(1)):
        raise ValueError("CPU starting model changed held-out predictions.")
    return metrics, float(np.abs(probability - expected).max())


def train_posttraining_fold(arrays, processor, manifest, fold, student_path, output_dir, *,
                            training: TrainingConfig, distillation: DistillationConfig,
                            targets, eligible, data_metadata, source_teacher,
                            save_plots=True):
    """Epoch 0 is eligible for checkpoint selection, preserving an unimproved start."""
    if training.mixup_probability:
        raise ValueError("This controlled post-training stage disables Mixup for both arms.")
    assert_preprocessor_matches(processor.state, manifest, fold)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Post-training output must be a fresh directory.")
    seed_everything(training.seed + fold, training.cpu_threads)
    device = choose_device(training.device)
    student, source_processor, source_checkpoint = load_cnn_checkpoint(student_path, device=str(device))
    if (student.model_name != "imu" or source_checkpoint["fold"] != fold
            or source_checkpoint["folds_sha256"] != manifest.fingerprint
            or not preprocessor_states_equal(source_processor.state, processor.state)):
        raise ValueError("Starting student differs from requested fold/input parameters.")
    validate_arrays(arrays, manifest, processor.max_length, student.tof_regions)
    train_indices = np.flatnonzero(manifest.table["fold"].to_numpy() != fold)
    val_indices = np.flatnonzero(manifest.table["fold"].to_numpy() == fold)
    if (source_teacher.get("teacher_fold") != fold
            or targets.shape != (len(manifest.table), len(ALL_GESTURES))
            or eligible.shape != (len(manifest.table),)
            or not np.isnan(targets[val_indices]).all() or eligible[val_indices].any()):
        raise ValueError("Teacher targets must come from this fold and exclude validation sequences.")
    dropout = SensorDropoutConfig(**source_checkpoint["sensor_dropout"])
    train_loader = DataLoader(DistillationDataset(arrays, train_indices, targets, eligible,
        seed=training.seed + fold, dropout=dropout), batch_size=training.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(training.seed + fold))
    val_loader = DataLoader(CNNTensorDataset(arrays, val_indices), batch_size=training.batch_size)
    criterion = CMIHierarchicalLoss(training)
    baseline, error = verify_starting_predictions(student, arrays, manifest, fold, student_path,
                                                 device, criterion, training.batch_size)
    provenance = {"method": "multisensor_to_imu_kd" if distillation.weight else "supervised_finetuning_control",
                  "student_sha256": hashlib.sha256(Path(student_path).read_bytes()).hexdigest(),
                  **source_teacher, "distillation": asdict(distillation),
                  "teacher_target_sequences": len(train_indices),
                  "eligible_training_sequences": int(eligible[train_indices].sum()),
                  "teacher_validation_target_sequences": 0,
                  "teacher_view": "original observed sensors; student view has train-only sensor dropout",
                  "teacher_bn": "frozen eval; teacher targets detached",
                  "starting_probability_max_error": error}
    output_dir.mkdir(parents=True)
    processor.save(output_dir / "preprocessor.json")

    def save_checkpoint(epoch, metrics):
        torch.save({"version": 1, "state_dict": student.state_dict(), "label_order": list(ALL_GESTURES),
            "model_metadata": student.metadata(), "preprocessor": processor.state,
            "fold": fold, "folds_sha256": manifest.fingerprint, "best_epoch": epoch,
            "training_config": asdict(training), "sensor_dropout": asdict(dropout),
            "validation_metrics": metrics, "data_metadata": data_metadata,
            "posttraining": provenance}, output_dir / "best.pt")

    save_checkpoint(0, baseline)
    stopping = EarlyStopping(training.early_stopping_patience, training.min_delta)
    stopping.update(baseline["score"], 0)
    optimizer = torch.optim.AdamW(student.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
        factor=training.lr_factor, patience=training.lr_patience, threshold=training.min_delta,
        threshold_mode="abs", min_lr=training.min_lr)
    history, started = [], time.perf_counter()
    print(f"POSTTRAIN fold {fold}: KD weight={distillation.weight}, baseline={baseline['score']:.6f}", flush=True)
    for epoch in range(1, training.epochs + 1):
        student.train()
        totals, count = np.zeros(3), 0
        used_lr = optimizer.param_groups[0]["lr"]
        epoch_start = time.perf_counter()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = student(batch)
            supervised = criterion(logits, batch["label"])
            kd = distillation_loss(logits, batch["teacher_logits"], batch["teacher_eligible"],
                                   distillation.temperature) if distillation.weight else logits.sum() * 0
            loss = supervised + distillation.weight * kd
            if not torch.isfinite(loss):
                raise ValueError("Non-finite post-training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), training.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            size = len(batch["label"])
            totals += np.array([float(loss.detach()), float(supervised.detach()), float(kd.detach())]) * size
            count += size
        metrics, _ = evaluate_model(student, val_loader, criterion, device)
        improved, stop = stopping.update(metrics["score"], epoch)
        if improved:
            save_checkpoint(epoch, metrics)
        scheduler.step(metrics["score"])
        row = {"epoch": epoch, "train_loss": totals[0] / count, "supervised_loss": totals[1] / count,
               "distillation_loss": totals[2] / count, "validation_loss": metrics["loss"],
               **{key: value for key, value in metrics.items() if key != "loss"},
               "learning_rate": used_lr, "seconds": time.perf_counter() - epoch_start,
               "best_checkpoint": improved}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"KD={distillation.weight} fold {fold} epoch {epoch}: CMI={metrics['score']:.6f}, "
              f"macro={metrics['macro_f1_9class']:.6f}, KDloss={row['distillation_loss']:.4f}, "
              f"{row['seconds']:.1f}s{' *' if improved else ''}", flush=True)
        if stop:
            break
    restored, _, checkpoint = load_cnn_checkpoint(output_dir / "best.pt", device=str(device))
    metrics, probability = evaluate_model(restored, val_loader, criterion, device)
    np.testing.assert_allclose(metrics["score"], checkpoint["validation_metrics"]["score"], rtol=0, atol=1e-12)
    predictions = pd.DataFrame({"sequence_id": arrays["sequence_id"][val_indices],
        "predicted_gesture": np.asarray(ALL_GESTURES)[probability.argmax(1)],
        "folds_sha256": manifest.fingerprint})
    predictions = pd.concat([predictions, pd.DataFrame(probability, columns=PROBABILITY_COLUMNS)], axis=1)
    write_fold_predictions(manifest, fold, predictions, output_dir / "predictions.csv", preprocessor_state=processor.state)
    result = {"model": "imu", "fold": fold, "folds_sha256": manifest.fingerprint,
              "scope": "Development held-out subject fold; original source and post-training checkpoints selected on validation",
              "baseline": baseline, "validation": metrics, "best_epoch": checkpoint["best_epoch"],
              "epochs_run": len(history), "training": asdict(training), "posttraining": provenance,
              "elapsed_seconds": time.perf_counter() - started, "device": str(device),
              "max_length": processor.max_length, "model_metadata": restored.metadata()}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if save_plots:
        from .cnn_training import plot_history
        plot_history(pd.DataFrame(history), output_dir / "learning_curve.png", checkpoint["best_epoch"])
    return result, predictions
