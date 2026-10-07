"""Reproducible v1 CNN training and held-out-subject evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cnn import CMI1DCNN, CNNConfig
from .cnn_data import CNNTensorDataset, prepare_cnn_fold, validate_arrays
from .dataset_analysis import PROJECT_DIR
from .evaluation import (ALL_GESTURES, METRIC_NAME, PROBABILITY_COLUMNS, cmi_metrics,
                         evaluate_oof_frames, write_fold_predictions, _checked_predictions)
from .preprocessing import FoldPreprocessor, PreprocessingConfig, SensorDropoutConfig
from .validation import FoldManifest, assert_preprocessor_matches, load_fold_manifest


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 45
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    label_smoothing: float = 0.03
    macro_loss_weight: float = 0.0
    binary_loss_weight: float = 0.0
    early_stopping_patience: int = 10
    min_delta: float = 0.0001
    lr_patience: int = 3
    lr_factor: float = 0.5
    min_lr: float = 0.00001
    gradient_clip: float = 1.0
    cpu_threads: int = 4
    device: str = "auto"

    def __post_init__(self):
        for name in ("epochs", "batch_size", "early_stopping_patience", "cpu_threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("learning_rate", "min_lr", "gradient_clip"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0 or not 0 <= self.label_smoothing < 1:
            raise ValueError("Invalid weight_decay/label_smoothing.")
        for name in ("macro_loss_weight", "binary_loss_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0.")
        if not 0 < self.lr_factor < 1 or self.lr_patience < 0 or self.min_lr > self.learning_rate:
            raise ValueError("Invalid learning rate scheduler settings.")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and >= 0.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda.")


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0001
    best_score: float = -float("inf")
    reference_score: float = -float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, score: float, epoch: int) -> tuple[bool, bool]:
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score.")
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch = score, epoch
        if score > self.reference_score + self.min_delta:
            self.reference_score, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


class CMIHierarchicalLoss(nn.Module):
    """18-class CE plus optional losses on the official label groupings.

    logsumexp aggregates probabilities, not average logits. No extra head or
    train-only phase/orientation label is required. Zero weights preserve v1.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.smoothing = config.label_smoothing
        self.macro_weight = config.macro_loss_weight
        self.binary_weight = config.binary_loss_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.cross_entropy(logits, target, label_smoothing=self.smoothing)
        if self.macro_weight or self.binary_weight:
            non_target = torch.logsumexp(logits[:, 8:], dim=1, keepdim=True)
            if self.macro_weight:
                nine_logits = torch.cat([logits[:, :8], non_target], dim=1)
                loss = loss + self.macro_weight * nn.functional.cross_entropy(
                    nine_logits, target.clamp_max(8), label_smoothing=self.smoothing)
            if self.binary_weight:
                odds = torch.logsumexp(logits[:, :8], dim=1) - non_target.squeeze(1)
                loss = loss + self.binary_weight * nn.functional.binary_cross_entropy_with_logits(
                    odds, (target < 8).to(logits.dtype))
        return loss


def seed_everything(seed: int, cpu_threads: int) -> None:
    # Required by deterministic CUDA matrix multiplication; set before CUDA use.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but this PyTorch environment has no CUDA device.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_model(model: CMI1DCNN, loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> tuple[dict, np.ndarray]:
    model.eval()
    probability_parts, target_parts, total_loss, count = [], [], 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = _move(batch, device)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("Non-finite validation logits/loss.")
            probability_parts.append(logits.softmax(dim=1).cpu().numpy())
            target_parts.append(batch["label"].cpu().numpy())
            total_loss += float(loss) * len(batch["label"])
            count += len(batch["label"])
    probabilities, targets = np.concatenate(probability_parts), np.concatenate(target_parts)
    labels = np.asarray(ALL_GESTURES)
    prediction = probabilities.argmax(axis=1)
    return {"loss": total_loss / count, "accuracy_18class": float((targets == prediction).mean()),
            **cmi_metrics(labels[targets], labels[prediction])}, probabilities


def load_cnn_checkpoint(path: Path, *, device: str = "cpu") -> tuple[CMI1DCNN, FoldPreprocessor, dict]:
    """Restore weights, class order, spatial pooling and train-fitted scalers."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("version") != 1 or checkpoint.get("label_order") != list(ALL_GESTURES):
        raise ValueError("Incompatible CNN checkpoint/class order.")
    metadata = checkpoint["model_metadata"]
    model = CMI1DCNN(metadata["model"], tof_regions=metadata["tof_regions"],
                     config=CNNConfig.from_dict(metadata["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    processor = FoldPreprocessor(PreprocessingConfig.from_dict(checkpoint["preprocessor"]["config"]))
    processor.state = checkpoint["preprocessor"]
    return model, processor, checkpoint


def plot_history(history: pd.DataFrame, path: Path, best_epoch: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["score"], label="CMI score")
    axes[1].plot(history["epoch"], history["macro_f1_9class"], label="Macro F1 (9 classes)")
    axes[1].plot(history["epoch"], history["binary_f1"], label="Binary F1")
    axes[1].axvline(best_epoch, color="grey", linestyle="--", label="best checkpoint")
    axes[1].set(xlabel="Epoch", ylabel="Held-out fold metric", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_comparison_report(output_dir: Path, summaries: list[dict], manifest: FoldManifest) -> None:
    """Compare only matching folds; a fold-0 preview is never a five-fold result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = sorted({s["fold"] for s in summaries})
    rows = [{"model": "CNN A (IMU)" if s["model"] == "imu" else "CNN B (IMU + THM + ToF)",
             "fold": s["fold"], **{k: s["validation"][k] for k in ("score", "binary_f1", "macro_f1_9class")},
             "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"]} for s in summaries]
    for name in ("lightgbm_imu", "xgboost_imu"):
        directory = PROJECT_DIR / "outputs/baseline" / name / "evaluation"
        if not (directory / "metrics.json").is_file() or not (directory / "fold_scores.csv").is_file():
            continue
        metadata = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metadata.get("folds_sha256") != manifest.fingerprint:
            continue
        scores = pd.read_csv(directory / "fold_scores.csv")
        for row in scores[scores["fold"].isin(folds)].to_dict("records"):
            rows.append({"model": name, **{k: row[k] for k in ("fold", "score", "binary_f1", "macro_f1_9class")}})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "baseline_comparison.csv", index=False)
    metrics = ["score", "binary_f1", "macro_f1_9class"]
    grouped = table.groupby("model", sort=False)[metrics].mean()
    figure, axis = plt.subplots(figsize=(10, 4.6))
    positions = np.arange(len(grouped))
    for i, (metric, label) in enumerate(zip(metrics, ("CMI score", "Binary F1", "Macro F1 (9 classes)"))):
        bars = axis.bar(positions + (i - 1) * 0.24, grouped[metric], 0.24, label=label)
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axis.set_xticks(positions, grouped.index, fontsize=8)
    axis.set(ylim=(0, 1.09), ylabel="Held-out subject score",
             title=f"CMI v1: matching folds {', '.join(map(str, folds))}")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(figure)
    complete = len(folds) == manifest.n_splits
    scope = "完整五折 OOF；表格逐折列出，图表为折均值。" if complete else f"第一版预览：仅 folds {folds}，不是完整五折结果。"
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>CMI 1D CNN v1</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#17212b}"
        "table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #dce2e8;padding:8px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}img{max-width:100%}code{background:#eef2f6;padding:2px 5px}</style>",
        "<h1>CMI 主实验 v1：1D CNN</h1>", f"<p>{html.escape(scope)}</p>",
        "<p>固定 subject folds，标准化和长度分位数仅由训练 subjects 拟合。CNN 的验证分数同时用于选择 checkpoint 和 early stopping；这是开发阶段验证结果。</p>",
        "<p>Model A：IMU encoder → masked mean/max pooling → FC。Model B：IMU / THM / ToF 独立 encoder → concatenate → FC。ToF 使用有效像素区域均值和有效像素比例后沿时间做 1D CNN。</p>",
        "<img src='comparison.png' alt='Matching-fold model comparison'>",
        table.to_html(index=False, float_format=lambda value: f"{value:.5f}", na_rep="—", border=0)]
    for summary in summaries:
        name, fold = summary["model"], summary["fold"]
        relative = f"{name}/fold_{fold}"
        parts.extend([f"<h2>{html.escape(name)} · fold {fold}</h2>",
            f"<p>sequence length={summary['max_length']}；最佳 epoch={summary['best_epoch']} / {summary['epochs_run']}；"
            f"参数={summary['model_metadata']['parameters']:,}；训练时间={summary['elapsed_seconds'] / 60:.1f} 分钟。</p>",
            f"<img src='{relative}/learning_curve.png' alt='Training and validation curves'>",
            f"<p><a href='{relative}/class_scores.csv'>逐类 precision / recall / F1</a> · "
            f"<a href='{relative}/confusion_matrix.csv'>混淆矩阵</a> · "
            f"<a href='{relative}/predictions.csv'>验证预测概率</a> · "
            f"<a href='{relative}/metrics.json'>完整参数和指标</a></p>"])
    parts.append("</html>")
    (output_dir / "report.html").write_text("\n".join(parts), encoding="utf-8")


def train_cnn_fold(arrays: dict, processor: FoldPreprocessor, manifest: FoldManifest,
                   fold: int, output_dir: Path, *, model_name: str = "imu", tof_regions: int = 2,
                   model_config: CNNConfig | None = None, training: TrainingConfig | None = None,
                   sensor_dropout: SensorDropoutConfig | None = None, data_metadata: dict | None = None,
                   save_plots: bool = True) -> tuple[dict, pd.DataFrame]:
    training, model_config = training or TrainingConfig(), model_config or CNNConfig()
    assert_preprocessor_matches(processor.state, manifest, fold)
    validate_arrays(arrays, manifest, processor.max_length, tof_regions)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("CNN fold output is non-empty; choose a fresh --output-dir.")
    seed_everything(training.seed + fold, training.cpu_threads)
    device = choose_device(training.device)
    model = CMI1DCNN(model_name, tof_regions=tof_regions, config=model_config).to(device)
    selected = manifest.table["fold"].to_numpy() == fold
    train_indices, val_indices = np.flatnonzero(~selected), np.flatnonzero(selected)
    generator = torch.Generator().manual_seed(training.seed + fold)
    train_loader = DataLoader(CNNTensorDataset(arrays, train_indices, training=True,
        seed=training.seed + fold, dropout=sensor_dropout), batch_size=training.batch_size,
        shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(CNNTensorDataset(arrays, val_indices), batch_size=training.batch_size,
                            shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=training.lr_factor,
        patience=training.lr_patience, threshold=training.min_delta, threshold_mode="abs", min_lr=training.min_lr)
    criterion = CMIHierarchicalLoss(training)
    stopping = EarlyStopping(training.early_stopping_patience, training.min_delta)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor.save(output_dir / "preprocessor.json")
    history, started = [], time.perf_counter()
    print(f"Training {model_name} fold {fold} on {device}: {len(train_indices):,} train / {len(val_indices):,} validation; "
          f"{model.metadata()['parameters']:,} parameters", flush=True)
    for epoch in range(1, training.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, count = 0.0, 0
        used_lr = optimizer.param_groups[0]["lr"]
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch["label"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite CNN training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch["label"])
            count += len(batch["label"])
        metrics, _ = evaluate_model(model, val_loader, criterion, device)
        improved, stop = stopping.update(metrics["score"], epoch)
        if improved:
            torch.save({"version": 1, "state_dict": model.state_dict(), "label_order": list(ALL_GESTURES),
                "model_metadata": model.metadata(), "training_config": asdict(training),
                "sensor_dropout": asdict(sensor_dropout or SensorDropoutConfig()),
                "preprocessor": processor.state, "data_metadata": data_metadata or {},
                "fold": fold, "folds_sha256": manifest.fingerprint, "best_epoch": epoch,
                "validation_metrics": metrics}, output_dir / "best.pt")
        scheduler.step(metrics["score"])
        row = {"epoch": epoch, "train_loss": total_loss / count, "validation_loss": metrics.pop("loss"),
               **metrics, "learning_rate": used_lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - epoch_start, "best_checkpoint": improved,
               "early_stopping_bad_epochs": stopping.bad_epochs}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"{model_name} fold {fold} epoch {epoch:02d}: train={row['train_loss']:.4f} "
              f"val={row['validation_loss']:.4f} CMI={row['score']:.5f} "
              f"macro={row['macro_f1_9class']:.5f} lr={used_lr:.2g} "
              f"({row['seconds']:.1f}s){' *' if improved else ''}", flush=True)
        if stop:
            print(f"Early stopping after {epoch} epochs; best epoch={stopping.best_epoch}", flush=True)
            break
    best_model, restored_processor, checkpoint = load_cnn_checkpoint(output_dir / "best.pt", device=str(device))
    assert_preprocessor_matches(restored_processor.state, manifest, fold)
    final_metrics, probabilities = evaluate_model(best_model, val_loader, criterion, device)
    validation = manifest.table.iloc[val_indices]
    predicted = np.asarray(ALL_GESTURES)[probabilities.argmax(axis=1)]
    predictions = pd.DataFrame({"sequence_id": arrays["sequence_id"][val_indices], "predicted_gesture": predicted,
                                "folds_sha256": manifest.fingerprint})
    predictions = pd.concat([predictions, pd.DataFrame(probabilities, columns=PROBABILITY_COLUMNS)], axis=1)
    write_fold_predictions(manifest, fold, predictions, output_dir / "predictions.csv", preprocessor_state=processor.state)
    pd.DataFrame(confusion_matrix(validation["gesture"], predicted, labels=ALL_GESTURES),
                 index=ALL_GESTURES, columns=ALL_GESTURES).to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    class_report = classification_report(validation["gesture"], predicted, labels=ALL_GESTURES,
                                        output_dict=True, zero_division=0)
    pd.DataFrame(class_report).transpose().to_csv(output_dir / "class_scores.csv", encoding="utf-8-sig")
    summary = {"model": model_name, "fold": fold, "metric": METRIC_NAME,
        "evaluation_scope": "single held-out subject fold; validation selects the checkpoint",
        "folds_sha256": manifest.fingerprint, "dataset_sha256": manifest.metadata["dataset_sha256"],
        "train_sequences": len(train_indices), "validation_sequences": len(val_indices),
        "best_epoch": checkpoint["best_epoch"], "epochs_run": len(history),
        "stopped_early": len(history) < training.epochs, "device": str(device),
        "elapsed_seconds": time.perf_counter() - started, "max_length": processor.max_length,
        "validation": final_metrics, "model_metadata": model.metadata(), "training": asdict(training),
        "sensor_dropout": asdict(sensor_dropout or SensorDropoutConfig()), "data_metadata": data_metadata or {},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if save_plots:
        plot_history(pd.DataFrame(history), output_dir / "learning_curve.png", checkpoint["best_epoch"])
    print(f"Best {model_name} fold {fold}: CMI={final_metrics['score']:.6f}; saved to {output_dir}", flush=True)
    return summary, predictions


def reuse_completed_fold(output_dir: Path, processor: FoldPreprocessor, manifest: FoldManifest,
                        fold: int, model_name: str, model_config: CNNConfig,
                        training: TrainingConfig, sensor_dropout: SensorDropoutConfig,
                        data_metadata: dict, tof_regions: int) -> tuple[dict, pd.DataFrame]:
    """Reuse a completed fold only when its effective settings and data match."""
    summary = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    _, restored, checkpoint = load_cnn_checkpoint(output_dir / "best.pt")
    expected = (checkpoint["fold"] == fold and checkpoint["folds_sha256"] == manifest.fingerprint
        and checkpoint["model_metadata"]["model"] == model_name
        and checkpoint["model_metadata"]["tof_regions"] == tof_regions
        and CNNConfig.from_dict(checkpoint["model_metadata"]["config"]) == model_config
        and TrainingConfig(**checkpoint["training_config"]) == training
        and checkpoint["sensor_dropout"] == asdict(sensor_dropout)
        and checkpoint["data_metadata"] == data_metadata
        and restored.state == processor.state
        and summary["fold"] == fold and summary["model"] == model_name
        and summary["folds_sha256"] == manifest.fingerprint
        and summary["validation"] == checkpoint["validation_metrics"])
    if not expected:
        raise ValueError("Completed CNN fold settings/data differ; use a fresh output directory.")
    predictions = _checked_predictions(manifest, fold,
        pd.read_csv(output_dir / "predictions.csv", dtype={"sequence_id": str}), require_fingerprint=True)
    actual = cmi_metrics(predictions["gesture"], predictions["predicted_gesture"])
    if any(not np.isclose(actual[key], summary["validation"][key], atol=1e-12, rtol=0) for key in actual):
        raise ValueError("Completed fold predictions disagree with checkpoint metrics.")
    print(f"Reused completed {model_name} fold {fold}: CMI={actual['score']:.6f}", flush=True)
    return summary, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Main experiment v1: IMU-only and multisensor temporal CNNs.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/cnn_v1.json")
    parser.add_argument("--model", choices=("imu", "multisensor", "both"), default="both")
    parser.add_argument("--fold", type=int, nargs="+", help="Defaults to config folds (v1 preview: fold 0).")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed folds with matching settings; does not resume partial epochs.")
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        training_settings = dict(settings.get("training", {}))
        for name in ("epochs", "batch_size", "learning_rate", "device"):
            if getattr(args, name) is not None:
                training_settings[name] = getattr(args, name)
        training = TrainingConfig(**training_settings)
        choose_device(training.device)
        model_settings = dict(settings.get("model_parameters", {}))
        if args.dropout is not None:
            model_settings["dropout"] = args.dropout
        model_config = CNNConfig.from_dict(model_settings)
        data_dir = (PROJECT_DIR / settings.get("data_dir", "data")).resolve()
        output_dir = (args.output_dir or PROJECT_DIR / settings["output_dir"]).resolve()
        cache_root = (args.cache_dir or PROJECT_DIR / settings["cache_dir"]).resolve()
        for path in (output_dir, cache_root):
            if path == data_dir or data_dir in path.parents:
                raise ValueError("CNN outputs/cache must be outside raw data.")
        if output_dir == cache_root or output_dir in cache_root.parents or cache_root in output_dir.parents:
            raise ValueError("Use separate, non-nested output and cache directories.")
        manifest = load_fold_manifest(PROJECT_DIR / settings.get("folds_path", "configs/folds.csv"))
        folds = args.fold if args.fold is not None else settings.get("folds", [0])
        if not folds or len(set(folds)) != len(folds) or any(f not in range(manifest.n_splits) for f in folds):
            raise ValueError("Invalid CNN fold selection.")
        names = ["imu", "multisensor"] if args.model == "both" else [args.model]
        if not args.prepare_only:
            if output_dir.exists() and any(output_dir.iterdir()):
                if not args.resume:
                    raise ValueError("CNN output directory is non-empty; choose a fresh --output-dir or --resume.")
                previous = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
                if (TrainingConfig(**previous["training"]) != training
                    or CNNConfig.from_dict(previous["model_parameters"]) != model_config
                    or previous["folds_sha256"] != manifest.fingerprint
                    or Path(previous["cache_dir"]).resolve() != cache_root
                    or previous["sequence_length"] != (args.sequence_length if args.sequence_length is not None else settings.get("sequence_length"))
                    or any(previous["settings"].get(key) != settings.get(key) for key in
                        ("data_dir", "folds_path", "preprocessing_config", "length_quantile", "tof_regions", "input_clip"))):
                    raise ValueError("Resume settings differ; choose a fresh output directory.")
            output_dir.mkdir(parents=True, exist_ok=True)
        input_settings = json.loads((PROJECT_DIR / settings.get("preprocessing_config", "configs/preprocessing.json")).read_text(encoding="utf-8"))
        preprocessing = PreprocessingConfig.from_dict(input_settings.get("preprocessing", {}))
        sensor_dropout = SensorDropoutConfig(**input_settings.get("sensor_dropout", {}))
        if "length_quantile" in settings:
            preprocessing = PreprocessingConfig.from_dict({**asdict(preprocessing), "length_quantile": settings["length_quantile"]})
        sequence_length = args.sequence_length if args.sequence_length is not None else settings.get("sequence_length")
        regions = settings.get("tof_regions", 2)
        summaries, predictions = [], {name: {} for name in names}
        if not args.prepare_only:
            (output_dir / "run_config.json").write_text(json.dumps({
                "settings": settings, "training": asdict(training), "model_parameters": asdict(model_config),
                "models": names, "folds": folds, "cache_dir": str(cache_root), "sequence_length": sequence_length,
                "folds_sha256": manifest.fingerprint}, indent=2, ensure_ascii=False), encoding="utf-8")
        for fold in folds:
            arrays, processor, data_metadata = prepare_cnn_fold(data_dir, cache_root / f"fold_{fold}", manifest, fold,
                preprocessing=preprocessing, tof_regions=regions, sequence_length=sequence_length,
                input_clip=settings.get("input_clip", 8.0), chunksize=settings.get("chunksize", 25000))
            if args.prepare_only:
                continue
            for name in names:
                folder = output_dir / name / f"fold_{fold}"
                if args.resume and folder.exists() and any(folder.iterdir()):
                    summary, prediction = reuse_completed_fold(folder, processor, manifest, fold,
                        name, model_config, training, sensor_dropout, data_metadata, regions)
                else:
                    summary, prediction = train_cnn_fold(arrays, processor, manifest, fold,
                        folder, model_name=name, tof_regions=regions,
                        model_config=model_config, training=training, sensor_dropout=sensor_dropout, data_metadata=data_metadata)
                summaries.append(summary)
                predictions[name][fold] = prediction
                pd.DataFrame([{"model": s["model"], "fold": s["fold"], **s["validation"],
                    "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"], "max_length": s["max_length"],
                    "parameters": s["model_metadata"]["parameters"]} for s in summaries]).to_csv(output_dir / "comparison.csv", index=False)
                write_comparison_report(output_dir, summaries, manifest)
        if args.prepare_only:
            return 0
        complete = set(folds) == set(range(manifest.n_splits))
        if complete:
            for name in names:
                evaluate_oof_frames(manifest, predictions[name], output_dir / name / "evaluation",
                    experiment_name=f"cnn_v1_{name}", experiment_config={"training": asdict(training),
                        "model_config": asdict(model_config), "checkpoint_selection": "held-out fold CMI score"})
        (output_dir / "summary.json").write_text(json.dumps({"version": "v1", "folds": folds,
            "complete_oof": complete, "evaluation_scope": "5-fold OOF" if complete else "held-out fold preview",
            "folds_sha256": manifest.fingerprint, "results": summaries}, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"Comparison: {output_dir / 'comparison.csv'}; complete five-fold OOF={complete}", flush=True)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
