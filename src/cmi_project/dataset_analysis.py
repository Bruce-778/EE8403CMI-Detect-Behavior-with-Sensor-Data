"""CMI Step 1: reproducible, chunked dataset inspection without changing raw data.

Run ``python scripts/analyze_dataset.py`` from project/.
Outputs go to outputs/dataset_analysis/.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html import escape
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SENSOR_COLUMNS = {
    "IMU": ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"],
    "THM": [f"thm_{sensor}" for sensor in range(1, 6)],
    "ToF": [f"tof_{sensor}_v{pixel}" for sensor in range(1, 6) for pixel in range(64)],
}
ALL_SENSOR_COLUMNS = [col for columns in SENSOR_COLUMNS.values() for col in columns]


@dataclass
class SequenceStats:
    length: int = 0
    subjects: set = field(default_factory=set)
    gestures: set = field(default_factory=set)
    sequence_types: set = field(default_factory=set)
    missing_subject_rows: int = 0
    missing_gesture_rows: int = 0
    # For each modality: NaN cells, -1 cells, any-unavailable rows, all-unavailable rows.
    sensor_counts: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int64))


def default_data_dir() -> Path:
    local = PROJECT_DIR / "data"
    return local if (local / "train.csv").is_file() else PROJECT_DIR.parent / "cmi-data"


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def write_csv(table: pd.DataFrame, path: Path) -> None:
    # BOM allows Excel on Windows to recognize Chinese text; values stay numeric.
    table.to_csv(path, index=False, encoding="utf-8-sig")


def scan_train(train_path: Path, chunksize: int) -> dict:
    """Scan all rows, preserving exact counts across split/non-contiguous sequences."""
    header = pd.read_csv(train_path, nrows=0).columns.tolist()
    required = ["sequence_id", "subject", "gesture", *ALL_SENSOR_COLUMNS]
    absent = sorted(set(required) - set(header))
    if absent:
        raise ValueError(f"train.csv 缺少必需列: {', '.join(absent)}")
    metadata = ["sequence_id", "subject", "gesture"]
    if "sequence_type" in header:
        metadata.append("sequence_type")
    dtypes = {col: "string" for col in metadata}
    dtypes.update({col: "float64" for col in ALL_SENSOR_COLUMNS})
    states: dict[str, SequenceStats] = defaultdict(SequenceStats)
    rows = 0
    null_metadata = Counter()
    subject_rows = Counter()
    gesture_rows = Counter()
    feature_nan = Counter()
    feature_minus1 = Counter()
    modality_rows = {name: Counter() for name in SENSOR_COLUMNS}

    reader = pd.read_csv(train_path, usecols=metadata + ALL_SENSOR_COLUMNS,
                         dtype=dtypes, chunksize=chunksize)
    for chunk_number, chunk in enumerate(reader, start=1):
        rows += len(chunk)
        null_metadata.update({col: int(chunk[col].isna().sum()) for col in metadata})
        subject_rows.update(chunk["subject"].value_counts().to_dict())
        gesture_rows.update(chunk["gesture"].value_counts().to_dict())
        for sequence_id, count in chunk.groupby("sequence_id", sort=False).size().items():
            states[sequence_id].length += int(count)
        # Inspect every distinct metadata combination, never take an unchecked first label.
        for values in chunk[metadata].drop_duplicates().itertuples(index=False, name=None):
            sequence_id, subject, gesture = values[:3]
            if pd.isna(sequence_id):
                continue
            state = states[sequence_id]
            if pd.notna(subject):
                state.subjects.add(subject)
            if pd.notna(gesture):
                state.gestures.add(gesture)
            if len(values) == 4 and pd.notna(values[3]):
                state.sequence_types.add(values[3])

        diagnostics = pd.DataFrame({"sequence_id": chunk["sequence_id"]})
        diagnostics["missing_subject_rows"] = chunk["subject"].isna().astype("int64")
        diagnostics["missing_gesture_rows"] = chunk["gesture"].isna().astype("int64")
        for modality, columns in SENSOR_COLUMNS.items():
            values = chunk[columns].to_numpy()
            nan_mask = np.isnan(values)
            minus1_mask = values == -1 if modality == "ToF" else np.zeros_like(nan_mask)
            unavailable = nan_mask | minus1_mask
            feature_nan.update(dict(zip(columns, nan_mask.sum(axis=0).astype(int))))
            feature_minus1.update(dict(zip(columns, minus1_mask.sum(axis=0).astype(int))))
            row_counts = modality_rows[modality]
            row_counts["rows_any_nan"] += int(nan_mask.any(axis=1).sum())
            row_counts["rows_all_nan"] += int(nan_mask.all(axis=1).sum())
            row_counts["rows_any_unavailable"] += int(unavailable.any(axis=1).sum())
            row_counts["rows_all_unavailable"] += int(unavailable.all(axis=1).sum())
            diagnostics[f"{modality}_nan_cells"] = nan_mask.sum(axis=1)
            diagnostics[f"{modality}_minus1_cells"] = minus1_mask.sum(axis=1)
            diagnostics[f"{modality}_any_unavailable_rows"] = unavailable.any(axis=1).astype(int)
            diagnostics[f"{modality}_all_unavailable_rows"] = unavailable.all(axis=1).astype(int)
        grouped = diagnostics.groupby("sequence_id", sort=False).sum(numeric_only=True)
        for values in grouped.itertuples(name=None):
            state = states[values[0]]
            state.missing_subject_rows += int(values[1])
            state.missing_gesture_rows += int(values[2])
            state.sensor_counts += np.asarray(values[3:], dtype=np.int64)
        if chunk_number == 1 or chunk_number % 10 == 0:
            print(f"已扫描 {rows:,} 行，发现 {len(states):,} 个 sequence", flush=True)

    if not rows:
        raise ValueError("train.csv 没有数据行。")
    return {"header": header, "rows": rows, "states": states,
            "null_metadata": null_metadata, "subject_rows": subject_rows,
            "gesture_rows": gesture_rows, "feature_nan": feature_nan,
            "feature_minus1": feature_minus1, "modality_rows": modality_rows}


def build_tables(scan: dict, demographics: pd.DataFrame, expected_gestures: int) -> dict:
    states = scan["states"]
    records = []
    subject_sequences = defaultdict(set)
    for sequence_id, state in sorted(states.items()):
        label_ok = len(state.gestures) == 1 and state.missing_gesture_rows == 0
        subject_ok = len(state.subjects) == 1 and state.missing_subject_rows == 0
        for subject in state.subjects:
            subject_sequences[subject].add(sequence_id)
        record = {
            "sequence_id": sequence_id, "subject": next(iter(state.subjects)) if subject_ok else None,
            "gesture": next(iter(state.gestures)) if label_ok else None,
            "sequence_type": next(iter(state.sequence_types)) if len(state.sequence_types) == 1 else None,
            "length": state.length, "gesture_count": len(state.gestures),
            "subject_count": len(state.subjects), "label_is_valid": label_ok,
            "subject_is_valid": subject_ok, "missing_gesture_rows": state.missing_gesture_rows,
            "missing_subject_rows": state.missing_subject_rows,
            "observed_gestures": json.dumps(sorted(state.gestures), ensure_ascii=False),
            "observed_subjects": json.dumps(sorted(state.subjects), ensure_ascii=False),
        }
        for i, (modality, columns) in enumerate(SENSOR_COLUMNS.items()):
            nan, invalid, any_rows, all_rows = map(int, state.sensor_counts[i * 4:i * 4 + 4])
            record.update({f"{modality}_nan_cells": nan, f"{modality}_minus1_cells": invalid,
                           f"{modality}_unavailable_ratio": (nan + invalid) / (state.length * len(columns)),
                           f"{modality}_any_unavailable_rows": any_rows,
                           f"{modality}_all_unavailable_rows": all_rows,
                           f"{modality}_fully_unavailable": all_rows == state.length})
        records.append(record)
    sequences = pd.DataFrame(records)
    label_valid = sequences[sequences["label_is_valid"]]
    gesture_records = []
    for gesture in sorted(scan["gesture_rows"]):
        group = label_valid[label_valid["gesture"] == gesture]
        participating_subjects = set()
        for sequence_id in group["sequence_id"]:
            participating_subjects.update(states[sequence_id].subjects)
        gesture_records.append({"gesture": gesture, "sequence_count": len(group),
                                "sequence_ratio": len(group) / len(sequences),
                                "row_count": scan["gesture_rows"][gesture],
                                "subject_count": len(participating_subjects)})
    gestures = pd.DataFrame(gesture_records, columns=["gesture", "sequence_count", "sequence_ratio",
                                                    "row_count", "subject_count"])
    gestures = gestures.sort_values(["sequence_count", "gesture"], ascending=[False, True])
    subject_records = [{"subject": subject, "sequence_count": len(subject_sequences[subject]),
                        "row_count": count} for subject, count in sorted(scan["subject_rows"].items())]
    subjects = pd.DataFrame(subject_records, columns=["subject", "sequence_count", "row_count"])
    demo_counts = demographics["subject"].value_counts()
    train_subjects = set(scan["subject_rows"])
    demo_subjects = set(demo_counts.index)
    coverage = pd.DataFrame([{"subject": subject, "in_train": subject in train_subjects,
                             "demographics_rows": int(demo_counts.get(subject, 0)),
                             "sequence_count": len(subject_sequences[subject])}
                            for subject in sorted(train_subjects | demo_subjects)])
    subjects = subjects.merge(coverage[["subject", "demographics_rows"]], on="subject", how="left",
                              validate="one_to_one")
    length_summary = sequences["length"].describe(percentiles=[.05, .25, .5, .75, .95, .99])
    lengths = length_summary.rename_axis("statistic").reset_index(name="length_in_rows")
    if len(label_valid):
        length_by_gesture = label_valid.groupby("gesture")["length"].agg(
            sequence_count="count", mean="mean", std="std", minimum="min", median="median", maximum="max"
        ).reset_index()
    else:
        length_by_gesture = pd.DataFrame(columns=["gesture", "sequence_count", "mean", "std",
                                                "minimum", "median", "maximum"])

    feature_records = []
    for modality, columns in SENSOR_COLUMNS.items():
        for col in columns:
            nan = int(scan["feature_nan"][col])
            invalid = int(scan["feature_minus1"][col])
            feature_records.append({"modality": modality, "feature": col,
                                    "total_cells": scan["rows"], "nan_cells": nan,
                                    "nan_ratio": nan / scan["rows"], "minus1_cells": invalid,
                                    "minus1_ratio": invalid / scan["rows"],
                                    "minus1_ratio_of_non_nan": fraction(invalid, scan["rows"] - nan)
                                    if modality == "ToF" else None,
                                    "unavailable_cells": nan + invalid,
                                    "unavailable_ratio": (nan + invalid) / scan["rows"]})
    feature_missing = pd.DataFrame(feature_records)
    sensor_records = []
    for modality, columns in SENSOR_COLUMNS.items():
        group = feature_missing[feature_missing["modality"] == modality]
        total = scan["rows"] * len(columns)
        nan, invalid = int(group["nan_cells"].sum()), int(group["minus1_cells"].sum())
        sensor_records.append({"modality": modality, "feature_count": len(columns), "total_cells": total,
                               "nan_cells": nan, "nan_ratio": nan / total,
                               "minus1_cells": invalid, "minus1_ratio": invalid / total,
                               "minus1_ratio_of_non_nan": fraction(invalid, total - nan)
                               if modality == "ToF" else None,
                               "unavailable_cells": nan + invalid, "unavailable_ratio": (nan + invalid) / total,
                               **scan["modality_rows"][modality],
                               "sequences_any_unavailable": int((sequences[f"{modality}_unavailable_ratio"] > 0).sum()),
                               "sequences_fully_unavailable": int(sequences[f"{modality}_fully_unavailable"].sum())})
    sensors = pd.DataFrame(sensor_records)
    device_records = []
    for device in range(1, 6):
        group = feature_missing[feature_missing["feature"].str.startswith(f"tof_{device}_")]
        total = int(group["total_cells"].sum())
        nan, invalid = int(group["nan_cells"].sum()), int(group["minus1_cells"].sum())
        device_records.append({"device": f"tof_{device}", "feature_count": len(group), "total_cells": total,
                               "nan_cells": nan, "nan_ratio": nan / total,
                               "minus1_cells": invalid, "minus1_ratio": invalid / total,
                               "minus1_ratio_of_non_nan": fraction(invalid, total - nan),
                               "unavailable_cells": nan + invalid, "unavailable_ratio": (nan + invalid) / total})
    devices = pd.DataFrame(device_records)
    demographics_missing = pd.DataFrame([{"feature": col, "total_rows": len(demographics),
                                          "nan_rows": int(demographics[col].isna().sum()),
                                          "nan_ratio": fraction(int(demographics[col].isna().sum()), len(demographics))}
                                         for col in demographics.columns])
    feature_schema = pd.DataFrame([{"modality": name, "feature": col,
                                    "component": "acceleration" if col.startswith("acc_") else
                                    "quaternion" if col.startswith("rot_") else
                                    "temperature" if name == "THM" else "distance_pixel",
                                    "invalid_sentinel": -1 if name == "ToF" else None}
                                   for name, columns in SENSOR_COLUMNS.items() for col in columns])

    checks = []
    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("sequence_id_not_null", scan["null_metadata"]["sequence_id"] == 0,
          f"missing rows = {scan['null_metadata']['sequence_id']}")
    invalid_labels = int((~sequences["label_is_valid"]).sum())
    invalid_subjects = int((~sequences["subject_is_valid"]).sum())
    check("one_non_null_gesture_per_sequence", invalid_labels == 0, f"invalid sequences = {invalid_labels}")
    check("one_non_null_subject_per_sequence", invalid_subjects == 0, f"invalid sequences = {invalid_subjects}")
    check("expected_gesture_count", len(gestures) == expected_gestures,
          f"observed = {len(gestures)}, expected = {expected_gestures}")
    check("demographics_subject_not_null", demographics["subject"].notna().all(),
          f"missing rows = {int(demographics['subject'].isna().sum())}")
    check("demographics_subject_unique", not demographics["subject"].duplicated().any(),
          f"duplicate subject keys = {int((demo_counts > 1).sum())}")
    missing_demo = sorted(train_subjects - demo_subjects)
    check("all_train_subjects_have_demographics", not missing_demo,
          f"missing subjects = {json.dumps(missing_demo)}")
    check("sequence_rows_reconcile", int(sequences["length"].sum()) == scan["rows"],
          f"sequence sum = {int(sequences['length'].sum())}, train rows = {scan['rows']}")
    check("gesture_sequences_reconcile", int(gestures["sequence_count"].sum()) == len(sequences),
          f"labeled sequences = {int(gestures['sequence_count'].sum())}, all sequences = {len(sequences)}")
    check("subject_sequences_reconcile", int(subjects["sequence_count"].sum()) == len(sequences),
          f"subject-sequence pairs = {int(subjects['sequence_count'].sum())}, all sequences = {len(sequences)}")
    integrity = pd.DataFrame(checks)
    summary_values = {
        "train_rows": scan["rows"], "sequence_count": len(sequences),
        "subject_count": len(subjects), "gesture_count": len(gestures),
        "demographics_rows": len(demographics), "demographics_subject_count": len(demo_subjects),
        "demographics_subjects_without_train": len(demo_subjects - train_subjects),
        "IMU_features": 7, "THM_features": 5, "ToF_features": 320, "total_sensor_features": 332,
        "sequence_length_min": int(sequences["length"].min()),
        "sequence_length_mean": float(sequences["length"].mean()),
        "sequence_length_std": float(sequences["length"].std()) if len(sequences) > 1 else None,
        "sequence_length_median": float(sequences["length"].median()),
        "sequence_length_p95": float(sequences["length"].quantile(.95)),
        "sequence_length_max": int(sequences["length"].max()),
        "invalid_label_sequences": invalid_labels, "invalid_subject_sequences": invalid_subjects,
        "missing_demographics_subjects": len(missing_demo),
        "all_integrity_checks_passed": bool(integrity["passed"].all()),
    }
    basic = pd.DataFrame([{"metric": key, "value": value} for key, value in summary_values.items()])
    return {"basic_statistics": basic, "gesture_distribution": gestures,
            "sequence_summary": sequences, "sequence_length_summary": lengths,
            "sequence_length_by_gesture": length_by_gesture, "subject_distribution": subjects,
            "sensor_missingness": sensors, "sensor_feature_missingness": feature_missing,
            "tof_device_missingness": devices, "demographics_missingness": demographics_missing,
            "subject_demographics_coverage": coverage, "feature_schema": feature_schema,
            "integrity_checks": integrity, "integrity_issues": integrity[~integrity["passed"]],
            "summary": summary_values}


def make_figures(tables: dict, output_dir: Path) -> list[tuple[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.axisbelow": True})
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    def save(fig, filename: str, title: str) -> None:
        fig.savefig(figures_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures.append((title, f"figures/{filename}"))

    gestures = tables["gesture_distribution"].iloc[::-1]
    sequences = tables["sequence_summary"]
    fig, ax = plt.subplots(figsize=(11, max(5, len(gestures) * .36)))
    bars = ax.barh(gestures["gesture"], gestures["sequence_count"], color="#3679a8")
    ax.bar_label(bars, padding=4, fmt="%d")
    ax.set(xlabel="Number of sequences", title="Gesture distribution (one sample per sequence)")
    ax.set_xlim(0, max(1, gestures["sequence_count"].max() if len(gestures) else 1) * 1.14)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    save(fig, "gesture_distribution.png", "18 类 gesture 的 sequence 样本分布")

    lengths = sequences["length"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(lengths, bins=50, color="#3679a8", edgecolor="white", linewidth=.4)
    axes[0].axvline(np.median(lengths), color="#c56a27", linestyle="--", label=f"Median = {np.median(lengths):g}")
    axes[0].set(xlabel="Sequence length (rows)", ylabel="Number of sequences", title="Sequence length distribution")
    axes[0].legend()
    sorted_lengths = np.sort(lengths)
    axes[1].step(sorted_lengths, np.arange(1, len(lengths) + 1) / len(lengths), where="post", color="#3679a8")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(xlabel="Sequence length (rows)", ylabel="Cumulative share of sequences", title="Sequence length ECDF")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "sequence_length_distribution.png", "Sequence 长度直方图与累计分布")

    fig, ax = plt.subplots(figsize=(11, max(5, len(gestures) * .36)))
    box_gestures = [gesture for gesture in gestures["gesture"] if (sequences["gesture"] == gesture).any()]
    if box_gestures:
        box = ax.boxplot([sequences.loc[sequences["gesture"] == gesture, "length"] for gesture in box_gestures],
                         vert=False, tick_labels=box_gestures, patch_artist=True,
                         flierprops={"markersize": 2, "alpha": .4})
        for patch in box["boxes"]:
            patch.set_facecolor("#b6d3e8")
    else:
        ax.text(.5, .5, "No sequences with a unique non-null gesture", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Sequence length (rows)", title="Sequence length by gesture")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    save(fig, "sequence_length_by_gesture.png", "各 gesture 的 sequence 长度箱线图")

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = tables["subject_distribution"]["sequence_count"]
    ax.hist(counts, bins="auto", color="#3679a8", edgecolor="white")
    ax.set(xlabel="Sequences per subject", ylabel="Number of subjects", title="Subject sample-count distribution")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save(fig, "subject_distribution.png", "各 subject 的 sequence 数量分布")

    sensors = tables["sensor_missingness"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sensors["modality"], 1 - sensors["unavailable_ratio"], label="Available", color="#84b59a")
    ax.barh(sensors["modality"], sensors["nan_ratio"], left=1 - sensors["unavailable_ratio"],
            label="NaN / blank", color="#efb96d")
    ax.barh(sensors["modality"], sensors["minus1_ratio"], left=1 - sensors["minus1_ratio"],
            label="ToF -1", color="#c96b68")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Share of sensor cells (rows x features)", xlim=(0, 1), title="Sensor availability")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "sensor_missingness.png", "三类传感器的空值、ToF -1 与可用值比例")

    fig, axes = plt.subplots(1, 5, figsize=(14, 3), layout="constrained")
    missing = tables["sensor_feature_missingness"].set_index("feature")
    for device, ax in enumerate(axes, start=1):
        pixel_rates = missing.loc[[f"tof_{device}_v{pixel}" for pixel in range(64)], "unavailable_ratio"]
        heatmap = ax.imshow(pixel_rates.to_numpy().reshape(8, 8), cmap="YlOrRd", vmin=0, vmax=1)
        ax.set(title=f"ToF {device}", xlabel="Pixel column", ylabel="Pixel row")
        ax.set_xticks([0, 3, 7])
        ax.set_yticks([0, 3, 7])
    fig.colorbar(heatmap, ax=axes, shrink=.8, format=PercentFormatter(1), label="NaN + -1 / all cells")
    save(fig, "tof_pixel_unavailability.png", "ToF 各像素的不可用比例（NaN 与 -1 的并集）")
    return figures


def write_report(tables: dict, figures: list, output_dir: Path) -> None:
    summary = tables["summary"]
    sensors = tables["sensor_missingness"]
    status = "所有检查通过" if summary["all_integrity_checks_passed"] else "发现数据完整性问题，请查看检查表"
    def html_table(name: str, columns=None) -> str:
        table = tables[name] if columns is None else tables[name][columns]
        return table.to_html(index=False, border=0, classes="data-table", escape=True,
                             float_format=lambda value: f"{value:,.4f}", na_rep="—")

    sensor_display = sensors[["modality", "feature_count", "nan_ratio", "minus1_ratio",
                              "unavailable_ratio", "sequences_fully_unavailable"]].copy()
    for col in ("nan_ratio", "minus1_ratio", "unavailable_ratio"):
        sensor_display[col] = sensor_display[col].map(lambda value: f"{value:.2%}")
    sensor_html = sensor_display.to_html(index=False, border=0, classes="data-table")
    charts = "".join(f'<figure><img src="{escape(path)}" alt="{escape(title)}"><figcaption>{escape(title)}</figcaption></figure>'
                     for title, path in figures)
    tof = sensors[sensors["modality"] == "ToF"].iloc[0]
    label_statement = ("每个 sequence 的所有行均对应同一个非空 gesture 标签。"
                       if summary["invalid_label_sequences"] == 0 else
                       f"有 {summary['invalid_label_sequences']:,} 个 sequence 缺少标签或存在多个 gesture；这些 sequence 未被强行赋予最终标签。")
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMI Dataset Analysis — Step 1</title><style>
body{{font-family:system-ui,'Microsoft YaHei',sans-serif;color:#203040;background:#f5f7fa;margin:0;line-height:1.65}}
main{{max-width:1120px;margin:32px auto;background:white;padding:36px;border-radius:12px}}
h1,h2{{line-height:1.3}} h2{{margin-top:36px}} p{{max-width:1000px}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #dfe5eb;white-space:nowrap}}
th{{background:#eef3f8}} figure{{margin:26px 0}} img{{max-width:100%;height:auto}}
figcaption{{color:#536578;font-size:14px}} code{{background:#eef3f8;padding:2px 5px}}
a{{color:#28638e}} @media(max-width:700px){{main{{margin:0;padding:18px;border-radius:0}}}}
</style></head><body><main>
<h1>CMI Dataset Analysis — Step 1</h1>
<p>完整扫描 <code>train.csv</code>，读取 <code>train_demographics.csv</code>。原始数据保持不变。</p>
<p>训练数据包含 <strong>{summary['train_rows']:,}</strong> 行、<strong>{summary['sequence_count']:,}</strong> 个 sequence、
<strong>{summary['subject_count']:,}</strong> 个 subject，以及 <strong>{summary['gesture_count']}</strong> 类 gesture。
Sequence 长度中位数为 <strong>{summary['sequence_length_median']:g}</strong> 行，范围为
<strong>{summary['sequence_length_min']:,}–{summary['sequence_length_max']:,}</strong> 行。</p>
<h2>统计口径与特征</h2>
<p>每个 sequence 是一个分类样本；gesture 分布按 sequence 数量统计，row_count 单独记录时间步数量。
长度单位为 CSV 行数（时间步），未假设采样频率。IMU 为 acc_x/y/z 与 rot_w/x/y/z，共 7 列；THM 为 thm_1–5，
共 5 列；ToF 为 tof_1–5 的 v0–63，共 320 列。Demographics 按 subject 核对唯一性与覆盖情况。</p>
<p>空值定义为 pandas 读取后的 NaN（包括 CSV 空字段）。仅 ToF 的 -1 视为无效；IMU 中的负数保留为有效数值。
cell 比例分母为总行数 × 该模态特征数。ToF 的 NaN 与 -1 互斥，不可用比例为两者之和；
minus1_ratio_of_non_nan 的分母仅包含非 NaN 单元格。整段不可用指该 sequence 的每一行、每一列均不可用。
像素图按 v0–v63 顺序重排为 8×8 网格，仅用于展示索引分布，未推断物理方向。</p>
<h2>基础数据统计</h2><div class="table-wrap">{html_table('basic_statistics')}</div>
<h2>Gesture 样本分布</h2><div class="table-wrap">{html_table('gesture_distribution')}</div>
<h2>Sequence 长度统计</h2><div class="table-wrap">{html_table('sequence_length_summary')}</div>
<h2>传感器缺失与无效值</h2><div class="table-wrap">{sensor_html}</div>
<p>ToF 空值比例为 {tof['nan_ratio']:.2%}，-1 比例为 {tof['minus1_ratio']:.2%}，合计不可用比例为
{tof['unavailable_ratio']:.2%}；在非空 ToF 单元格中，-1 占 {tof['minus1_ratio_of_non_nan']:.2%}。
共有 {int(tof['sequences_fully_unavailable']):,} 个 sequence 的全部 ToF 单元格不可用。
以上均为原始数据统计，后续实验可据此制定缺失值处理方式。</p>
<div class="table-wrap">{html_table('tof_device_missingness')}</div>
<h2>标签和 subject 完整性</h2><p><strong>{status}</strong>。{label_statement}</p>
<p>有 {summary['missing_demographics_subjects']} 个训练 subject 缺少 demographics；
demographics 中有 {summary['demographics_subjects_without_train']} 个 subject 未出现在训练数据中。
异常 sequence 的全部观测标签与 subject 保留在 sequence_summary.csv 的 observed_* 列。</p>
<div class="table-wrap">{html_table('integrity_checks')}</div>
<h2>Demographics 缺失值</h2><div class="table-wrap">{html_table('demographics_missingness')}</div>
<h2>分布图</h2>{charts}
<h2>可下载统计表</h2><ul>
{''.join(f'<li><a href="tables/{escape(name)}.csv">{escape(name)}.csv</a></li>' for name in tables if name != 'summary')}
</ul><p>详细运行信息与统计数值见 <a href="summary.json">summary.json</a>。
本报告中的 gesture 分布仅包含标签唯一且非空的 sequence；若存在异常，比例分母仍为全部 sequence，检查表会报告未能对齐的计数。</p>
</main></body></html>"""
    (output_dir / "dataset_analysis.html").write_text(report, encoding="utf-8")


def analyze(data_dir: Path, output_dir: Path, chunksize: int = 25000,
            expected_gestures: int = 18, plots: bool = True) -> dict:
    if chunksize <= 0 or expected_gestures <= 0:
        raise ValueError("chunksize 和 expected_gestures 必须为正整数。")
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    train_path, demo_path = data_dir / "train.csv", data_dir / "train_demographics.csv"
    for path in (train_path, demo_path):
        if not path.is_file():
            raise FileNotFoundError(f"数据文件不存在: {path}")
    # Prevent overwriting an input file with a generated table.
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("输出目录必须位于原始数据目录之外。")
    demographics = pd.read_csv(demo_path, dtype={"subject": "string"})
    if "subject" not in demographics:
        raise ValueError("train_demographics.csv 缺少 subject 列。")
    scan = scan_train(train_path, chunksize)
    tables = build_tables(scan, demographics, expected_gestures)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name != "summary":
            write_csv(table, tables_dir / f"{name}.csv")
    figures = make_figures(tables, output_dir) if plots else []
    write_report(tables, figures, output_dir)
    summary = {**tables["summary"], "sources": {"train": str(train_path), "demographics": str(demo_path)},
               "source_file_bytes": {"train": train_path.stat().st_size, "demographics": demo_path.stat().st_size},
               "chunksize": chunksize, "expected_gestures": expected_gestures,
               "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
               "sensor_statistics": tables["sensor_missingness"].astype(object)
               .where(pd.notna(tables["sensor_missingness"]), None).to_dict(orient="records"),
               "integrity_checks": tables["integrity_checks"].to_dict(orient="records"),
               "figures": [path for _, path in figures]}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"完成: {summary['sequence_count']:,} sequences / {summary['subject_count']} subjects / {summary['gesture_count']} gestures")
    print(f"报告: {output_dir / 'dataset_analysis.html'}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="CMI Step 1：分块扫描训练数据，输出统计表、分布图与 HTML 报告。")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "dataset_analysis.json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunksize", type=int)
    parser.add_argument("--expected-gestures", type=int)
    parser.add_argument("--no-plots", action="store_true", help="仅输出表格、JSON 和不含图的 HTML。")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config_data_dir = config.get("data_dir")
        data_dir = args.data_dir or (PROJECT_DIR / config_data_dir if config_data_dir else default_data_dir())
        output_dir = args.output_dir or PROJECT_DIR / config.get("output_dir", "outputs/dataset_analysis")
        chunksize = args.chunksize if args.chunksize is not None else int(config.get("chunksize", 25000))
        expected = args.expected_gestures if args.expected_gestures is not None else int(config.get("expected_gestures", 18))
        summary = analyze(data_dir, output_dir, chunksize, expected, not args.no_plots)
    except (ValueError, FileNotFoundError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2
    if not summary["all_integrity_checks_passed"]:
        print("数据完整性检查未全部通过，诊断结果已写入 integrity_issues.csv。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
