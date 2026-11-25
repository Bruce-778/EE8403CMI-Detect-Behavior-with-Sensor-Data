"""Compare time-unordered acceleration summaries and recording length on fixed CMI folds.

Pass a local competition train.csv, the project checkout and an output directory.
The optional --latex argument writes Table 6 in LaTeX.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

AXES = ["acc_x", "acc_y", "acc_z"]
SUMMARIES = [f"{axis}_{stat}" for stat in ("mean", "sd") for axis in AXES]
INPUTS = {
    "Training priors": [],
    "Length": ["length"],
    "Acceleration": SUMMARIES,
    "Length + acceleration": ["length", *SUMMARIES],
}
METRICS = ["score", "binary_f1", "macro_f1_9class", "binary_auroc", "target_identity_accuracy"]


def most_common(values):
    """Resolve frequency ties by lexical order, independently inside each training fold."""
    counts = values.value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def load_data(csv_path, manifest_path):
    metadata = ["subject", "gesture", "sequence_type"]
    frames = pd.read_csv(csv_path, usecols=["sequence_id", *metadata, *AXES])
    if frames.isna().any().any() or not np.isfinite(frames[AXES].to_numpy()).all():
        raise ValueError("Expected complete sequence metadata and finite raw acceleration")
    by_id = frames.groupby("sequence_id", sort=True)
    if not by_id[metadata].nunique().eq(1).all().all():
        raise ValueError("A sequence contains inconsistent labels or subject IDs")
    seq = pd.concat([
        by_id[metadata].first(),
        by_id.size().rename("length"),
        by_id[AXES].mean().add_suffix("_mean"),
        by_id[AXES].std(ddof=0).add_suffix("_sd"),
    ], axis=1)
    folds = pd.read_csv(manifest_path).set_index("sequence_id", verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(seq[["subject", "gesture", "length"]], folds[["subject", "gesture", "length"]])
    seq["fold"] = folds.fold
    if set(seq.fold) != set(range(5)) or not seq.groupby("subject").fold.nunique().eq(1).all():
        raise ValueError("Invalid five-fold participant partition")
    return seq


def evaluate(train, valid, features, score_gestures):
    target = train.sequence_type.eq("Target")
    truth_target = valid.sequence_type.eq("Target")
    if features:
        settings = dict(max_depth=3, min_samples_leaf=100, random_state=42)
        detect = DecisionTreeClassifier(**settings).fit(train[features], target)
        identify = DecisionTreeClassifier(**settings).fit(train.loc[target, features], train.loc[target, "gesture"])
        detected = detect.predict(valid[features])
        prob = detect.predict_proba(valid[features])[:, list(detect.classes_).index(True)]
        identity = identify.predict(valid[features])
    else:
        detected = np.full(len(valid), most_common(target))
        prob = np.full(len(valid), target.mean())
        identity = np.full(len(valid), most_common(train.loc[target, "gesture"]))
    negative_label = most_common(train.loc[~target, "gesture"])
    prediction = np.where(detected, identity, negative_label)
    return {
        **score_gestures(valid.gesture, prediction),
        "binary_auroc": roc_auc_score(truth_target, prob),
        "target_identity_accuracy": accuracy_score(valid.loc[truth_target, "gesture"], identity[truth_target]),
    }


def to_latex(summary, path):
    header = r"""\begin{table}[htbp]
\centering\small
\caption{Post hoc controls on the same five participant-disjoint folds. Inputs are total frame count and/or per-axis acceleration means and standard deviations. Score is the official composite and SD its sample standard deviation across folds. The remaining columns are fold means. Target ID is eight-way accuracy of the identity stage on true target sequences, with the detector bypassed.}
\label{tab:length-control}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrrr@{}}\toprule
Input & Score & SD & Binary F1 & Macro F1$_9$ & AUROC & Target ID\\\midrule"""
    lines = [header]
    for name, row in summary.iterrows():
        cols = ["score_mean", "score_sd", *[f"{metric}_mean" for metric in METRICS[1:]]]
        numbers = [f"{row[col]:.6f}".lstrip("0") for col in cols]
        lines.append(" & ".join([name, *numbers]) + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--latex", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str((args.repo / "src").resolve()))
    from cmi_project.evaluation import TARGET_GESTURES, cmi_metrics

    seq = load_data(args.data, args.repo / "configs/folds.csv")
    if not seq.sequence_type.eq("Target").equals(seq.gesture.isin(TARGET_GESTURES)):
        raise ValueError("Target indicators disagree with the official scoring groups")
    scores = []
    for fold in range(5):
        train, valid = seq.loc[seq.fold.ne(fold)], seq.loc[seq.fold.eq(fold)]
        for name, features in INPUTS.items():
            metrics = evaluate(train, valid, features, cmi_metrics)
            scores.append(dict(fold=fold, inputs=name, sequences=len(valid), participants=valid.subject.nunique(),
                               targets=valid.sequence_type.eq("Target").sum(), **metrics))
    folded = pd.DataFrame(scores)
    grouped = folded.groupby("inputs", sort=False)
    summary = grouped[METRICS].mean().add_suffix("_mean")
    summary["score_sd"] = grouped.score.std(ddof=1)
    args.out.mkdir(parents=True, exist_ok=True)
    folded.to_csv(args.out / "acceleration-control-folds.csv", index=False)
    summary.to_csv(args.out / "acceleration-control-summary.csv")
    if args.latex:
        to_latex(summary, args.latex)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
