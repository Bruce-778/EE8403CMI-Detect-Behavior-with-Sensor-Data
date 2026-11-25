"""Recover aggregate target-detection counts from the archived, full-precision F1 scores.

Run with a checkout of the pinned project revision and an output CSV path.
Rows labelled 'pooled' are solved independently and checked against the five fold totals.
"""
import argparse
import csv
import json
from collections import Counter
from math import isclose
from pathlib import Path
from typing import NamedTuple


INPUT_LABELS = {"observed": "Recorded", "aux_dropout50": "Half masked", "imu_only": "IMU only"}


class Counts(NamedTuple):
    tp: int
    fn: int
    fp: int
    tn: int


def recover(f1, positives, negatives):
    # F1 = 2 TP / (positives + TP + FP). For these archived scores, two
    # adjacent admissible FP values differ in F1 by much more than 1e-13.
    if not 0 < f1 <= 1 or positives == 0:
        raise ValueError("Expected a positive F1 score and positive target support.")
    matches = []
    for tp in range(positives + 1):
        fp = round(2 * tp / f1 - positives - tp)
        if 0 <= fp <= negatives:
            candidate_f1 = 2 * tp / (positives + tp + fp)
            if isclose(candidate_f1, f1, abs_tol=1e-13, rel_tol=0):
                matches.append(Counts(tp, positives - tp, fp, negatives - fp))
    if len(matches) != 1:
        raise ValueError(f"F1={f1!r}, P={positives}, N={negatives} gives {len(matches)} integer solutions.")
    return matches[0]


def write_latex(path, pooled_counts):
    header = r"""\begin{tabular}{@{}lrrrr@{}}\toprule
&\multicolumn{2}{c}{Missed targets}&\multicolumn{2}{c}{False alarms}\\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Input & Base & Modified & Base & Modified \\\midrule"""
    lines = [header]
    for policy, label in INPUT_LABELS.items():
        base = pooled_counts[policy, "base"]
        modified = pooled_counts[policy, "modified"]
        lines.append(f"{label} & {base.fn} & {modified.fn} & {base.fp} & {modified.fp}" + r"\\")
    lines[-1] += r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, help="Local checkout of the course project")
    parser.add_argument("output", type=Path, help="Output CSV path")
    parser.add_argument("--latex", type=Path, help="Write the pooled counts as a LaTeX tabular for Table 10")
    args = parser.parse_args()
    results = args.repository / "experiments/results"
    records = {
        "base": json.loads((results / "second_place_base_five_fold_merged_v1.json").read_text(encoding="utf-8")),
        "modified": json.loads((results / "second_place_hybrid_imu_group_loss_v1.json").read_text(encoding="utf-8")),
    }
    targets = set(records["base"]["diagnostics"]["observed"]["official_9class_oof_f1"]["new"])
    targets.remove("non_target")
    if len(targets) != 8:
        raise ValueError("Expected eight target gestures in the scoring ontology.")
    fold_size, positive = Counter(), Counter()
    sequence_ids = set()
    with (args.repository / "configs/folds.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["sequence_id"] in sequence_ids:
                raise ValueError(f"Repeated sequence {row['sequence_id']}.")
            sequence_ids.add(row["sequence_id"])
            fold = int(row["fold"])
            fold_size[fold] += 1
            positive[fold] += row["gesture"] in targets

    rows = []
    pooled_counts = {}
    for policy in INPUT_LABELS:
        for model, record in records.items():
            summary = record["scenarios"][policy]
            folds = sorted(summary["folds"], key=lambda row: row["fold"])
            if [row["fold"] for row in folds] != sorted(fold_size) or len(folds) != 5:
                raise ValueError("Fold identifiers differ between the scores and manifest.")
            fold_counts = []
            for row in folds:
                fold = row["fold"]
                if row["sequences"] != fold_size[fold]:
                    raise ValueError(f"Sequence count differs in fold {fold}.")
                p, n = positive[fold], fold_size[fold] - positive[fold]
                f1 = row["new"]["binary_f1"]
                counts = recover(f1, p, n)
                rows.append((policy, model, fold, p, n, *counts, f1))
                fold_counts.append(counts)
            if summary["new"]["oof_sequences"] != len(sequence_ids):
                raise ValueError("Pooled record and manifest differ in sequence count.")
            p, n = sum(positive.values()), len(sequence_ids) - sum(positive.values())
            f1 = summary["new"]["oof"]["binary_f1"]
            pooled = recover(f1, p, n)
            if tuple(map(sum, zip(*fold_counts))) != pooled:
                raise ValueError(f"Fold totals and pooled inference disagree for {policy}, {model}.")
            pooled_counts[policy, model] = pooled
            rows.append((policy, model, "pooled", p, n, *pooled, f1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("scenario", "model", "fold", "positives", "negatives", *Counts._fields, "binary_f1"))
        writer.writerows(rows)
    if args.latex:
        write_latex(args.latex, pooled_counts)


if __name__ == "__main__":
    main()
