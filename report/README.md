# Reproducing Tables 6 and 10

These two CPU analyses accompany the EE8403 report on CMI wrist gesture recognition. Run the commands below from the root of this repository. Each command writes its results under `outputs/report-check/`. The reference CSVs and LaTeX tables are in [expected/](expected/).

## Table 6 — Recording length and acceleration

Can a shallow classifier distinguish gestures using only the duration of a recording or the movement it contains? The four controls use the same five participant-disjoint folds as the neural models. Acceleration supplies six features per recording, the mean and population standard deviation on each axis. Only the training participants in each fold are used to fit the two decision trees. The first tree detects targets and the second predicts which of the eight target gestures occurred. The target-ID diagnostic evaluates the second tree on known targets with the detector bypassed.

Use the extracted competition `train.csv` from the [CMI data page](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data). The command assumes it is at `data/train.csv`. You can pass another path with `--data`. The original file checksum is recorded in [the data manifest](../configs/data_source_hashes.json). The verified analysis used Python 3.13.2 with the package versions below.

```sh
python -m pip install numpy==2.3.1 pandas==2.3.0 scikit-learn==1.7.1
python report/acceleration_control.py --repo . --data "data/train.csv" --out outputs/report-check --latex outputs/report-check/acceleration-control.tex
```

| Inputs | Mean competition score | Target-ID accuracy on true targets |
| --- | ---: | ---: |
| Training priors | 0.393545 | 0.124779 |
| Length | 0.411226 | 0.146105 |
| Acceleration | 0.448951 | 0.185607 |
| Length and acceleration | 0.445922 | 0.178967 |

The [20 fold results](expected/acceleration-control-folds.csv) contain one row per input and fold. The [four summary rows](expected/acceleration-control-summary.csv) give the equal-weight fold means and the sample standard deviation of the competition score. Each score gives equal weight to binary target F1 and nine-class macro F1. The acceleration summaries outperform length alone in all five folds. This comparison measures the information available to these shallow classifiers.

## Table 10 — Missed targets and false alarms

A standard Python installation and a checkout of this repository are sufficient for this table. The script uses the [fold manifest](../configs/folds.csv) and the full-precision binary F1 values in the [base](../experiments/results/second_place_base_five_fold_merged_v1.json) and [modified](../experiments/results/second_place_hybrid_imu_group_loss_v1.json) result records. In each record it reads the independent predictions reported under `new`.

```sh
python report/binary_error_counts.py . outputs/report-check/binary-error-counts.csv --latex outputs/report-check/binary-error-counts.tex
```

| Input | Missed targets, base | Missed targets, modified | False alarms, base | False alarms, modified |
| --- | ---: | ---: | ---: | ---: |
| Sensors as recorded | 20 | 21 | 24 | 23 |
| Half masked | 39 | 35 | 29 | 29 |
| IMU only | 58 | 52 | 35 | 41 |

For known target support P, binary F1 equals 2TP / (P + TP + FP). Across these records, each archived value admits one integer solution at an absolute tolerance of 1e-13. The script checks the five fold totals against an independent reconstruction from the pooled F1. The [36 result rows](expected/binary-error-counts.csv) cover three conditions and two models, each with five folds and one pooled row. With IMU alone both systems make 93 target-status errors. These aggregate counts do not identify the individual sequences or the gestures involved in a confusion.

To check a rerun, compare the generated LaTeX tables with [Table 6](expected/acceleration-control.tex) and [Table 10](expected/binary-error-counts.tex). Their displayed values should match. For the CSVs, compare folds, labels and counts exactly. Allow for floating-point serialization when comparing unrounded scores.

With the verified package versions and a fresh output directory, the following comparison covers both LaTeX tables and all three CSVs. A successful match produces no diff.

```sh
git diff --no-index --exit-code --ignore-space-at-eol report/expected outputs/report-check
```
