# CMI wrist gesture recognition with IMU auxiliary losses

This EE8403 course project studies how well a wrist-worn device can distinguish eight target gestures from everyday movements when proximity measurements are missing. We use the [Child Mind Institute sensor benchmark](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data) and reproduce the base architecture from [Yamato Arai's second-place solution](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution).

Our [main experiment](configs/main_experiment.json) asks whether training the inertial measurement unit (IMU) branches to recognize both the gesture group and its target status improves the competition score. The earlier `cnn_dynamics_mixup` implementation is in the [experiment archive](experiments/legacy/cnn_dynamics_mixup/).

## Five-fold validation results

Both systems use the same five participant-disjoint folds and the same prediction rule. Every one of the 8,151 sequences has an out-of-fold (OOF) prediction in each input condition. The competition score gives equal weight to binary target-detection F1 and macro F1 over nine groups (eight target gestures and one group for all everyday movements).

| Input condition | Reproduced second-place base | Base with grouped IMU losses |
| --- | ---: | ---: |
| Sensors as recorded | 0.891362 ± 0.012919 | 0.891716 ± 0.012698 |
| Auxiliary sensors masked for about half the sequences (fixed subset) | 0.861310 ± 0.013242 | 0.861624 ± 0.011197 |
| IMU only | 0.831799 ± 0.013723 | 0.832369 ± 0.012404 |

Values are means across the five folds ± sample standard deviation (`ddof=1`). The mean gain with IMU alone was 0.000569. In that condition, three of the five folds improved. Separate experiments with participant-history post-processing produced changes in both directions. The hybrid has development-fold results only. The Kaggle private leaderboard score of **0.833362** belongs to the earlier frozen dynamics CNN.

## What changed in training

The model predicts a joint label for posture, gesture and initial movement. We retain the upstream architecture, physical features and movement-phase supervision. The two IMU branches minimize the following objective.

```text
joint cross-entropy + phase cross-entropy
    + 0.5 × nine-group soft cross-entropy
    + 0.1 × binary cross-entropy (BCE) for target status
```

Both auxiliary targets come from the joint Mixup soft labels. We sum their probability mass first by scoring group and then by target status. At inference, both systems return the gesture of the highest-scoring joint class.

Each fold has four branches. Across five folds, we trained ten IMU models (with and without rotation features) for 50 epochs each and retained the final checkpoints. For the ten time-of-flight (ToF) branches, we reused the original base checkpoints after verifying the fold, training sequence IDs and joint-class ordering. This design measures the combined effect of the two auxiliary losses in the IMU branches.

Training uses phase-wise Mixup (alpha 0.5), Adam with learning rate 0.001 and weight decay 0.0001, batch size 32, gradient-norm clipping at 1, and a 10% warmup followed by cosine decay. Inputs retain the last 200 frames at most and use right padding. The seed is `42 + fold`. The upstream source is pinned to [commit cb53f8d0cc82f33b86403ba35c5a580930475528](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution/tree/cb53f8d0cc82f33b86403ba35c5a580930475528).

## Code and entry points

| Path | Purpose |
| --- | --- |
| [configs/main_experiment.json](configs/main_experiment.json) | Main experiment settings, audit record and checkpoint provenance |
| [scripts/main_experiment.py](scripts/main_experiment.py) | Show status, verify local files, export an offline notebook or start training |
| [scripts/train_second_place_hybrid.py](scripts/train_second_place_hybrid.py) | Train the two IMU branches with grouped losses on all five folds |
| [scripts/evaluate_second_place_hybrid.py](scripts/evaluate_second_place_hybrid.py) | Import and validate the new IMU models, combine them with the original ToF branches and evaluate the routed model |
| [src/cmi_project/second_place.py](src/cmi_project/second_place.py) | Physical features, training cache, fold-specific joint labels and checkpoint loading |
| [src/cmi_project/second_place_hybrid.py](src/cmi_project/second_place_hybrid.py) | Grouped nine-class and binary losses for the joint classifier |
| [src/cmi_project/vendor/second_place/](src/cmi_project/vendor/second_place/) | Pinned upstream code with author attribution and source hashes |
| [experiments/legacy/cnn_dynamics_mixup/](experiments/legacy/cnn_dynamics_mixup/) | Earlier model code, configurations, tests and documentation |
| [experiments/results/](experiments/results/) | Experiment records, including unsuccessful runs and fold-by-fold comparisons |

## Environment and usage

Use the [`cmi` Conda environment](environment.yml) and run the commands below from the repository root. The original Windows setup used `D:\anaconda\envs\cmi\python.exe` with the `-s` flag. Place the raw competition files under `data/`. All experiments use [configs/folds.csv](configs/folds.csv). Training and validation participants are disjoint. Each fold uses its training partition for fitted preprocessing and construction of the joint-label vocabulary.

```powershell
# Display the recorded main experiment.
python -s scripts/main_experiment.py

# Verify the byte-level hashes of the local model files and all 15 complete OOF files.
python -s scripts/main_experiment.py status --verify

# Run the test suite.
python -s -m unittest discover -s tests -v

# Export an offline training notebook. This command only writes the notebook.
python -s scripts/main_experiment.py export --run-name hybrid_imu_next --notebook-name hybrid-imu-next.ipynb

# Train all ten IMU models for 50 epochs each. Use a fresh output directory.
python -s -u scripts/main_experiment.py train --device cuda --output outputs/second_place/hybrid_imu_next
```

Full evaluation routes sequences to either an IMU or a ToF branch. After a training run, import its ZIP archive with `evaluate_second_place_hybrid.py` and supply the matching base ToF checkpoints. The [hybrid experiment log](experiments/SECOND_PLACE_HYBRID.md) gives the import command. The original audited model files are stored locally under `outputs/kaggle_training/imported/second_place_hybrid_imu_group_loss_v1/`. Raw data, checkpoints, caches and complete prediction files stay in the local `data/` and `outputs/` directories. Full checksum verification requires those files.

## Experiment records and follow-up work

- [Second-place reproduction](experiments/SECOND_PLACE_REPRODUCTION.md) documents the original base, fixed folds and adaptations to the upstream code.
- [Hybrid experiment log](experiments/SECOND_PLACE_HYBRID.md) covers fusion experiments, training batches, the GPU run, ZIP import and audit results.
- [Main result record](experiments/results/second_place_hybrid_imu_group_loss_v1.json) contains per-fold scores for all three sensor conditions, all predefined participant-history orderings and model provenance.
- [Migration record](experiments/MAIN_EXPERIMENT_MIGRATION.md) identifies the Git snapshot before cleanup, the migration commits and the files retained.
- The [experiment index](experiments/README.md) and [legacy implementation notes](experiments/legacy/cnn_dynamics_mixup/README.md) document the earlier models and their results.

The project treats probability fusion, causal participant-history decoding and online pseudo-labelling as separate experiments. For the online pseudo-labelling comparison, the reference is the frozen base model using matched CPU inference at each sequence's actual length. Each participant, fold, sensor condition and arrival order has separate model state, buffers, prediction history and random-number-generator state. Predictions use only the current and earlier arrivals. Validation gesture, posture and phase labels are excluded from inference. Previously returned decisions stay fixed, and all predefined arrival orders are retained. The archived experiment record lists this comparison as ongoing.

Open experiments include the upstream simple and deep architectures, the original ten-fold ensemble, the 34-feature dynamics and thermopile configuration, and retraining the ToF branches with auxiliary losses. At the time of the recorded run, the remaining GPU quota was insufficient for another full training run at the fixed 50-epoch budget. The main-experiment migration preserved the earlier frozen production weights and made no new competition submission.
