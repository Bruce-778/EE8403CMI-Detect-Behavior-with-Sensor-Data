# Second-place base with project improvements

The user authorized integrating the stronger public method into this project
and measuring whether our own additions improve it. The original reproduction
and running CPU online experiment remain separate controls.

The seven source files at commit `cb53f8d0cc82f33b86403ba35c5a580930475528`
are now tracked unchanged in `src/cmi_project/vendor/second_place`. Their author
README/comments and source SHA manifest are retained. Project additions belong
in `second_place_hybrid.py` and the separate hybrid runners.

## Completed fixed-weight complementarity screening

`second_place_hybrid_fusion_v2.json` validates all 8151 samples in all three
missing-sensor scenarios. Our contribution weights 0.1, 0.2 and 0.5 were fixed
in code before execution and applied equally to every fold. Every preset result
is reported; no per-fold weight or best arrival order is chosen. This is
development screening, not fresh independent validation or an online score.

We compare the original joint argmax with a marginal-probability decision
control, then fuse our gesture probabilities into the second-place joint axis
while retaining its conditional orientation/initial-behavior distribution.
The latter supports the existing causal history decoder without mixing joint
axes across folds. All three history orders have independently reset state and
only same-subject already-arrived predictions.

| Scenario | Second-place joint | Ours 10% | Ours 20% | Ours 50% |
|---|---:|---:|---:|---:|
| observed | 0.891362 | 0.892128 | 0.894091 | 0.889713 |
| aux_dropout50 | 0.861310 | 0.863378 | 0.864297 | 0.858076 |
| imu_only | 0.831799 | 0.833555 | 0.833266 | 0.824751 |

These are fold means; every fold, sample SD and full OOF is saved in the JSON
and `outputs/second_place/hybrid/second_place_hybrid_fusion_v2`.
20% observed causal history means for orders42/142/242:
0.922322 / 0.922236 / 0.923810. This is a probability fusion result, not proof
that transplanted training modules improve the second-place network.

First attempt v1 failed because merged routed folders contain implicit raw
metadata logit order rather than an explicit `joint_sequence_ids.npy`. The
missing path was `routed/observed/fold_0/joint_sequence_ids.npy`. The failed
record and commit00d3238 are preserved. v2 uses the existing audited alignment
helper, verifies reconstructed probabilities and exact baseline fold means,
and saves source hashes. No source weights or existing predictions were changed.

## Registered training ablation: IMU metric-group auxiliary losses

Keep the author's base model, train-only 102-class ontology, phase CE,
phase-wise Mixup0.5, Adam0.001/wd0.0001, batch32, warmup/cosine, seed42+fold,
200-frame processing and final50epoch checkpoint. Add only our official-group
objectives: joint soft CE + 0.5 official9 soft CE + 0.1 binary BCE. Group targets
are derived from training joint Mixup targets, not validation annotations;
the existing phase CE is unchanged. No extra label smoothing is introduced.

Train both IMU and rotation-missing IMU arms for all five fixed folds (ten new
models), then combine with the ten already-audited original ToF weights for a
new routed comparison. This specifically measures the IMU loss addition;
it does not claim that the new loss has been tested on retrained ToF arms.
Dynamic34D features, THM and other architecture changes remain untested in this
hybrid and will require separate registered comparisons.

The loss test checks joint-axis permutation, soft Mixup targets, finite
gradients and exact zero-auxiliary equivalence. Two real fold0 **training-only**
batches passed forward/backward/Adam updates, finite gradients and unchanged
phase loss. Record `second_place_hybrid_training_batch_v1.json` is a CPU
functionality check, not a CV score.

Before launch, the normal Kaggle account UI explicitly shows GPU3h59m
available of30h. The original ten IMU arm histories total3.17125h, leaving
about0.812h for additional loss overhead, setup and export. No shortened50epoch
training is used. A complete ToF retraining would exceed this demonstrated
budget, so it is not started. Kaggle runs privately/offline in the original
environment; no competition submission, push or frozen production replacement.

## Actual GPU launch: private version356690453

Normal browser Save & Run All accepted on2026-10-09T08:17:10Z. The unique private notebook is Running: https://www.kaggle.com/code/mingweiwei03/cmi-second-place-hybrid-imu-fixed-folds?scriptVersionId=356690453. Official CMI competition input attached, T4x2, internet off, editor Pin to original environment. Logs label the container Latest Container Image, but actual torch2.11.0+cu128/TeslaT4 matches the completed base controls; this display difference is recorded rather than hidden. Fixed-fold byte SHA and both raw CSV byte SHAs passed in actual startup logs. Cache generation and training have completed; the strict full result audit is recorded below.

All ten IMU models retain50epochs. After completion download second_place_hybrid_experiments.zip via normalUI into a new recovery/import directory, audit each arm and new-loss provenance, then preserve and merge the original ten all/all_rot arms with identical per-fold joint axes/trainIDs/seed. Report all8151 sequence OOF, eachfold/mean/sampleSD, three missing-sensor scenarios and all three fixed causal history orders. The existing CPU online PID2040 remains running and its source is unchanged.

## Hybrid result importer prepared (real archive still pending)

`scripts/evaluate_second_place_hybrid.py` checks the exact registered sourceURL,
all ten new IMU arms and their auxiliary-loss/unchanged hyperparameter
provenance, then rechecks every original copied file SHA and all20 original
models before copying the unchanged ToF arms into a fresh package. It rejects
joint-axis, train/validation identity, source and seed drift. Rebuilt routes
include explicit sequenceIDs. The existing strict auditor then verifies the
20-arm package and all OOF/history predictions; a final matched base replay
recomputes original independent and three-history mean/sampleSD controls.
The separate routing-audit record precedes the final hybrid comparison, so a
post-audit failure cannot masquerade as a completed comparison.

Two protocol/identity/axis guard tests passed. `--help` loads the importer.
These are implementation checks; the real download and full importer have not
yet run. Preserve all source files and use a new name if import fails.

```powershell
& 'D:\anaconda\envs\cmi\python.exe' -s scripts/evaluate_second_place_hybrid.py --archive <new-recovered-ZIP> --work-dir outputs/second_place/hybrid_import_v1 --output-dir outputs/kaggle_training/imported/second_place_hybrid_imu_group_loss_v1 --name second_place_hybrid_imu_group_loss_v1 --source-url 'https://www.kaggle.com/code/mingweiwei03/cmi-second-place-hybrid-imu-fixed-folds?scriptVersionId=356690453'
```

## Actual complete hybrid comparison: 2026-10-09

Version356690453 completed in11678.5seconds (3h14m38.5s). The newZIP is129107788bytes, SHA2567ce9736f233ea71b4633bc31439f9dbb9d88c3cee62ec39f51f66b9560cc410e. NormalUI download event timed out, but the actual Downloads file was checked and copied to a fresh recovery directory. All ten new IMU50epoch last weights/history/provenance/IDs and original arms passed; the new20arm package preserves original ToF weights, per-fold axes/trainIDs/seed and all source archive/copied-file hashes. Final matched base replay also passed.

Record: `experiments/results/second_place_hybrid_imu_group_loss_v1.json`; separate strict routing audit retained. All15 saved OOF files (including frozen-method controls) each cover8151 unique sequences; every fold mean and sampleSD was recomputed from the saved OOF. No truth was supplied to causal inference; all fixed orders were reset and reported.

| Scenario | Original base mean±sampleSD | Hybrid mean±sampleSD | Delta |
|---|---:|---:|---:|
| observed | 0.891362 ± 0.012919 | 0.891716 ± 0.012698 | +0.000354 |
| aux_dropout50 | 0.861310 ± 0.013242 | 0.861624 ± 0.011197 | +0.000314 |
| imu_only | 0.831799 ± 0.013723 | 0.832369 ± 0.012404 | +0.000569 |

Independent hybrid scores by fold:
| Fold | observed | aux_dropout50 | imu_only |
|---|---:|---:|---:|
| 0 | 0.889852 | 0.862009 | 0.837305 |
| 1 | 0.894932 | 0.864715 | 0.839346 |
| 2 | 0.874979 | 0.848246 | 0.814194 |
| 3 | 0.910194 | 0.878060 | 0.845310 |
| 4 | 0.888622 | 0.855091 | 0.825688 |

Causal history: all fixed orders, no selection. Each cell is hybrid mean±sampleSD (delta from original base).
| Order | observed | aux_dropout50 | imu_only |
|---|---:|---:|---:|
| 42 | 0.921325 ± 0.010253 (+0.000437) | 0.892203 ± 0.010703 (-0.000997) | 0.866801 ± 0.013070 (+0.002219) |
| 142 | 0.920567 ± 0.010315 (+0.000846) | 0.893480 ± 0.011102 (+0.000014) | 0.866270 ± 0.010560 (+0.001758) |
| 242 | 0.921900 ± 0.009483 (+0.000317) | 0.891393 ± 0.009337 (-0.002972) | 0.864564 ± 0.008767 (+0.000121) |

Independent gains are small (+0.000314–0.000569), and history effects are mixed: aux_dropout50 orders42/242 decrease. This trial does not establish a robust universal improvement or an online gain. Probability fusion remains a separate completed screening result; online pseudo-label contribution is still running on CPU. Dynamic34D/THM and loss retraining on ToF remain untested. Original three architectures/ten-fold weights are not reproduced.

Normal accountUI after completion explicitly shows GPU0h45m available of30h, saved as `outputs/second_place/evidence/kaggle_hybrid_v1_complete_quota.jpg`. This does not support new complete architecture training; no newGPU run is started. No push, competition submission or frozen production replacement. Keep the cmi follow-up while the CPU experiment remains active.
