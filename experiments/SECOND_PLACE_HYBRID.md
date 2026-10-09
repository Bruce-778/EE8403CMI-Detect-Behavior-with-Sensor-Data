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

Normal browser Save & Run All accepted on2026-10-09T08:17:10Z. The unique private notebook is Running: https://www.kaggle.com/code/mingweiwei03/cmi-second-place-hybrid-imu-fixed-folds?scriptVersionId=356690453. Official CMI competition input attached, T4x2, internet off, editor Pin to original environment. Logs label the container Latest Container Image, but actual torch2.11.0+cu128/TeslaT4 matches the completed base controls; this display difference is recorded rather than hidden. Fixed-fold byte SHA and both raw CSV byte SHAs passed in actual startup logs. Cache generation is still in progress; no CV result is claimed.

All ten IMU models retain50epochs. After completion download second_place_hybrid_experiments.zip via normalUI into a new recovery/import directory, audit each arm and new-loss provenance, then preserve and merge the original ten all/all_rot arms with identical per-fold joint axes/trainIDs/seed. Report all8151 sequence OOF, eachfold/mean/sampleSD, three missing-sensor scenarios and all three fixed causal history orders. A dedicated strict hybrid merge helper still needs implementation. The existing CPU online PID2040 remains running and its source is unchanged.
