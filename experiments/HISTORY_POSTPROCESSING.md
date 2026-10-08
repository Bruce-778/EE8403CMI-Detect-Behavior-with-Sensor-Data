# H1: causal subject-history assignment

## Predeclared design

This is an evaluation-only feasibility experiment, motivated by the second and
third place solutions reviewed in TOP_SOLUTIONS_REVIEW.md. It is a different
information protocol from independent single-sequence classification. No weights
are trained, no inference bundle is changed and no competition submission is made.

Use the frozen original routed OOF from `outputs/winner_comparison/candidate/routed/`
and the unchanged five subject-wise folds. For each held-out fold, estimate each
of the 18 gesture capacities as its maximum count per **training** subject. Do
not read validation class counts to fit the capacities. For a new sequence, use
only that subject's previously arrived probability vectors and its current vector.
Hungarian assignment maximizes their joint log probability with duplicated class
slots. Return only the newest row's label. Never rewrite a previously returned
prediction. No true orientation, phase, gesture or future rows are predictor inputs.

The latent assignments of past rows may change when the current row arrives;
therefore the internally constrained prefix does not guarantee that previously
returned labels satisfy the final quotas. This matches the distinction between
prefix re-optimization and retroactively rewriting past outputs. If the available
slots are exhausted, fall back to the original probability argmax and record it.

Predeclare all three arrival seeds **42, 142, 242**, with seed + fold used for
each held-out fold. Sort IDs before shuffling; orders do not depend on labels.
Report all orders and their range, never select the best order. Evaluate observed,
fixed aux_dropout50 and imu_only inputs. Keep all 8,151 sequences for each result,
report every fold and sample standard deviation, and save complete decoded OOF
and arrival traces under `outputs/history_postprocessing/pilot_v1/`.

This gesture-only approximation is weaker than the winners' predicted
gesture/orientation/initial-behavior joint classes; it is not their exact method.
The real test call order and number of arrived sequences can differ. This checks
feasibility on development OOF, not the Kaggle score or deployment validity.

```powershell
python -s -m unittest discover -s tests -p test_history_postprocessing.py -v
python -s -u scripts/evaluate_history_postprocessing.py
```

Compact evidence will be saved as `experiments/results/history_postprocessing_v1.json`.
The source OOF hashes, fold fingerprint, fitted capacities, changed prediction
counts, overflow fallbacks and all fixed-order results are retained there.

## Completed result

All nine scenario/order evaluations completed with all 8,151 held-out sequences,
unchanged fold fingerprints and zero capacity-overflow fallbacks. Means and sample
standard deviations below describe five folds; the three seeds describe arrival
orders, not independent retrainings.

| Scenario | Arrival seed | Baseline mean ± sample SD | History mean ± sample SD | Delta mean | Changed labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| observed | 42 | 0.852029 ± 0.008719 | 0.856516 ± 0.009778 | +0.004488 | 360 |
| observed | 142 | 0.852029 ± 0.008719 | 0.857156 ± 0.009911 | +0.005127 | 340 |
| observed | 242 | 0.852029 ± 0.008719 | 0.856433 ± 0.008676 | +0.004404 | 355 |
| aux_dropout50 | 42 | 0.821455 ± 0.011675 | 0.825773 ± 0.010533 | +0.004318 | 387 |
| aux_dropout50 | 142 | 0.821455 ± 0.011675 | 0.824980 ± 0.010653 | +0.003525 | 383 |
| aux_dropout50 | 242 | 0.821455 ± 0.011675 | 0.825021 ± 0.010440 | +0.003566 | 396 |
| imu_only | 42 | 0.787194 ± 0.014102 | 0.791760 ± 0.015203 | +0.004566 | 435 |
| imu_only | 142 | 0.787194 ± 0.014102 | 0.792055 ± 0.015046 | +0.004860 | 435 |
| imu_only | 242 | 0.787194 ± 0.014102 | 0.791344 ± 0.014331 | +0.004150 | 465 |

Across all three fixed orders the mean deltas are +0.004673, +0.003803 and
+0.004525 respectively. Every order improved its scenario's fold mean. This does
not imply every fold improved, statistical significance or an online gain.
The source network probabilities and weights were not modified. The original
scored ensemble remains the production default.

Decision: retain this development evaluation prototype and its evidence; do not
promote it to production on this result alone. It is useful but smaller than the
joint-label/context gains reported by the leading teams, and its protocol must
be validated against real inference call order and observable history.

Detailed per-fold scores, fitted training-fold capacities and source hashes:
[history_postprocessing_v1.json](results/history_postprocessing_v1.json).
Complete OOF and arrival traces: `outputs/history_postprocessing/pilot_v1/`.
Five focused tests check training-only quota fitting, subject separation,
annotation isolation, causal prefix behavior, input copy safety and invalid inputs.
The complete 73-test project suite passed (54.198 seconds). Independent rechecks
recomputed all 45 fold scores and sample SDs and verified all 45 arrival traces.
The original submitted bundle SHA256 remains
`e7af6cd296b54d75cdd9c135f36d78b7dd684dee4437975401722548b8720e19`.
