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
