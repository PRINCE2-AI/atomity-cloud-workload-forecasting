# Reproducibility record

## Execution sequence

1. Downloaded the starter at upstream revision `6e5ac0e4d2f1d81fab0903ce0a974037745db35b` and fixed `data-v1` archive (116,513,018 bytes).
2. Read DATA_CARD and contracts. Selected a compact candidate set and three 24-step validation folds before inspecting held-out scores.
3. Ran the two original starter tests, then the extended regression suite.
4. Executed the pipeline. It selected persistence for count and ridge AR(6), penalty 10, for CPU from the first 110 points. The pipeline saved selection before writing predictions and before computing final test metrics.
5. After final scores were available, an independent read-only reviewer inspected methodology and recomputed reported metrics. Fixes concerned dependency declarations, relative-metric output, input checks, consistent valid-reading EDA, smoke behavior, documentation and tests. Candidate models, ridge lag/penalty, fold origins and selection rule were NOT changed.
6. Re-ran the deterministic pipeline with those robustness changes; selections and predictions remained the same. The expanded suite passed 20 tests.
7. Generated the report from artifacts and exported a verified five-page PDF. A separate clean-copy/fresh-environment verification was then run, reusing the checksum-verified release files rather than downloading them a second time.

This record describes actual execution ordering, not an independent time-stamped pre-registration. The public holdout must not be reused for subsequent model tuning.

## Interpretation of reported metrics

Validation summary MAE is the mean over three equal-length validation windows. Validation summary RMSE is the mean of three fold RMSEs, not a pooled root mean square. Final test RMSE is the ordinary root mean squared error over 24 predictions. Errors for CPU are measured in percentage points. Relative error is against last-value persistence; relative metrics with a zero-error baseline are explicitly undefined.

The residual headroom buffer uses selected-model validation residuals and is heuristic, with no guaranteed 95% coverage. The final test contains no CPU values above the training 90th percentile. Peak-performance evidence therefore comes from validation, not that test window.

## Environment

Python 3.12.7 on Windows; exact installed package versions are in requirements-lock.txt. The ordinary clean-copy reproduction command is `python run.py --data <verified release directory> --output predictions.csv`. No GPU, paid training service, proprietary data or external model API was used.
