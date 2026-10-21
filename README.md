# Cloud workload forecasting — Prince Maurya

A leakage-safe two-hour forecast for the fixed Atomity `data-v1` Azure VM subset. The solution selects persistence for the represented VM count and a six-lag ridge autoregression for average CPU. Selection uses only the first 110 aggregate observations. The final 24 labels are evaluated only after selection and prediction generation.

## Reproduce from a clean checkout

Tested on Windows with Python 3.12.7. Use Python 3.12 or newer for the pinned reproduction below (the pinned NumPy 2.5.4 requires it). The source package retains Python 3.11 compatibility via its dependency ranges, but that alternative environment has not been execution-tested. No GPU, external ML API, private dataset or credentials are required.

```bash
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or on Linux/macOS with `source .venv/bin/activate`. Then:

```bash
python -m pip install -r requirements-lock.txt
python scripts/download_data.py
python -m pytest -q
python run.py --data challenge_data --output predictions.csv
```

The last line is the single prediction-generation command. It verifies the release hashes, audits inputs, constructs targets, runs temporal model selection, writes predictions, and creates analysis/results/figures. No manual editing of predictions is needed. `run.py` imports the local `src/` implementation, so an editable package installation is unnecessary. `pyproject.toml` also supports ordinary package installation outside this sandbox. A sandbox-specific setuptools editable-build failure was avoided using this portable entry point.

`requirements-lock.txt` pins the tested environment, including transitive dependencies; `requirements.txt` lists the four direct dependencies. The downloader fetches the exact ~116.5 MB archive from https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1, extracts only the required files, and checks SHA-256 values. Alternatively place `vm_cpu_readings.csv.gz`, `vm_metadata.csv.gz`, `DATA_CARD.md`, `SHA256SUMS` under `challenge_data/`.

## Results and decisions

- Training: 110 points; holdout: 24 points at five-minute cadence.
- Expanding origins: 38, 62, 86 observations, each with a disjoint 24-step validation window.
- Candidates: last value; means over 6, 12, 24 points; exponential smoothing with alpha 0.2/0.5; damped recent trend; ridge AR(6), penalty 10.
- Select mean validation MAE separately by target; ties favor the simpler candidate in declared order.
- Selected CPU holdout MAE: 0.098737 percentage points; persistence: 0.146782 (32.73% lower). RMSE: 0.120286 vs 0.184785.
- Active VM count is constant at 10,000. Both persistence and selected count forecast have zero error. Relative error ratios are undefined, not evidence of extraordinary generalization.
- Dataset construction retains complete VM traces, removing observed population churn. The workload is not a representative fleet-arrival/departure benchmark.
- Metadata categories/core buckets are retrospective EDA labels only. Source CPU summaries, deletion timestamps and metadata are not forecast features.
- Time is source-relative seconds, not a real calendar date. No daily/weekly seasonality claim is made from 11.1 hours.

All model settings above were fixed before final holdout evaluation. Re-running the deterministic code repeats the same training-only selection; it is not permission to tune using public test labels.

## Outputs

- `predictions.csv`: exactly 24 rows and the required three columns, source-second timestamps preserved.
- `reports/final_report.md` / `reports/final_report.pdf`: five-part report with workload, methods, validation, failures and operational policy.
- `reports/artifacts/selection.json`: upstream starter revision, input hashes, folds, frozen choices and residual buffer.
- `reports/artifacts/audit.json`, `training_analysis.json`, `segments.csv`, `core_segments.csv`: quantified input/workload analysis.
- `reports/artifacts/validation_folds.csv`, `validation_predictions.csv`: all candidates and forecasts, not only winners.
- `reports/artifacts/holdout_metrics.csv`: absolute errors, relative comparisons, training-threshold peak diagnostics and buffer exceedances.
- `reports/artifacts/figures/`: forecast, workload distribution, segments and validation comparison.
- A separate local `notes/interview_guide.md` is provided to the candidate, not included in the submission archive.

`python scripts/build_report.py` refreshes the Markdown report and print-ready report source from generated artifacts. It does not change predictions or model selection. The checked-in PDF is a convenience export; reproducing predictions does not require PDF tooling.

## Missingness and failure behavior

Duplicate VM/timestamp pairs are rejected rather than silently double-weighted. A valid CPU observation has finite `avg_cpu` in [0,100]; invalid averages are excluded from both targets. Timestamp groups without any valid observations and missing five-minute timestamps fail loudly, with no invented targets. Source metadata identifiers must be unique and match the CPU population. All CPU summary EDA uses the same valid-average definition. Raw missingness and min/max consistency remain separately audited.

Counts are lower-bounded at zero; CPU forecasts are clipped to [0,100] identically during validation and final prediction. Counts need not be integer-valued in the general implementation (expected counts), although this release produces exact integers.

## Small replacement / smoke data

Official mode verifies hashes and enforces the documented 10,000 × 134 shape and 24-step horizon. To test different data intentionally, with the same two compressed CSV schemas and category/core metadata columns:

```bash
python run.py --data tiny_data --skip-checksums --horizon 8 --output tiny_predictions.csv --artifacts smoke_artifacts
```

Without `--horizon`, the forecast horizon remains 24; a 48-point training prefix is sufficient for one temporal fold at that horizon. Smaller inputs require a reduced horizon. `--no-evaluate` still audits inputs and performs training-only selection/EDA, but never computes final held-out errors or plots held-out actuals. The tests mutate final labels and confirm identical predictions and choices, exercise a changing smaller VM population size, and run the full evaluation/plot path on tiny data.

## Operational limits

The +0.421842 percentage-point CPU buffer is an empirical one-sided 95th percentile of selected-model validation residuals, not a calibrated 95% prediction interval. Only 72 residuals, dependent time observations and model-selection reuse limit confidence. Mean CPU is not core-weighted demand, server count, SLA health or a safe consolidation ratio. Do not scale down production resources on this signal alone.

## Provenance and assistance

Challenge source: https://github.com/atomityhq/forecasting-challenge at upstream commit `6e5ac0e4d2f1d81fab0903ce0a974037745db35b`. Data: its fixed `data-v1` release, derived from Microsoft Azure Public Dataset V2. Starter utility files and original assignment documents are retained. Implementation, analysis and documentation were developed with AI-assistant support; all reported metrics were computed by the included code. The candidate should review and be able to explain the submission before sending it.

## Submission archive

`python scripts/package_submission.py` produces `dist/prince-maurya-atomity-forecasting.zip` with source, results, report and per-file SHA-256 manifest. Raw release data, environments, caches and private interview/email notes are excluded.
