# Paper-ready expanded experimental results

This directory contains results from 180 completed experimental runs using five seeds (0--4) and 50 global rounds.

## Important interpretation
- FedAvg and FedProx are algorithmic baselines. In this configuration they were numerically indistinguishable; do not claim FedProx superiority.
- The framework profile is not treated as a competing algorithm because its ideal-network optimization is FedProx.
- Connectivity scenarios are synthetic stress abstractions, not rural Indian network traces.
- Gaussian update noise is a utility stress test, not a differential-privacy guarantee.

## Files
- `combined_raw_runs_180.csv`: all 180 final-run records.
- `table_baseline_non_iid.csv`: centralized/local/FedAvg/FedProx comparison over alpha values.
- `table_dropout.csv`: 0--40% dropout results.
- `table_connectivity.csv`: synthetic connectivity results.
- `table_noise.csv`: update-noise stress results.
- `MANUSCRIPT_RESULTS_DISCUSSION.tex`: replacement Results/Discussion section based only on measured outputs.
- PNG files: publication figures.
