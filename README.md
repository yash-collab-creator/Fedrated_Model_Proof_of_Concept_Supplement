# Expanded Federated Learning Experimental Supplement

This supplement extends the original proof-of-concept repository for the manuscript:

**A Privacy and Connectivity-Aware Federated Learning Framework for Rural Healthcare in India: Integrating Non-IID optimisation, Tiered Aggregation, and Consent-Aware Governance**

## What was added

- Centralized baseline
- Local-only baseline
- FedAvg baseline
- FedProx baseline
- Framework-compatible operational profile
- Dirichlet alpha: 100, 10, 1, 0.1
- Client dropout: 0%, 10%, 20%, 30%, 40%
- Synthetic connectivity scenarios: ideal, moderate, poor, intermittent, severe
- Latency, availability and stale-update stress simulation
- Model communication-byte accounting
- Five seeds by default; ten in `--profile full`
- Mean +/- standard deviation summaries
- Convergence CSVs
- Publication-oriented PNG plots

## Important interpretation rule

`framework_profile` is **not a new FL algorithm**. It uses FedProx as the optimization mechanism and evaluates it under the operational resilience settings represented by the framework. Do not describe it as a newly invented optimizer.

Connectivity values are synthetic stress scenarios. They are not measurements of rural Indian PHC networks.

The Gaussian update-noise experiment is a utility stress test only. It does not provide differential privacy and does not calculate epsilon or delta.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Smoke test

```bash
python expanded_federated_experiments.py --profile smoke --outdir smoke_results
```

## Publication-oriented experiment

The default `paper` profile uses 5 seeds, 50 rounds, all four alpha values, the five dropout levels, and the five connectivity scenarios. To avoid an unnecessarily huge Cartesian product, connectivity scenarios are evaluated at zero dropout while dropout robustness is evaluated separately under the ideal network.

```bash
python expanded_federated_experiments.py --profile paper --outdir expanded_results
```

## Full stress sweep

```bash
python expanded_federated_experiments.py --profile full --outdir full_results
```

The full profile uses 10 seeds and all configured stress dimensions and can take substantially longer.

## Outputs

- `raw_runs.csv`: one final result row per seed/condition
- `summary_mean_sd.csv`: mean and sample SD across seeds
- `convergence_raw.csv`: per-round seed-level results
- `convergence_mean_sd.csv`: per-round mean/SD
- `publication_final_round_results.csv`: compact table for manuscript processing
- `experiment_config.json`: exact configuration used
- `figure_convergence_alpha_0_1.png`: convergence plot
- `figure_worst_client_vs_alpha.png`: worst-client heterogeneity plot

## Scientific reporting

The paper should report these as controlled simulation results. They should not be presented as clinical validation, real rural connectivity measurements, legal compliance, or evidence of differential privacy unless separate experiments establish those claims.
