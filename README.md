# Proof-of-Concept Supplementary Artefacts

This folder contains the reproducibility artefacts for the proof-of-concept
simulation described in the manuscript **"A Privacy and Connectivity-Aware
Federated Learning Framework for Rural Healthcare in India: Integrating Non-IID
optimisation, Tiered Aggregation, and Consent-Aware Governance."**

## Files

- `federated_proof_of_concept.py` — complete executable simulation.
- `experiment_config.json` — machine-readable configuration corresponding to the manuscript.
- `raw_runs.csv` — seed-level results for the five conditions reported in Table V.
- `summary.csv` — aggregated mean ± standard deviation inputs for Table V.
- `summary_by_alpha.csv` — same machine-readable summary organised by partition alpha.

## Reproduction

Recommended environment:

- Python 3.10+
- NumPy
- pandas
- scikit-learn
- PyTorch

Run:

```bash
python federated_proof_of_concept.py
```

The script uses the Breast Cancer Wisconsin (Diagnostic) benchmark through
`sklearn.datasets.load_breast_cancer`. No patient-level or private clinical
dataset is included.

## Manuscript alignment

The supplied manuscript specifies:

- 569 instances and 30 real-valued diagnostic features.
- 80/20 global train/test split.
- 10 synthetic client slots.
- Class-wise Dirichlet partitions with alpha = 100 and alpha = 0.1.
- 25 global rounds.
- 2 local epochs per round.
- SGD with learning rate 0.03.
- Three random seeds (0–2).
- 20% client dropout as an operational stress condition.
- Gaussian update perturbation with standard deviation 0.01 as a utility stress test.

The supplement reports only the five conditions shown in manuscript Table V.

### Important client-partition note

For alpha = 0.1, the class-wise Dirichlet allocation can create extremely
small client allocations. The implementation therefore excludes allocations
below four samples before the local 75/25 split. Consequently, the number of
**retained** clients can be below the target of 10 for some seeds. This is
recorded explicitly in `raw_runs.csv`.

### Privacy limitation

The 0.01 Gaussian perturbation is **not claimed to provide differential
privacy**. No privacy accountant, epsilon, or delta is calculated. The
experiment is only a utility-stress test, consistent with the manuscript.

### Scope limitation

These artefacts reproduce a methodological proof of concept. They do not
validate rural Indian PHC deployment, clinical effectiveness, real network
latency/power conditions, secure aggregation, consent workflows, legal
compliance, or clinical decision-making.
