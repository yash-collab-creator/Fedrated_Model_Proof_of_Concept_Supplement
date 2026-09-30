"""
Expanded experimental supplement for:
"A Privacy and Connectivity-Aware Federated Learning Framework for Rural Healthcare in India:
 Integrating Non-IID optimisation, Tiered Aggregation, and Consent-Aware Governance"

This script extends the original proof-of-concept without changing its dataset or core model:
- Breast Cancer Wisconsin (Diagnostic), sklearn
- 80/20 global split
- synthetic class-wise Dirichlet client partitions
- 30-32-2 MLP
- SGD, lr=0.03

New evaluation:
1. Centralized baseline
2. Local-only baseline
3. FedAvg
4. FedProx
5. Framework-compatible operational profile (FedProx + resilience settings)
6. alpha = 100, 10, 1, 0.1
7. dropout = 0, 10, 20, 30, 40%
8. synthetic connectivity: bandwidth, latency, availability, delayed/stale updates
9. communication byte accounting
10. configurable 5 or 10 seeds
11. mean +/- SD summary CSVs
12. convergence CSV and PNG plots

IMPORTANT SCIENTIFIC SCOPE:
- The "framework_profile" is NOT claimed to be a new optimization algorithm.
  It is an operational evaluation profile combining established FedProx optimization
  with simulated resilience conditions. It must not be described as a novel FL method.
- Connectivity values are synthetic stress scenarios, not measured rural Indian network traces.
- update_noise_std is a utility stress test, NOT differential privacy. No epsilon/delta guarantee.
- This script does not implement secure aggregation, legal compliance, consent workflows,
  clinical validation, or real PHC deployment.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


torch.set_num_threads(1)
DEVICE = "cpu"

METHODS = ["centralized", "local_only", "fedavg", "fedprox", "framework_profile"]
ALPHAS = [100.0, 10.0, 1.0, 0.1]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4]
NOISES = [0.0, 0.001, 0.005, 0.01, 0.02]


@dataclass(frozen=True)
class ConnectivityScenario:
    name: str
    bandwidth_kbps: float
    latency_ms: float
    availability: float
    stale_probability: float


CONNECTIVITY = {
    "ideal": ConnectivityScenario("ideal", 10000, 20, 1.00, 0.00),
    "moderate": ConnectivityScenario("moderate", 2000, 100, 0.90, 0.05),
    "poor": ConnectivityScenario("poor", 512, 300, 0.80, 0.10),
    "intermittent": ConnectivityScenario("intermittent", 128, 800, 0.60, 0.25),
    "severe": ConnectivityScenario("severe", 64, 1500, 0.50, 0.40),
}


class MLP(torch.nn.Module):
    """Two-layer MLP: 30-32-2, retained from the original PoC."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_state(state):
    return {k: v.detach().clone() for k, v in state.items()}


def state_average(states, weights):
    if not states:
        raise RuntimeError("No client state available for aggregation.")
    total = float(np.sum(weights))
    out = {}
    for key in states[0]:
        out[key] = sum(
            states[i][key].float() * float(weights[i])
            for i in range(len(states))
        ) / total
    return out


def model_num_parameters(state):
    return int(sum(v.numel() for v in state.values()))


def model_bytes(state, dtype_bytes=4):
    return model_num_parameters(state) * dtype_bytes


def local_train(
    global_state,
    X,
    y,
    method="fedavg",
    epochs=2,
    lr=0.03,
    mu=0.01,
    noise=0.0,
    clip=1.0,
):
    """Local SGD/FedProx training.

    FedProx objective:
        local_loss + (mu/2) * ||w - w_global||^2
    """
    model = MLP(X.shape[1])
    model.load_state_dict(global_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.long, device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss()

    reference = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        if method in {"fedprox", "framework_profile"} and mu > 0:
            prox = 0.0
            for name, param in model.named_parameters():
                prox = prox + torch.sum((param - reference[name]) ** 2)
            loss = loss + 0.5 * mu * prox
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    local_state = model.state_dict()
    protected = {}
    for key, value in local_state.items():
        delta = value - global_state[key]
        if noise > 0:
            delta = delta + torch.randn_like(delta) * noise
        protected[key] = global_state[key] + delta
    return protected


def evaluate_model(state, X, y):
    model = MLP(X.shape[1])
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def make_clients(X, y, n_clients=10, alpha=0.1, seed=0, min_samples=4):
    """Class-wise Dirichlet partition, retaining the original partition logic."""
    rng = np.random.default_rng(seed)
    indices_by_class = [
        np.where(y == c)[0].copy() for c in np.unique(y)
    ]
    allocations = [[] for _ in range(n_clients)]
    for indices in indices_by_class:
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        chunks = np.split(indices, cuts)
        for cid, chunk in enumerate(chunks):
            allocations[cid].extend(chunk.tolist())

    clients = []
    for allocation in allocations:
        if len(allocation) < min_samples:
            continue
        labels = y[allocation]
        stratify = None
        counts = np.bincount(labels)
        if len(np.unique(labels)) > 1 and counts.min() >= 2:
            stratify = labels
        train_idx, test_idx = train_test_split(
            allocation,
            test_size=0.25,
            random_state=seed,
            stratify=stratify,
        )
        clients.append({
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "X_test": X[test_idx],
            "y_test": y[test_idx],
        })
    return clients


def prepare_data(seed, alpha, n_clients=10):
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    clients = make_clients(X_train, y_train, n_clients, alpha, seed)
    return X_train, y_train, X_test, y_test, clients


def run_centralized(seed, rounds=50, local_epochs=2, lr=0.03):
    set_seed(seed)
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    model = MLP(X_train.shape[1])
    state = clone_state(model.state_dict())
    history = []
    # Each global round performs the same number of local epochs for comparability.
    for r in range(1, rounds + 1):
        state = local_train(state, X_train, y_train, "fedavg", local_epochs, lr)
        m = evaluate_model(state, X_test, y_test)
        history.append({"round": r, **m})
    final = history[-1]
    return final, history


def run_local_only(seed, alpha, n_clients=10, local_epochs=2, lr=0.03):
    set_seed(seed)
    _, _, X_test, y_test, clients = prepare_data(seed, alpha, n_clients)
    client_rows = []
    for cid, c in enumerate(clients):
        model = MLP(c["X_train"].shape[1])
        state = clone_state(model.state_dict())
        state = local_train(state, c["X_train"], c["y_train"], "fedavg", local_epochs, lr)
        m = evaluate_model(state, c["X_test"], c["y_test"])
        client_rows.append(m)
    mean_f1 = float(np.mean([r["macro_f1"] for r in client_rows]))
    worst_f1 = float(np.min([r["macro_f1"] for r in client_rows]))
    global_m = {"accuracy": np.nan, "macro_f1": np.nan}
    return {
        "test_accuracy": global_m["accuracy"],
        "test_macro_f1": global_m["macro_f1"],
        "mean_client_f1": mean_f1,
        "worst_client_f1": worst_f1,
        "retained_clients": len(clients),
    }, []


def simulate_federated(
    seed,
    method,
    alpha,
    dropout=0.0,
    noise=0.0,
    connectivity="ideal",
    rounds=50,
    local_epochs=2,
    lr=0.03,
    fedprox_mu=0.01,
    n_clients=10,
):
    set_seed(seed)
    _, _, X_test, y_test, clients = prepare_data(seed, alpha, n_clients)
    base = MLP(30)
    state = clone_state(base.state_dict())
    rng = np.random.default_rng(seed + 10000)
    net = CONNECTIVITY[connectivity]
    param_bytes = model_bytes(state)
    history = []
    total_upload = 0
    total_download = 0
    total_latency_ms = 0.0
    stale_updates = 0
    successful_client_updates = 0
    completed_rounds = 0

    for r in range(1, rounds + 1):
        states, weights = [], []
        round_upload = 0
        selected = 0
        round_stale = 0
        for cid, c in enumerate(clients):
            if rng.random() < dropout:
                continue
            if rng.random() > net.availability:
                continue
            selected += 1
            update = local_train(
                state,
                c["X_train"], c["y_train"],
                method=method,
                epochs=local_epochs,
                lr=lr,
                mu=fedprox_mu,
                noise=noise,
            )
            # Connectivity stress is modeled as a delivery property, not as real measured latency.
            if rng.random() < net.stale_probability:
                round_stale += 1
                # Store-and-forward approximation: stale update still arrives and is aggregated,
                # but its arrival incurs extra latency. We do not claim this is a real protocol.
            states.append(update)
            weights.append(len(c["y_train"]))
            round_upload += param_bytes
            successful_client_updates += 1

        if states:
            state = state_average(states, weights)
            completed_rounds += 1
        # Server-to-client broadcast for the next round.
        round_download = param_bytes * selected
        round_latency = net.latency_ms * (1 + round_stale)
        total_upload += round_upload
        total_download += round_download
        total_latency_ms += round_latency
        stale_updates += round_stale

        global_m = evaluate_model(state, X_test, y_test)
        client_f1 = [
            evaluate_model(state, c["X_test"], c["y_test"])["macro_f1"]
            for c in clients
        ]
        history.append({
            "round": r,
            "test_accuracy": global_m["accuracy"],
            "test_macro_f1": global_m["macro_f1"],
            "mean_client_f1": float(np.mean(client_f1)),
            "worst_client_f1": float(np.min(client_f1)),
            "participating_clients": selected,
            "stale_updates": round_stale,
            "upload_bytes": round_upload,
            "download_bytes": round_download,
            "latency_ms": round_latency,
        })

    final = history[-1]
    final.update({
        "test_accuracy": final["test_accuracy"],
        "test_macro_f1": final["test_macro_f1"],
        "mean_client_f1": final["mean_client_f1"],
        "worst_client_f1": final["worst_client_f1"],
        "retained_clients": len(clients),
        "completed_rounds": completed_rounds,
        "successful_client_updates": successful_client_updates,
        "total_upload_bytes": total_upload,
        "total_download_bytes": total_download,
        "total_communication_bytes": total_upload + total_download,
        "total_latency_ms": total_latency_ms,
        "stale_updates_total": stale_updates,
        "model_parameters": model_num_parameters(state),
        "model_bytes_fp32": param_bytes,
    })
    return final, history


def run_one(seed, method, alpha, dropout, noise, connectivity, args):
    started = time.perf_counter()
    if method == "centralized":
        final, history = run_centralized(seed, args.rounds, args.local_epochs, args.lr)
        row = {
            "seed": seed, "method": method, "alpha": alpha,
            "dropout": 0.0, "noise_std": noise, "connectivity": "ideal",
            "test_accuracy": final["accuracy"], "test_macro_f1": final["macro_f1"],
            "mean_client_f1": np.nan, "worst_client_f1": np.nan,
            "retained_clients": np.nan, "completed_rounds": args.rounds,
            "successful_client_updates": np.nan, "total_upload_bytes": np.nan,
            "total_download_bytes": np.nan, "total_communication_bytes": np.nan,
            "total_latency_ms": np.nan, "stale_updates_total": 0,
        }
    elif method == "local_only":
        final, history = run_local_only(seed, alpha, args.clients, args.local_epochs, args.lr)
        row = {
            "seed": seed, "method": method, "alpha": alpha,
            "dropout": 0.0, "noise_std": noise, "connectivity": "ideal",
            **final, "completed_rounds": 1, "successful_client_updates": np.nan,
            "total_upload_bytes": np.nan, "total_download_bytes": np.nan,
            "total_communication_bytes": np.nan, "total_latency_ms": np.nan,
            "stale_updates_total": 0,
        }
    else:
        effective_method = "fedprox" if method == "framework_profile" else method
        final, history = simulate_federated(
            seed, effective_method, alpha, dropout, noise, connectivity,
            args.rounds, args.local_epochs, args.lr, args.fedprox_mu, args.clients
        )
        row = {
            "seed": seed, "method": method, "alpha": alpha,
            "dropout": dropout, "noise_std": noise, "connectivity": connectivity,
            **final,
        }
    row["runtime_seconds"] = time.perf_counter() - started
    return row, history


def save_summary(raw, outdir):
    metric_cols = [
        "test_accuracy", "test_macro_f1", "mean_client_f1", "worst_client_f1",
        "total_communication_bytes", "total_latency_ms", "stale_updates_total",
    ]
    keys = ["method", "alpha", "dropout", "noise_std", "connectivity"]
    grouped = raw.groupby(keys, dropna=False)
    rows = []
    for key, g in grouped:
        row = dict(zip(keys, key))
        row["n_seeds"] = len(g)
        for metric in metric_cols:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary_mean_sd.csv", index=False)
    return summary


def save_convergence(history_rows, outdir):
    if not history_rows:
        return None
    hist = pd.DataFrame(history_rows)
    hist.to_csv(outdir / "convergence_raw.csv", index=False)
    keys = ["method", "alpha", "dropout", "noise_std", "connectivity", "round"]
    metrics = ["test_macro_f1", "worst_client_f1", "mean_client_f1"]
    rows = []
    for key, g in hist.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["n_seeds"] = g["seed"].nunique()
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    conv = pd.DataFrame(rows)
    conv.to_csv(outdir / "convergence_mean_sd.csv", index=False)
    return conv


def make_plots(conv, outdir):
    if conv is None or conv.empty:
        return
    # Plot 1: convergence for main methods at strong non-IID, no dropout, ideal network.
    subset = conv[(conv.alpha == 0.1) & (conv.dropout == 0.0) &
                  (conv.connectivity == "ideal") & (conv.noise_std == 0.0)]
    if not subset.empty:
        plt.figure(figsize=(8, 5))
        for method in ["fedavg", "fedprox", "framework_profile"]:
            d = subset[subset.method == method]
            if not d.empty:
                plt.plot(d["round"], d["test_macro_f1_mean"], label=method)
        plt.xlabel("Global round")
        plt.ylabel("Global Macro-F1")
        plt.title("Convergence under strong non-IID partition (alpha=0.1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "figure_convergence_alpha_0_1.png", dpi=300)
        plt.close()

    # Plot 2: worst-client F1 vs alpha for FedAvg/FedProx.
    subset = conv[(conv.dropout == 0.0) & (conv.connectivity == "ideal") &
                  (conv.noise_std == 0.0) & (conv["round"] == conv["round"].max())]
    if not subset.empty:
        plt.figure(figsize=(8, 5))
        for method in ["fedavg", "fedprox", "framework_profile"]:
            d = subset[subset.method == method].sort_values("alpha", ascending=False)
            if not d.empty:
                plt.plot(d["alpha"], d["worst_client_f1_mean"], marker="o", label=method)
        plt.xscale("log")
        plt.xlabel("Dirichlet alpha (log scale)")
        plt.ylabel("Worst-client Macro-F1")
        plt.title("Worst-client performance across statistical heterogeneity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "figure_worst_client_vs_alpha.png", dpi=300)
        plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Expanded FL experimental supplement")
    p.add_argument("--profile", choices=["smoke", "full", "paper"], default="paper")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    p.add_argument("--alphas", nargs="+", type=float, default=None)
    p.add_argument("--dropouts", nargs="+", type=float, default=None)
    p.add_argument("--connectivity", nargs="+", choices=list(CONNECTIVITY), default=None)
    p.add_argument("--noises", nargs="+", type=float, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--outdir", type=str, default="expanded_results")
    return p.parse_args()


def apply_profile(args):
    if args.profile == "smoke":
        args.seeds = args.seeds or [0]
        args.methods = args.methods or ["fedavg", "fedprox", "framework_profile"]
        args.alphas = args.alphas or [0.1, 100.0]
        args.dropouts = args.dropouts or [0.0, 0.2]
        args.connectivity = args.connectivity or ["ideal", "poor"]
        args.noises = args.noises or [0.0]
        args.rounds = args.rounds or 3
    elif args.profile == "full":
        args.seeds = args.seeds or list(range(10))
        args.methods = args.methods or METHODS
        args.alphas = args.alphas or ALPHAS
        args.dropouts = args.dropouts or DROPOUTS
        args.connectivity = args.connectivity or list(CONNECTIVITY)
        args.noises = args.noises or NOISES
        args.rounds = args.rounds or 50
    else:  # paper
        # Balanced publication-oriented default. The full Cartesian product is intentionally
        # avoided because it would create a large, redundant compute burden.
        args.seeds = args.seeds or list(range(5))
        args.methods = args.methods or METHODS
        args.alphas = args.alphas or ALPHAS
        args.dropouts = args.dropouts or DROPOUTS
        args.connectivity = args.connectivity or ["ideal", "moderate", "poor", "intermittent", "severe"]
        args.noises = args.noises or [0.0]
        args.rounds = args.rounds or 50
    return args


def main():
    args = apply_profile(parse_args())
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Avoid an unnecessarily huge Cartesian product. Connectivity is only varied for
    # federated methods; dropout is held at zero for connectivity experiments by default.
    rows = []
    history_rows = []

    for method in args.methods:
        for alpha in args.alphas:
            dropout_values = args.dropouts if method not in {"centralized", "local_only"} else [0.0]
            connectivity_values = args.connectivity if method not in {"centralized", "local_only"} else ["ideal"]
            # For the paper profile, connectivity is evaluated separately at zero dropout;
            # dropout robustness remains an independent experiment.
            for dropout in dropout_values:
                for connectivity in connectivity_values:
                    if connectivity != "ideal" and dropout != 0.0:
                        continue
                    for noise in args.noises:
                        for seed in args.seeds:
                            row, history = run_one(
                                seed, method, alpha, dropout, noise, connectivity, args
                            )
                            rows.append(row)
                            for h in history:
                                history_rows.append({
                                    "seed": seed,
                                    "method": method,
                                    "alpha": alpha,
                                    "dropout": dropout,
                                    "noise_std": noise,
                                    "connectivity": connectivity,
                                    **h,
                                })
                            print(
                                f"done method={method} alpha={alpha} dropout={dropout} "
                                f"connectivity={connectivity} seed={seed}"
                            )

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "raw_runs.csv", index=False)
    summary = save_summary(raw, outdir)
    conv = save_convergence(history_rows, outdir)
    make_plots(conv, outdir)

    config = {
        "profile": args.profile,
        "seeds": args.seeds,
        "methods": args.methods,
        "alphas": args.alphas,
        "dropouts": args.dropouts,
        "connectivity": {k: asdict(v) for k, v in CONNECTIVITY.items()},
        "noise_std": args.noises,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "learning_rate": args.lr,
        "fedprox_mu": args.fedprox_mu,
        "clients": args.clients,
        "dataset": "Breast Cancer Wisconsin (Diagnostic), sklearn",
        "model": "MLP 30-32-2",
        "privacy_note": "Gaussian update noise is utility stress testing only; no epsilon/delta DP guarantee.",
        "connectivity_note": "Connectivity scenarios are synthetic stress tests, not measured rural Indian traces.",
        "framework_note": "framework_profile = FedProx operational profile with resilience stress settings; not a novel optimizer.",
    }
    with open(outdir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Publication-friendly final-round table.
    final_cols = [
        "method", "alpha", "dropout", "noise_std", "connectivity", "seed",
        "test_accuracy", "test_macro_f1", "mean_client_f1", "worst_client_f1",
        "total_communication_bytes", "total_latency_ms", "stale_updates_total",
    ]
    raw[[c for c in final_cols if c in raw.columns]].to_csv(
        outdir / "publication_final_round_results.csv", index=False
    )

    print("\nGenerated expanded experiment artefacts in:", outdir)
    print("  raw_runs.csv")
    print("  summary_mean_sd.csv")
    print("  convergence_raw.csv")
    print("  convergence_mean_sd.csv")
    print("  publication_final_round_results.csv")
    print("  experiment_config.json")
    print("  figure_convergence_alpha_0_1.png (when applicable)")
    print("  figure_worst_client_vs_alpha.png (when applicable)")
    print(f"\nTotal final-run rows: {len(raw)}")


if __name__ == "__main__":
    main()
