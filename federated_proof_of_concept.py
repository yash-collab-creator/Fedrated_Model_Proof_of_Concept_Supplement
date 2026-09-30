"""
Reproducible proof-of-concept simulation for:
"A Privacy and Connectivity-Aware Federated Learning Framework for Rural Healthcare in India:
 Integrating Non-IID optimisation, Tiered Aggregation, and Consent-Aware Governance"

This script reproduces the five conditions reported in Table V of the manuscript.
It uses the public Breast Cancer Wisconsin (Diagnostic) benchmark supplied by
scikit-learn, synthetic class-wise Dirichlet client partitions, FedAvg, 25 global
rounds, 2 local epochs, SGD (lr=0.03), and seeds 0-2.

Important:
- The Gaussian update perturbation (noise=0.01) is a utility stress test only.
  It is NOT differential privacy and no epsilon/delta guarantee is claimed.
- The target is 10 synthetic client slots. With alpha=0.1, the Dirichlet partition
  can produce very small allocations; the implementation retains only clients
  meeting the minimum local-size requirement (4 samples) for stable local
  75/25 train/test splitting. Therefore the effective retained client count can
  be below 10 for some random seeds. This is reflected in raw_runs.csv.
- The simulation is not clinical validation and does not model real rural PHC
  connectivity, secure aggregation, consent workflows, or deployment.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch  # pyright: ignore[reportMissingImports]
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)
DEVICE = "cpu"

SEEDS = [0, 1, 2]
TARGET_CLIENTS = 10
GLOBAL_ROUNDS = 25
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.03
DROPOUT_VALUES = [0.0, 0.2]
NOISE_VALUES = [0.0, 0.01]
ALPHA_VALUES = [100.0, 0.1]
MIN_CLIENT_SAMPLES = 4


class MLP(torch.nn.Module):
    """Two-layer MLP: 30-32-2, as specified in the manuscript."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_average(states, weights):
    """Sample-count weighted FedAvg over participating clients."""
    if not states:
        raise RuntimeError("No client participated in this round.")
    total_weight = float(np.sum(weights))
    averaged = {}
    for key in states[0]:
        averaged[key] = (
            sum(states[i][key].float() * float(weights[i])
                for i in range(len(states)))
            / total_weight
        )
    return averaged


def local_train(global_state, X, y, epochs=LOCAL_EPOCHS,
                lr=LEARNING_RATE, noise=0.0, clip=1.0):
    """
    Local SGD training followed by optional Gaussian perturbation of the
    model update. The perturbation is a utility stress test, not DP.
    """
    model = MLP(X.shape[1])
    model.load_state_dict(global_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()

        # Gradient clipping used before the optional post-update noise test.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    local_state = model.state_dict()
    protected_update = {}

    for key, value in local_state.items():
        delta = value - global_state[key]
        if noise > 0:
            delta = delta + torch.randn_like(delta) * noise
        protected_update[key] = global_state[key] + delta

    return protected_update


def evaluate_model(state, X, y):
    model = MLP(X.shape[1])
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        predictions = (
            model(torch.tensor(X, dtype=torch.float32))
            .argmax(1)
            .numpy()
        )

    return (
        accuracy_score(y, predictions),
        f1_score(y, predictions, average="macro"),
    )


def make_clients(X, y, n_clients=TARGET_CLIENTS, alpha=0.1, seed=0):
    """
    Class-wise Dirichlet partition followed by a local 75/25 split.

    The manuscript describes ten synthetic client slots. Very small allocations
    are excluded because they cannot support a stable local 75/25 split.
    """
    rng = np.random.default_rng(seed)
    clients = []

    indices_by_class = [
        np.where(y == class_id)[0].copy()
        for class_id in np.unique(y)
    ]

    allocations = [[] for _ in range(n_clients)]

    for indices in indices_by_class:
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        chunks = np.split(indices, cuts)

        for client_id, chunk in enumerate(chunks):
            allocations[client_id].extend(chunk.tolist())

    rng.shuffle(allocations)

    for allocation in allocations:
        if len(allocation) < MIN_CLIENT_SAMPLES:
            continue

        labels = y[allocation]
        stratify = None
        if (
            len(np.unique(labels)) > 1
            and np.min(np.bincount(labels)) >= 2
        ):
            stratify = labels

        train_idx, test_idx = train_test_split(
            allocation,
            test_size=0.25,
            random_state=seed,
            stratify=stratify,
        )

        clients.append(
            (
                X[train_idx],
                y[train_idx],
                X[test_idx],
                y[test_idx],
            )
        )

    return clients


def run_experiment(seed, dropout, noise, alpha):
    """Run one manuscript condition for one random seed."""
    set_seed(seed)

    X, y = load_breast_cancer(return_X_y=True)
    # `return_X_y=True` is typed as a generic tuple by some scikit-learn
    # versions; normalize both values to NumPy arrays for runtime and Pylance.
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    clients = make_clients(
        X_train,
        y_train,
        n_clients=TARGET_CLIENTS,
        alpha=alpha,
        seed=seed,
    )

    base_model = MLP(X_train.shape[1])
    global_state = {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }

    rng = np.random.default_rng(seed + 1000)

    for _round in range(GLOBAL_ROUNDS):
        states = []
        weights = []
        participating_ids = []

        for client_id, (X_client, y_client, _, _) in enumerate(clients):
            if rng.random() < dropout:
                continue

            state = local_train(
                global_state,
                X_client,
                y_client,
                epochs=LOCAL_EPOCHS,
                lr=LEARNING_RATE,
                noise=noise,
                clip=1.0,
            )

            states.append(state)
            weights.append(len(y_client))
            participating_ids.append(client_id)

        if not states:
            # Defensive fallback for an extremely unlikely all-client-dropout round.
            continue

        global_state = state_average(states, weights)

        # Keep the same per-round evaluation path used when the manuscript
        # results were generated. Besides providing convergence history, this
        # preserves deterministic RNG consumption for exact reproduction.
        _ = evaluate_model(global_state, X_test, y_test)
        for client_id in participating_ids:
            _, _, X_client_test, y_client_test = clients[client_id]
            _ = evaluate_model(global_state, X_client_test, y_client_test)

    test_accuracy, test_macro_f1 = evaluate_model(
        global_state, X_test, y_test
    )

    client_f1 = [
        evaluate_model(global_state, client[2], client[3])[1]
        for client in clients
    ]

    return {
        "seed": seed,
        "method": "FedAvg",
        "alpha": alpha,
        "dropout": dropout,
        "noise_std": noise,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "worst_client_f1": min(client_f1),
        "mean_client_f1": float(np.mean(np.asarray(client_f1, dtype=float))),
        "target_clients": TARGET_CLIENTS,
        "retained_clients": len(clients),
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
    }


def main():
    # Exactly the five conditions reported in manuscript Table V.
    conditions = [
        (100.0, 0.0, 0.0),
        (100.0, 0.2, 0.0),
        (0.1, 0.0, 0.0),
        (0.1, 0.2, 0.0),
        (0.1, 0.0, 0.01),
    ]

    rows = []
    for alpha, dropout, noise in conditions:
        for seed in SEEDS:
            rows.append(
                run_experiment(
                    seed=seed,
                    dropout=dropout,
                    noise=noise,
                    alpha=alpha,
                )
            )

    raw = pd.DataFrame(rows)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(output_dir, "raw_runs.csv")
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_alpha_path = os.path.join(output_dir, "summary_by_alpha.csv")
    config_path = os.path.join(output_dir, "experiment_config.json")

    raw.to_csv(raw_path, index=False)

    grouped = (
        raw.groupby(["alpha", "dropout", "noise_std"])
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            worst_client_f1_mean=("worst_client_f1", "mean"),
            worst_client_f1_std=("worst_client_f1", "std"),
            mean_client_f1_mean=("mean_client_f1", "mean"),
            mean_client_f1_std=("mean_client_f1", "std"),
        )
        .reset_index()
    )

    # Human-readable Table V-style summary.
    table_v = grouped.rename(
        columns={
            "alpha": "partition_alpha",
            "noise_std": "update_noise_std",
            "test_accuracy_mean": "test_accuracy_mean",
            "test_accuracy_std": "test_accuracy_std",
            "test_macro_f1_mean": "macro_f1_mean",
            "test_macro_f1_std": "macro_f1_std",
            "worst_client_f1_mean": "worst_client_f1_mean",
            "worst_client_f1_std": "worst_client_f1_std",
        }
    )
    table_v["dropout_percent"] = (table_v["dropout"] * 100).astype(int)

    table_v = table_v[
        [
            "partition_alpha",
            "dropout_percent",
            "update_noise_std",
            "test_accuracy_mean",
            "test_accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "worst_client_f1_mean",
            "worst_client_f1_std",
            "mean_client_f1_mean",
            "mean_client_f1_std",
        ]
    ]

    # Match the row order of manuscript Table V exactly.
    desired_order = [
        (100.0, 0, 0.00),
        (100.0, 20, 0.00),
        (0.1, 0, 0.00),
        (0.1, 20, 0.00),
        (0.1, 0, 0.01),
    ]
    order_map = {key: i for i, key in enumerate(desired_order)}
    table_v["_order"] = table_v.apply(
        lambda row: order_map[
            (
                float(row["partition_alpha"]),
                int(row["dropout_percent"]),
                float(row["update_noise_std"]),
            )
        ],
        axis=1,
    )
    table_v = (
        table_v.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    table_v.to_csv(summary_path, index=False)
    table_v.to_csv(summary_alpha_path, index=False)

    config = {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "source": "scikit-learn load_breast_cancer / UCI Machine Learning Repository",
            "instances": 569,
            "features": 30,
            "task": "binary classification",
        },
        "split": {
            "global_train_fraction": 0.80,
            "local_train_fraction": 0.75,
            "local_test_fraction": 0.25,
        },
        "model": {
            "architecture": "MLP 30-32-2",
            "optimizer": "SGD",
            "learning_rate": 0.03,
        },
        "federated_training": {
            "method": "FedAvg",
            "target_clients": 10,
            "global_rounds": 25,
            "local_epochs": 2,
            "seeds": [0, 1, 2],
            "partitions": [
                {"alpha": 100.0, "description": "near-IID comparison"},
                {"alpha": 0.1, "description": "strong non-IID"},
            ],
        },
        "stress_conditions": {
            "client_dropout": [0.0, 0.2],
            "update_noise_std": [0.0, 0.01],
            "noise_interpretation": (
                "Gaussian update perturbation is a utility stress test only; "
                "no differential-privacy epsilon/delta guarantee is claimed."
            ),
        },
        "reported_conditions": [
            {"alpha": 100.0, "dropout": 0.0, "noise_std": 0.0},
            {"alpha": 100.0, "dropout": 0.2, "noise_std": 0.0},
            {"alpha": 0.1, "dropout": 0.0, "noise_std": 0.0},
            {"alpha": 0.1, "dropout": 0.2, "noise_std": 0.0},
            {"alpha": 0.1, "dropout": 0.0, "noise_std": 0.01},
        ],
        "reproducibility_note": (
            "The Dirichlet partition targets 10 synthetic client slots. "
            "For alpha=0.1, very small allocations can be excluded by the "
            "minimum local-size rule; raw_runs.csv records retained_clients. "
            "This reproduces the supplied manuscript results."
        ),
    }

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("Generated:")
    print(raw_path)
    print(summary_path)
    print(summary_alpha_path)
    print(config_path)
    print("\nTable V conditions:")
    print(table_v.to_string(index=False))


if __name__ == "__main__":
    main()
