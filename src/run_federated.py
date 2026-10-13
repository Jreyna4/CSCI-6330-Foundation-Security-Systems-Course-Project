"""Conditions 2-4: federated training in a single-process simulation.

Each round, every client starts from the current global model, trains
`local_epochs` on its own partition, and returns its weights. The server
combines them with the configured aggregation rule. Attacker clients (if
enabled) train on label-flipped data.

Logged per round: validation macro F1 / accuracy of the global model, mean
client training loss, round time. Also logged: bytes exchanged per round
(communication proxy) and, at the end, the global model's macro F1 on each
client's clean local data (client-level dispersion).

Usage:
    python src/run_federated.py --config configs/fedavg_noniid.yaml
    python src/run_federated.py --config configs/fedavg_noniid.yaml --seeds 42
    python src/run_federated.py --config configs/trimmed_noniid_poisoned.yaml --final
"""

import argparse
import time

import numpy as np
import torch

from aggregation import aggregate
from attacks import flip_labels
from common import get_device, load_config, repo_path, rss_mb, save_json, set_seed
from data import load_all, load_partition
from engine import get_weights, make_loader, predict_proba, set_weights, train_epochs
from labels import FAMILIES, FAMILY_TO_ID
from metrics import compute_metrics
from model import build_model


def build_clients(cfg, X, y, client_ids, rng):
    atk = cfg["attack"]
    clients = []
    for c in range(cfg["partition"]["num_clients"]):
        idx = np.flatnonzero(client_ids == c)
        y_clean = y[idx]
        y_used, n_flipped = y_clean, 0
        if atk["enabled"] and c in atk["attacker_clients"]:
            y_used, n_flipped = flip_labels(
                y_clean, atk["flip_fraction"], FAMILY_TO_ID[atk["target_label"]], rng
            )
        clients.append({"X": X[idx], "y_clean": y_clean, "y_train": y_used, "n_flipped": n_flipped})
    return clients


def run(cfg, data, client_ids, seed, final, device):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    t, fed = cfg["train"], cfg["federated"]

    clients = build_clients(cfg, X_train, y_train, client_ids, rng)
    for c, cl in enumerate(clients):
        print(f"client {c}: {len(cl['y_train'])} rows, {cl['n_flipped']} labels flipped")

    global_model = build_model(cfg, X_train.shape[1], len(FAMILIES)).to(device)
    local_model = build_model(cfg, X_train.shape[1], len(FAMILIES)).to(device)
    model_bytes = sum(w.nbytes for w in get_weights(global_model))
    bytes_per_round = 2 * model_bytes * len(clients)  # download + upload per client

    start = time.perf_counter()
    history = []
    for rnd in range(1, fed["rounds"] + 1):
        t0 = time.perf_counter()
        global_weights = get_weights(global_model)
        updates, losses = [], []
        for cl in clients:
            set_weights(local_model, global_weights)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=t["lr"])
            loader = make_loader(cl["X"], cl["y_train"], t["batch_size"])
            losses.append(train_epochs(local_model, loader, optimizer, t["local_epochs"], device))
            updates.append((get_weights(local_model), len(cl["y_train"])))
        set_weights(global_model, aggregate(updates, fed))

        val = compute_metrics(y_val, predict_proba(global_model, X_val, device))
        history.append({
            "round": rnd,
            "val_macro_f1": val["macro_f1"],
            "val_accuracy": val["accuracy"],
            "val_binary_fnr": val["binary_fnr"],
            "mean_client_loss": float(np.mean(losses)),
            "round_sec": time.perf_counter() - t0,
        })
        print(f"seed {seed} | round {rnd:3d} | val macro F1 {val['macro_f1']:.4f}")
    runtime = time.perf_counter() - start

    per_client = []
    for c, cl in enumerate(clients):
        m = compute_metrics(cl["y_clean"], predict_proba(global_model, cl["X"], device))
        per_client.append({"client": c, "rows": len(cl["y_clean"]), "n_flipped": cl["n_flipped"],
                           "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})

    result = {
        "experiment": cfg["experiment"],
        "seed": seed,
        "config": cfg,
        "history": history,
        "val": val,
        "per_client": per_client,
        "client_macro_f1_std": float(np.std([p["macro_f1"] for p in per_client])),
        "bytes_per_round": bytes_per_round,
        "runtime_sec": runtime,
        "rss_mb": rss_mb(),
    }
    if final:
        X_test, y_test = data["test"]
        result["test"] = compute_metrics(y_test, predict_proba(global_model, X_test, device))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--final", action="store_true", help="also evaluate on the test set")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device()

    data = load_all(cfg)
    client_ids = load_partition(cfg)
    if len(client_ids) != len(data["train"][1]):
        raise ValueError("Partition does not match train_pool. Re-run src/make_partitions.py.")
    num_clients = cfg["partition"]["num_clients"]
    if cfg["attack"]["enabled"] and any(c >= num_clients for c in cfg["attack"]["attacker_clients"]):
        raise ValueError(f"attacker_clients must be < num_clients ({num_clients})")

    out_dir = repo_path(cfg["paths"]["results_dir"]) / cfg["experiment"]
    for seed in args.seeds or cfg["seeds"]:
        result = run(cfg, data, client_ids, seed, args.final, device)
        save_json(result, out_dir / f"seed_{seed}.json")


if __name__ == "__main__":
    main()
