"""Conditions 2-4: federated training in a single-process simulation.

Each round, every client starts from the current global model, trains
`local_epochs` on its own partition with a fresh Adam optimizer, and returns
its full weights. The server combines them with the configured aggregation
rule. All clients participate in every round.

Threat model (fixed and reproducible, per instructor feedback):
  * Attacker choice: `attack.selection: most_attack_rows` picks the
    `attack.num_attackers` clients with the most attack rows in the TRAIN
    partition (ties -> lower id). It depends only on the frozen partition, so
    the SAME clients are malicious for every seed and every condition.
    `selection: fixed` uses `attack.attacker_clients` instead.
  * Eligibility: each attacker must hold >= `min_attack_families` attack
    families with >= `min_rows_per_family` rows each, otherwise the run stops.
  * Poisoned records: `flip_fraction` (50%) of the attacker's attack rows,
    drawn per attack family (so every family is flipped at the same rate),
    relabeled as Benign. Drawn ONCE per (seed, client) from its own RNG, and
    reused unchanged in every round. Same seed -> same poisoned records in the
    FedAvg and trimmed-mean conditions (paired).
  * The attacker only changes labels. It trains, and sends its update,
    exactly like an honest client. Evaluation always uses clean labels.

Seeds: data.seed (42) fixes the subset, splits and partitions for ALL runs.
Each entry of `seeds` changes only (a) model initialization, (b) minibatch
order and dropout masks, and (c) which attack rows are poisoned. Clean,
attacked and defended conditions with the same seed therefore start from
identical initial weights (recorded as init_weights_sha256).

Logged: per-round validation metrics; final global validation (and test with
--final) metrics with support counts; per-client metrics on each client's
held-out records (assigned by the frozen partitioner); communication bytes;
runtime with hardware description.

Usage:
    python src/run_federated.py --config configs/fedavg_noniid.yaml
    python src/run_federated.py --config configs/fedavg_noniid.yaml --seeds 42
    python src/run_federated.py --config configs/pilot.yaml configs/fedavg_noniid.yaml
    python src/run_federated.py --config configs/fedavg_noniid_poisoned.yaml \
        --set attack.num_attackers=2 experiment=fedavg_noniid_poisoned_m2
    python src/run_federated.py --config configs/trimmed_noniid_poisoned.yaml --final
"""

import argparse
import time
import warnings

import numpy as np
import torch

from common import (
    BENIGN_ID,
    FAMILIES,
    FAMILY_TO_ID,
    build_model,
    build_optimizer,
    compute_metrics,
    count_parameters,
    get_device,
    get_weights,
    hardware_info,
    load_all,
    load_config,
    load_manifest,
    load_partition,
    make_loader,
    predict_proba,
    repo_path,
    rss_mb,
    save_json,
    set_seed,
    set_weights,
    sha256_array,
    train_epochs,
)


# --- Attack: label flipping ---------------------------------------------------

def flip_labels(y, flip_fraction, target_id, rng, per_family=True):
    """Return (poisoned copy of y, sorted indices of flipped rows).

    Candidates are rows whose label is not `target_id`. With per_family, the
    fraction is applied within each attack family separately.
    """
    y = y.copy()
    candidates = np.flatnonzero(y != target_id)
    groups = [candidates[y[candidates] == c] for c in np.unique(y[candidates])] if per_family else [candidates]
    chosen = []
    for g in groups:
        n_flip = int(round(flip_fraction * len(g)))
        if n_flip:
            chosen.append(rng.choice(g, size=n_flip, replace=False))
    idx = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int64)
    y[idx] = target_id
    return y, idx


def attack_profile(y, num_clients, client_ids, target_id=BENIGN_ID):
    """Attack rows and per-family counts per client (train partition)."""
    prof = []
    for c in range(num_clients):
        yc = y[client_ids == c]
        fam = np.bincount(yc, minlength=len(FAMILIES))
        fam[target_id] = 0
        prof.append({"client": c, "attack_rows": int(fam.sum()), "family_counts": fam})
    return prof


def eligible_attackers(atk, prof):
    """Clients that hold enough attack families for a meaningful flip."""
    return [p["client"] for p in prof
            if int((p["family_counts"] >= atk["min_rows_per_family"]).sum())
            >= atk["min_attack_families"]]


def select_attackers(atk, y, client_ids, num_clients, data_seed=42):
    """Deterministic attacker choice; does not depend on the run seed.

    selection:
      most_attack_rows  clients with the most attack rows (default; in our
                        Dirichlet split these are also the largest clients)
      median_size       eligible clients whose size is closest to the median
                        client size: a typical-sized attacker
      random            a seeded random draw among eligible clients, fixed by
                        (data seed, attack.random_draw), so the same clients are
                        attackers in every condition and run seed
      fixed             attack.attacker_clients
    Separating attacker size from poisoning was a predicted Milestone 2
    feedback point: with most_attack_rows the attacker is also the largest
    client, which FedAvg weights most.
    """
    target = FAMILY_TO_ID[atk["target_label"]]
    prof = attack_profile(y, num_clients, client_ids, target)
    m = atk["num_attackers"]
    sel = atk["selection"]
    if sel == "fixed":
        chosen = list(atk["attacker_clients"])
    elif sel == "most_attack_rows":
        order = sorted(prof, key=lambda p: (-p["attack_rows"], p["client"]))
        chosen = [p["client"] for p in order[:m]]
    elif sel in ("median_size", "random"):
        pool = eligible_attackers(atk, prof)
        if len(pool) < m:
            raise ValueError(f"only {len(pool)} eligible attacker clients, need {m}")
        if sel == "median_size":
            sizes = np.bincount(client_ids, minlength=num_clients)
            med = float(np.median(sizes))
            chosen = sorted(pool, key=lambda c: (abs(sizes[c] - med), c))[:m]
        else:
            rng = np.random.default_rng([data_seed, 104729, int(atk.get("random_draw", 0))])
            chosen = rng.choice(sorted(pool), size=m, replace=False).tolist()
    else:
        raise ValueError(f"Unknown attack.selection: {sel}")
    if any(c >= num_clients or c < 0 for c in chosen):
        raise ValueError(f"attacker ids {chosen} must be in [0, {num_clients})")

    for c in chosen:
        fam = prof[c]["family_counts"]
        n_ok = int((fam >= atk["min_rows_per_family"]).sum())
        if n_ok < atk["min_attack_families"]:
            msg = (f"attacker client {c} has only {n_ok} attack families with >= "
                   f"{atk['min_rows_per_family']} rows; a 50% flip would not be meaningful.")
            if atk.get("allow_weak_attacker", False):
                warnings.warn(msg)
            else:
                raise ValueError(msg + " Choose another client or set attack.allow_weak_attacker.")
    return sorted(chosen)


# --- Aggregation rules --------------------------------------------------------
# Each rule takes `updates`, a list of (weights, num_examples) pairs where
# weights is a list of numpy arrays (one per model tensor), and returns the
# new global weights in the same format.

def fedavg(updates):
    """FedAvg (McMahan et al., 2017): mean weighted by client dataset size."""
    total = sum(n for _, n in updates)
    return [
        sum(w[i] * n for w, n in updates) / total
        for i in range(len(updates[0][0]))
    ]


def unweighted_mean(updates):
    """Plain mean of the client models, ignoring client size.

    Control condition (instructor feedback): FedAvg weights clients by size,
    trimmed mean does not. Comparing FedAvg -> mean isolates the weighting;
    mean -> trimmed mean isolates the trimming.
    """
    n = len(updates)
    return [sum(w[i] for w, _ in updates) / n for i in range(len(updates[0][0]))]


def trim_count(n, beta):
    """Number of values removed from EACH end per coordinate."""
    return int(np.floor(beta * n + 1e-9))


def trimmed_mean(updates, beta):
    """Coordinate-wise trimmed mean (Yin et al., 2018).

    For every parameter, sort the n client values, drop the k = floor(beta*n)
    largest and k smallest, and take the UNWEIGHTED mean of the remaining
    n - 2k values (client sizes are ignored, unlike FedAvg). With n = 10 and
    beta = 0.2: k = 2, so 4 of 10 values are dropped and 6 averaged.
    """
    n = len(updates)
    k = trim_count(n, beta)
    if 2 * k >= n:
        raise ValueError(f"trim_beta={beta} trims all {n} clients")
    if k == 0:
        warnings.warn(f"trim_beta={beta} trims nothing with {n} clients (plain mean)")
    out = []
    for i in range(len(updates[0][0])):
        stacked = np.sort(np.stack([w[i] for w, _ in updates]), axis=0)
        out.append(stacked[k : n - k].mean(axis=0))
    return out


def median(updates):
    """Coordinate-wise median (extension condition)."""
    return [
        np.median(np.stack([w[i] for w, _ in updates]), axis=0)
        for i in range(len(updates[0][0]))
    ]


def _flat(weights):
    return np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w in weights])


def krum_scores(updates, f):
    """Krum score of each client (Blanchard et al., 2017).

    Every client starts the round from the same global model, so distances
    between client models equal distances between their updates. Score_i =
    sum of squared distances to its n - f - 2 nearest other clients; a low
    score means the update sits among the others.
    """
    n = len(updates)
    if n - f - 2 < 1:
        raise ValueError(f"Krum needs n - f - 2 >= 1 (n={n}, f={f})")
    V = np.stack([_flat(w) for w, _ in updates])
    sq = (V * V).sum(axis=1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * V @ V.T, 0.0)
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, : n - f - 2].sum(axis=1)


def multi_krum(updates, f, m):
    """Unweighted mean of the m clients with the lowest Krum scores (m=1: Krum)."""
    keep = np.argsort(krum_scores(updates, f), kind="stable")[:m]
    kept = [updates[i] for i in keep]
    return unweighted_mean(kept), sorted(int(i) for i in keep)


def aggregate(updates, cfg_fed):
    rule = cfg_fed["aggregation"]
    if rule == "fedavg":
        return fedavg(updates)
    if rule == "mean":
        return unweighted_mean(updates)
    if rule == "trimmed_mean":
        return trimmed_mean(updates, cfg_fed["trim_beta"])
    if rule == "median":
        return median(updates)
    if rule in ("krum", "multi_krum"):
        n, f = len(updates), int(cfg_fed.get("krum_f", 2))
        m = 1 if rule == "krum" else int(cfg_fed.get("multi_krum_m") or n - f)
        return multi_krum(updates, f, m)[0]
    raise ValueError(f"Unknown aggregation rule: {rule}")


def client_diagnostics(updates, global_weights, trim_k, krum_f):
    """Per-client numbers that show whether an update looks unusual.

    trimmed_share  share of the client's coordinates that fall in the k lowest
                   or k highest of the n values, i.e. that coordinate-wise
                   trimming would remove (2k/n = 0.4 for n=10, k=2 if values
                   were in random order)
    dist_to_median L2 distance of the client's update to the coordinate-wise
                   median update
    cos_to_others  cosine similarity of the client's update to the mean update
                   of the other clients
    krum_rank      rank of the client's Krum score (0 = most typical)
    Computed every round for every rule, so the same diagnostic is available
    for FedAvg and the robust rules alike.
    """
    g = _flat(global_weights)
    U = np.stack([_flat(w) - g for w, _ in updates])
    n = len(U)
    ranks = np.argsort(np.argsort(U, axis=0, kind="stable"), axis=0, kind="stable")
    trimmed = ((ranks < trim_k) | (ranks >= n - trim_k)).mean(axis=1) if trim_k else np.zeros(n)
    med = np.median(U, axis=0)
    dist = np.linalg.norm(U - med, axis=1)
    total = U.sum(axis=0)
    cos = []
    for i in range(n):
        others = (total - U[i]) / (n - 1)
        den = np.linalg.norm(U[i]) * np.linalg.norm(others)
        cos.append(float(U[i] @ others / den) if den > 0 else 0.0)
    scores = krum_scores(updates, krum_f)
    krank = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
    return {"trimmed_share": trimmed.round(4).tolist(), "dist_to_median": dist.round(5).tolist(),
            "cos_to_others": [round(c, 4) for c in cos], "krum_rank": krank.tolist(),
            "update_norm": np.linalg.norm(U, axis=1).round(5).tolist()}


# --- Federated simulation -----------------------------------------------------

def build_clients(cfg, X, y, client_ids, attackers, seed):
    atk = cfg["attack"]
    target = FAMILY_TO_ID[atk["target_label"]]
    clients = []
    for c in range(cfg["partition"]["num_clients"]):
        idx = np.flatnonzero(client_ids == c)
        y_clean = y[idx]
        y_used, flipped = y_clean, np.array([], dtype=np.int64)
        if atk["enabled"] and c in attackers:
            # Own RNG per (seed, client): independent of torch randomness, so
            # the clean/attacked/defended runs stay paired.
            rng = np.random.default_rng([seed, c, 7919])
            y_used, flipped = flip_labels(y_clean, atk["flip_fraction"], target, rng,
                                          atk["flip_per_family"])
        clients.append({"X": X[idx], "y_clean": y_clean, "y_train": y_used,
                        "flipped_local_idx": flipped, "pool_idx": idx})
    return clients


def poisoning_record(clients, attackers):
    rec = []
    for c in attackers:
        cl = clients[c]
        fl = cl["flipped_local_idx"]
        orig = cl["y_clean"][fl]
        attack_total = np.bincount(cl["y_clean"], minlength=len(FAMILIES))
        flipped = np.bincount(orig, minlength=len(FAMILIES))
        rec.append({
            "client": c,
            "attack_rows": int(attack_total.sum() - attack_total[BENIGN_ID]),
            "flipped_rows": int(len(fl)),
            "flipped_per_family": {f: int(flipped[i]) for i, f in enumerate(FAMILIES) if i != BENIGN_ID},
            "attack_rows_per_family": {f: int(attack_total[i]) for i, f in enumerate(FAMILIES) if i != BENIGN_ID},
            "flipped_pool_indices_sha256": sha256_array(cl["pool_idx"][fl]),
        })
    return rec


def heldout_client_metrics(model, X, y, client_ids, num_clients, device):
    """Global model evaluated on each client's held-out records."""
    proba = predict_proba(model, X, device)
    out = []
    for c in range(num_clients):
        m = client_ids == c
        res = compute_metrics(y[m], proba[m])
        out.append({"client": c, **{k: res.get(k) for k in
                                    ("n", "macro_f1", "f1_classes", "accuracy",
                                     "binary_fpr", "binary_fnr", "binary_counts",
                                     "macro_pr_auc", "pr_auc_classes", "pr_auc_excluded",
                                     "per_class", "confusion_matrix")}})
    return out


def run(cfg, data, part, seed, final, device):
    """`part` holds client ids for train/val/(test) rows and the attacker list."""
    set_seed(seed)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    t, fed, atk = cfg["train"], cfg["federated"], cfg["attack"]
    k_clients = cfg["partition"]["num_clients"]
    attackers = part["attackers"] if atk["enabled"] else []

    clients = build_clients(cfg, X_train, y_train, part["train"], attackers, seed)
    for c, cl in enumerate(clients):
        tag = " (ATTACKER)" if c in attackers else ""
        print(f"client {c}: {len(cl['y_train'])} rows, {len(cl['flipped_local_idx'])} labels flipped{tag}")

    global_model = build_model(cfg, X_train.shape[1], len(FAMILIES)).to(device)
    init_hash = sha256_array(*get_weights(global_model))
    local_model = build_model(cfg, X_train.shape[1], len(FAMILIES)).to(device)
    model_bytes = int(sum(w.nbytes for w in get_weights(global_model)))
    comm = {
        "unit": "bytes of raw float32 model tensors (no compression, no protocol overhead)",
        "model_parameters": count_parameters(global_model),
        "download_per_client_per_round": model_bytes,  # server -> client (global model)
        "upload_per_client_per_round": model_bytes,    # client -> server (full local model)
        "participating_clients_per_round": k_clients,
        "total_per_round": 2 * model_bytes * k_clients,
        "cumulative_per_run": 2 * model_bytes * k_clients * fed["rounds"],
        "downloads_included": True,
    }

    start = time.perf_counter()
    history = []
    for rnd in range(1, fed["rounds"] + 1):
        t0 = time.perf_counter()
        global_weights = get_weights(global_model)
        updates, losses = [], []
        for cl in clients:
            set_weights(local_model, global_weights)
            optimizer = build_optimizer(cfg, local_model.parameters())  # fresh each round
            loader = make_loader(cl["X"], cl["y_train"], t["batch_size"])
            losses.append(train_epochs(local_model, loader, optimizer, t["local_epochs"], device))
            updates.append((get_weights(local_model), len(cl["y_train"])))
        diag = client_diagnostics(updates, global_weights,
                                  trim_count(k_clients, fed["trim_beta"]), int(fed.get("krum_f", 2)))
        set_weights(global_model, aggregate(updates, fed))

        val = compute_metrics(y_val, predict_proba(global_model, X_val, device))
        history.append({
            "round": rnd,
            "val_macro_f1": val["macro_f1"],
            "val_accuracy": val["accuracy"],
            "val_binary_fnr": val["binary_fnr"],
            "val_binary_fpr": val["binary_fpr"],
            "mean_client_loss": float(np.mean(losses)),
            "client_losses": [float(x) for x in losses],
            "client_diagnostics": diag,
            "round_sec": time.perf_counter() - t0,
            "cumulative_bytes": comm["total_per_round"] * rnd,
        })
        print(f"seed {seed} | round {rnd:3d} | val macro F1 {val['macro_f1']:.4f}")
    runtime = time.perf_counter() - start

    per_client_val = heldout_client_metrics(global_model, X_val, y_val, part["val"], k_clients, device)
    f1s = [p["macro_f1"] for p in per_client_val if p["macro_f1"] is not None]

    result = {
        "experiment": cfg["experiment"],
        "seed": seed,
        "config": cfg,
        "data_manifest_files": (load_manifest(cfg) or {}).get("files"),
        "init_weights_sha256": init_hash,
        "attack": {
            "enabled": atk["enabled"],
            "attackers": attackers,
            "malicious_fraction": len(attackers) / k_clients,
            "selection": atk["selection"],
            "poisoned": poisoning_record(clients, attackers),
        },
        "aggregation": {
            "rule": fed["aggregation"],
            "weighting": "by client size" if fed["aggregation"] == "fedavg" else "unweighted",
            **({"trim_beta": fed["trim_beta"],
                "trimmed_per_side": trim_count(k_clients, fed["trim_beta"]),
                "values_averaged": k_clients - 2 * trim_count(k_clients, fed["trim_beta"]),
                "stress_test": len(attackers) > trim_count(k_clients, fed["trim_beta"]),
                } if fed["aggregation"] == "trimmed_mean" else {}),
            **({"krum_f": int(fed.get("krum_f", 2)),
                "clients_kept": 1 if fed["aggregation"] == "krum"
                else int(fed.get("multi_krum_m") or k_clients - int(fed.get("krum_f", 2)))}
               if fed["aggregation"] in ("krum", "multi_krum") else {}),
        },
        "model_selection": "global model after the last round",
        "history": history,
        "val": val,
        "per_client_val": per_client_val,
        "client_macro_f1_std": float(np.std(f1s)) if f1s else None,
        "client_macro_f1_min": float(np.min(f1s)) if f1s else None,
        "communication": comm,
        "runtime_sec": runtime,
        "runtime_note": "single-machine simulation time",
        "hardware": hardware_info(),
        "rss_mb": rss_mb(),
    }
    if final:
        X_test, y_test = data["test"]
        result["test"] = compute_metrics(y_test, predict_proba(global_model, X_test, device))
        result["per_client_test"] = heldout_client_metrics(
            global_model, X_test, y_test, part["test"], k_clients, device)
    return result


def check_trim_vs_attackers(cfg, attackers):
    fed = cfg["federated"]
    if fed["aggregation"] == "trimmed_mean" and cfg["attack"]["enabled"]:
        k = trim_count(cfg["partition"]["num_clients"], fed["trim_beta"])
        if len(attackers) > k:
            print(f"STRESS TEST: {len(attackers)} attackers > {k} values trimmed per side. "
                  "Attacker values can survive trimming in every coordinate, so this run "
                  "probes failure, not a setting with a robustness guarantee.")
        else:
            print(f"NOTE: {len(attackers)} attacker(s) <= {k} trimmed per side. Even so, "
                  "label flipping need not make the attacker's values the extreme ones, so "
                  "trimming is not guaranteed to remove them.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True,
                        help="one or more YAML files merged on top of default.yaml, in order")
    parser.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE",
                        help="override config keys, e.g. attack.num_attackers=2")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--final", action="store_true", help="also evaluate on the test set")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = get_device()

    data = load_all(cfg)
    part = {s: load_partition(cfg, s) for s in ("train", "val", "test")}
    if len(part["train"]) != len(data["train"][1]) or len(part["val"]) != len(data["val"][1]):
        raise ValueError("Partition does not match the splits. Re-run src/prepare_data.py.")
    k = cfg["partition"]["num_clients"]
    part["attackers"] = (select_attackers(cfg["attack"], data["train"][1], part["train"], k,
                                          cfg["data"]["seed"])
                         if cfg["attack"]["enabled"] else [])
    if part["attackers"]:
        print(f"attackers: {part['attackers']} ({len(part['attackers'])}/{k} clients, "
              f"selection={cfg['attack']['selection']})")
    check_trim_vs_attackers(cfg, part["attackers"])

    out_dir = repo_path(cfg["paths"]["results_dir"]) / cfg["experiment"]
    for seed in args.seeds or cfg["seeds"]:
        result = run(cfg, data, part, seed, args.final, device)
        save_json(result, out_dir / f"seed_{seed}.json")


if __name__ == "__main__":
    main()
