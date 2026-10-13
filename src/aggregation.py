"""Server-side aggregation rules.

Each rule takes `updates`, a list of (weights, num_examples) pairs where
weights is a list of numpy arrays (one per model tensor), and returns the new
global weights in the same format.
"""

import warnings

import numpy as np


def fedavg(updates):
    """FedAvg (McMahan et al., 2017): mean weighted by client dataset size."""
    total = sum(n for _, n in updates)
    return [
        sum(w[i] * n for w, n in updates) / total
        for i in range(len(updates[0][0]))
    ]


def trimmed_mean(updates, beta):
    """Coordinate-wise trimmed mean (Yin et al., 2018).

    For every parameter, drop the k = floor(beta * n) largest and k smallest
    client values, then average the rest (unweighted). With n = 4 clients,
    beta = 0.25 gives k = 1.
    """
    n = len(updates)
    k = int(np.floor(beta * n + 1e-9))
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


def aggregate(updates, cfg_fed):
    rule = cfg_fed["aggregation"]
    if rule == "fedavg":
        return fedavg(updates)
    if rule == "trimmed_mean":
        return trimmed_mean(updates, cfg_fed["trim_beta"])
    if rule == "median":
        return median(updates)
    raise ValueError(f"Unknown aggregation rule: {rule}")
