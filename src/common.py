"""Shared code used by every script: config, labels, data loading, the model,
training helpers and metrics."""

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


# --- Config, seeding, paths ---------------------------------------------------

def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None):
    """Load configs/default.yaml, then merge the experiment file on top."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path is not None:
        path = Path(path)
        if not path.exists():
            path = REPO_ROOT / path  # allow running scripts from any folder
    if path is not None and path.resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    return cfg


def repo_path(relative):
    """Resolve a path from the config relative to the repository root."""
    return REPO_ROOT / relative


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rss_mb():
    """Resident memory of this process in MB (resource-consumption metric)."""
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


# --- Labels: 33 CICIoT2023 attack labels -> 7 families + benign ---------------
# Family grouping follows the CICIoT2023 paper (Neto et al., 2023). Label
# strings match the `label` column of the CSV files. Unknown labels raise an
# error so a typo or a different dataset version is caught.

FAMILIES = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]
FAMILY_TO_ID = {name: i for i, name in enumerate(FAMILIES)}
BENIGN_ID = FAMILY_TO_ID["Benign"]

_EXPLICIT = {
    "BenignTraffic": "Benign",
    "VulnerabilityScan": "Recon",
    "BrowserHijacking": "Web",
    "Backdoor_Malware": "Web",
    "XSS": "Web",
    "Uploading_Attack": "Web",
    "SqlInjection": "Web",
    "CommandInjection": "Web",
    "DictionaryBruteForce": "BruteForce",
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
}
_PREFIXES = {"DDoS-": "DDoS", "DoS-": "DoS", "Recon-": "Recon", "Mirai-": "Mirai"}


def to_family(label):
    """Map one original CICIoT2023 label to its family name."""
    if label in _EXPLICIT:
        return _EXPLICIT[label]
    for prefix, family in _PREFIXES.items():
        if label.startswith(prefix):
            return family
    raise ValueError(f"Unknown CICIoT2023 label: {label!r}")


# --- Data loading and preprocessing -------------------------------------------

NON_FEATURE_COLS = {"family", "y"}


def load_split(cfg, name):
    """Return (X, y, feature_cols) for 'train_pool', 'val' or 'test'."""
    path = repo_path(cfg["paths"]["processed_dir"]) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/prepare_data.py first.")
    df = pd.read_parquet(path)
    feature_cols = [
        c for c in df.columns if c not in NON_FEATURE_COLS | {cfg["data"]["label_col"]}
    ]
    # copy=True: pandas 3 returns read-only arrays, which PyTorch warns about.
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["y"].to_numpy(dtype=np.int64, copy=True)
    return X, y, feature_cols


class Preprocessor:
    """Optional signed log transform followed by standardization.

    Fit on the training pool only, then apply the same transform to the
    validation and test sets.
    """

    def __init__(self, log_transform=True):
        self.log_transform = log_transform
        self.scaler = StandardScaler()

    def _log(self, X):
        return np.sign(X) * np.log1p(np.abs(X)) if self.log_transform else X

    def fit(self, X):
        self.scaler.fit(self._log(X))
        return self

    def transform(self, X):
        return self.scaler.transform(self._log(X)).astype(np.float32)


def load_all(cfg):
    """Load all three splits and preprocess them with a scaler fit on train_pool."""
    X_train, y_train, feature_cols = load_split(cfg, "train_pool")
    X_val, y_val, _ = load_split(cfg, "val")
    X_test, y_test, _ = load_split(cfg, "test")
    prep = Preprocessor(cfg["data"]["log_transform"]).fit(X_train)
    return {
        "train": (prep.transform(X_train), y_train),
        "val": (prep.transform(X_val), y_val),
        "test": (prep.transform(X_test), y_test),
        "feature_cols": feature_cols,
    }


def partition_path(cfg):
    p = cfg["partition"]
    return repo_path(cfg["paths"]["processed_dir"]) / "partitions" / f"{p['method']}_k{p['num_clients']}.npy"


def load_partition(cfg):
    """Client id (0..num_clients-1) for every row of train_pool."""
    path = partition_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/prepare_data.py first.")
    return np.load(path)


# --- Model (held constant across all conditions) ------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(cfg, in_dim, num_classes):
    m = cfg["model"]
    return MLP(in_dim, num_classes, m["hidden_sizes"], m["dropout"])


# --- Training helpers ---------------------------------------------------------

def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epochs(model, loader, optimizer, epochs, device):
    """Train in place. Returns the mean training loss over the last epoch."""
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model, X, device, batch_size=4096):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def get_weights(model):
    """Model parameters as a list of numpy arrays (what a client 'sends')."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model, weights):
    keys = model.state_dict().keys()
    model.load_state_dict({k: torch.from_numpy(w) for k, w in zip(keys, weights)})


# --- Metrics ------------------------------------------------------------------
# Primary: macro F1 over the 8 families. Also per-class precision / recall /
# FNR, the binary benign-vs-attack FPR and FNR the spec asks for, one-vs-rest
# PR-AUC, accuracy and the confusion matrix.

def compute_metrics(y_true, proba, class_names=FAMILIES, benign_id=BENIGN_ID):
    labels = np.arange(len(class_names))
    y_pred = proba.argmax(axis=1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = support > 0  # classes with no true samples are left out of macro averages

    # Binary view: any attack family counts as "attack".
    true_attack = y_true != benign_id
    pred_attack = y_pred != benign_id
    fpr = (pred_attack & ~true_attack).sum() / max((~true_attack).sum(), 1)
    fnr = (~pred_attack & true_attack).sum() / max(true_attack.sum(), 1)

    pr_auc = {
        class_names[c]: float(average_precision_score(y_true == c, proba[:, c]))
        for c in labels[present]
    }

    return {
        "macro_f1": float(f1[present].mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_fpr": float(fpr),
        "binary_fnr": float(fnr),
        "macro_pr_auc": float(np.mean(list(pr_auc.values()))),
        "per_class": {
            name: {
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "fnr": float(1 - rec[c]) if present[c] else None,
                "f1": float(f1[c]),
                "pr_auc": pr_auc.get(name),
                "support": int(support[c]),
            }
            for c, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
