"""Shared code used by every script: config, labels, data loading, the model,
training helpers, metrics and run metadata (hardware, hashes)."""

import copy
import hashlib
import json
import os
import platform
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


def _resolve(path):
    path = Path(path)
    return path if path.exists() else REPO_ROOT / path  # run from any folder


def load_config(paths=None, overrides=None):
    """Load configs/default.yaml, then merge each given file on top, in order.

    `paths` may be None, one path, or a list (e.g. [pilot.yaml, fedavg_noniid.yaml]).
    `overrides` is a list of "dotted.key=value" strings (values parsed as YAML).
    """
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if paths is None:
        paths = []
    elif isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        p = _resolve(p)
        if p.resolve() == DEFAULT_CONFIG.resolve():
            continue
        with open(p) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg, overrides):
    """Apply ["attack.num_attackers=2", "experiment=foo"] style overrides."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def repo_path(relative):
    """Resolve a config path relative to the repository root (absolute paths pass through)."""
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


def hardware_info():
    """Machine description stored with every run, so runtimes can be interpreted.

    All runtimes are single-machine simulation time, not real network time.
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"Saved {path}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_array(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# --- Labels: 33 CICIoT2023 attack labels -> 7 families + benign ---------------
# Family grouping follows the seven attack categories of the CICIoT2023 paper
# (Neto et al., 2023, Table of attacks): DDoS, DoS, Recon, Web-based, Brute
# Force, Spoofing, Mirai. Matching is case-insensitive because some releases
# of the CSVs use upper-case labels (e.g. "DDOS-ICMP_FLOOD"). Unknown labels
# raise an error so a typo or a different dataset version is caught.
# The full table is written to data/processed/label_mapping.csv by
# prepare_data.py and reproduced in the README.

FAMILIES = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]
FAMILY_TO_ID = {name: i for i, name in enumerate(FAMILIES)}
BENIGN_ID = FAMILY_TO_ID["Benign"]

# All 34 labels in the CICIoT2023 CSVs (33 attacks + benign), as spelled in
# the dataset, with their family.
LABEL_TO_FAMILY = {
    "BenignTraffic": "Benign",
    # DDoS (12)
    "DDoS-RSTFINFlood": "DDoS", "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    # DoS (4)
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Mirai (3)
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    # Recon (5)
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon", "Recon-HostDiscovery": "Recon",
    # Spoofing (2)
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    # Web-based (6)
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web", "XSS": "Web",
    "Uploading_Attack": "Web", "SqlInjection": "Web", "CommandInjection": "Web",
    # Brute force (1)
    "DictionaryBruteForce": "BruteForce",
}
def _norm(label):
    """Lower-case and drop everything but letters and digits, so that
    'BenignTraffic', 'BENIGN_TRAFFIC' and 'DDOS-ICMP_FLOOD' all match."""
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


_LOOKUP = {_norm(k): v for k, v in LABEL_TO_FAMILY.items()}
# Spellings used by other releases of the CSVs (e.g. the MERGED_CSV files).
_LOOKUP.update({"benign": "Benign"})
# Any other label with one of these prefixes belongs to that family.
_PREFIX_FAMILIES = [("ddos", "DDoS"), ("dos", "DoS"), ("recon", "Recon"), ("mirai", "Mirai")]


def to_family(label):
    """Map one original CICIoT2023 label to its family name.

    Matching ignores case and punctuation; unknown DDoS-/DoS-/Recon-/Mirai-
    variants fall back to their prefix family.
    """
    key = _norm(label)
    if key in _LOOKUP:
        return _LOOKUP[key]
    for prefix, family in _PREFIX_FAMILIES:
        if key.startswith(prefix):
            return family
    raise ValueError(f"Unknown CICIoT2023 label: {label!r}")


def label_mapping_table():
    rows = [{"label": k, "family": v} for k, v in LABEL_TO_FAMILY.items()]
    return pd.DataFrame(rows).sort_values(["family", "label"]).reset_index(drop=True)


def find_column(columns, name):
    """Case-insensitive column lookup ('label' matches 'Label')."""
    for c in columns:
        if c.strip().lower() == name.lower():
            return c
    raise KeyError(f"Column {name!r} not found. Columns: {list(columns)[:10]}...")


# --- Data loading and preprocessing -------------------------------------------

NON_FEATURE_COLS = {"family", "y", "source_file"}


def processed_dir(cfg):
    return repo_path(cfg["paths"]["processed_dir"])


def load_split(cfg, name):
    """Return (X, y, feature_cols) for 'train_pool', 'val' or 'test'."""
    path = processed_dir(cfg) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/prepare_data.py first.")
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS | {"label"}]
    # copy=True: pandas 3 returns read-only arrays, which PyTorch warns about.
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["y"].to_numpy(dtype=np.int64, copy=True)
    return X, y, feature_cols


class Preprocessor:
    """Optional signed log transform followed by standardization.

    Fit on the training pool only, then applied unchanged (frozen) to the
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


def partition_path(cfg, split="train"):
    """Client ids for train_pool (split='train') or held-out rows ('val' / 'test')."""
    p = cfg["partition"]
    suffix = "" if split == "train" else f"_{split}"
    return processed_dir(cfg) / "partitions" / f"{p['method']}_k{p['num_clients']}{suffix}.npy"


def load_partition(cfg, split="train"):
    """Client id (0..num_clients-1) for every row of the given split.

    Held-out (val/test) ids come from the frozen train-time partitioner
    (scaler + centroids for k-means, class proportions for Dirichlet); they
    are never refit on held-out data.
    """
    path = partition_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/prepare_data.py first.")
    return np.load(path)


def load_manifest(cfg):
    path = processed_dir(cfg) / "pipeline_manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


# --- Model (held constant across all conditions) ------------------------------

class MLP(nn.Module):
    """Linear -> ReLU -> Dropout for each hidden layer, then a linear output layer
    (logits; softmax is applied by the loss / at prediction time)."""

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


def build_optimizer(cfg, params):
    t = cfg["train"]
    if t.get("optimizer", "adam").lower() != "adam":
        raise ValueError("Only the Adam optimizer is implemented")
    return torch.optim.Adam(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))


# --- Training helpers ---------------------------------------------------------

def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epochs(model, loader, optimizer, epochs, device):
    """Train in place with unweighted cross-entropy. Returns the mean loss of the last epoch."""
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    total, n = 0.0, 0
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
    if not out:
        return np.zeros((0, len(FAMILIES)), dtype=np.float32)
    return np.concatenate(out)


def get_weights(model):
    """Model parameters as a list of numpy arrays (what a client 'sends')."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model, weights):
    keys = model.state_dict().keys()
    model.load_state_dict({k: torch.from_numpy(np.asarray(w)) for k, w in zip(keys, weights)})


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


# --- Metrics ------------------------------------------------------------------
# Primary: macro F1 over the families present in y_true. Also, per class:
# precision, recall, FNR, one-vs-rest FPR, F1, PR-AUC, each with its support
# (positives) and number of negatives, so every rate can be read with its
# denominator. Plus binary benign-vs-attack FPR/FNR (with counts), accuracy and
# the confusion matrix.

def compute_metrics(y_true, proba, class_names=FAMILIES, benign_id=BENIGN_ID):
    y_true = np.asarray(y_true)
    labels = np.arange(len(class_names))
    if len(y_true) == 0:
        return {"n": 0, "macro_f1": None, "accuracy": None}
    y_pred = proba.argmax(axis=1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = support > 0  # classes with no true samples are left out of macro averages
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    predicted = cm.sum(axis=0)
    negatives = len(y_true) - support
    fp = predicted - np.diag(cm)

    # Binary view: any attack family counts as "attack".
    true_attack = y_true != benign_id
    pred_attack = y_pred != benign_id
    n_benign, n_attack = int((~true_attack).sum()), int(true_attack.sum())
    fp_bin = int((pred_attack & ~true_attack).sum())
    fn_bin = int((~pred_attack & true_attack).sum())

    pr_auc = {}
    for c in labels[present]:
        pos = y_true == c
        # PR-AUC is undefined if every row belongs to this class.
        pr_auc[class_names[c]] = (
            float(average_precision_score(pos, proba[:, c])) if not pos.all() else None
        )

    pr_vals = [v for v in pr_auc.values() if v is not None]
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1[present].mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_fpr": fp_bin / max(n_benign, 1),
        "binary_fnr": fn_bin / max(n_attack, 1),
        "binary_counts": {"benign": n_benign, "attack": n_attack,
                          "false_positives": fp_bin, "false_negatives": fn_bin},
        "macro_pr_auc": float(np.mean(pr_vals)) if pr_vals else None,
        "per_class": {
            name: {
                "support": int(support[c]),
                "negatives": int(negatives[c]),
                "predicted": int(predicted[c]),
                "precision": float(prec[c]),
                "recall": float(rec[c]) if present[c] else None,
                "fnr": float(1 - rec[c]) if present[c] else None,
                "fpr": float(fp[c] / negatives[c]) if negatives[c] > 0 else None,
                "f1": float(f1[c]) if present[c] else None,
                "pr_auc": pr_auc.get(name),
            }
            for c, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }
