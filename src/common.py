"""Shared helpers: config loading, seeding, paths, resource measurement."""

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


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
