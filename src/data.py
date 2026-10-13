"""Load the processed splits and apply the shared feature preprocessing."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import repo_path

NON_FEATURE_COLS = {"family", "y"}


def load_split(cfg, name):
    """Return (X, y, feature_cols) for 'train_pool', 'val' or 'test'."""
    path = repo_path(cfg["paths"]["processed_dir"]) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/make_subset.py first.")
    df = pd.read_parquet(path)
    feature_cols = [
        c for c in df.columns if c not in NON_FEATURE_COLS | {cfg["data"]["label_col"]}
    ]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int64)
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
        raise FileNotFoundError(f"{path} not found. Run src/make_partitions.py first.")
    return np.load(path)
