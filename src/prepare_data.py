"""Step 1: build the subset, the splits and the client partitions.

1. Subset: read the CICIoT2023 CSVs in data/raw/, map labels to the 8
   families, and cap each family at `data.max_per_class` rows (rare families
   are kept whole).
2. Splits: hold out a stratified test set and validation set (seed
   `data.seed`). The rest is the training pool.
3. Partitions: split the training pool into `partition.num_clients` clients.
     kmeans : non-IID. k-means on the preprocessed features. No labels, IPs,
              ports, timestamps or row indices are used.
     iid    : control. Random equal-size split.
   A leakage report (client sizes, per-client family counts, client-label
   mutual information) is saved next to each partition. High mutual
   information means the client id alone says a lot about the label.

Writes data/processed/{train_pool,val,test}.parquet, subset_summary.json and
data/processed/partitions/.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --partitions-only    # e.g. after changing num_clients
"""

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from common import (
    FAMILIES,
    FAMILY_TO_ID,
    Preprocessor,
    load_config,
    load_split,
    partition_path,
    repo_path,
    save_json,
    to_family,
)

MIN_CLIENT_SHARE = 0.05  # warn if a client holds less than 5% of the pool


# --- 1. Subset ------------------------------------------------------------------

def cap_per_family(df, cap, seed):
    """Random sample of at most `cap` rows per family (all rows if fewer)."""
    return df.sample(frac=1, random_state=seed).groupby("family").head(cap)


def read_raw(cfg):
    raw_dir = repo_path(cfg["paths"]["raw_dir"])
    files = sorted(raw_dir.glob("*.csv"))
    if cfg["data"]["max_files"]:
        files = files[: cfg["data"]["max_files"]]
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}. See README > Data.")

    label_col = cfg["data"]["label_col"]
    cap = cfg["data"]["max_per_class"]
    # Sample a little from every file so the subset covers all capture files,
    # not just the first few. The final cap is applied after concatenation.
    per_file_cap = math.ceil(2 * cap / len(files))

    parts = []
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        df["family"] = df[label_col].map(to_family)
        df = cap_per_family(df, per_file_cap, seed=cfg["data"]["seed"] + i)
        parts.append(df)
        print(f"[{i + 1}/{len(files)}] {path.name}: kept {len(df)} rows")
    return pd.concat(parts, ignore_index=True)


# --- 2. Splits ------------------------------------------------------------------

def make_subset_and_splits(cfg):
    seed = cfg["data"]["seed"]
    label_col = cfg["data"]["label_col"]

    df = read_raw(cfg)

    # Clean: drop identifier columns and rows with inf / NaN features.
    df = df.drop(columns=[c for c in cfg["data"]["drop_cols"] if c in df.columns])
    feature_cols = [c for c in df.columns if c not in (label_col, "family")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    n_dropped_nan = n_before - len(df)

    # Final per-family cap.
    cap = cfg["data"]["max_per_class"]
    df = cap_per_family(df, cap, seed).reset_index(drop=True)
    df["y"] = df["family"].map(FAMILY_TO_ID).astype("int64")

    # Hold out test first, then validation, both stratified by family.
    test_frac, val_frac = cfg["data"]["test_frac"], cfg["data"]["val_frac"]
    rest, test = train_test_split(df, test_size=test_frac, stratify=df["y"], random_state=seed)
    train_pool, val = train_test_split(
        rest, test_size=val_frac / (1 - test_frac), stratify=rest["y"], random_state=seed
    )

    out_dir = repo_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train_pool": train_pool, "val": val, "test": test}
    for name, split in splits.items():
        split.reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)

    summary = {
        "seed": seed,
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,
        "dropped_cols": cfg["data"]["drop_cols"],
        "rows_dropped_inf_nan": int(n_dropped_nan),
        "max_per_class": cap,
        "family_counts": {
            name: {f: int((split["family"] == f).sum()) for f in FAMILIES}
            for name, split in splits.items()
        },
        "original_label_counts": df[label_col].value_counts().to_dict(),
    }
    save_json(summary, out_dir / "subset_summary.json")
    print(pd.DataFrame(summary["family_counts"]))


# --- 3. Partitions --------------------------------------------------------------

def kmeans_partition(X, num_clients, seed, log_transform):
    Xs = Preprocessor(log_transform).fit(X).transform(X)
    km = KMeans(n_clusters=num_clients, n_init=10, random_state=seed)
    return km.fit_predict(Xs)


def iid_partition(n_rows, num_clients, seed):
    rng = np.random.default_rng(seed)
    clients = np.arange(n_rows) % num_clients
    rng.shuffle(clients)
    return clients


def leakage_report(clients, y, num_clients):
    counts = pd.crosstab(
        pd.Series(clients, name="client"),
        pd.Series(np.array(FAMILIES)[y], name="family"),
    ).reindex(index=range(num_clients), columns=FAMILIES, fill_value=0)
    sizes = counts.sum(axis=1)
    return {
        "client_sizes": sizes.to_dict(),
        "client_family_counts": counts.to_dict(orient="index"),
        "mutual_info_nats": float(mutual_info_score(clients, y)),
        "normalized_mutual_info": float(normalized_mutual_info_score(clients, y)),
        "smallest_client_share": float(sizes.min() / sizes.sum()),
    }, counts


def make_partitions(cfg):
    X, y, _ = load_split(cfg, "train_pool")
    k, seed = cfg["partition"]["num_clients"], cfg["data"]["seed"]

    for method in ["kmeans", "iid"]:
        if method == "kmeans":
            clients = kmeans_partition(X, k, seed, cfg["data"]["log_transform"])
        else:
            clients = iid_partition(len(y), k, seed)

        cfg["partition"]["method"] = method
        out = partition_path(cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, clients)

        report, counts = leakage_report(clients, y, k)
        save_json(report, out.with_suffix(".json"))
        print(f"\n== {method}, {k} clients ==")
        print(counts.assign(total=counts.sum(axis=1)))
        print(f"client-label MI: {report['mutual_info_nats']:.3f} nats "
              f"(normalized {report['normalized_mutual_info']:.3f})")
        if report["smallest_client_share"] < MIN_CLIENT_SHARE:
            print(f"WARNING: smallest client has only {report['smallest_client_share']:.1%} "
                  "of the pool. Consider fewer clients or a different partitioner.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--partitions-only", action="store_true",
                        help="reuse the existing splits and only redo the client partitions")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if not args.partitions_only:
        make_subset_and_splits(cfg)
    make_partitions(cfg)


if __name__ == "__main__":
    main()
