"""Split the training pool into simulated clients.

  kmeans : non-IID. k-means (k = num_clients) on the preprocessed features of
           the training pool. No labels, IPs, ports, timestamps or row indices
           are used, so clients are defined by traffic shape only.
  iid    : control. Random equal-size split.

Also writes a leakage report: client sizes, per-client family counts and the
mutual information between client id and label. High MI means the client id
alone says a lot about the label, which should be discussed in the report.

Usage:
    python src/make_partitions.py                 # both methods, num_clients from config
    python src/make_partitions.py --method kmeans
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from common import load_config, save_json
from data import Preprocessor, load_split, partition_path
from labels import FAMILIES

MIN_CLIENT_SHARE = 0.05  # warn if a client holds less than 5% of the pool


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", choices=["kmeans", "iid", "both"], default="both")
    args = parser.parse_args()
    cfg = load_config(args.config)

    X, y, _ = load_split(cfg, "train_pool")
    k, seed = cfg["partition"]["num_clients"], cfg["data"]["seed"]
    methods = ["kmeans", "iid"] if args.method == "both" else [args.method]

    for method in methods:
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


if __name__ == "__main__":
    main()
