"""Step 1: build the subset, the splits and the client partitions.

The order of operations is fixed and documented (instructor feedback):

1. Read: every CSV in data/raw/. Each row keeps its `source_file` (the only
   capture/session grouping CICIoT2023's CSVs expose). Original label counts
   are recorded BEFORE any sampling.
2. Sample: per file, at most a few rows per family, so the subset covers all
   capture files.
3. Clean: drop identifier/shortcut columns (`data.drop_cols`), drop rows with
   inf/NaN, remove exact duplicate (feature + label) rows, and drop feature
   vectors that appear with conflicting labels. All counts are logged.
4. Cap: at most `data.max_per_class` rows per family (rare families kept
   whole). The resulting prevalence is NOT the natural prevalence; both are
   written to subset_summary.json.
5. Split: hold out test (20%) then validation (10%). With `data.group_split`,
   whole source files are assigned to one split each, so the realized
   fractions move in file-sized steps: with N files each file is about 1/N of
   the rows. Among `data.split_search_tries` random file assignments (seeded),
   the one closest to the target fractions and to the global family mix is
   kept; every family must appear in every split. The file list and the
   per-file and per-family counts of each split are written to
   split_files.csv and subset_summary.json. All of this happens BEFORE any
   scaler or partitioner is fit.
   Caveat: file grouping guards against chunk-level leakage; it does not
   prove capture/session independence. The MERGED_CSV files each contain
   (nearly) every attack type, so one capture can contribute rows to several
   files, and the release has no capture id, IPs or timestamps to group on.
   A nearest-neighbour check between splits (split_diagnostics.json) shows
   how close held-out rows are to training rows.
6. Partition the training pool into `partition.num_clients` clients with
   every method (dirichlet, kmeans, iid). The scaler and k-means centroids are
   fit on the training pool only; held-out val/test rows are assigned to
   clients with the FROZEN scaler + centroids (k-means) or the frozen
   per-class client proportions (Dirichlet), never refit. Note: the Dirichlet
   assignment of held-out rows uses their labels. That is label-informed
   benchmark construction (it fixes which client "owns" each held-out row, so
   per-client evaluation matches the client's training mix), not a rule a
   deployed system could use to identify clients. It is acceptable here
   because labels are never model inputs and the test set is never used for
   tuning.
7. Report: per-client size, benign/attack balance, family distribution,
   distance from the global distribution, label purity, client-label mutual
   information, and a pass/fail check that the clients are balanced enough,
   trainable and not label-pure. Also written: feature_schema.csv (the
   official 46 CICIoT2023 features vs. the columns actually read and used,
   with the transform and training-pool statistics of each).
8. Freeze: pipeline_manifest.json stores the config and SHA-256 hashes of
   every output, and each training run records it.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --config configs/pilot.yaml   # small end-to-end pilot
    python src/prepare_data.py --partitions-only             # e.g. after changing num_clients
"""

import argparse
import math
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from common import (
    FAMILIES,
    FAMILY_TO_ID,
    Preprocessor,
    feature_schema,
    find_column,
    label_mapping_table,
    load_config,
    load_split,
    partition_path,
    processed_dir,
    repo_path,
    save_json,
    sha256_file,
    to_family,
)

PARTITION_METHODS = ["dirichlet", "kmeans", "iid"]
MIN_GROUPS_FOR_GROUP_SPLIT = 10


# --- 1-2. Read and sample -------------------------------------------------------

def cap_per_family(df, cap, seed):
    """Random sample of at most `cap` rows per family (all rows if fewer)."""
    return df.sample(frac=1, random_state=seed).groupby("family", sort=False).head(cap)


def read_raw(cfg):
    raw_dir = repo_path(cfg["paths"]["raw_dir"])
    files = sorted(raw_dir.glob("*.csv"))
    if cfg["data"]["max_files"]:
        files = files[: cfg["data"]["max_files"]]
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}. See README > Data.")

    cap = cfg["data"]["max_per_class"]
    # Sample a little from every file so the subset covers all capture files,
    # not just the first few. The final cap is applied after cleaning.
    per_file_cap = math.ceil(cfg["data"]["per_file_oversample"] * cap / len(files))

    raw_label_counts = pd.Series(dtype="int64")
    raw_rows = 0
    parts, per_file, read_cols = [], [], None
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        label_col = find_column(df.columns, cfg["data"]["label_col"])
        df = df.rename(columns={label_col: "label"})
        df.columns = [c.strip() for c in df.columns]
        # Rows with an empty label (blank or truncated lines in a CSV) cannot
        # be used; drop them here and count them per file.
        labels = df["label"].astype("string").str.strip()
        missing = labels.isna() | (labels == "")
        n_missing = int(missing.sum())
        if n_missing:
            print(f"  {path.name}: dropped {n_missing} rows with an empty label")
        df = df[~missing].copy()
        df["label"] = labels[~missing].astype(str)
        raw_rows += len(df)
        raw_label_counts = raw_label_counts.add(df["label"].value_counts(), fill_value=0)
        unknown = []
        for lab in df["label"].unique():
            try:
                to_family(lab)
            except ValueError:
                unknown.append(lab)
        if unknown:
            raise ValueError(f"{path.name}: unknown labels {sorted(map(str, unknown))}. "
                             "Add them to LABEL_TO_FAMILY in src/common.py.")
        df["family"] = df["label"].map(to_family)
        if read_cols is None:
            read_cols = [c for c in df.columns if c not in ("label", "family")]
        fam = df["family"].value_counts().reindex(FAMILIES, fill_value=0)
        per_file.append({"file": path.name, "raw_rows": len(df),
                         "rows_dropped_empty_label": n_missing,
                         **{f"raw_{f}": int(fam[f]) for f in FAMILIES}})
        df = cap_per_family(df, per_file_cap, seed=cfg["data"]["seed"] + i)
        df["source_file"] = path.name
        parts.append(df)
        print(f"[{i + 1}/{len(files)}] {path.name}: kept {len(df)} rows")
    raw_label_counts = raw_label_counts.astype("int64").sort_values(ascending=False)
    return (pd.concat(parts, ignore_index=True), files, raw_label_counts, raw_rows,
            pd.DataFrame(per_file), read_cols)


# --- 3. Clean -------------------------------------------------------------------

def clean(df, cfg):
    """Drop identifier columns, inf/NaN rows, duplicate rows and label conflicts."""
    log = {}
    drop = [c for c in cfg["data"]["drop_cols"] if c in df.columns]
    df = df.drop(columns=drop)
    log["dropped_cols"] = drop
    feature_cols = [c for c in df.columns if c not in ("label", "family", "source_file")]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    n = len(df)
    df = df.dropna(subset=feature_cols)
    log["rows_dropped_inf_nan"] = n - len(df)

    # Exact duplicates of (features + label), across all files, are removed
    # BEFORE splitting so the same record cannot land in train and test.
    n = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["label"], keep="first")
    log["rows_dropped_exact_duplicates"] = n - len(df)

    # Identical feature vectors with different labels cannot be learned and
    # would leak across splits; drop all copies.
    n = len(df)
    if cfg["data"]["drop_conflicting_duplicates"]:
        conflict = df.duplicated(subset=feature_cols, keep=False)
        df = df[~conflict]
    log["rows_dropped_conflicting_labels"] = n - len(df)
    return df.reset_index(drop=True), feature_cols, log


# --- 5. Splits ------------------------------------------------------------------

def search_group_split(y, groups, test_frac, val_frac, seed, tries):
    """Assign whole source files to test, val or train.

    For each of `tries` seeded random file orders, files are added to test
    while that moves the test share closer to test_frac, then to val the same
    way; the rest is train. Score = |test share - target| + |val share - target|
    + 0.25 * (largest family-share deviation from the overall mix, summed over
    test and val). Assignments that leave any family of the subset out of a
    split are rejected. The lowest score wins; the search is deterministic given `seed`.
    """
    files = np.unique(groups)
    pos = {g: i for i, g in enumerate(files)}
    gid = np.array([pos[g] for g in groups])
    m = np.zeros((len(files), len(FAMILIES)), dtype=np.int64)
    np.add.at(m, (gid, y), 1)
    file_rows = m.sum(axis=1)
    total = m.sum()
    overall = m.sum(axis=0) / total
    present = m.sum(axis=0) > 0  # families that exist in the subset at all
    rng = np.random.default_rng(seed)

    best = None
    for t in range(tries):
        order = rng.permutation(len(files))
        assign = np.full(len(files), 2)  # 0 = test, 1 = val, 2 = train
        for split_id, target in ((0, test_frac), (1, val_frac)):
            rows = 0
            for f in order:
                if assign[f] == 2 and abs((rows + file_rows[f]) / total - target) < abs(rows / total - target):
                    assign[f] = split_id
                    rows += file_rows[f]
        sums = [m[assign == s].sum(axis=0) for s in (0, 1, 2)]
        if any((s[present] == 0).any() for s in sums):
            continue
        score = abs(sums[0].sum() / total - test_frac) + abs(sums[1].sum() / total - val_frac)
        score += 0.25 * sum(np.abs(s / s.sum() - overall).max() for s in sums[:2])
        if best is None or score < best[0]:
            best = (score, assign.copy(), t)
    if best is None:
        raise RuntimeError("No file assignment puts every family in every split; "
                           "use more files or set data.group_split: false.")
    score, assign, t = best
    a = assign[gid]
    idx = np.arange(len(y))
    info = {"method": "seeded search over whole-file assignments (see search_group_split)",
            "tries": int(tries), "best_try": int(t), "score": float(score)}
    return idx[a == 2], idx[a == 1], idx[a == 0], info


def split_indices(y, groups, test_frac, val_frac, seed, group_split, tries=2000):
    """Return (train_idx, val_idx, test_idx, grouped, info)."""
    idx = np.arange(len(y))
    n_groups = len(np.unique(groups))
    use_groups = group_split and n_groups >= MIN_GROUPS_FOR_GROUP_SPLIT
    if group_split and not use_groups:
        print(f"WARNING: only {n_groups} source files; falling back to a non-grouped "
              f"stratified split (need >= {MIN_GROUPS_FOR_GROUP_SPLIT}).")

    if use_groups:
        tr, va, te, info = search_group_split(y, groups, test_frac, val_frac, seed, tries)
        return tr, va, te, True, info

    rest, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    tr, va = train_test_split(
        rest, test_size=val_frac / (1 - test_frac), stratify=y[rest], random_state=seed
    )
    return tr, va, test, False, {"method": "row-level stratified split (not grouped)"}


def make_subset_and_splits(cfg):
    seed = cfg["data"]["seed"]
    df, files, raw_label_counts, raw_rows, per_file, read_cols = read_raw(cfg)
    n_sampled = len(df)
    df, feature_cols, clean_log = clean(df, cfg)

    cap = cfg["data"]["max_per_class"]
    df = cap_per_family(df, cap, seed).reset_index(drop=True)
    df["y"] = df["family"].map(FAMILY_TO_ID).astype("int64")

    tr, va, te, grouped, split_info = split_indices(
        df["y"].to_numpy(), df["source_file"].to_numpy(),
        cfg["data"]["test_frac"], cfg["data"]["val_frac"], seed, cfg["data"]["group_split"],
        cfg["data"].get("split_search_tries", 2000),
    )
    splits = {"train_pool": df.iloc[tr], "val": df.iloc[va], "test": df.iloc[te]}

    if grouped:
        sets = {k: set(v["source_file"]) for k, v in splits.items()}
        assert not (sets["train_pool"] & sets["val"] or sets["train_pool"] & sets["test"]
                    or sets["val"] & sets["test"]), "source file appears in two splits"

    out_dir = processed_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)

    mapping = label_mapping_table()
    mapping.to_csv(out_dir / "label_mapping.csv", index=False)

    # Which file went where, with raw and subset counts per family.
    split_of_file = {f: name for name, sp in splits.items() for f in sp["source_file"].unique()}
    sub = (df.groupby(["source_file", "family"]).size().unstack(fill_value=0)
           .reindex(columns=FAMILIES, fill_value=0))
    per_file = per_file.set_index("file")
    per_file.insert(0, "split", [split_of_file.get(f, "unused") for f in per_file.index])
    per_file["subset_rows"] = sub.sum(axis=1).reindex(per_file.index, fill_value=0)
    for f in FAMILIES:
        per_file[f"subset_{f}"] = sub[f].reindex(per_file.index, fill_value=0)
    per_file["raw_families_present"] = (per_file[[f"raw_{f}" for f in FAMILIES]] > 0).sum(axis=1)
    per_file.sort_values(["split", "file"]).to_csv(out_dir / "split_files.csv")

    # Feature schema: official 46 vs. what was read and used, with raw
    # (untransformed) training-pool statistics of every used column.
    tp = splits["train_pool"][feature_cols]
    stats = {c: {"mean": float(tp[c].mean()), "std": float(tp[c].std()),
                 "nonzero_frac": float((tp[c] != 0).mean())} for c in feature_cols}
    schema = feature_schema(read_cols, feature_cols, clean_log["dropped_cols"], stats)
    schema.to_csv(out_dir / "feature_schema.csv", index=False)
    constant = [c for c in feature_cols if not stats[c]["std"] > 0]

    raw_family_counts = (
        raw_label_counts.rename(index=lambda s: to_family(s)).groupby(level=0).sum()
        .reindex(FAMILIES, fill_value=0)
    )
    subset_family_counts = df["family"].value_counts().reindex(FAMILIES, fill_value=0)
    summary = {
        "seed": seed,
        "selection_procedure": (
            f"Read {len(files)} CSV files ({raw_rows} rows). From each file, a random "
            f"sample of at most {math.ceil(cfg['data']['per_file_oversample'] * cap / len(files))} "
            f"rows per family (seed {seed} + file index). Then removed inf/NaN rows, exact "
            f"(feature+label) duplicates and conflicting-label duplicates, then a random sample "
            f"of at most {cap} rows per family (seed {seed}). Families with fewer rows are kept "
            "whole, so the subset is closer to balanced than the raw data: its prevalence is "
            "a design choice, not the natural prevalence."
        ),
        "source_files": [p.name for p in files],
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,
        "feature_schema": {
            "official_count": 46,
            "read_from_release": len(read_cols),
            "used_as_inputs": len(feature_cols),
            "status_counts": schema["status"].value_counts().to_dict(),
            "constant_in_training_pool": constant,
            "selection": ("No feature selection: every numeric column of the release is used "
                          "except data.drop_cols. None is an identifier (the release has no "
                          "IPs, ports, MACs or timestamps). See feature_schema.csv."),
        },
        "rows_raw_total": int(raw_rows),
        "rows_dropped_empty_label": int(per_file["rows_dropped_empty_label"].sum()),
        "rows_after_per_file_sampling": int(n_sampled),
        **{k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in clean_log.items()},
        "rows_final_subset": int(len(df)),
        "max_per_class": cap,
        "split": {
            "test_frac_target": cfg["data"]["test_frac"],
            "val_frac_target": cfg["data"]["val_frac"],
            "grouped_by_source_file": grouped,
            "assignment": split_info,
            "why_fractions_differ": (
                "Whole files are assigned to one split, so the realized fractions move in "
                f"steps of about 1/{len(files)} of the rows; the closest assignment found is used."
                if grouped else "Row-level stratified split."),
            "files": {k: sorted(v["source_file"].unique().tolist()) for k, v in splits.items()},
            "rows": {k: int(len(v)) for k, v in splits.items()},
            "realized_frac": {k: len(v) / len(df) for k, v in splits.items()},
            "source_files_per_split": {k: int(v["source_file"].nunique()) for k, v in splits.items()},
        },
        "raw_label_counts_before_sampling": raw_label_counts.to_dict(),
        "raw_family_counts_before_sampling": raw_family_counts.astype(int).to_dict(),
        "raw_family_prevalence": (raw_family_counts / raw_family_counts.sum()).to_dict(),
        "subset_label_counts": df["label"].value_counts().to_dict(),
        "subset_family_counts": subset_family_counts.astype(int).to_dict(),
        "subset_family_prevalence": (subset_family_counts / subset_family_counts.sum()).to_dict(),
        "family_counts_per_split": {
            name: {f: int((split["family"] == f).sum()) for f in FAMILIES}
            for name, split in splits.items()
        },
    }
    save_json(summary, out_dir / "subset_summary.json")
    table = pd.DataFrame({
        "raw": raw_family_counts.astype(int),
        "raw_%": (100 * raw_family_counts / raw_family_counts.sum()).round(2),
        "subset": subset_family_counts.astype(int),
        "subset_%": (100 * subset_family_counts / subset_family_counts.sum()).round(2),
        **{k: pd.Series(v) for k, v in summary["family_counts_per_split"].items()},
    })
    table.to_csv(out_dir / "family_counts.csv")
    print(table)
    print({k: v for k, v in clean_log.items()})
    sp = summary["split"]
    print("split: " + ", ".join(
        f"{k} {100 * v:.1f}% ({sp['source_files_per_split'][k]} files)"
        for k, v in sp["realized_frac"].items()))
    print(f"features: read {len(read_cols)}, used {len(feature_cols)}; "
          + ", ".join(f"{k}: {v}" for k, v in summary["feature_schema"]["status_counts"].items()))
    if constant:
        print(f"constant in the training pool (kept; they standardize to 0): {constant}")


# --- 5b. Split diagnostics ------------------------------------------------------

def _dist_summary(d):
    return {"median": float(np.median(d)), "p10": float(np.percentile(d, 10)),
            "p90": float(np.percentile(d, 90)),
            "exact_duplicates": int((d < 1e-9).sum()),
            "share_within_0.01": float((d < 0.01).mean())}


def split_diagnostics(cfg, n_query=2000):
    """How close are held-out rows to the training pool?

    For a seeded sample of val and test rows, the distance to the nearest
    training row (after the train-fit transform) is compared with the
    distance between a training row and its nearest OTHER training row. If
    the held-out distances are about as small, file grouping has not made
    held-out rows less similar to training rows than training rows are to
    each other, so it does not by itself show capture independence. Exact
    cross-split duplicates should be 0 (they are removed before splitting).
    """
    X_tr, _, _ = load_split(cfg, "train_pool")
    prep = Preprocessor(cfg["data"]["log_transform"]).fit(X_tr)
    Z_tr = prep.transform(X_tr)
    rng = np.random.default_rng(cfg["data"]["seed"])
    nn = NearestNeighbors(n_neighbors=2).fit(Z_tr)
    out = {"n_query_per_split": n_query,
           "distance": "euclidean, on signed-log + standardized features (train-fit)"}
    q = rng.choice(len(Z_tr), size=min(n_query, len(Z_tr)), replace=False)
    d, _ = nn.kneighbors(Z_tr[q])
    out["train_to_other_train"] = _dist_summary(d[:, 1])
    for name in ("val", "test"):
        X, _, _ = load_split(cfg, name)
        Z = prep.transform(X)
        q = rng.choice(len(Z), size=min(n_query, len(Z)), replace=False)
        d, _ = nn.kneighbors(Z[q], n_neighbors=1)
        out[f"{name}_to_train"] = _dist_summary(d[:, 0])
    save_json(out, processed_dir(cfg) / "split_diagnostics.json")
    print("median distance to nearest training row: " + ", ".join(
        f"{k} {v['median']:.3f}" for k, v in out.items() if isinstance(v, dict)))
    return out


# --- 6. Partitions --------------------------------------------------------------

def iid_partition(n_rows, num_clients, rng):
    clients = np.arange(n_rows) % num_clients
    rng.shuffle(clients)
    return clients


def dirichlet_partition(y, num_clients, alpha, rng, min_client_rows, max_tries=200):
    """Label-skewed split (Hsu et al., 2019): for each family c, draw client
    proportions p_c ~ Dir(alpha) and deal that family's rows out accordingly.

    Smaller alpha = more skew. Redraws until every client has at least
    `min_client_rows` rows. Returns (client ids, realized per-family client
    proportions [n_families x num_clients], attempts used).
    """
    n_classes = len(FAMILIES)
    for attempt in range(1, max_tries + 1):
        clients = np.empty(len(y), dtype=np.int64)
        for c in range(n_classes):
            idx = np.flatnonzero(y == c)
            if len(idx) == 0:
                continue
            idx = rng.permutation(idx)
            p = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(p)[:-1] * len(idx)).round().astype(int)
            for k, part in enumerate(np.split(idx, cuts)):
                clients[part] = k
        sizes = np.bincount(clients, minlength=num_clients)
        if sizes.min() >= min_client_rows:
            break
    else:
        raise RuntimeError(
            f"Dirichlet(alpha={alpha}) never gave every client >= {min_client_rows} rows "
            f"in {max_tries} tries. Increase alpha or lower partition.min_client_share."
        )
    counts = np.zeros((n_classes, num_clients))
    np.add.at(counts, (y, clients), 1)
    totals = counts.sum(axis=1, keepdims=True)
    props = np.divide(counts, totals, out=np.full_like(counts, 1 / num_clients), where=totals > 0)
    return clients, props, attempt


def assign_heldout_dirichlet(y, props, rng):
    """Assign held-out rows with the FROZEN per-family client proportions."""
    out = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        out[idx] = rng.choice(props.shape[1], size=len(idx), p=props[c])
    return out


def kmeans_fit(X_train, num_clients, seed, log_transform):
    prep = Preprocessor(log_transform).fit(X_train)
    km = KMeans(n_clusters=num_clients, n_init=10, random_state=seed)
    clients = km.fit_predict(prep.transform(X_train))
    return clients, prep, km


def partition_report(clients, y, num_clients):
    """Per-client size, balance, family distribution and distance from global."""
    counts = pd.crosstab(
        pd.Series(clients, name="client"),
        pd.Series(np.array(FAMILIES)[y], name="family"),
    ).reindex(index=range(num_clients), columns=FAMILIES, fill_value=0)
    sizes = counts.sum(axis=1)
    global_p = counts.sum(axis=0) / counts.values.sum()
    per_client = {}
    for k in range(num_clients):
        row = counts.loc[k]
        n = int(row.sum())
        p = row / n if n else row * 0
        m = 0.5 * (p + global_p)
        js = 0.5 * (_kl(p, m) + _kl(global_p, m))
        attack = row.drop("Benign")
        per_client[k] = {
            "rows": n,
            "share_of_pool": n / max(int(sizes.sum()), 1),
            "benign_rows": int(row["Benign"]),
            "attack_rows": int(attack.sum()),
            "benign_fraction": float(row["Benign"] / n) if n else None,
            "family_counts": {f: int(row[f]) for f in FAMILIES},
            "family_fraction": {f: float(p[f]) for f in FAMILIES},
            "label_purity": float(p.max()) if n else None,
            "tv_distance_from_global": float(0.5 * np.abs(p - global_p).sum()),
            "js_divergence_from_global_bits": float(js),
        }
    return {
        "num_clients": num_clients,
        "global_family_fraction": {f: float(global_p[f]) for f in FAMILIES},
        "clients": per_client,
        "client_sizes": {int(k): int(v) for k, v in sizes.items()},
        "smallest_client_share": float(sizes.min() / sizes.sum()),
        "size_ratio_max_min": float(sizes.max() / max(sizes.min(), 1)),
        "mutual_info_nats": float(mutual_info_score(clients, y)),
        "normalized_mutual_info": float(normalized_mutual_info_score(clients, y)),
    }, counts


def _kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def partition_checks(report, pcfg):
    """Is every client big enough, trainable (>= 2 families) and not label-pure?"""
    issues = []
    if report["smallest_client_share"] < pcfg["min_client_share"]:
        issues.append(f"smallest client holds {report['smallest_client_share']:.1%} of the pool "
                      f"(< {pcfg['min_client_share']:.0%})")
    if report["size_ratio_max_min"] > pcfg["max_size_ratio"]:
        issues.append(f"largest/smallest client size ratio {report['size_ratio_max_min']:.1f} "
                      f"(> {pcfg['max_size_ratio']})")
    for k, c in report["clients"].items():
        fams = sum(v >= pcfg["min_family_rows"] for v in c["family_counts"].values())
        if fams < 2:
            issues.append(f"client {k} has only {fams} family with >= {pcfg['min_family_rows']} rows "
                          "(not trainable as a classifier)")
        if c["label_purity"] is not None and c["label_purity"] > pcfg["max_label_purity"]:
            issues.append(f"client {k} is nearly label-pure ({c['label_purity']:.1%} one family)")
    return issues


def make_partitions(cfg):
    X, y, _ = load_split(cfg, "train_pool")
    X_val, y_val, _ = load_split(cfg, "val")
    X_test, y_test, _ = load_split(cfg, "test")
    pcfg = cfg["partition"]
    k, seed = pcfg["num_clients"], cfg["data"]["seed"]
    out_dir = processed_dir(cfg) / "partitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts, rows = {}, []

    for method in PARTITION_METHODS:
        rng = np.random.default_rng(seed)
        extra = {}
        if method == "iid":
            clients = iid_partition(len(y), k, rng)
            val_c = rng.integers(0, k, len(y_val))
            test_c = rng.integers(0, k, len(y_test))
            extra["heldout_assignment"] = "uniform random"
        elif method == "dirichlet":
            min_rows = math.ceil(pcfg["min_client_share"] * len(y))
            clients, props, attempts = dirichlet_partition(
                y, k, pcfg["dirichlet_alpha"], rng, min_rows)
            val_c = assign_heldout_dirichlet(y_val, props, rng)
            test_c = assign_heldout_dirichlet(y_test, props, rng)
            extra.update({
                "alpha": pcfg["dirichlet_alpha"], "draw_attempts": attempts,
                "family_client_proportions": {f: props[i].tolist() for i, f in enumerate(FAMILIES)},
                "heldout_assignment": "frozen per-family client proportions from the train split",
                "note": ("Uses labels by design: label skew is the controlled heterogeneity "
                         "variable. Held-out rows are assigned to clients using their labels: "
                         "label-informed benchmark construction, not a client-identification "
                         "rule. Labels are never model inputs; the test set is never used "
                         "for tuning."),
                "label_informed_heldout_assignment": True,
            })
        else:  # kmeans
            clients, prep, km = kmeans_fit(X, k, seed, cfg["data"]["log_transform"])
            val_c = km.predict(prep.transform(X_val))
            test_c = km.predict(prep.transform(X_test))
            np.savez(out_dir / f"kmeans_k{k}_model.npz", centroids=km.cluster_centers_,
                     scaler_mean=prep.scaler.mean_, scaler_scale=prep.scaler.scale_)
            extra.update({
                "heldout_assignment": "frozen train-fit scaler + centroids (nearest centroid)",
                "note": "No labels used. Fit on train_pool only.",
            })

        cfg_m = {**cfg, "partition": {**pcfg, "method": method}}
        np.save(partition_path(cfg_m, "train"), clients)
        np.save(partition_path(cfg_m, "val"), val_c)
        np.save(partition_path(cfg_m, "test"), test_c)

        report, counts = partition_report(clients, y, k)
        report["heldout_val_sizes"] = np.bincount(val_c, minlength=k).tolist()
        report["heldout_test_sizes"] = np.bincount(test_c, minlength=k).tolist()
        issues = partition_checks(report, pcfg)
        report.update(method=method, seed=seed, checks_passed=not issues, issues=issues, **extra)
        save_json(report, partition_path(cfg_m).with_suffix(".json"))
        verdicts[method] = issues

        print(f"\n== {method}, {k} clients ==")
        show = counts.assign(total=counts.sum(axis=1))
        show["benign_%"] = (100 * counts["Benign"] / show["total"]).round(1)
        show["TV_dist"] = [round(report["clients"][i]["tv_distance_from_global"], 3) for i in range(k)]
        print(show)
        print(f"client-label MI: {report['mutual_info_nats']:.3f} nats "
              f"(normalized {report['normalized_mutual_info']:.3f})")
        print("checks: " + ("PASS" if not issues else "FAIL\n  - " + "\n  - ".join(issues)))
        for i in range(k):
            c = report["clients"][i]
            rows.append({"method": method, "client": i, "rows": c["rows"],
                         "share": round(c["share_of_pool"], 4),
                         "benign_fraction": round(c["benign_fraction"] or 0, 4),
                         "attack_rows": c["attack_rows"],
                         "families_present": sum(v > 0 for v in c["family_counts"].values()),
                         "label_purity": round(c["label_purity"] or 0, 4),
                         "tv_from_global": round(c["tv_distance_from_global"], 4),
                         "js_from_global_bits": round(c["js_divergence_from_global_bits"], 4),
                         "heldout_val_rows": report["heldout_val_sizes"][i],
                         **{f: c["family_counts"][f] for f in FAMILIES}})

    pd.DataFrame(rows).to_csv(out_dir / f"partition_table_k{k}.csv", index=False)
    chosen = pcfg["method"]
    print(f"\nConfigured non-IID method: partition.method = {chosen}")
    if verdicts.get("kmeans"):
        print("k-means failed the partition checks; keep partition.method = dirichlet "
              "(instructor feedback: use a controlled Dirichlet split in that case).")
    if verdicts.get(chosen):
        print(f"WARNING: the configured method '{chosen}' failed its checks. Adjust "
              "partition.dirichlet_alpha / num_clients before running experiments.")


# --- 8. Freeze ------------------------------------------------------------------

def write_manifest(cfg):
    out_dir = processed_dir(cfg)
    files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "pipeline_manifest.json")
    pkgs = {}
    for pkg in ["numpy", "pandas", "scikit-learn", "torch", "pyarrow"]:
        try:
            pkgs[pkg] = version(pkg)
        except PackageNotFoundError:
            pkgs[pkg] = None
    manifest = {
        "data_config": cfg["data"],
        "partition_config": cfg["partition"],
        "packages": pkgs,
        "files": {str(p.relative_to(out_dir)): sha256_file(p) for p in files},
    }
    save_json(manifest, out_dir / "pipeline_manifest.json")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=None)
    parser.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE",
                        help="override config keys, e.g. partition.num_clients=12")
    parser.add_argument("--partitions-only", action="store_true",
                        help="reuse the existing splits and only redo the client partitions")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    if not args.partitions_only:
        make_subset_and_splits(cfg)
        split_diagnostics(cfg)
    make_partitions(cfg)
    write_manifest(cfg)


if __name__ == "__main__":
    main()
