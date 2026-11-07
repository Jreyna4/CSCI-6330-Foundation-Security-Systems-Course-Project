"""End-to-end smoke test on small synthetic CSVs shaped like CICIoT2023.

Runs data prep (all partition methods), one centralized run, a clean and a
poisoned federated run, and the summary, in a temporary folder. Takes a few
seconds and needs no real data. It checks wiring, determinism and pairing,
not model quality.
"""

import json

import numpy as np
import pandas as pd
import pytest

import common
import error_analysis
import prepare_data
import run_federated
import summarize_results
import train_centralized
from common import LABEL_TO_FAMILY, load_config

FEATURES = ["flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
            "Tot sum", "Min", "Max", "AVG", "IAT"]


def make_synthetic_csvs(raw_dir, n_files=12, rows=600, seed=0):
    rng = np.random.default_rng(seed)
    labels = list(LABEL_TO_FAMILY)
    fam_ids = {f: i for i, f in enumerate(common.FAMILIES)}
    for f in range(n_files):
        lab = rng.choice(labels, size=rows)
        shift = np.array([fam_ids[LABEL_TO_FAMILY[x]] for x in lab])[:, None]
        X = rng.normal(size=(rows, len(FEATURES))) + shift
        df = pd.DataFrame(np.abs(X) * 10, columns=FEATURES)
        df["label"] = lab
        if f == 0:  # an exact duplicate, an inf, and a label conflict
            df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
            df.loc[1, "Rate"] = np.inf
            conflict = df.iloc[[2]].copy()
            conflict["label"] = "BenignTraffic" if df.loc[2, "label"] != "BenignTraffic" else "XSS"
            df = pd.concat([df, conflict], ignore_index=True)
        df.to_csv(raw_dir / f"part-{f:05d}.csv", index=False)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    raw = tmp / "raw"
    raw.mkdir()
    make_synthetic_csvs(raw)
    cfg = load_config(None, [
        f"paths.raw_dir={raw}", f"paths.processed_dir={tmp / 'processed'}",
        f"paths.results_dir={tmp / 'results'}", "data.max_per_class=600",
        "partition.num_clients=5", "partition.min_client_share=0.02",
        "train.epochs=2", "federated.rounds=2", "seeds=[42]",
        "attack.min_rows_per_family=5",
    ])
    prepare_data.make_subset_and_splits(cfg)
    prepare_data.split_diagnostics(cfg, n_query=200)
    prepare_data.make_partitions(cfg)
    prepare_data.write_manifest(cfg)
    return cfg


def test_prep_outputs(cfg):
    out = common.processed_dir(cfg)
    s = json.loads((out / "subset_summary.json").read_text())
    assert s["rows_dropped_exact_duplicates"] >= 1
    assert s["rows_dropped_inf_nan"] >= 1
    assert s["rows_dropped_conflicting_labels"] >= 2
    assert s["split"]["grouped_by_source_file"]
    assert sum(s["raw_label_counts_before_sampling"].values()) == s["rows_raw_total"]
    assert (out / "label_mapping.csv").exists()
    for m in ("dirichlet", "kmeans", "iid"):
        rep = json.loads((out / "partitions" / f"{m}_k5.json").read_text())
        assert "checks_passed" in rep and len(rep["clients"]) == 5
    assert (out / "partitions" / "kmeans_k5_model.npz").exists()
    assert (out / "pipeline_manifest.json").exists()

    # Grouped split: closest file assignment, every file in exactly one split,
    # realized fractions within one file of the targets.
    sp = s["split"]
    files = sp["files"]
    allf = files["train_pool"] + files["val"] + files["test"]
    assert len(allf) == len(set(allf)) == 12
    one_file = 1 / 12
    assert abs(sp["realized_frac"]["test"] - 0.20) <= one_file
    assert abs(sp["realized_frac"]["val"] - 0.10) <= one_file
    sf = pd.read_csv(out / "split_files.csv")
    assert set(sf["split"]) == {"train_pool", "val", "test"} and len(sf) == 12
    assert sf["subset_rows"].sum() == s["rows_final_subset"]

    # Feature schema: 46 official rows; the synthetic CSVs carry 10 of them.
    fs = pd.read_csv(out / "feature_schema.csv")
    assert (fs["official_index"].notna()).sum() == 46
    assert fs["model_input"].sum() == s["num_features"] == len(FEATURES)
    assert s["feature_schema"]["used_as_inputs"] == len(FEATURES)

    d = json.loads((out / "split_diagnostics.json").read_text())
    assert d["test_to_train"]["exact_duplicates"] == 0
    dp = json.loads((out / "partitions" / "dirichlet_k5.json").read_text())
    assert dp["label_informed_heldout_assignment"] is True


def test_runs_are_paired_and_summarized(cfg):
    data = common.load_all(cfg)
    part = {s: common.load_partition(cfg, s) for s in ("train", "val", "test")}
    device = common.get_device()

    clean_cfg = {**cfg, "experiment": "fedavg_noniid"}
    part["attackers"] = []
    clean = run_federated.run(clean_cfg, data, part, 42, True, device)

    pois_cfg = common.apply_overrides(cfg, ["experiment=fedavg_noniid_poisoned", "attack.enabled=true"])
    part["attackers"] = run_federated.select_attackers(
        pois_cfg["attack"], data["train"][1], part["train"], 5)
    pois = run_federated.run(pois_cfg, data, part, 42, False, device)
    pois2 = run_federated.run(pois_cfg, data, part, 42, False, device)

    trim_cfg = common.apply_overrides(pois_cfg, ["experiment=trimmed_noniid_poisoned",
                                                 "federated.aggregation=trimmed_mean"])
    trim = run_federated.run(trim_cfg, data, part, 42, False, device)

    # Same seed -> same initial weights and same poisoned records across conditions.
    assert clean["init_weights_sha256"] == pois["init_weights_sha256"] == trim["init_weights_sha256"]
    assert pois["attack"]["poisoned"] == trim["attack"]["poisoned"]
    assert pois["attack"]["poisoned"][0]["flipped_rows"] > 0
    # Fully deterministic rerun.
    assert pois["val"]["macro_f1"] == pois2["val"]["macro_f1"]
    assert trim["aggregation"]["trimmed_per_side"] == 1
    assert len(clean["per_client_val"]) == 5 and "per_client_test" in clean
    assert clean["communication"]["cumulative_per_run"] == 2 * clean["communication"]["total_per_round"]

    mean_cfg = common.apply_overrides(pois_cfg, ["experiment=mean_noniid_poisoned",
                                                 "federated.aggregation=mean"])
    mean_p = run_federated.run(mean_cfg, data, part, 42, False, device)
    part["attackers"] = []
    mean_c = run_federated.run(common.apply_overrides(cfg, ["experiment=mean_noniid",
                                                            "federated.aggregation=mean"]),
                               data, part, 42, False, device)
    trim_c = run_federated.run(common.apply_overrides(cfg, ["experiment=trimmed_noniid",
                                                            "federated.aggregation=trimmed_mean"]),
                               data, part, 42, False, device)
    assert mean_p["aggregation"]["weighting"] == "unweighted"
    assert clean["aggregation"]["weighting"] == "by client size"
    assert mean_p["init_weights_sha256"] == clean["init_weights_sha256"]
    assert "macro_pr_auc" in clean["per_client_val"][0]

    res = common.repo_path(cfg["paths"]["results_dir"])
    for r in (clean, pois, trim, mean_p, mean_c, trim_c):
        common.save_json(r, res / r["experiment"] / "seed_42.json")
    cen = train_centralized.run({**cfg, "experiment": "centralized"}, data, 42, False, device)
    common.save_json(cen, res / "centralized" / "seed_42.json")

    summarize_results.main(["--config", str(_write_cfg(cfg, res.parent))])
    assert (res / "runs.csv").exists() and (res / "paired_deltas.csv").exists()
    assert (res / "convergence_val_f1.png").exists()
    deltas = pd.read_csv(res / "paired_deltas.csv")
    assert {"trimming_gain_under_attack", "unweighting_gain_under_attack"} <= set(deltas.comparison)

    error_analysis.main(["--config", str(_write_cfg(cfg, res.parent))])
    an = res / "analysis"
    fp = pd.read_csv(an / "per_family_paired.csv")
    assert {"attack_cost_fedavg", "attack_cost_trimmed", "attack_cost_mean"} <= set(fp.comparison)
    cp = pd.read_csv(an / "per_client_paired.csv")
    assert cp["client"].nunique() == 5
    assert (an / "family_missed_as_benign.png").exists() and (an / "client_attack_cost.png").exists()


def _write_cfg(cfg, folder):
    import yaml
    p = folder / "cfg.yaml"
    p.write_text(yaml.safe_dump({"paths": cfg["paths"]}))
    return p
