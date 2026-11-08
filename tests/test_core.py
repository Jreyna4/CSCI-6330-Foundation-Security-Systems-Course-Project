import numpy as np
import pytest

from common import (
    BENIGN_ID,
    FAMILIES,
    FAMILY_TO_ID,
    LABEL_TO_FAMILY,
    apply_overrides,
    compute_metrics,
    label_mapping_table,
    to_family,
)
from prepare_data import (
    assign_heldout_dirichlet,
    dirichlet_partition,
    partition_checks,
    partition_report,
    split_indices,
)
from run_federated import (
    aggregate,
    fedavg,
    flip_labels,
    median,
    select_attackers,
    trim_count,
    trimmed_mean,
)


# --- Labels -------------------------------------------------------------------

def test_mapping_covers_34_labels_and_8_families():
    assert len(LABEL_TO_FAMILY) == 34
    assert set(LABEL_TO_FAMILY.values()) == set(FAMILIES)
    counts = label_mapping_table().family.value_counts().to_dict()
    # Family sizes of the CICIoT2023 taxonomy (Neto et al., 2023).
    assert counts == {"DDoS": 12, "Web": 6, "Recon": 5, "DoS": 4, "Mirai": 3,
                      "Spoofing": 2, "BruteForce": 1, "Benign": 1}


def test_mapping_is_case_insensitive():
    assert to_family("DDOS-ICMP_FLOOD") == "DDoS"
    assert to_family(" benigntraffic ") == "Benign"
    assert to_family("VULNERABILITYSCAN") == "Recon"


def test_merged_csv_spellings():
    # Labels as written in the MERGED_CSV release of CICIoT2023.
    assert to_family("BENIGN") == "Benign"
    assert to_family("DDOS-SYNONYMOUSIP_FLOOD") == "DDoS"
    assert to_family("DOS-HTTP_FLOOD") == "DoS"
    assert to_family("MITM-ARPSPOOFING") == "Spoofing"
    assert to_family("DNS_SPOOFING") == "Spoofing"
    assert to_family("DICTIONARYBRUTEFORCE") == "BruteForce"
    assert to_family("BACKDOOR_MALWARE") == "Web"
    assert to_family("RECON-PINGSWEEP") == "Recon"
    assert to_family("MIRAI-GREIP_FLOOD") == "Mirai"
    assert to_family("DDoS-Some_New_Variant") == "DDoS"  # prefix fallback


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        to_family("NotARealLabel")


def test_overrides():
    cfg = apply_overrides({"attack": {"num_attackers": 1}}, ["attack.num_attackers=3", "experiment=x"])
    assert cfg["attack"]["num_attackers"] == 3 and cfg["experiment"] == "x"


# --- Attack -------------------------------------------------------------------

def test_flip_labels_flips_half_of_each_attack_family():
    y = np.array([BENIGN_ID] * 10 + [1] * 20 + [2] * 30)
    poisoned, idx = flip_labels(y, 0.5, BENIGN_ID, np.random.default_rng(0))
    assert len(idx) == 25
    assert (y[idx] == 1).sum() == 10 and (y[idx] == 2).sum() == 15  # 50% per family
    assert (poisoned == BENIGN_ID).sum() == 35
    assert (poisoned[y == BENIGN_ID] == BENIGN_ID).all()  # benign rows untouched
    assert (y == BENIGN_ID).sum() == 10  # input not modified


def test_flip_labels_is_deterministic_per_seed():
    y = np.repeat(np.arange(8), 50)
    _, a = flip_labels(y, 0.5, BENIGN_ID, np.random.default_rng([42, 3, 7919]))
    _, b = flip_labels(y, 0.5, BENIGN_ID, np.random.default_rng([42, 3, 7919]))
    _, c = flip_labels(y, 0.5, BENIGN_ID, np.random.default_rng([43, 3, 7919]))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


ATK = {"target_label": "Benign", "selection": "most_attack_rows", "num_attackers": 1,
       "attacker_clients": [], "min_attack_families": 2, "min_rows_per_family": 5}


def test_attacker_is_client_with_most_attack_rows():
    y = np.array([0] * 50 + [1] * 10 + [2] * 10 + [1] * 30 + [3] * 30)
    clients = np.array([0] * 70 + [1] * 60)
    assert select_attackers(ATK, y, clients, 2) == [1]
    assert select_attackers({**ATK, "num_attackers": 2}, y, clients, 2) == [0, 1]


def test_attacker_needs_multiple_attack_families():
    y = np.array([0] * 10 + [1] * 100)
    clients = np.zeros(110, dtype=int)
    with pytest.raises(ValueError):
        select_attackers(ATK, y, clients, 1)


# --- Metrics ------------------------------------------------------------------

def test_metrics_with_support_and_fpr():
    # 4 benign (1 flagged as attack), 4 attack (2 missed as benign).
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 2, 0])
    proba = np.eye(len(FAMILIES))[y_pred]
    m = compute_metrics(y_true, proba)
    assert m["binary_fpr"] == pytest.approx(0.25)
    assert m["binary_fnr"] == pytest.approx(0.5)
    assert m["binary_counts"] == {"benign": 4, "attack": 4, "false_positives": 1, "false_negatives": 2}
    assert m["accuracy"] == pytest.approx(5 / 8)
    benign = m["per_class"]["Benign"]
    assert benign["support"] == 4 and benign["negatives"] == 4
    assert benign["fpr"] == pytest.approx(2 / 4)  # 2 attacks predicted benign
    assert m["per_class"]["DDoS"]["fpr"] == pytest.approx(1 / 6)
    assert m["per_class"]["Mirai"]["fnr"] is None  # absent class
    assert m["per_class"]["Mirai"]["support"] == 0


# --- Splits and partitions ----------------------------------------------------

def test_grouped_split_keeps_files_apart():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 3000)
    groups = np.repeat(np.arange(30), 100)
    tr, va, te, grouped, _ = split_indices(y, groups, 0.2, 0.1, 42, True)
    assert grouped
    assert len(set(tr) | set(va) | set(te)) == 3000
    g = [set(groups[i]) for i in (tr, va, te)]
    assert not (g[0] & g[1] or g[0] & g[2] or g[1] & g[2])


def test_dirichlet_partition_sizes_and_heldout_assignment():
    y = np.repeat(np.arange(8), 500)
    clients, props, _ = dirichlet_partition(y, 10, 0.5, np.random.default_rng(42), 40)
    assert np.bincount(clients, minlength=10).min() >= 40
    np.testing.assert_allclose(props.sum(axis=1), 1.0)
    # Same seed -> same partition.
    again, _, _ = dirichlet_partition(y, 10, 0.5, np.random.default_rng(42), 40)
    np.testing.assert_array_equal(clients, again)
    # Held-out rows follow the frozen train proportions.
    y_val = np.repeat(np.arange(8), 2000)
    val_c = assign_heldout_dirichlet(y_val, props, np.random.default_rng(1))
    for c in range(8):
        emp = np.bincount(val_c[y_val == c], minlength=10) / 2000
        np.testing.assert_allclose(emp, props[c], atol=0.04)


def test_partition_checks_flag_label_pure_client():
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)
    clients = np.array([0] * 100 + [1] * 200)  # client 0 is all benign
    report, _ = partition_report(clients, y, 2)
    cfg = {"min_client_share": 0.02, "max_size_ratio": 20,
           "max_label_purity": 0.95, "min_family_rows": 10}
    issues = partition_checks(report, cfg)
    assert any("client 0" in i for i in issues)
    assert report["clients"][0]["benign_fraction"] == 1.0
    assert report["clients"][1]["tv_distance_from_global"] == pytest.approx(1 / 3)


# --- Aggregation rules --------------------------------------------------------

def make_updates(values, sizes=None):
    """One-layer 'models' whose single weight array is filled with `value`."""
    sizes = sizes or [1] * len(values)
    return [([np.full((2, 3), v, dtype=np.float32)], n) for v, n in zip(values, sizes)]


def test_fedavg_weights_by_client_size():
    out = fedavg(make_updates([0.0, 10.0], sizes=[3, 1]))
    np.testing.assert_allclose(out[0], 2.5)


def test_trim_count_for_ten_clients():
    assert trim_count(10, 0.2) == 2
    assert trim_count(10, 0.1) == 1
    assert trim_count(4, 0.25) == 1


def test_trimmed_mean_ignores_two_outliers_with_ten_clients():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 1000, -1000]
    out = trimmed_mean(make_updates([float(v) for v in vals]), beta=0.2)
    np.testing.assert_allclose(out[0], np.mean([2, 3, 4, 5, 6, 7]))  # drops -1000, 1, 8, 1000


def test_trimmed_mean_is_coordinate_wise_and_unweighted():
    a = [np.array([0.0, 9.0])]
    b = [np.array([1.0, 1.0])]
    c = [np.array([2.0, 2.0])]
    d = [np.array([9.0, 0.0])]
    out = trimmed_mean([(a, 100), (b, 1), (c, 1), (d, 1)], beta=0.25)
    np.testing.assert_allclose(out[0], [1.5, 1.5])


def test_trimmed_mean_rejects_trimming_everything():
    with pytest.raises(ValueError):
        trimmed_mean(make_updates([1.0, 2.0, 3.0, 4.0]), beta=0.5)


def test_trimmed_mean_warns_when_nothing_is_trimmed():
    with pytest.warns(UserWarning):
        trimmed_mean(make_updates([1.0, 2.0, 3.0, 4.0]), beta=0.2)


def test_median():
    out = median(make_updates([1.0, 2.0, 100.0]))
    np.testing.assert_allclose(out[0], 2.0)


def test_aggregate_dispatch():
    updates = make_updates([1.0, 2.0, 3.0, 1000.0])
    out = aggregate(updates, {"aggregation": "trimmed_mean", "trim_beta": 0.25})
    np.testing.assert_allclose(out[0], 2.5)
    with pytest.raises(ValueError):
        aggregate(updates, {"aggregation": "nope"})


def test_unweighted_mean_ignores_client_size():
    import run_federated
    a = [np.array([0.0, 0.0])]
    b = [np.array([10.0, 2.0])]
    out = run_federated.unweighted_mean([(a, 1), (b, 99)])
    assert np.allclose(out[0], [5.0, 1.0])
    fed = run_federated.fedavg([(a, 1), (b, 99)])
    assert np.allclose(fed[0], [9.9, 1.98])
    assert np.allclose(run_federated.aggregate([(a, 1), (b, 99)], {"aggregation": "mean"})[0], [5.0, 1.0])


def test_pr_auc_excludes_zero_support_classes():
    from common import FAMILIES, compute_metrics
    y = np.array([0, 0, 1, 1])
    proba = np.full((4, len(FAMILIES)), 0.01)
    proba[[0, 1], 0] = 0.9
    proba[[2, 3], 1] = 0.9
    m = compute_metrics(y, proba)
    assert m["pr_auc_classes"] == ["Benign", "DDoS"]
    assert m["pr_auc_excluded"]["Mirai"] == "no support"
    assert m["macro_pr_auc"] == 1.0
    assert m["f1_classes"] == ["Benign", "DDoS"]
    # One class only: no negatives, so PR-AUC is undefined and excluded.
    m1 = compute_metrics(np.array([1, 1]), proba[2:])
    assert m1["pr_auc_excluded"]["DDoS"] == "no negatives" and m1["macro_pr_auc"] is None


def test_feature_schema_maps_merged_release():
    from common import OFFICIAL_ABSENT_IN_MERGED, feature_schema
    merged = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate", "fin_flag_number",
              "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
              "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count",
              "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
              "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
              "Tot size", "IAT", "Number", "Variance"]
    assert len(merged) == 39
    fs = feature_schema(merged, merged, [])
    counts = fs["status"].value_counts().to_dict()
    assert counts == {"kept": 37, "absent in release": 8, "kept (renamed)": 1,
                      "release-only (kept)": 1}
    absent = set(fs.loc[fs.status == "absent in release", "official_name"])
    assert absent == set(OFFICIAL_ABSENT_IN_MERGED)
    assert fs.loc[fs.official_name == "Duration", "release_name"].item() == "Time_To_Live"
    assert fs["model_input"].sum() == 39


def test_group_split_search_hits_targets():
    import prepare_data
    rng = np.random.default_rng(0)
    groups = np.repeat([f"f{i:02d}" for i in range(40)], 100)
    y = rng.integers(0, 8, size=len(groups))
    tr, va, te, info = prepare_data.search_group_split(y, groups, 0.2, 0.1, 42, 500)
    assert len(tr) + len(va) + len(te) == len(y)
    assert abs(len(te) / len(y) - 0.2) <= 0.025 and abs(len(va) / len(y) - 0.1) <= 0.025
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[va]) & set(groups[te]))
    tr2, va2, te2, _ = prepare_data.search_group_split(y, groups, 0.2, 0.1, 42, 500)
    assert (te == te2).all()  # deterministic


def test_krum_drops_the_outlier():
    import run_federated
    honest = [([np.array([1.0, 1.0]) + 0.01 * i], 10) for i in range(8)]
    bad = [([np.array([50.0, -50.0])], 100), ([np.array([40.0, -45.0])], 100)]
    ups = honest + bad
    out, kept = run_federated.multi_krum(ups, f=2, m=1)
    assert kept[0] < 8 and np.allclose(out[0], [1.0, 1.0], atol=0.1)
    out_m, kept_m = run_federated.multi_krum(ups, f=2, m=8)
    assert kept_m == list(range(8))
    agg = run_federated.aggregate(ups, {"aggregation": "multi_krum", "krum_f": 2, "multi_krum_m": None})
    assert np.allclose(agg[0], out_m[0])


def test_client_diagnostics_flag_the_outlier():
    import run_federated
    g = [np.zeros(50)]
    rng = np.random.default_rng(0)
    ups = [([rng.normal(0, 0.1, 50)], 10) for _ in range(9)] + [([np.full(50, 5.0)], 10)]
    d = run_federated.client_diagnostics(ups, g, trim_k=2, krum_f=2)
    assert d["trimmed_share"][9] == 1.0          # always in the top tail
    assert d["krum_rank"][9] == 9                # least typical
    assert np.argmax(d["dist_to_median"]) == 9
    assert all(0.0 <= t <= 1.0 for t in d["trimmed_share"])


def test_attacker_selection_modes():
    import run_federated
    atk = {"target_label": "Benign", "num_attackers": 1, "min_rows_per_family": 1,
           "min_attack_families": 2, "selection": "median_size", "attacker_clients": []}
    # client sizes 10, 20, 30, 40, 50 (median 30); families 1 and 2 everywhere
    clients = np.repeat(np.arange(5), [10, 20, 30, 40, 50])
    y = np.tile([1, 2], len(clients) // 2)
    assert run_federated.select_attackers(atk, y, clients, 5) == [2]
    r1 = run_federated.select_attackers({**atk, "selection": "random", "random_draw": 0}, y, clients, 5, 42)
    r2 = run_federated.select_attackers({**atk, "selection": "random", "random_draw": 0}, y, clients, 5, 42)
    assert r1 == r2 and len(r1) == 1                      # deterministic
    draws = {tuple(run_federated.select_attackers({**atk, "selection": "random", "random_draw": d},
                                                  y, clients, 5, 42)) for d in range(6)}
    assert len(draws) > 1                                 # different draws differ


def test_clean_baseline_and_counterpart_names():
    from summarize_results import clean_baseline, fedavg_counterpart
    names = {"fedavg_noniid", "fedavg_noniid_r60", "fedavg_noniid_poisoned_r60", "krum_noniid",
             "krum_noniid_poisoned_atkrand1", "fedavg_noniid_poisoned_atkrand1"}
    assert clean_baseline("fedavg_noniid_poisoned_r60", names) == "fedavg_noniid_r60"
    assert clean_baseline("krum_noniid_poisoned_atkrand1", names) == "krum_noniid"
    assert clean_baseline("fedavg_noniid", names) is None
    assert fedavg_counterpart("krum_noniid_poisoned_atkrand1", names) == "fedavg_noniid_poisoned_atkrand1"


def test_targeted_flip_only_touches_source_family():
    import run_federated
    y = np.array([0, 1, 1, 3, 3, 3, 3, 6, 6])
    y2, idx = run_federated.flip_labels(y, 1.0, 0, np.random.default_rng(0), True, [3])
    assert (y2[[3, 4, 5, 6]] == 0).all() and (y2[[1, 2, 7, 8]] == y[[1, 2, 7, 8]]).all()
    assert list(idx) == [3, 4, 5, 6]


def test_most_source_rows_picks_holder_of_target_family():
    import run_federated
    atk = {"target_label": "Benign", "num_attackers": 1, "min_rows_per_family": 1,
           "min_attack_families": 1, "selection": "most_source_rows",
           "source_families": ["Recon"], "attacker_clients": []}
    # client 0: many DDoS; client 1: most Recon
    y = np.array([1] * 50 + [3] * 2 + [3] * 10 + [1] * 1)
    clients = np.array([0] * 52 + [1] * 11)
    assert run_federated.select_attackers(atk, y, clients, 2) == [1]


def test_balanced_loss_weights():
    from common import loss_weights
    y = np.array([0] * 6 + [1] * 2 + [2] * 2)
    assert loss_weights({"train": {"loss_weighting": "none"}}, y, 4) is None
    w = loss_weights({"train": {"loss_weighting": "balanced"}}, y, 4).numpy()
    assert np.allclose(w[:3], [10 / 24, 10 / 8, 10 / 8]) and w[3] == 0.0
