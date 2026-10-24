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
    tr, va, te, grouped = split_indices(y, groups, 0.2, 0.1, 42, True)
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
