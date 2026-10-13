import numpy as np
import pytest

from attacks import flip_labels
from labels import BENIGN_ID, FAMILIES, to_family
from metrics import compute_metrics

# All 34 labels in the CICIoT2023 CSVs (33 attacks + benign).
CICIOT2023_LABELS = [
    "BenignTraffic",
    "DDoS-RSTFINFlood", "DDoS-PSHACK_Flood", "DDoS-SYN_Flood", "DDoS-UDP_Flood",
    "DDoS-TCP_Flood", "DDoS-ICMP_Flood", "DDoS-SynonymousIP_Flood",
    "DDoS-ACK_Fragmentation", "DDoS-UDP_Fragmentation", "DDoS-ICMP_Fragmentation",
    "DDoS-SlowLoris", "DDoS-HTTP_Flood",
    "DoS-UDP_Flood", "DoS-SYN_Flood", "DoS-TCP_Flood", "DoS-HTTP_Flood",
    "Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain",
    "Recon-PingSweep", "Recon-OSScan", "Recon-PortScan", "VulnerabilityScan",
    "Recon-HostDiscovery",
    "DNS_Spoofing", "MITM-ArpSpoofing",
    "BrowserHijacking", "Backdoor_Malware", "XSS", "Uploading_Attack",
    "SqlInjection", "CommandInjection",
    "DictionaryBruteForce",
]


def test_every_label_maps_to_one_of_eight_families():
    assert len(CICIOT2023_LABELS) == 34
    families = {to_family(label) for label in CICIOT2023_LABELS}
    assert families == set(FAMILIES)


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        to_family("NotARealLabel")


def test_flip_labels_flips_half_of_attack_rows_to_benign():
    y = np.array([BENIGN_ID] * 10 + [1] * 20 + [2] * 20)
    poisoned, n = flip_labels(y, 0.5, BENIGN_ID, np.random.default_rng(0))
    assert n == 20
    assert (poisoned == BENIGN_ID).sum() == 30
    assert (poisoned[y == BENIGN_ID] == BENIGN_ID).all()  # benign rows untouched
    assert (y == BENIGN_ID).sum() == 10  # input not modified


def test_binary_fpr_fnr():
    # 4 benign (1 flagged as attack), 4 attack (2 missed as benign).
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 2, 0])
    proba = np.eye(len(FAMILIES))[y_pred]
    m = compute_metrics(y_true, proba)
    assert m["binary_fpr"] == pytest.approx(0.25)
    assert m["binary_fnr"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["per_class"]["Mirai"]["fnr"] is None  # absent class
