"""Collapse the 33 CICIoT2023 attack labels into 7 families plus benign.

Family grouping follows the CICIoT2023 paper (Neto et al., 2023). Label strings match the
`label` column of the CICIoT2023 CSV files. Unknown labels raise an error so a
typo or a different dataset version is caught instead of silently dropped.
"""

FAMILIES = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]
FAMILY_TO_ID = {name: i for i, name in enumerate(FAMILIES)}
BENIGN_ID = FAMILY_TO_ID["Benign"]

# Labels that do not start with a family prefix.
_EXPLICIT = {
    "BenignTraffic": "Benign",
    "VulnerabilityScan": "Recon",
    "BrowserHijacking": "Web",
    "Backdoor_Malware": "Web",
    "XSS": "Web",
    "Uploading_Attack": "Web",
    "SqlInjection": "Web",
    "CommandInjection": "Web",
    "DictionaryBruteForce": "BruteForce",
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
}

# DDoS-*, DoS-*, Recon-*, Mirai-* labels.
_PREFIXES = {"DDoS-": "DDoS", "DoS-": "DoS", "Recon-": "Recon", "Mirai-": "Mirai"}


def to_family(label):
    """Map one original CICIoT2023 label to its family name."""
    if label in _EXPLICIT:
        return _EXPLICIT[label]
    for prefix, family in _PREFIXES.items():
        if label.startswith(prefix):
            return family
    raise ValueError(f"Unknown CICIoT2023 label: {label!r}")
