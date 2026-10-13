# Federated Intrusion Detection Under Label-Flipping Poisoning

A simulated multi-site study on CICIoT2023.
Course project for **CSCI 6330: Foundations of Intelligent Security Systems** (Fall 2026), The University of Texas Rio Grande Valley.

> **Status:** Proposal approved stage. Setting up the repository. No results yet.

## Team

- Salvador Leon Lopez
- Elena Victoria Sanchez
- Joe Reyna
- April Flores

| Area | Owner |
|---|---|
| Data preparation (subset, splits, client partitions) | TBD |
| Centralized reference model | TBD |
| Federated baseline (FedAvg, IID and non-IID) | TBD |
| Attack (label flipping) and defense (trimmed mean) | TBD |

## Research question

In a six-client federated IDS trained on non-IID partitions of CICIoT2023, how much global detection performance is lost when one client performs a label-flipping attack, and does robust aggregation recover it?

**Expectation (falsifiable):** Under FedAvg, global macro F1 falls by more than 3 points versus the clean run, and trimmed-mean aggregation recovers at least half of that loss. Smaller effects would falsify this.

## Experimental conditions

1. Centralized reference model on pooled training data
2. Clean federated baseline: FedAvg on non-IID and IID partitions
3. Poisoned: one of six clients relabels 50% of its attack samples as benign each round
4. Robust aggregation: trimmed mean on the poisoned and clean non-IID setups

Extensions if time allows: local-only models, median aggregation.

## Planned repository layout

```
.
├── data/               # Not committed. Raw and processed CICIoT2023 files go here.
│   ├── raw/
│   └── processed/
├── src/                # Subset, split, partition, model, and FL code
├── configs/            # Experiment settings (seeds, rounds, clients, aggregation)
├── notebooks/          # Exploration only; results come from scripts
├── results/            # Metrics, plots, logs from each run
├── reports/            # PDF milestone reports and slides
├── requirements.txt    # Pinned dependencies
└── README.md
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<user-or-org>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Core dependencies (exact versions will be pinned in `requirements.txt`):

- PyTorch
- Flower (`flwr[simulation]`)
- scikit-learn
- pandas, NumPy, Matplotlib

## Data

This repository does **not** include the dataset.

1. Download CICIoT2023 from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Place the CSV files in `data/raw/`.

Notes:

- The official page states no license and asks for citation. We use it for coursework research only (accessed Sep. 20, 2026).
- We build a documented stratified subset of about 300,000 flows and collapse the 33 attack types into 7 families plus benign: DDoS, DoS, Recon, Web-based, Brute force, Spoofing, Mirai.
- Identifier fields (IPs, ports, timestamps, row indices), if present, are dropped to prevent shortcut learning.

## Reproducibility

- Hold out a stratified 20% test set and 10% validation set first (seed 42). The test set is used only for final reporting; all settings are chosen on validation.
- The remaining 70% is split into six simulated clients with k-means (k = 6, seed 42) on standardized features fit on that pool only. No labels are used to build the partition. An IID split is the control.
- Each condition runs 30 rounds, one local epoch, all six clients participating, with 3 seeds (report mean and standard deviation).
- Every reported number will have a script and command listed here.

Run commands will be added as the code is written:

```bash
# Example placeholders
python src/make_subset.py
python src/make_partitions.py
python src/train_centralized.py
python src/run_federated.py --config configs/fedavg_noniid.yaml
```

## Evaluation

Primary metric: macro F1 on the global test set. Also reported: per-class precision, recall, and FNR; FPR; PR-AUC; confusion matrices; per-round performance; client-level dispersion; bytes per round; runtime.

## Milestones

| Date | Milestone |
|---|---|
| Oct. 25 | Milestone 1: subset, partitions, centralized model |
| Nov. 15 | Milestone 2: clean vs. poisoned federated runs |
| Nov. 29 | Final slides due |
| Nov. 30 | Final presentation and demo |
| Dec. 7 | Final report and code due |

## Safety

All experiments are simulated on static public data using our own code. No real services, traffic generation, university or third-party systems, or personal data are involved. Poisoning is a controlled label change always compared against a clean baseline.

## References

1. B. McMahan et al., "Communication-efficient learning of deep networks from decentralized data," *Proc. AISTATS*, 2017.
2. V. Tolpegin, S. Truex, M. E. Gursoy, and L. Liu, "Data poisoning attacks against federated learning systems," *Proc. ESORICS*, 2020.
3. D. Yin, Y. Chen, K. Ramchandran, and P. Bartlett, "Byzantine-robust distributed learning," *Proc. ICML*, 2018.
4. E. C. P. Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, 2023.
5. D. J. Beutel et al., "Flower: A friendly federated learning framework," arXiv:2007.14390, 2020.
