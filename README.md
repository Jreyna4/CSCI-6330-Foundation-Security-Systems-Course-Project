# Federated Intrusion Detection Under Label-Flipping Poisoning

A simulated multi-site study on CICIoT2023.
Course project for **CSCI 6330: Foundations of Intelligent Security Systems** (Fall 2026), The University of Texas Rio Grande Valley.

> **Status:** Repository scaffold is in place (data prep, partitioning, centralized model, federated loop, attack, defenses, metrics). No results yet.

## Team

- Salvador Leon Lopez
- Elena Victoria Sanchez
- Joe Reyna
- April Flores

| Area | Owner | Main files |
|---|---|---|
| Data preparation (subset, splits, client partitions) | TBD | `src/labels.py`, `src/make_subset.py`, `src/make_partitions.py` |
| Centralized reference model | TBD | `src/model.py`, `src/train_centralized.py` |
| Federated baseline (FedAvg, IID and non-IID) | TBD | `src/run_federated.py`, `src/aggregation.py` |
| Attack (label flipping) and defense (trimmed mean) | TBD | `src/attacks.py`, `src/aggregation.py` |

## Changes since the proposal

- **Four clients instead of six.** The instructor said k = 6 was too many, so the k-means partition now uses k = 4 (`partition.num_clients` in `configs/default.yaml`). With one attacker, that is 1 of 4 clients (25%) instead of 1 of 6.
- **Trimmed mean setting.** With 4 clients, `trim_beta = 0.25` trims the single highest and lowest value per coordinate. A smaller beta (for example 0.2) trims nothing at 4 clients and is just a plain mean.
- **Federated framework.** The FL loop is a small single-process simulation in `src/run_federated.py` instead of Flower. The project spec allows "a clearly documented minimal simulation written by the team", and Flower 1.39 has deprecated its `run_simulation()` API in favor of a SuperLink-based setup. FedAvg, trimmed mean and median are implemented in `src/aggregation.py` and tested in `tests/`. Mention this in the Milestone 1 report.

## Research question

In a four-client federated IDS trained on non-IID partitions of CICIoT2023, how much global detection performance is lost when one client performs a label-flipping attack, and does robust aggregation recover it?

**Expectation (falsifiable):** Under FedAvg, global macro F1 falls by more than 3 points versus the clean run, and trimmed-mean aggregation recovers at least half of that loss. Smaller effects would falsify this.

## Experimental conditions

| # | Condition | Config |
|---|---|---|
| 1 | Centralized reference on pooled training data | `configs/centralized.yaml` |
| 2a | Clean FedAvg, IID split (control) | `configs/fedavg_iid.yaml` |
| 2b | Clean FedAvg, non-IID split | `configs/fedavg_noniid.yaml` |
| 3 | FedAvg, non-IID, one label-flipping client | `configs/fedavg_noniid_poisoned.yaml` |
| 4a | Trimmed mean, non-IID, clean | `configs/trimmed_noniid.yaml` |
| 4b | Trimmed mean, non-IID, one label-flipping client | `configs/trimmed_noniid_poisoned.yaml` |

Extensions if time allows: local-only models, median aggregation (`federated.aggregation: median` is already implemented).

## Repository layout

```
.
├── configs/
│   ├── default.yaml              # all shared settings; experiment files override a few keys
│   └── *.yaml                    # one file per experimental condition
├── data/                         # not committed
│   ├── raw/                      # CICIoT2023 CSVs go here
│   └── processed/                # train_pool / val / test parquet + partitions/
├── src/
│   ├── common.py                 # config loading, seeding, paths
│   ├── labels.py                 # 33 attack labels -> 7 families + benign
│   ├── make_subset.py            # subset + stratified train_pool / val / test split
│   ├── data.py                   # loading + preprocessing (log1p + standardize)
│   ├── make_partitions.py        # k-means (non-IID) and IID client splits + leakage report
│   ├── model.py                  # the MLP used everywhere
│   ├── engine.py                 # train / predict / get and set weights
│   ├── metrics.py                # macro F1, FPR, FNR, PR-AUC, confusion matrix
│   ├── attacks.py                # label flipping
│   ├── aggregation.py            # FedAvg, trimmed mean, median
│   ├── train_centralized.py      # condition 1
│   ├── run_federated.py          # conditions 2-4
│   └── summarize_results.py      # mean/std table + convergence plot
├── tests/                        # pytest unit tests
├── notebooks/                    # exploration only; reported numbers come from scripts
├── results/                      # run outputs (git-ignored)
├── reports/                      # proposal, milestone reports, slides
├── requirements.txt              # pinned dependencies
└── README.md
```

## Setup

Requires Python 3.11, 3.12 or 3.13.

```bash
git clone https://github.com/Jreyna4/CSCI-6330-Foundation-Security-Systems-Course-Project.git
cd CSCI-6330-Foundation-Security-Systems-Course-Project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Linux only, optional: CPU-only PyTorch avoids a ~2 GB CUDA download
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pytest                           # unit tests, no dataset needed
```

## Data

This repository does **not** include the dataset.

1. Download the CSV files of CICIoT2023 from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Place the CSV files in `data/raw/`.

Notes:

- The official page states no license and asks for citation. We use it for coursework research only (accessed Sep. 20, 2026).
- We build a documented subset of about 300,000 flows and collapse the 33 attack types into 7 families plus benign: DDoS, DoS, Recon, Web-based, Brute force, Spoofing, Mirai. Rare families are kept whole and common ones are capped at `data.max_per_class` rows. The exact counts are written to `data/processed/subset_summary.json`.
- Identifier fields (IPs, ports, timestamps, row indices), if present, are dropped to prevent shortcut learning (`data.drop_cols`).
- For a quick test, set `data.max_files: 2` in `configs/default.yaml` to read only two CSVs.

## How to run

```bash
# 1. Subset and splits (writes data/processed/)
python src/make_subset.py

# 2. Client partitions: k-means (non-IID) and IID, 4 clients, plus leakage report
python src/make_partitions.py

# 3. Centralized reference
python src/train_centralized.py

# 4. Federated conditions
python src/run_federated.py --config configs/fedavg_iid.yaml
python src/run_federated.py --config configs/fedavg_noniid.yaml
python src/run_federated.py --config configs/fedavg_noniid_poisoned.yaml
python src/run_federated.py --config configs/trimmed_noniid.yaml
python src/run_federated.py --config configs/trimmed_noniid_poisoned.yaml

# 5. Table (results/summary.csv) and convergence plot
python src/summarize_results.py
```

Each run writes `results/<experiment>/seed_<seed>.json`. Add `--seeds 42` for a single quick run. Scripts report **validation** metrics only. Add `--final` to also evaluate on the test set; do that once, after all settings are frozen.

## Reproducibility

- A stratified 20% test set and 10% validation set are held out first (seed 42). The test set is used only for final reporting; all settings are chosen on validation.
- The remaining 70% is split into four simulated clients with k-means (k = 4, seed 42) on standardized features fit on that pool only. No labels are used to build the partition. An IID split is the control.
- `src/make_partitions.py` prints per-client family counts and the client-label mutual information, and saves them next to each partition (`data/processed/partitions/*.json`). It warns if a client gets less than 5% of the pool.
- Each condition runs 30 rounds, one local epoch, all four clients participating, with 3 seeds (report mean and standard deviation).

## Evaluation

Primary metric: macro F1 on the global test set. Also reported: per-class precision, recall, and FNR; benign-vs-attack FPR and FNR; PR-AUC; confusion matrices; per-round performance; client-level dispersion; bytes per round; runtime and memory.

## Open decisions (TODO)

- Which client is the attacker (`attack.attacker_clients`). With k-means, clients differ in how many attack rows they hold, so pick one deliberately (for example the client with the most attack rows) and justify it, or run each client as the attacker.
- `data.max_per_class`: check the resulting family counts after the first `make_subset.py` run.
- Check timing-like features (for example `IAT`) for shortcut behavior before training; list anything dropped in `data.drop_cols`.
- Model selection: scripts currently keep the model from the last epoch / round.

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
