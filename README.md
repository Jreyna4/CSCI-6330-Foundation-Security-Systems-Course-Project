# Federated Intrusion Detection Under Label-Flipping Poisoning

A simulated multi-site study on CICIoT2023.
Course project for **CSCI 6330: Foundations of Intelligent Security Systems** (Fall 2026), The University of Texas Rio Grande Valley.

> **Status:** Pipeline revised to address the proposal feedback (see [`reports/feedback_response.md`](reports/feedback_response.md)). Tested end-to-end on synthetic data; no results on CICIoT2023 yet.

## Team

- Salvador Leon Lopez
- Elena Victoria Sanchez
- Joe Reyna
- April Flores

| Area | Owner | Main file |
|---|---|---|
| Data preparation (subset, splits, client partitions) | TBD | `src/prepare_data.py` |
| Centralized reference model | TBD | `src/train_centralized.py` |
| Federated baseline (FedAvg, IID and non-IID) | TBD | `src/run_federated.py` (simulation loop, `fedavg`) |
| Attack (label flipping) and defense (trimmed mean) | TBD | `src/run_federated.py` (`flip_labels`, `select_attackers`, `trimmed_mean`) |

## Changes since the proposal

- **Ten clients instead of six.** The written feedback recommends 10-20 clients for clearer statistics, so `partition.num_clients: 10`. One attacker is 10% of clients; the attacker sweep also runs 2 and 3 attackers (20%, 30%).
- **Dirichlet partition as the primary non-IID split.** The feedback notes that k-means assumes spherical, similar-size clusters, is sensitive to outliers, and can give unequal or label-pure clients. The main non-IID split is now a controlled Dirichlet label-skew split (alpha = 0.5). The k-means partition is still built, checked and reported as an alternative. The result is *simulated statistical heterogeneity*, not evidence about real organizations.
- **Trimmed mean setting.** With 10 clients, `trim_beta = 0.2` removes the 2 highest and 2 lowest values per coordinate and averages the remaining 6, unweighted.
- **Data pipeline order.** Exact duplicates are removed before splitting, splits are grouped by source CSV file, and held-out rows are assigned to clients with the frozen train-time partitioner. The order is listed below.
- **Federated framework.** The FL loop is a small single-process simulation in `src/run_federated.py` instead of Flower. The project spec allows "a clearly documented minimal simulation written by the team", and Flower 1.39 has deprecated its `run_simulation()` API in favor of a SuperLink-based setup.

## Research question

In a ten-client federated IDS trained on non-IID partitions of CICIoT2023, how much global detection performance is lost when one client performs a label-flipping attack, and does robust aggregation recover it?

**Expectation (falsifiable):** Under FedAvg, global macro F1 falls by more than 3 points versus the clean run, and trimmed-mean aggregation recovers at least half of that loss. Smaller effects would falsify this. Label flipping may not create coordinate-wise extreme updates, so weak recovery would itself be a meaningful result.

## Experimental conditions

| # | Condition | Config |
|---|---|---|
| 1 | Centralized reference on pooled training data | `configs/centralized.yaml` |
| 2a | Clean FedAvg, IID split (control) | `configs/fedavg_iid.yaml` |
| 2b | Clean FedAvg, non-IID split | `configs/fedavg_noniid.yaml` |
| 3 | FedAvg, non-IID, one label-flipping client | `configs/fedavg_noniid_poisoned.yaml` |
| 4a | Trimmed mean, non-IID, clean | `configs/trimmed_noniid.yaml` |
| 4b | Trimmed mean, non-IID, one label-flipping client | `configs/trimmed_noniid_poisoned.yaml` |
| S | Conditions 3 and 4b with 2 and 3 attackers | `scripts/run_attacker_sweep.sh` |

Everything not named in a condition is identical across conditions: selected records, partitions, initial weights (per seed), participating clients, local epochs, optimizer, learning rate, batch size, rounds, preprocessing and test data.

Extensions if time allows: local-only models, median aggregation (`federated.aggregation: median`), Dirichlet alpha sweep (0.1 / 0.5 / 1.0), k-means partition instead of Dirichlet.

## Repository layout

```
.
├── configs/
│   ├── default.yaml              # all shared settings; experiment files override a few keys
│   ├── pilot.yaml                # small end-to-end pilot (stack it before any other config)
│   └── *.yaml                    # one file per experimental condition
├── data/                         # not committed
│   ├── raw/                      # CICIoT2023 CSVs go here
│   └── processed/                # splits, partitions/, summaries, manifest
├── src/
│   ├── common.py                 # config, label mapping, data loading, MLP, training, metrics
│   ├── prepare_data.py           # step 1: subset, dedup, grouped splits, partitions + checks
│   ├── train_centralized.py      # step 2: condition 1
│   ├── run_federated.py          # step 3: conditions 2-4 (FL loop, attack, FedAvg / trimmed mean / median)
│   └── summarize_results.py      # step 4: per-run, summary, paired-delta and per-client tables + plot
├── scripts/
│   ├── run_all.sh                # whole pipeline (add "pilot" for the pilot)
│   └── run_attacker_sweep.sh     # 2 and 3 attackers
├── tests/                        # unit tests + synthetic end-to-end test
├── reports/                      # proposal, feedback response, milestone reports, slides
├── results/                      # run outputs (git-ignored)
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
pytest                           # unit tests + synthetic end-to-end test, no dataset needed
```

## Data

This repository does **not** include the dataset.

1. Download the CSV files of CICIoT2023 from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Place the CSV files in `data/raw/`.

The official page states no license and asks for citation. We use it for coursework research only (accessed Sep. 20, 2026). It is lab-generated, single-testbed data, so all claims are limited to this simulation.

### Label mapping (33 attacks + benign -> 7 families + benign)

Follows the seven attack categories of the CICIoT2023 paper (Neto et al., 2023). Matching is case-insensitive. The table is also written to `data/processed/label_mapping.csv`, and a unit test checks the family sizes.

| Family | Original labels |
|---|---|
| Benign (1) | BenignTraffic |
| DDoS (12) | DDoS-ACK_Fragmentation, DDoS-HTTP_Flood, DDoS-ICMP_Flood, DDoS-ICMP_Fragmentation, DDoS-PSHACK_Flood, DDoS-RSTFINFlood, DDoS-SYN_Flood, DDoS-SlowLoris, DDoS-SynonymousIP_Flood, DDoS-TCP_Flood, DDoS-UDP_Flood, DDoS-UDP_Fragmentation |
| DoS (4) | DoS-HTTP_Flood, DoS-SYN_Flood, DoS-TCP_Flood, DoS-UDP_Flood |
| Recon (5) | Recon-HostDiscovery, Recon-OSScan, Recon-PingSweep, Recon-PortScan, VulnerabilityScan |
| Web-based (6) | Backdoor_Malware, BrowserHijacking, CommandInjection, SqlInjection, Uploading_Attack, XSS |
| Brute force (1) | DictionaryBruteForce |
| Spoofing (2) | DNS_Spoofing, MITM-ArpSpoofing |
| Mirai (3) | Mirai-greeth_flood, Mirai-greip_flood, Mirai-udpplain |

### How the ~300,000-row subset is built (in this order)

1. **Read** every CSV; record the original label counts **before any sampling** and keep each row's source file.
2. **Sample per file:** at most `ceil(2 × max_per_class / n_files)` rows per family from each file (seed 42 + file index), so the subset covers all capture files.
3. **Clean:** drop `data.drop_cols`, rows with inf/NaN, **exact duplicate (feature + label) rows**, and feature vectors that appear with **conflicting labels**. Every count is logged.
4. **Cap:** at most `max_per_class` = 45,000 rows per family (seed 42); rarer families are kept whole. **The subset prevalence is therefore a design choice, not the natural prevalence.** Both are reported (`family_counts.csv`, `subset_summary.json`).
5. **Split:** test 20%, then validation 10%, stratified by family and **grouped by source CSV file** (no file contributes rows to two splits). CICIoT2023's CSVs have no device or session column, so the source file is the only capture grouping available.
6. **Partition** the 70% training pool into 10 clients. The scaler and k-means are fit on training rows only.
7. **Assign held-out rows to clients** with the **frozen** partitioner: nearest train-fit centroid for k-means, the train-time per-family client proportions for Dirichlet. Nothing is refit on held-out data.
8. **Freeze:** `pipeline_manifest.json` stores the config, package versions and SHA-256 of every output; each run records these hashes.

### Client partitions and checks

`prepare_data.py` builds three partitions every time:

| Method | How | Role |
|---|---|---|
| `dirichlet` | For each family, client shares ~ Dir(0.5); redrawn until every client has >= 2% of the pool | **Primary non-IID** split; uses labels by design, since label skew is the controlled variable |
| `kmeans` | k-means (k = 10) on log-transformed, standardized features; no labels | Reported alternative |
| `iid` | Random equal split | Control |

For each partition it reports and saves, per client: size and share, benign/attack balance, family counts and fractions, label purity, total-variation and Jensen-Shannon distance from the global distribution, held-out size, and client-label mutual information (`partitions/*.json`, `partitions/partition_table_k10.csv`). It then runs **PASS/FAIL checks**: every client holds >= 2% of the pool, largest/smallest size ratio <= 20, no client is > 95% one family, and every client has >= 2 families with >= 50 rows (trainable). If the configured method fails, change `partition.dirichlet_alpha` or `num_clients` before training.

## Model and training hyperparameters

| Setting | Value |
|---|---|
| Input | 46 numeric flow features (exact list in `subset_summary.json`), `sign(x)·log1p(|x|)` then standardization fit on the training pool |
| MLP | in -> 128 -> 64 -> 8; ReLU after each hidden layer; dropout 0.1 after each hidden layer; output = 8 logits |
| Loss | Cross-entropy, no class weights |
| Optimizer | Adam, lr 0.001, betas (0.9, 0.999), weight decay 0 |
| Batch size | 256 (centralized and each client's local batch) |
| Centralized | 30 epochs |
| Federated | 30 rounds × 1 local epoch, all 10 clients every round, fresh Adam state per client per round |
| Model selection | Last epoch / last round (no early stopping) |

## Threat model and attack protocol

- **Attacker choice:** the client(s) with the most attack rows in the training partition (`attack.selection: most_attack_rows`). This depends only on the frozen partition, so **the same client is malicious for every seed and every condition**.
- **Eligibility:** the attacker must hold >= 2 attack families with >= 100 rows each, otherwise the run stops. This keeps a 50% flip meaningful.
- **Poisoned records:** 50% of the attacker's attack rows, **within each attack family**, relabeled as Benign. They are drawn **once per seed** from a dedicated RNG and reused unchanged in **every round**. The same seed gives the same poisoned records under FedAvg and trimmed mean. A hash of the poisoned row indices is saved in each result.
- **Behavior:** the attacker changes **labels only**. It trains and sends updates exactly like an honest client. Evaluation always uses clean labels.
- **Malicious ratio:** `attack.num_attackers` (1 of 10 by default). The sweep runs 2 and 3. Three attackers exceed the 2 values trimmed per side, which the run notes.

## Seeds and pairing

`data.seed = 42` fixes the subset, splits and partitions for **all** runs. Each run seed (42, 43, 44) changes only:

- model initialization
- minibatch order and dropout masks
- which attack rows are poisoned

Conditions with the same seed are **paired**. They start from identical initial weights (`init_weights_sha256` in each result) and use the same poisoned records. `summarize_results.py` reports every individual run, mean ± std (sample std), and **per-seed paired differences**. Three seeds are a minimum, not a basis for distributional claims.

## How to run

```bash
# 0. Pilot first: 12 CSVs, <= 3,000 rows per family, 5 rounds, 1 seed (separate folders)
bash scripts/run_all.sh pilot

# 1. Subset, splits, partitions (dirichlet, kmeans, iid) + checks + manifest
python src/prepare_data.py
#    After changing num_clients or alpha, redo only the partitions:
#    python src/prepare_data.py --partitions-only

# 2. Centralized reference
python src/train_centralized.py

# 3. Federated conditions
python src/run_federated.py --config configs/fedavg_iid.yaml
python src/run_federated.py --config configs/fedavg_noniid.yaml
python src/run_federated.py --config configs/fedavg_noniid_poisoned.yaml
python src/run_federated.py --config configs/trimmed_noniid.yaml
python src/run_federated.py --config configs/trimmed_noniid_poisoned.yaml
bash scripts/run_attacker_sweep.sh          # 2 and 3 attackers

# 4. Tables and convergence plot
python src/summarize_results.py
```

Steps 1-4 are also `bash scripts/run_all.sh`. On Windows without bash, run the Python commands above.

Any config key can be overridden from the command line, for example `--set attack.num_attackers=2 experiment=my_run`. Several config files can be stacked, as in `--config configs/pilot.yaml configs/fedavg_noniid.yaml`.

Scripts report **validation** metrics only. Add `--final` (or `FINAL=1 bash scripts/run_all.sh`) to also evaluate on the test set. Do that once, after every setting is frozen, then run `python src/summarize_results.py --split test`.

## Evaluation

- **Global:** macro F1 on the pooled test set (primary); per-family precision, recall, FNR, one-vs-rest FPR, F1 and PR-AUC, **each with its support and number of negatives**; benign-vs-attack FPR/FNR with counts; confusion matrix; per-round validation curves.
- **Client-level:** the global model evaluated on each client's held-out records (assigned by the frozen partitioner), with per-family support, plus the minimum and spread across clients. A pooled test set can hide failures on minority clients or families, so both are reported.
- **Communication proxy:** raw float32 model bytes. Per participating client per round, one download (global model) plus one upload (full local model). Also reported per round for all clients and cumulatively per run. Downloads are included; compression and protocol overhead are not.
- **Runtime:** single-machine simulation time, stored with a hardware description (CPU, cores, RAM, GPU if any, torch version and threads) and peak resident memory.

Output tables in `results/`: `runs.csv`, `summary.csv`, `paired_deltas.csv`, `hypothesis_check.csv`, `per_client.csv`, `per_class.csv`, `convergence_val_f1.png`.

## Open decisions (TODO)

- Run the pilot, then check the Dirichlet and k-means checks on the real data. If Dirichlet fails, adjust `partition.dirichlet_alpha`.
- Check the family counts in `data/processed/family_counts.csv` (settles `max_per_class`).
- Check timing-like features (for example `IAT`) for shortcut behavior; list anything dropped in `data.drop_cols` and in the report.
- Confirm which client was chosen as attacker and quote its family counts in the Milestone 2 report.

## Milestones

| Date | Milestone |
|---|---|
| Oct. 25 | Milestone 1: subset, partitions, centralized model, small FedAvg pilot |
| Nov. 15 | Milestone 2: clean vs. poisoned federated runs |
| Nov. 29 | Final slides due |
| Nov. 30 | Final presentation and demo |
| Dec. 7 | Final report and code due |

## Safety

All experiments are simulated on static public data using our own code. No real services, traffic generation, university or third-party systems, or personal data are involved. Poisoning is a controlled label change, always compared against a clean baseline.

## References

1. B. McMahan et al., "Communication-efficient learning of deep networks from decentralized data," *Proc. AISTATS*, 2017.
2. V. Tolpegin, S. Truex, M. E. Gursoy, and L. Liu, "Data poisoning attacks against federated learning systems," *Proc. ESORICS*, 2020.
3. D. Yin, Y. Chen, K. Ramchandran, and P. Bartlett, "Byzantine-robust distributed learning," *Proc. ICML*, 2018.
4. E. C. P. Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, 2023.
5. T.-M. H. Hsu, H. Qi, and M. Brown, "Measuring the effects of non-identical data distribution for federated visual classification," arXiv:1909.06335, 2019.
6. D. J. Beutel et al., "Flower: A friendly federated learning framework," arXiv:2007.14390, 2020.
