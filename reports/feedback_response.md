# Response to Proposal Feedback (Group 8)

This file maps each point of the instructor's proposal feedback to the change made in the repository. Use it for the "Changes since the proposal" section of the Milestone 1 report.

## 1. K-means partition and Dirichlet alternative

**Feedback:** k-means assumes spherical, similar-size clusters and is sensitive to outliers. Report each cluster's size, benign/attack balance, attack-family distribution and distance from the global distribution. If clusters are very unequal or nearly label-pure, use a controlled Dirichlet split. Call the result simulated statistical heterogeneity.

**Change:**
- Added a Dirichlet label-skew partition (alpha = 0.5, Hsu et al. 2019) and made it the primary non-IID split (`partition.method: dirichlet`).
- k-means is still built and reported as an alternative.
- `prepare_data.py` reports, for every client of every partition:
  - size and share of the pool
  - benign/attack rows and benign fraction
  - family counts and fractions
  - label purity
  - total-variation and Jensen-Shannon distance from the global distribution
  - client-label mutual information
- It then runs PASS/FAIL checks: minimum share, size ratio, label purity, and at least 2 trainable families per client.
- Output: `data/processed/partitions/*.json` and `partition_table_k10.csv`.
- The README and report wording now use "simulated statistical heterogeneity".

## 2. Number of clients and malicious ratio

**Feedback:** use more than six clients, at least 10-20, and vary the ratio of benign to malicious clients.

**Change:**
- `partition.num_clients: 10`.
- `scripts/run_attacker_sweep.sh` runs 1, 2 and 3 attackers (10%, 20%, 30%) under FedAvg and trimmed mean.
- `trim_beta` is now 0.2, which trims 2 values per side at 10 clients.

## 3. Order of the data pipeline

**Feedback:** remove exact duplicate feature-label rows before splitting, and preserve capture/device/session grouping. Fit scaling and k-means on training records only. Assign held-out records to clients with the frozen scaler and centroids, without refitting.

**Change:**
- Exact (feature + label) duplicates and conflicting-label duplicates are removed before the split, with counts logged.
- The split is grouped by source CSV file. That is the only capture grouping the CSVs expose, since there is no device or session column. An assertion checks that no file appears in two splits.
- The scaler and k-means are fit on the training pool only.
- Validation and test rows are assigned to clients with the frozen scaler and centroids (k-means) or the frozen per-family client proportions (Dirichlet). These assignments are saved as `*_val.npy` and `*_test.npy`.

## 4. Global and client-level evaluation

**Feedback:** report global macro F1 and performance on held-out records for each client. Include support counts for every per-class precision, recall, FNR and FPR value.

**Change:**
- Every run stores global metrics plus per-client held-out metrics: `per_client_val`, and `per_client_test` with `--final`.
- Each per-class entry has `support`, `negatives`, `predicted`, precision, recall, FNR, one-vs-rest FPR, F1 and PR-AUC.
- The binary FPR/FNR come with their raw counts.
- `summarize_results.py` writes `per_client.csv` and `per_class.csv`.

## 5. Deterministic, reproducible attack

**Feedback:** state how the malicious client is chosen, whether it is the same across seeds, whether the 50% subset is fixed per seed or changes every round, and whether the attacker changes only labels. Verify the attacker has enough examples from several attack families.

**Change:**
- **Choice:** the client with the most attack rows in the frozen training partition. It is the same for every seed and every condition.
- **Poisoned subset:** 50% within each attack family, drawn once per seed from a dedicated RNG and reused in all rounds. A hash of the poisoned indices is stored in each result.
- **Behavior:** labels only; training and updates are honest.
- **Eligibility:** the run stops unless the attacker has at least 2 attack families with at least 100 rows each.
- The README has a "Threat model and attack protocol" section.

## 6. Trimmed-mean specification

**Feedback:** state the trimming parameter, how many high and low values are removed, and how the rest are averaged. Apply the defense to the clean non-IID condition. Weak recovery is a meaningful result.

**Change:**
- beta = 0.2 with n = 10 means k = floor(0.2 × 10) = 2 values removed from each end per coordinate, and an **unweighted** mean of the remaining 6.
- These values are recorded in each result's `aggregation` field.
- Condition 4a (clean trimmed mean) is kept.
- The expectation in the README now says that weak recovery would itself be a meaningful result.

## 7. Paired factors and seeds

**Feedback:** keep everything else paired across conditions. Show individual runs as well as mean and std. Define what the three seeds change, and use paired seeds.

**Change:**
- `data.seed` fixes records, splits and partitions for all runs.
- A run seed changes only model initialization, minibatch order and dropout, and the poisoned-record draw.
- Initial weights are identical across conditions for the same seed. This is verified by `init_weights_sha256` and by a test.
- `summarize_results.py` writes `runs.csv` (each run), `summary.csv` (mean, sample std, min, max), `paired_deltas.csv` (per-seed differences) and `hypothesis_check.csv`.

## 8. Pilot, label mapping, class counts and hyperparameters

**Feedback:** start with a smaller end-to-end pilot, publish and verify the 33-to-7 mapping, and report class counts before and after sampling. Don't describe a capped subset's prevalence as natural, and specify all hyperparameters.

**Change:**
- `configs/pilot.yaml` with `bash scripts/run_all.sh pilot` runs the whole pipeline on 12 files with 5 rounds, in separate folders.
- The mapping table is in the README and `label_mapping.csv`, and a test checks the family sizes against the paper's taxonomy.
- Raw label and family counts are recorded before sampling, alongside the subset counts and prevalence (`family_counts.csv`, `subset_summary.json`), with a written selection procedure that says the subset prevalence is not natural.
- All hyperparameters are in the README table and in `configs/default.yaml`.

## 9. Communication proxy and runtime

**Feedback:** report bytes sent and received per participating client per round and cumulative bytes per run, and state whether downloads are included. Identify the hardware and describe runtime as single-machine simulation time.

**Change:**
- Each result's `communication` block has download and upload bytes per client per round, the total per round, and the cumulative total per run, with downloads included (raw float32 tensors).
- Each round logs cumulative bytes.
- Each result stores a `hardware` block and is labeled "single-machine simulation time".

## Priority actions before Milestone 1

| Action from the feedback | Status |
|---|---|
| Test that clients are balanced, trainable and meaningfully heterogeneous; replace with a controlled partition if needed | Partition checks implemented; Dirichlet is primary. **Run on real data.** |
| Freeze the dedup, grouped split, scaling, clustering and client-assignment pipeline | Done (`pipeline_manifest.json`) |
| Publish exact client counts, class distributions, source files and the family mapping | Written automatically by `prepare_data.py` |
| Define the malicious client, fixed poisoned records, trimmed-mean parameter and paired-seed protocol | Done (README sections) |
| Small centralized and clean IID/non-IID FedAvg pilot before poisoning and defense runs | `bash scripts/run_all.sh pilot` |
| Report global, client-level and per-class outcomes; limit claims to the single-testbed simulation | Done in code and README |
