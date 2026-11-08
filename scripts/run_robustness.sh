#!/usr/bin/env bash
# Robustness runs for the final report (after scripts/run_all.sh). They answer
# the points we expect in the Milestone 2 feedback. Reuses the prepared data
# in data/processed; nothing is overwritten (new experiment names or folders).
#
#   1. Two more seeds (45, 46) for every main condition -> 5 paired seeds.
#   2. More defenses: median, Krum, multi-Krum (clean and attacked, 5 seeds).
#   3. Attacker size vs. poisoning: the same attack from a typical-sized client
#      (selection: median_size) and from randomly drawn clients (random draws
#      0, 1, 3), under FedAvg, unweighted mean, trimmed mean and multi-Krum.
#   4. IAT ablation at model-input time (same rows, clients and attacker).
#   Then the summary tables and the error analysis (incl. update diagnostics).
#
# Usage:  caffeinate -i bash scripts/run_robustness.sh      (about 75 minutes)
set -euo pipefail
cd "$(dirname "$0")/.."

FL="python src/run_federated.py"

echo "== 1. seeds 45 and 46 for the main conditions =="
python src/train_centralized.py --config configs/centralized.yaml --seeds 45 46
for c in fedavg_iid fedavg_noniid mean_noniid trimmed_noniid \
         fedavg_noniid_poisoned mean_noniid_poisoned trimmed_noniid_poisoned; do
  $FL --config "configs/$c.yaml" --seeds 45 46
done

echo "== 2. more defenses: median, Krum, multi-Krum =="
for c in median_noniid median_noniid_poisoned krum_noniid krum_noniid_poisoned \
         multikrum_noniid multikrum_noniid_poisoned; do
  $FL --config "configs/$c.yaml" --seeds 42 43 44 45 46
done

echo "== 3. attacker size vs. poisoning (3 seeds each) =="
for rule in fedavg mean trimmed multikrum; do
  $FL --config "configs/${rule}_noniid_poisoned.yaml" --seeds 42 43 44 \
      --set attack.selection=median_size "experiment=${rule}_noniid_poisoned_atkmedian"
  for d in 0 1 3; do
    $FL --config "configs/${rule}_noniid_poisoned.yaml" --seeds 42 43 44 \
        --set attack.selection=random attack.random_draw=$d "experiment=${rule}_noniid_poisoned_atkrand$d"
  done
done

echo "== 4. IAT ablation (same rows, clients and attacker) =="
python src/train_centralized.py --config configs/ablate_iat.yaml configs/centralized.yaml --seeds 42 43 44 45 46
for c in fedavg_noniid fedavg_noniid_poisoned; do
  $FL --config configs/ablate_iat.yaml "configs/$c.yaml" --seeds 42 43 44 45 46
done

echo "== summaries =="
python src/summarize_results.py
python src/error_analysis.py
python src/summarize_results.py --config configs/ablate_iat.yaml
