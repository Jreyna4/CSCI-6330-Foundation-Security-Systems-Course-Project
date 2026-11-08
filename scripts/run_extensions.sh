#!/usr/bin/env bash
# Extension runs for the final report (after scripts/run_robustness.sh).
# Reuses data/processed; every run has a new name or its own results folder,
# so nothing is overwritten. Seeds 42-44, paired with the existing clean runs.
#
#   A. Attack strength: flip 25% and 100% of the attacker's attack rows.
#   B. Targeted attack: flip only Recon -> Benign (100%), from the client
#      holding the most Recon rows.
#   C. Better model: class-balanced loss (results_balanced/).
#   D. 20 clients with 2 attackers (results_k20/).
#
# Usage:  caffeinate -i bash scripts/run_extensions.sh      (about 50 minutes)
set -euo pipefail
cd "$(dirname "$0")/.."
FL="python src/run_federated.py"
SEEDS="--seeds 42 43 44"

echo "== A. attack strength (25% and 100%) =="
for rule in fedavg trimmed multikrum; do
  for p in 25 100; do
    frac=$( [ "$p" = 100 ] && echo 1.0 || echo 0.25 )
    $FL --config "configs/${rule}_noniid_poisoned.yaml" $SEEDS \
        --set attack.flip_fraction=$frac "experiment=${rule}_noniid_poisoned_flip$p"
  done
done

echo "== B. targeted attack: Recon -> Benign =="
for rule in fedavg mean trimmed multikrum; do
  $FL --config "configs/${rule}_noniid_poisoned.yaml" $SEEDS \
      --set attack.selection=most_source_rows "attack.source_families=[Recon]" \
            attack.flip_fraction=1.0 "experiment=${rule}_noniid_poisoned_recon"
done

echo "== C. class-balanced loss =="
python src/train_centralized.py --config configs/balanced_loss.yaml configs/centralized.yaml $SEEDS
for c in fedavg_noniid fedavg_noniid_poisoned trimmed_noniid trimmed_noniid_poisoned \
         multikrum_noniid multikrum_noniid_poisoned; do
  $FL --config configs/balanced_loss.yaml "configs/$c.yaml" $SEEDS
done

echo "== D. 20 clients, 2 attackers =="
python src/prepare_data.py --partitions-only --config configs/clients20.yaml
for c in fedavg_noniid fedavg_noniid_poisoned mean_noniid mean_noniid_poisoned \
         trimmed_noniid trimmed_noniid_poisoned multikrum_noniid multikrum_noniid_poisoned; do
  $FL --config configs/clients20.yaml "configs/$c.yaml" $SEEDS
done

echo "== summaries =="
python src/summarize_results.py
python src/error_analysis.py
for c in balanced_loss clients20; do
  python src/summarize_results.py --config "configs/$c.yaml"
  python src/error_analysis.py --config "configs/$c.yaml"
done
