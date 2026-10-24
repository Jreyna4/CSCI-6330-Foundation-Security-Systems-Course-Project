#!/usr/bin/env bash
# Vary the ratio of malicious to benign clients (instructor feedback):
# m = 1, 2, 3 attackers out of 10 under FedAvg and trimmed mean. m = 1 is the
# main condition; trim_beta = 0.2 trims 2 per side, so m = 3 is past its limit.
#   bash scripts/run_attacker_sweep.sh          # real runs
#   bash scripts/run_attacker_sweep.sh pilot    # pilot folders
set -euo pipefail
cd "$(dirname "$0")/.."
PRE=()
if [[ "${1:-}" == "pilot" ]]; then PRE=(configs/pilot.yaml); fi

for m in 2 3; do
  python src/run_federated.py --config ${PRE[@]+"${PRE[@]}"} configs/fedavg_noniid_poisoned.yaml \
    --set attack.num_attackers=$m experiment=fedavg_noniid_poisoned_m$m
  python src/run_federated.py --config ${PRE[@]+"${PRE[@]}"} configs/trimmed_noniid_poisoned.yaml \
    --set attack.num_attackers=$m experiment=trimmed_noniid_poisoned_m$m
done
python src/summarize_results.py --config ${PRE[@]+"${PRE[@]}"} configs/default.yaml
