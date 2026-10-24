#!/usr/bin/env bash
# Run the full pipeline. Usage:
#   bash scripts/run_all.sh            # real runs (all CSVs, 3 seeds)
#   bash scripts/run_all.sh pilot      # small end-to-end pilot first
#   FINAL=1 bash scripts/run_all.sh    # also evaluate on the test set (once, at the end)
set -euo pipefail
cd "$(dirname "$0")/.."

PRE=()
if [[ "${1:-}" == "pilot" ]]; then PRE=(configs/pilot.yaml); fi
EXTRA=()
if [[ "${FINAL:-0}" == "1" ]]; then EXTRA=(--final); fi

python src/prepare_data.py --config ${PRE[@]+"${PRE[@]}"} configs/default.yaml
python src/train_centralized.py --config ${PRE[@]+"${PRE[@]}"} configs/centralized.yaml ${EXTRA[@]+"${EXTRA[@]}"}
for c in fedavg_iid fedavg_noniid fedavg_noniid_poisoned trimmed_noniid trimmed_noniid_poisoned; do
  python src/run_federated.py --config ${PRE[@]+"${PRE[@]}"} "configs/$c.yaml" ${EXTRA[@]+"${EXTRA[@]}"}
done
python src/summarize_results.py --config ${PRE[@]+"${PRE[@]}"} configs/default.yaml
