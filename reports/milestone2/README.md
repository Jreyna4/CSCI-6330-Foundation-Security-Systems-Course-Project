# Milestone 2 (Progress Presentation 2)

- `milestone2_report.pdf` / `.tex`: the 3-page report (IEEE journal template). Figures: `families_benign.png`, `clients_fnr.png`.
- `results/`: frozen copy of the run outputs the report is based on (all 63 files, seeds 42-44, validation set). The live `results/` folder at the repo root is ignored by git, so this copy keeps the report's numbers traceable.
  - `runs.csv`, `summary.csv`, `paired_deltas.csv`, `hypothesis_check.csv`, `per_class.csv`, `per_client.csv`: from `src/summarize_results.py`. Includes the main 8 conditions, the attacker sweep (`*_m2`, `*_m3`) and the 60-round check (`*_r60`).
  - `analysis/`: per-family and per-client paired error analysis from `src/error_analysis.py`.
  - `no_iat/`: the sensitivity run without the IAT feature (`configs/no_iat.yaml`).
- `data/`: data-prep summaries (no raw data): split per file and family, feature schema (46 official -> 39 used), family counts, split diagnostics, client partition table.

## How these results were produced

```bash
bash scripts/run_all.sh                       # main paired comparison + error analysis
bash scripts/run_attacker_sweep.sh            # 2 and 3 attackers (stress test)
python src/error_analysis.py

# 60 rounds instead of 30
for c in fedavg_noniid fedavg_noniid_poisoned trimmed_noniid trimmed_noniid_poisoned; do
  python src/run_federated.py --config configs/$c.yaml --set federated.rounds=60 experiment=${c}_r60
done
python src/summarize_results.py

# Without IAT (separate data and results folders)
python src/prepare_data.py --config configs/no_iat.yaml
python src/train_centralized.py --config configs/no_iat.yaml configs/centralized.yaml
python src/run_federated.py --config configs/no_iat.yaml configs/fedavg_noniid.yaml
python src/run_federated.py --config configs/no_iat.yaml configs/fedavg_noniid_poisoned.yaml
python src/summarize_results.py --config configs/no_iat.yaml
```

Hardware: Apple MacBook Pro, CPU only, Python 3.14, PyTorch 2.14.1. About 33 s per 30-round federated run.
