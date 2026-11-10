# Robustness and extension results (for the final report)

Frozen copy of the outputs of `scripts/run_robustness.sh` and `scripts/run_extensions.sh` (all 63 files, validation set). The live results folders are ignored by git, so this copy keeps the numbers traceable.

| Folder | What | Produced by |
|---|---|---|
| `results/` | Everything that uses the main 10-client data: main conditions with 5 seeds (42-46); median, Krum and multi-Krum; typical-sized (`_atkmedian`) and random (`_atkrand0/1/3`) attackers; attacker sweep (`_m2`, `_m3`); 60 rounds (`_r60`); attack strength (`_flip25`, `_flip100`); targeted Recon->Benign attack (`_recon`) | both scripts |
| `results/analysis/` | Per-family and per-client paired error analysis, and `update_diagnostics.csv` (does the attacker's update look unusual?) | `src/error_analysis.py` |
| `ablate_iat/` | Same clients and attacker, IAT removed at model input | `configs/ablate_iat.yaml` |
| `balanced/` | Class-balanced loss | `configs/balanced_loss.yaml` |
| `k20/` | 20 clients, 2 attackers | `configs/clients20.yaml` |

Start with `attack_table.csv` in each folder: every attacked run against its own clean run, and against FedAvg under the same attack.

## Main findings

- Damage follows the number of poisoned rows, not attacker size: FedAvg's extra missed attacks are +3.4 points (client 5), +2.9 (typical-sized client 0), +0.7 to +1.3 (random clients 6, 2, 3).
- Attack strength (FedAvg): flipping 25 / 50 / 100% of the attacker's attack rows costs -0.6 / -1.1 / -4.7 macro-F1 points and adds +1.6 / +3.4 / +11.2 points of missed attacks. At 100% trimmed mean cuts the increase to +4.0 and multi-Krum to +1.9 (multi-Krum varies by about 2 F1 points across seeds).
- Targeted Recon->Benign attack: macro F1 unchanged (+0.01) while Recon rows predicted Benign rise from 13% to 25%; multi-Krum holds Recon at 16%, trimmed mean at 23%.
- The defenses detect the attacker (Krum rank about 8.9 of 9; 68% of its parameters in the trimmed tails vs. 37% for honest clients) but help clearly only against strong attacks.
- IAT ablation: no change (centralized .742 both ways; attack cost -1.05). Class-balanced loss: better rare-family recall, same attack effect (+3.3 points missed). 20 clients: same pattern; robust rules roughly halve the increase in missed attacks.
