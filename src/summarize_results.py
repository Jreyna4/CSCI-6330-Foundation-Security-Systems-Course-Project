"""Collect results/<experiment>/seed_*.json into one table and one plot.

Writes:
  results/summary.csv               mean and std over seeds for each experiment
  results/convergence_val_f1.png    validation macro F1 per round (federated runs)

Usage:
    python src/summarize_results.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd

from common import load_config, repo_path

# Fixed color per experiment, so a color always means the same condition
# across every figure in the report.
COLORS = {
    "fedavg_iid": "#2a78d6",
    "fedavg_noniid": "#eb6834",
    "fedavg_noniid_poisoned": "#1baf7a",
    "trimmed_noniid": "#eda100",
    "trimmed_noniid_poisoned": "#e87ba4",
    # Add new experiments here in this order: "#008300", "#6250d6", "#e34948"
}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def load_runs(results_dir):
    runs = {}
    for exp_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        seeds = [json.loads(f.read_text()) for f in sorted(exp_dir.glob("seed_*.json"))]
        if seeds:
            runs[exp_dir.name] = seeds
    return runs


def summary_table(runs):
    rows = []
    for exp, seeds in runs.items():
        row = {"experiment": exp, "n_seeds": len(seeds)}
        for split in ("val", "test"):
            if all(split in s for s in seeds):
                for metric in ("macro_f1", "binary_fpr", "binary_fnr", "accuracy"):
                    vals = [s[split][metric] for s in seeds]
                    row[f"{split}_{metric}_mean"] = np.mean(vals)
                    row[f"{split}_{metric}_std"] = np.std(vals)
        row["runtime_sec_mean"] = np.mean([s["runtime_sec"] for s in seeds])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_convergence(runs, path):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for exp, seeds in runs.items():
        if "round" not in seeds[0]["history"][0]:
            continue  # centralized run
        curves = np.array([[h["val_macro_f1"] for h in s["history"]] for s in seeds])
        rounds = np.arange(1, curves.shape[1] + 1)
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        color = COLORS.get(exp, MUTED)
        ax.plot(rounds, mean, color=color, linewidth=2, label=exp)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Round", color=TEXT)
    ax.set_ylabel("Validation macro F1", color=TEXT)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, labelcolor=TEXT, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def main():
    cfg = load_config()
    results_dir = repo_path(cfg["paths"]["results_dir"])
    runs = load_runs(results_dir)
    if not runs:
        print(f"No results in {results_dir}")
        return

    df = summary_table(runs)
    df.to_csv(results_dir / "summary.csv", index=False)
    print(df.to_string(index=False, float_format="%.4f"))
    plot_convergence(runs, results_dir / "convergence_val_f1.png")


if __name__ == "__main__":
    main()
