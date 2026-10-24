"""Collect results/<experiment>/seed_*.json into tables and one plot.

Writes (into the results directory):
  runs.csv                  one row per (experiment, seed): every individual run
  summary.csv               mean, std, min and max over seeds per experiment
  paired_deltas.csv         per-seed paired differences for the key comparisons
                            (same seed = same initial weights and poisoned records)
  per_client.csv            each client's held-out macro F1 / FNR per run
  per_class.csv             per-family precision / recall / FNR / FPR with support
  convergence_val_f1.png    validation macro F1 per round: each seed (thin) + mean

With 3 seeds, report the individual runs next to mean and std; three
observations do not support strong distributional claims.

Usage:
    python src/summarize_results.py
    python src/summarize_results.py --split test     # after the --final runs
    python src/summarize_results.py --config configs/pilot.yaml
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from common import FAMILIES, load_config, repo_path

# Fixed color per experiment, so a color always means the same condition
# across every figure in the report.
COLORS = {
    "fedavg_iid": "#2a78d6",
    "fedavg_noniid": "#eb6834",
    "fedavg_noniid_poisoned": "#1baf7a",
    "trimmed_noniid": "#eda100",
    "trimmed_noniid_poisoned": "#e87ba4",
}
EXTRA_COLORS = ["#008300", "#6250d6", "#e34948", "#7a5c2e", "#3a9fb0", "#b04e9c"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

# (baseline, treatment, name). Deltas are treatment - baseline, paired by seed.
PAIRS = [
    ("fedavg_noniid", "fedavg_noniid_poisoned", "attack_cost_fedavg"),
    ("fedavg_noniid", "trimmed_noniid_poisoned", "attack_cost_trimmed"),
    ("fedavg_noniid_poisoned", "trimmed_noniid_poisoned", "defense_gain_under_attack"),
    ("fedavg_noniid", "trimmed_noniid", "defense_cost_clean"),
    ("fedavg_iid", "fedavg_noniid", "heterogeneity_cost"),
]
METRICS = ("macro_f1", "binary_fpr", "binary_fnr", "accuracy", "macro_pr_auc")


def load_runs(results_dir):
    runs = {}
    for exp_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        seeds = [json.loads(f.read_text()) for f in sorted(exp_dir.glob("seed_*.json"))]
        if seeds:
            runs[exp_dir.name] = seeds
    return runs


def runs_table(runs, split):
    rows = []
    for exp, seeds in runs.items():
        for s in seeds:
            if split not in s:
                continue
            row = {"experiment": exp, "seed": s["seed"]}
            for m in METRICS:
                row[f"{split}_{m}"] = s[split].get(m)
            atk = s.get("attack", {})
            row["attackers"] = ",".join(map(str, atk.get("attackers", []))) or "-"
            row["malicious_fraction"] = atk.get("malicious_fraction", 0.0)
            row["client_macro_f1_min"] = s.get("client_macro_f1_min")
            row["client_macro_f1_std"] = s.get("client_macro_f1_std")
            comm = s.get("communication", {})
            row["bytes_per_round"] = comm.get("total_per_round")
            row["cumulative_bytes"] = comm.get("cumulative_per_run")
            row["runtime_sec"] = s["runtime_sec"]
            row["rss_mb"] = s.get("rss_mb")
            row["init_weights_sha256"] = s.get("init_weights_sha256", "")[:12]
            rows.append(row)
    return pd.DataFrame(rows)


def summary_table(df):
    num = [c for c in df.columns if c not in ("experiment", "seed", "attackers", "init_weights_sha256")]
    g = df.groupby("experiment")[num]
    out = g.agg(["mean", "std", "min", "max"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.insert(0, "n_seeds", df.groupby("experiment").size())
    # std is the sample standard deviation (ddof=1).
    return out.reset_index()


def paired_deltas(df, split):
    col = f"{split}_macro_f1"
    rows = []
    for base, treat, name in PAIRS:
        a = df[df.experiment == base].set_index("seed")[col]
        b = df[df.experiment == treat].set_index("seed")[col]
        for seed in sorted(set(a.index) & set(b.index)):
            rows.append({"comparison": name, "baseline": base, "treatment": treat,
                         "seed": seed, "delta_macro_f1_points": 100 * (b[seed] - a[seed])})
    return pd.DataFrame(rows)


def hypothesis_check(df, split):
    """H: FedAvg attack cost > 3 points AND trimmed mean recovers >= half of it."""
    col = f"{split}_macro_f1"
    need = {"fedavg_noniid", "fedavg_noniid_poisoned", "trimmed_noniid_poisoned"}
    if not need <= set(df.experiment):
        return None
    t = df.pivot_table(index="seed", columns="experiment", values=col)
    t = t.dropna(subset=list(need))
    if t.empty:
        return None
    loss = 100 * (t["fedavg_noniid"] - t["fedavg_noniid_poisoned"])
    recovered = 100 * (t["trimmed_noniid_poisoned"] - t["fedavg_noniid_poisoned"])
    frac = recovered / loss.where(loss != 0)
    return pd.DataFrame({"attack_loss_points": loss, "recovered_points": recovered,
                         "recovered_fraction": frac,
                         "loss_gt_3": loss > 3, "recovers_half": frac >= 0.5})


def per_client_table(runs, key):
    rows = []
    for exp, seeds in runs.items():
        for s in seeds:
            attackers = set(s.get("attack", {}).get("attackers", []))
            for p in s.get(key, []):
                rows.append({"experiment": exp, "seed": s["seed"], "client": p["client"],
                             "attacker": p["client"] in attackers, "n": p["n"],
                             "macro_f1": p["macro_f1"], "binary_fnr": p.get("binary_fnr"),
                             "binary_fpr": p.get("binary_fpr")})
    return pd.DataFrame(rows)


def per_class_table(runs, split):
    rows = []
    for exp, seeds in runs.items():
        for s in seeds:
            if split not in s:
                continue
            for fam in FAMILIES:
                pc = s[split]["per_class"][fam]
                rows.append({"experiment": exp, "seed": s["seed"], "family": fam, **pc})
    return pd.DataFrame(rows)


def plot_convergence(runs, path):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    extra = iter(EXTRA_COLORS)
    for exp, seeds in runs.items():
        if not seeds[0]["history"] or "round" not in seeds[0]["history"][0]:
            continue  # centralized run
        curves = np.array([[h["val_macro_f1"] for h in s["history"]] for s in seeds])
        rounds = np.arange(1, curves.shape[1] + 1)
        color = COLORS.get(exp) or next(extra, MUTED)
        for c in curves:
            ax.plot(rounds, c, color=color, linewidth=0.8, alpha=0.35)
        ax.plot(rounds, curves.mean(axis=0), color=color, linewidth=2, label=exp)

    ax.set_xlabel("Round", color=TEXT)
    ax.set_ylabel("Validation macro F1", color=TEXT)
    ax.set_title("Thin lines: individual seeds; thick: mean", color=MUTED, fontsize=9, loc="left")
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
    plt.close(fig)
    print(f"Saved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    results_dir = repo_path(cfg["paths"]["results_dir"])
    if not results_dir.exists() or not (runs := load_runs(results_dir)):
        print(f"No results in {results_dir}")
        return

    pd.set_option("display.width", 200)
    df = runs_table(runs, args.split)
    if df.empty:
        print(f"No runs contain '{args.split}' metrics (use --final for test).")
        return
    df.to_csv(results_dir / "runs.csv", index=False)
    summary = summary_table(df)
    summary.to_csv(results_dir / "summary.csv", index=False)
    print("\n== Individual runs ==")
    print(df[["experiment", "seed", f"{args.split}_macro_f1", f"{args.split}_binary_fnr",
              "attackers", "client_macro_f1_min"]].to_string(index=False, float_format="%.4f"))

    deltas = paired_deltas(df, args.split)
    if not deltas.empty:
        deltas.to_csv(results_dir / "paired_deltas.csv", index=False)
        print("\n== Paired deltas (macro F1 points, treatment - baseline) ==")
        print(deltas.groupby("comparison")["delta_macro_f1_points"]
              .agg(["count", "mean", "std", "min", "max"]).to_string(float_format="%.2f"))

    hyp = hypothesis_check(df, args.split)
    if hyp is not None:
        hyp.to_csv(results_dir / "hypothesis_check.csv")
        print("\n== Hypothesis check per seed ==")
        print(hyp.to_string(float_format="%.2f"))

    key = "per_client_val" if args.split == "val" else "per_client_test"
    pc = per_client_table(runs, key)
    if not pc.empty:
        pc.to_csv(results_dir / "per_client.csv", index=False)
    per_class_table(runs, args.split).to_csv(results_dir / "per_class.csv", index=False)
    plot_convergence(runs, results_dir / "convergence_val_f1.png")


if __name__ == "__main__":
    main()
