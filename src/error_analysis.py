"""Per-family and per-client paired error analysis (Milestone 2 priority).

Reads results/<experiment>/seed_*.json and the frozen Dirichlet partition, and
answers: which attack families does the label-flipping client hurt, which
clients does it hurt, and how much of that does each aggregation rule undo?
Every comparison is paired by seed (same initial weights, same poisoned rows),
so deltas are treatment - baseline within a seed, then summarized over seeds.

Writes into <results_dir>/analysis/:
  per_family.csv           one row per (experiment, seed, family): support,
                           recall, FNR, precision, F1, PR-AUC, and
                           missed_as_benign = share of that family's rows the
                           model labels Benign (the attacker's target)
  per_family_paired.csv    per comparison and family: paired deltas of recall,
                           F1 and missed_as_benign (mean, std, min, max, n)
  per_client.csv           one row per (experiment, seed, client): held-out
                           size, macro F1, binary FNR/FPR, macro PR-AUC, the
                           client's training mix and whether it is an attacker
  per_client_paired.csv    per comparison and client: paired deltas
  family_missed_as_benign.png   attack families x key conditions (black/white)
  client_attack_cost.png        per-client paired macro-F1 cost of the attack

Usage:
    python src/error_analysis.py
    python src/error_analysis.py --split test          # after --final runs
    python src/error_analysis.py --config configs/pilot.yaml
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BENIGN_ID, FAMILIES, load_config, processed_dir, repo_path
from summarize_results import load_runs

ATTACK_FAMILIES = [f for f in FAMILIES if f != "Benign"]

# (baseline, treatment, name); deltas = treatment - baseline, paired by seed.
PAIRS = [
    ("fedavg_noniid", "fedavg_noniid_poisoned", "attack_cost_fedavg"),
    ("mean_noniid", "mean_noniid_poisoned", "attack_cost_mean"),
    ("trimmed_noniid", "trimmed_noniid_poisoned", "attack_cost_trimmed"),
    ("fedavg_noniid_poisoned", "mean_noniid_poisoned", "unweighting_gain_under_attack"),
    ("mean_noniid_poisoned", "trimmed_noniid_poisoned", "trimming_gain_under_attack"),
    ("fedavg_noniid_poisoned", "trimmed_noniid_poisoned", "defense_gain_under_attack"),
    ("fedavg_iid", "fedavg_noniid", "heterogeneity_cost"),
]
# Black-and-white styles for figures, one per condition.
STYLE = {
    "fedavg_noniid": dict(color="white", edgecolor="black", hatch=""),
    "fedavg_noniid_poisoned": dict(color="black", edgecolor="black", hatch=""),
    "mean_noniid_poisoned": dict(color="#8c8c8c", edgecolor="black", hatch=""),
    "trimmed_noniid_poisoned": dict(color="white", edgecolor="black", hatch="////"),
}
LABEL = {
    "fedavg_noniid": "FedAvg, clean",
    "fedavg_noniid_poisoned": "FedAvg, attacked",
    "mean_noniid_poisoned": "Unweighted mean, attacked",
    "trimmed_noniid_poisoned": "Trimmed mean, attacked",
}


def missed_as_benign(cm, fam_id):
    row = np.asarray(cm)[fam_id]
    return float(row[BENIGN_ID] / row.sum()) if row.sum() else None


def per_family_table(runs, split):
    rows = []
    for exp, seeds in runs.items():
        for s in seeds:
            if split not in s:
                continue
            res = s[split]
            for i, fam in enumerate(FAMILIES):
                pc = res["per_class"][fam]
                rows.append({
                    "experiment": exp, "seed": s["seed"], "family": fam,
                    "support": pc["support"], "recall": pc["recall"], "fnr": pc["fnr"],
                    "precision": pc["precision"], "f1": pc["f1"], "pr_auc": pc["pr_auc"],
                    "fpr": pc["fpr"],
                    "missed_as_benign": missed_as_benign(res["confusion_matrix"], i)
                    if fam != "Benign" else None,
                })
    return pd.DataFrame(rows)


def client_mix(cfg):
    """Training rows per family for each client of the frozen Dirichlet partition."""
    k = cfg["partition"]["num_clients"]
    path = processed_dir(cfg) / "partitions" / f"dirichlet_k{k}.json"
    if not path.exists():
        return {}
    rep = json.loads(path.read_text())
    return {int(c): v for c, v in rep["clients"].items()} if isinstance(rep["clients"], dict) \
        else {i: v for i, v in enumerate(rep["clients"])}


def per_client_table(runs, split, mix):
    key = "per_client_val" if split == "val" else "per_client_test"
    rows = []
    for exp, seeds in runs.items():
        for s in seeds:
            attackers = set(s.get("attack", {}).get("attackers", []))
            for p in s.get(key, []):
                c = p["client"]
                m = mix.get(c, {})
                fc = m.get("family_counts", {})
                train_rows = sum(fc.values()) if fc else None
                rows.append({
                    "experiment": exp, "seed": s["seed"], "client": c,
                    "attacker": c in attackers, "heldout_n": p["n"],
                    "macro_f1": p["macro_f1"], "f1_n_classes": len(p.get("f1_classes") or []),
                    "binary_fnr": p.get("binary_fnr"), "binary_fpr": p.get("binary_fpr"),
                    "macro_pr_auc": p.get("macro_pr_auc"),
                    "train_rows": train_rows,
                    "train_benign_share": (fc.get("Benign", 0) / train_rows) if train_rows else None,
                    **{f"train_{f}": fc.get(f) for f in FAMILIES},
                })
    return pd.DataFrame(rows)


def paired(df, keys, metrics, scale=100.0):
    """Paired deltas (treatment - baseline) per seed, summarized over seeds."""
    out = []
    for base, treat, name in PAIRS:
        a = df[df.experiment == base].set_index(["seed", *keys])
        b = df[df.experiment == treat].set_index(["seed", *keys])
        common = a.index.intersection(b.index)
        if common.empty:
            continue
        for metric in metrics:
            d = (b.loc[common, metric].astype(float) - a.loc[common, metric].astype(float)) * scale
            g = d.groupby(level=list(range(1, len(keys) + 1)))
            summ = g.agg(["count", "mean", "std", "min", "max"]).reset_index()
            summ.insert(0, "metric", metric)
            summ.insert(0, "comparison", name)
            summ["per_seed"] = [
                ";".join(f"{v:.2f}" for v in d.xs(k if isinstance(k, tuple) else (k,),
                                                     level=list(range(1, len(keys) + 1))).values)
                for k in (summ[keys].itertuples(index=False, name=None))
            ]
            out.append(summ)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def bw_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2)
    ax.grid(axis="y", lw=0.4, color="#cccccc")
    ax.set_axisbelow(True)


def plot_family(fam, path):
    conds = [c for c in STYLE if c in set(fam.experiment)]
    if len(conds) < 2:
        return
    d = fam[fam.family.isin(ATTACK_FAMILIES) & fam.experiment.isin(conds)]
    g = d.groupby(["experiment", "family"])["missed_as_benign"].agg(["mean", "min", "max"])
    x = np.arange(len(ATTACK_FAMILIES))
    w = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(7.0, 2.4), dpi=300)
    for j, c in enumerate(conds):
        m = g.loc[c].reindex(ATTACK_FAMILIES)
        err = np.vstack([m["mean"] - m["min"], m["max"] - m["mean"]]) * 100
        ax.bar(x + (j - (len(conds) - 1) / 2) * w, m["mean"] * 100, w, yerr=err,
               error_kw=dict(lw=0.6, capsize=1.5), lw=0.5, label=LABEL[c], **STYLE[c])
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_FAMILIES)
    ax.set_ylabel("Rows predicted Benign (%)")
    ax.legend(frameon=False, ncol=len(conds), fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    bw_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_clients(cp, pc, path):
    names = [n for n in ("attack_cost_fedavg", "attack_cost_mean", "attack_cost_trimmed")
             if n in set(cp.comparison)]
    if not names:
        return
    d = cp[(cp.metric == "macro_f1") & cp.comparison.isin(names)]
    k = int(d.client.max()) + 1
    attackers = set(pc.loc[pc.attacker, "client"])
    styles = [dict(color="black"), dict(color="#8c8c8c"), dict(color="white", hatch="////")]
    labels = {"attack_cost_fedavg": "FedAvg", "attack_cost_mean": "Unweighted mean",
              "attack_cost_trimmed": "Trimmed mean"}
    x = np.arange(k)
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.0, 2.3), dpi=300)
    for j, n in enumerate(names):
        m = d[d.comparison == n].set_index("client").reindex(range(k))
        err = np.vstack([m["mean"] - m["min"], m["max"] - m["mean"]])
        ax.bar(x + (j - (len(names) - 1) / 2) * w, m["mean"], w, yerr=err, edgecolor="black",
               lw=0.5, error_kw=dict(lw=0.6, capsize=1.5), label=labels[n], **styles[j])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}*" if c in attackers else str(c) for c in x])
    ax.set_xlabel("Client (* = attacker)")
    ax.set_ylabel("Macro F1 change\nunder attack (points)")
    ax.legend(frameon=False, ncol=len(names), fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    bw_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    results_dir = repo_path(cfg["paths"]["results_dir"])
    runs = load_runs(results_dir) if results_dir.exists() else {}
    runs = {k: v for k, v in runs.items() if k != "analysis"}
    if not runs:
        print(f"No results in {results_dir}")
        return
    out = results_dir / "analysis"
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    fam = per_family_table(runs, args.split)
    fam.to_csv(out / "per_family.csv", index=False)
    fp = paired(fam, ["family"], ["recall", "f1", "missed_as_benign"])
    if not fp.empty:
        fp.to_csv(out / "per_family_paired.csv", index=False)
        show = fp[(fp.metric == "missed_as_benign") & fp.family.isin(ATTACK_FAMILIES)]
        print("\n== Change in attack rows predicted Benign (percentage points, paired) ==")
        print(show.pivot_table(index="family", columns="comparison", values="mean")
              .reindex(ATTACK_FAMILIES).to_string(float_format="%.1f"))

    pc = per_client_table(runs, args.split, client_mix(cfg))
    pc.to_csv(out / "per_client.csv", index=False)
    cp = paired(pc, ["client"], ["macro_f1", "binary_fnr"])
    if not cp.empty:
        cp.to_csv(out / "per_client_paired.csv", index=False)
        show = cp[cp.metric == "macro_f1"]
        print("\n== Per-client change in held-out macro F1 (points, paired) ==")
        print(show.pivot_table(index="client", columns="comparison", values="mean")
              .to_string(float_format="%.1f"))

    plot_family(fam, out / "family_missed_as_benign.png")
    if not cp.empty:
        plot_clients(cp, pc, out / "client_attack_cost.png")


if __name__ == "__main__":
    main()
