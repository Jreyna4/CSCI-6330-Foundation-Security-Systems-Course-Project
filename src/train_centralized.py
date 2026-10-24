"""Condition 1: centralized reference model trained on the pooled training data.

Same MLP, optimizer, learning rate, batch size and preprocessing as the
federated runs. Evaluates on the validation set after every epoch; the test
set is only touched with --final, once all settings are frozen. For
comparison with the federated runs, the final model is also evaluated on
each client's held-out records under the configured non-IID partition.

Usage:
    python src/train_centralized.py
    python src/train_centralized.py --seeds 42                     # one quick run
    python src/train_centralized.py --config configs/pilot.yaml configs/centralized.yaml
    python src/train_centralized.py --final                         # also report test metrics
"""

import argparse
import time

from common import (
    FAMILIES,
    build_model,
    build_optimizer,
    compute_metrics,
    count_parameters,
    get_device,
    hardware_info,
    load_all,
    load_config,
    load_manifest,
    load_partition,
    make_loader,
    predict_proba,
    repo_path,
    rss_mb,
    save_json,
    set_seed,
    train_epochs,
)
from run_federated import heldout_client_metrics


def run(cfg, data, seed, final, device):
    set_seed(seed)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    t = cfg["train"]

    model = build_model(cfg, X_train.shape[1], len(FAMILIES)).to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    loader = make_loader(X_train, y_train, t["batch_size"])

    start = time.perf_counter()
    history = []
    for epoch in range(1, t["epochs"] + 1):
        loss = train_epochs(model, loader, optimizer, 1, device)
        val = compute_metrics(y_val, predict_proba(model, X_val, device))
        history.append({"epoch": epoch, "train_loss": loss,
                        "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]})
        print(f"seed {seed} | epoch {epoch:3d} | loss {loss:.4f} | val macro F1 {val['macro_f1']:.4f}")
    runtime = time.perf_counter() - start

    result = {
        "experiment": cfg["experiment"],
        "seed": seed,
        "config": cfg,
        "data_manifest_files": (load_manifest(cfg) or {}).get("files"),
        "model_parameters": count_parameters(model),
        "model_selection": "model after the last epoch",
        "history": history,
        "val": val,
        "runtime_sec": runtime,
        "runtime_note": "single-machine time",
        "hardware": hardware_info(),
        "rss_mb": rss_mb(),
    }
    k = cfg["partition"]["num_clients"]
    try:
        result["per_client_val"] = heldout_client_metrics(
            model, X_val, y_val, load_partition(cfg, "val"), k, device)
        result["per_client_partition"] = cfg["partition"]["method"]
    except FileNotFoundError:
        pass
    if final:
        X_test, y_test = data["test"]
        result["test"] = compute_metrics(y_test, predict_proba(model, X_test, device))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=["configs/centralized.yaml"])
    parser.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--final", action="store_true", help="also evaluate on the test set")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = get_device()

    data = load_all(cfg)
    out_dir = repo_path(cfg["paths"]["results_dir"]) / cfg["experiment"]
    for seed in args.seeds or cfg["seeds"]:
        result = run(cfg, data, seed, args.final, device)
        save_json(result, out_dir / f"seed_{seed}.json")


if __name__ == "__main__":
    main()
