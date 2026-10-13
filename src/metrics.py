"""Evaluation metrics.

Primary: macro F1 over the 8 families. Also per-class precision / recall /
FNR, the binary benign-vs-attack FPR and FNR the spec asks for, one-vs-rest
PR-AUC, accuracy and the confusion matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from labels import BENIGN_ID, FAMILIES


def compute_metrics(y_true, proba, class_names=FAMILIES, benign_id=BENIGN_ID):
    labels = np.arange(len(class_names))
    y_pred = proba.argmax(axis=1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = support > 0  # classes with no true samples are left out of macro averages

    # Binary view: any attack family counts as "attack".
    true_attack = y_true != benign_id
    pred_attack = y_pred != benign_id
    fpr = (pred_attack & ~true_attack).sum() / max((~true_attack).sum(), 1)
    fnr = (~pred_attack & true_attack).sum() / max(true_attack.sum(), 1)

    pr_auc = {
        class_names[c]: float(average_precision_score(y_true == c, proba[:, c]))
        for c in labels[present]
    }

    return {
        "macro_f1": float(f1[present].mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_fpr": float(fpr),
        "binary_fnr": float(fnr),
        "macro_pr_auc": float(np.mean(list(pr_auc.values()))),
        "per_class": {
            name: {
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "fnr": float(1 - rec[c]) if present[c] else None,
                "f1": float(f1[c]),
                "pr_auc": pr_auc.get(name),
                "support": int(support[c]),
            }
            for c, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
