"""Evaluation metrics for anomaly detection.

Accuracy is meaningless when anomalies are ~2% of the data, so we report
precision / recall / F1 at a chosen threshold plus **PR-AUC** (average precision),
which summarizes performance across all thresholds on the rare-positive class.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def classification_at_threshold(y_true, scores, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def pr_auc(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")  # undefined without both classes
    return float(average_precision_score(y_true, np.asarray(scores)))


def best_f1_threshold(y_true, scores, *, n: int = 200) -> float:
    """Oracle threshold: the one maximizing F1 on this set (a reference upper bound)."""
    scores = np.asarray(scores)
    grid = np.unique(np.quantile(scores, np.linspace(0, 1, n)))
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        f1 = classification_at_threshold(y_true, scores, t)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t
