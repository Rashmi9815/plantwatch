import numpy as np

from plantwatch.evaluation.metrics import (
    best_f1_threshold,
    classification_at_threshold,
    pr_auc,
)
from plantwatch.evaluation.thresholds import quantile_threshold


def test_classification_perfect_separation():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.9, 0.8]
    m = classification_at_threshold(y, scores, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_pr_auc_perfect_and_degenerate():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]) == 1.0
    assert np.isnan(pr_auc([0, 0, 0], [0.1, 0.2, 0.3]))  # no positives


def test_best_f1_threshold_recovers_separator():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.zeros(50), np.ones(50)])
    t = best_f1_threshold(y, scores)
    assert classification_at_threshold(y, scores, t)["f1"] == 1.0


def test_quantile_threshold():
    normal = np.arange(100)
    assert quantile_threshold(normal, q=0.99) >= 98
