"""Threshold selection.

In production you do *not* get to see test labels, so the realistic way to pick a
cutoff is from the score distribution on known-normal data: flag anything above a high
quantile. :func:`quantile_threshold` does that; the oracle ``best_f1_threshold`` in
``metrics`` is reported alongside only as a reference upper bound.
"""

from __future__ import annotations

import numpy as np


def quantile_threshold(normal_scores, q: float = 0.99) -> float:
    """Cutoff at the q-quantile of scores observed on normal operation."""
    return float(np.quantile(np.asarray(normal_scores), q))
