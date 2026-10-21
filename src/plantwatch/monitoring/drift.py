"""Data-drift detection via the Population Stability Index (PSI).

PSI compares a feature's distribution now against a reference (training) distribution.
Rule of thumb: < 0.1 no shift, 0.1-0.2 moderate, > 0.2 significant. When features drift,
the model is scoring inputs it never learned — the trigger for retraining in Phase 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODERATE = 0.1
SIGNIFICANT = 0.2


def population_stability_index(expected, actual, *, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant reference; nothing meaningful to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf  # catch values outside the reference range

    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)

    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    bins: int = 10,
    threshold: float = SIGNIFICANT,
) -> dict[str, dict]:
    """Per-column PSI and a drifted flag. Columns are the engineered features."""
    report = {}
    for col in reference.columns:
        psi = population_stability_index(reference[col], current[col], bins=bins)
        report[col] = {"psi": psi, "drifted": psi > threshold}
    return report
