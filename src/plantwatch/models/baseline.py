"""Isolation Forest baseline on rolling-window features.

Fast, unsupervised, no GPU. This is the bar the deep model has to clear — a challenger
that cannot beat Isolation Forest is not worth its operational cost.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from plantwatch.models.base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    name = "isolation_forest"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        contamination: float | str = "auto",
        random_state: int = 42,
    ) -> None:
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X) -> IsolationForestDetector:
        self.model.fit(self.scaler.fit_transform(np.asarray(X)))
        return self

    def score(self, X) -> np.ndarray:
        # score_samples: higher = more normal. Negate so higher = more anomalous.
        return -self.model.score_samples(self.scaler.transform(np.asarray(X)))
