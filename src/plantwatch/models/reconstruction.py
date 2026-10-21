"""PCA reconstruction-error detector.

A reconstruction-based challenger with no deep-learning dependency: fit PCA on normal
windows, then score each window by how badly it reconstructs. Normal operation lives on
the learned low-dimensional manifold and reconstructs well; faults don't, so their error
is high. Same idea as the autoencoder, but it trains in milliseconds and always runs.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plantwatch.models.base import AnomalyDetector


class PCAReconstructionDetector(AnomalyDetector):
    name = "pca_reconstruction"

    def __init__(self, *, n_components: float | int = 0.95, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)

    def fit(self, X) -> PCAReconstructionDetector:
        Xs = self.scaler.fit_transform(np.asarray(X))
        self.pca.fit(Xs)
        return self

    def score(self, X) -> np.ndarray:
        Xs = self.scaler.transform(np.asarray(X))
        recon = self.pca.inverse_transform(self.pca.transform(Xs))
        return ((Xs - recon) ** 2).mean(axis=1)
