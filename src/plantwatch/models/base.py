"""Shared detector interface.

Every anomaly detector in PlantWatch follows the same contract: ``fit`` learns what
*normal* looks like, and ``score`` returns one number per sample where **higher means
more anomalous**. Keeping the sign consistent lets the evaluation and thresholding code
treat Isolation Forest, PCA, and the LSTM autoencoder identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AnomalyDetector(ABC):
    name: str = "detector"

    @abstractmethod
    def fit(self, X: Any) -> AnomalyDetector: ...

    @abstractmethod
    def score(self, X: Any) -> np.ndarray:
        """Return an anomaly score per row; higher = more anomalous."""
