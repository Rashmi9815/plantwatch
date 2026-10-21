"""Turn a cleaned sensor frame into model-ready windows.

Produces both representations the detectors need, perfectly aligned to the same set of
window-end timestamps:

* ``X_feat`` — rolling summary statistics, one row per window (for Isolation Forest / PCA).
* ``X_seq`` — the raw window as a sequence ``(window, n_sensors)`` (for the LSTM autoencoder).
* ``y`` — a window label: 1 if *any* reading inside the window is anomalous.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plantwatch.features.windowing import sliding_window_features


@dataclass
class WindowedDataset:
    X_feat: pd.DataFrame  # (n_windows, n_sensors * n_stats)
    X_seq: np.ndarray  # (n_windows, window, n_sensors) float32
    y: np.ndarray  # (n_windows,) int
    index: pd.Index  # window-end timestamps

    def __len__(self) -> int:
        return len(self.index)


def build_windows(
    df: pd.DataFrame,
    sensors: Sequence[str],
    *,
    window: int = 30,
    stats: Sequence[str] = ("mean", "std", "min", "max"),
    label_col: str = "anomaly",
) -> WindowedDataset:
    sensors = list(sensors)
    if len(df) < window:
        raise ValueError(f"need at least {window} rows, got {len(df)}")

    feats = sliding_window_features(df, sensors, window=window, stats=stats)
    index = feats.index
    n = len(index)

    raw = df[sensors].to_numpy(dtype=np.float32)
    seq = np.lib.stride_tricks.sliding_window_view(raw, window, axis=0)  # (n, n_sensors, window)
    seq = np.transpose(seq, (0, 2, 1))[:n]  # -> (n, window, n_sensors)

    if label_col in df.columns:
        labels = df[label_col].to_numpy()
        windowed = np.lib.stride_tricks.sliding_window_view(labels, window)[:n]
        y = (windowed.max(axis=1) > 0).astype(int)
    else:
        y = np.zeros(n, dtype=int)

    return WindowedDataset(X_feat=feats, X_seq=seq, y=y, index=index)


def time_split(
    df: pd.DataFrame, *, train: float = 0.6, val: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split — never shuffle time series."""
    df = df.sort_index()
    n = len(df)
    i1, i2 = int(n * train), int(n * (train + val))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
