"""Cleaning and sliding-window feature engineering for sensor time series.

Two steps that every downstream model depends on:

* :func:`clean` — de-duplicate timestamps, sort, and fill gaps by interpolation, so a
  dropped reading does not become a phantom anomaly.
* :func:`sliding_window_features` — turn raw readings into rolling statistics
  (mean / std / min / max) over a fixed window. Reconstruction- and distance-based
  anomaly detectors learn on these, not on single noisy samples.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def clean(
    df: pd.DataFrame,
    sensors: Sequence[str],
    *,
    resample: str | None = None,
) -> pd.DataFrame:
    """Sort, drop duplicate timestamps, optionally resample, and interpolate gaps."""
    out = df[~df.index.duplicated(keep="last")].sort_index().copy()
    if resample is not None:
        out = out.resample(resample).mean()
    method = "time" if isinstance(out.index, pd.DatetimeIndex) else "linear"
    out[list(sensors)] = out[list(sensors)].interpolate(method=method, limit_direction="both")
    return out


def sliding_window_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    *,
    window: int = 10,
    stats: Sequence[str] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Rolling-window statistics per sensor, aligned to each window's right edge.

    The first ``window - 1`` rows have an incomplete window and are dropped, so every
    returned row is backed by a full window of data.
    """
    roll = df[list(sensors)].rolling(window=window, min_periods=window)
    frames = []
    for stat in stats:
        agg = getattr(roll, stat)()
        agg.columns = [f"{col}_{stat}" for col in sensors]
        frames.append(agg)
    return pd.concat(frames, axis=1).dropna()
