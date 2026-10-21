"""Scoring service — the logic the Lambda wraps.

A trained detector is saved as a :class:`ModelBundle` (the fitted model plus everything
needed to reproduce its inputs: the threshold, window length, sensor order, and feature
stats). :func:`score_payload` is a pure function — given a bundle and a request dict it
returns the decision — so it is trivially unit-testable without AWS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from plantwatch.features.windowing import sliding_window_features


@dataclass
class ModelBundle:
    detector: object  # anything implementing .score()
    threshold: float
    model_version: str
    window: int
    sensors: Sequence[str]
    stats: Sequence[str]
    sequence_model: bool


def save_bundle(bundle: ModelBundle, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return str(path)


def load_bundle(path: str | Path) -> ModelBundle:
    return joblib.load(path)


def score_payload(payload: dict, bundle: ModelBundle) -> dict:
    """Score one window. ``payload["window"]`` is a list of ``window`` sensor rows."""
    rows = payload.get("window")
    if rows is None:
        raise ValueError("payload missing 'window'")

    arr = np.asarray(rows, dtype=float)
    n_sensors = len(bundle.sensors)
    if arr.ndim != 2 or arr.shape[1] != n_sensors:
        raise ValueError(f"window must be shaped (rows, {n_sensors}); got {arr.shape}")
    if arr.shape[0] != bundle.window:
        raise ValueError(f"window must have {bundle.window} rows; got {arr.shape[0]}")

    if bundle.sequence_model:
        features = arr.astype(np.float32)[None, ...]  # (1, T, F)
    else:
        frame = pd.DataFrame(arr, columns=list(bundle.sensors))
        features = sliding_window_features(
            frame, bundle.sensors, window=bundle.window, stats=bundle.stats
        )

    score = float(bundle.detector.score(features)[0])
    return {
        "machine_id": payload.get("machine_id"),
        "model_version": bundle.model_version,
        "score": score,
        "threshold": bundle.threshold,
        "is_anomaly": bool(score >= bundle.threshold),
    }
