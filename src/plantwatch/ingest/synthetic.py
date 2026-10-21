"""Synthetic plant-telemetry generator.

Produces a multivariate sensor frame that looks like a running machine: each sensor
is a slow oscillation around its own baseline with a little noise. Faults are then
injected as labeled windows of three kinds — spike, drift, and stuck-sensor — so the
whole pipeline (and the models in later phases) has a ground-truth ``anomaly`` column
to validate against. Fully deterministic for a given ``seed``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_SENSORS: tuple[str, ...] = (
    "motor_current",
    "bearing_temp",
    "vibration",
    "throughput",
    "line_pressure",
)


def generate_sensor_frame(
    n_rows: int = 5000,
    sensors: Sequence[str] = DEFAULT_SENSORS,
    *,
    freq: str = "1s",
    anomaly_rate: float = 0.02,
    seed: int = 42,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Return a DatetimeIndex frame of sensor readings plus an ``anomaly`` label (0/1)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=n_rows, freq=freq, name="timestamp")

    columns: dict[str, np.ndarray] = {}
    for i, sensor in enumerate(sensors):
        baseline = 50.0 + 10.0 * i
        oscillation = np.sin(np.linspace(0, 6 * np.pi, n_rows) + i) * 2.0
        noise = rng.normal(0, 0.8, n_rows)
        columns[sensor] = baseline + oscillation + noise

    label = np.zeros(n_rows, dtype=int)
    low, high = 10, n_rows - 30
    candidates = np.arange(low, high)
    # Too few rows to place a fault window safely — return clean data.
    if candidates.size == 0:
        frame = pd.DataFrame(columns, index=index)
        frame["anomaly"] = label
        return frame

    n_events = min(max(1, int(n_rows * anomaly_rate) // 5), candidates.size)
    event_starts = rng.choice(candidates, size=n_events, replace=False)
    max_cols = min(2, len(sensors))

    for event_start in event_starts:
        length = int(rng.integers(5, 25))
        end = min(int(event_start) + length, n_rows)
        span = end - int(event_start)
        kind = int(rng.integers(0, 3))
        hit = rng.choice(np.array(sensors), size=int(rng.integers(1, max_cols + 1)), replace=False)
        for sensor in hit:
            series = columns[sensor]
            if kind == 0:  # spike
                series[event_start:end] += rng.normal(15, 3, span)
            elif kind == 1:  # drift
                series[event_start:end] += np.linspace(0, 12, span)
            else:  # stuck sensor
                series[event_start:end] = series[event_start]
        label[event_start:end] = 1

    frame = pd.DataFrame(columns, index=index)
    frame["anomaly"] = label
    return frame
