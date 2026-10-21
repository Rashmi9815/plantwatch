import io
import json

import numpy as np
import pandas as pd

from plantwatch.monitoring.drift import drift_report, population_stability_index
from plantwatch.monitoring.metrics import build_emf, emit


def test_emf_structure():
    result = {
        "machine_id": "M1",
        "model_version": "pca@w30",
        "score": 0.42,
        "is_anomaly": True,
    }
    emf = build_emf(result)
    assert emf["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "PlantWatch"
    assert emf["MachineId"] == "M1"
    assert isinstance(emf["AnomalyScore"], float)
    assert emf["AnomalyFlag"] == 1


def test_emit_writes_valid_json_line():
    buffer = io.StringIO()
    emit(
        {"machine_id": "M1", "model_version": "x", "score": 0.1, "is_anomaly": False}, stream=buffer
    )
    parsed = json.loads(buffer.getvalue().strip())
    assert parsed["AnomalyFlag"] == 0


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    data = rng.normal(size=5000)
    assert population_stability_index(data, data.copy()) < 0.01


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=5000)
    shifted = rng.normal(3, 1, size=5000)
    assert population_stability_index(reference, shifted) > 0.2


def test_drift_report_flags_shifted_column():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    current = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(5, 1, 2000)})
    report = drift_report(reference, current)
    assert report["a"]["drifted"] is False
    assert report["b"]["drifted"] is True
