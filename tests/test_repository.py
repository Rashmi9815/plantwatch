import pandas as pd
import pytest

from plantwatch.db import repository as repo


@pytest.fixture
def engine():
    eng = repo.get_engine()  # in-memory SQLite
    repo.create_all(eng)
    repo.upsert_machine(eng, "M1", "Line 1")
    return eng


def test_insert_and_recent_anomalies(engine):
    index = pd.date_range("2024-01-01", periods=5, freq="min")
    scores = [0.1, 0.9, 0.2, 0.95, 0.3]
    rows = repo.predictions_from_scores("M1", index, scores, threshold=0.5, model_version="pca@w30")
    assert repo.insert_predictions(engine, rows) == 5

    anomalies = repo.recent_anomalies(engine)
    assert len(anomalies) == 2
    assert all(a["is_anomaly"] for a in anomalies)
    assert anomalies[0]["window_end"] >= anomalies[1]["window_end"]  # newest first


def test_anomaly_count_by_model(engine):
    index = pd.date_range("2024-01-01", periods=4, freq="min")
    repo.insert_predictions(
        engine,
        repo.predictions_from_scores(
            "M1", index, [0.9, 0.9, 0.1, 0.1], threshold=0.5, model_version="if@w30"
        ),
    )
    repo.insert_predictions(
        engine,
        repo.predictions_from_scores(
            "M1", index, [0.9, 0.1, 0.1, 0.1], threshold=0.5, model_version="pca@w30"
        ),
    )
    counts = repo.anomaly_count_by_model(engine)
    assert counts == {"if@w30": 2, "pca@w30": 1}


def test_upsert_machine_is_idempotent(engine):
    repo.upsert_machine(engine, "M1", "Line 1 renamed")
    assert repo.count_machines(engine) == 1
