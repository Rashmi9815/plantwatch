import json

import pytest

from plantwatch.ingest.synthetic import DEFAULT_SENSORS, generate_sensor_frame
from plantwatch.pipelines.export import export_model
from plantwatch.serving.service import load_bundle, score_payload

WINDOW = 20
N_SENSORS = len(DEFAULT_SENSORS)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    path, _ = export_model(model="pca", out=str(out), n_rows=4000, window=WINDOW, seed=5)
    return path


def _normal_window():
    """A real slice of normal operation, so feature stats are in-distribution."""
    df = generate_sensor_frame(n_rows=500, seed=99)
    normal = df[df["anomaly"] == 0][list(DEFAULT_SENSORS)]
    return normal.iloc[:WINDOW].to_numpy().tolist()


def test_bundle_scores_payload(model_path):
    bundle = load_bundle(model_path)
    result = score_payload({"machine_id": "M1", "window": _normal_window()}, bundle)
    assert result["machine_id"] == "M1"
    assert isinstance(result["score"], float)
    assert isinstance(result["is_anomaly"], bool)
    assert result["model_version"].startswith("pca_reconstruction")


def test_bad_window_shape_raises(model_path):
    bundle = load_bundle(model_path)
    with pytest.raises(ValueError):
        score_payload({"window": [[1, 2, 3]]}, bundle)  # wrong n_sensors and n_rows


def test_handler_returns_200(model_path, monkeypatch):
    monkeypatch.setenv("PLANTWATCH_MODEL_PATH", str(model_path))
    from plantwatch.serving import handler as h

    h._BUNDLE = None  # reset cold-start cache
    event = {"body": json.dumps({"machine_id": "M1", "window": _normal_window()})}
    resp = h.handler(event)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert "score" in body and "is_anomaly" in body


def test_handler_returns_400_on_bad_request(model_path, monkeypatch):
    monkeypatch.setenv("PLANTWATCH_MODEL_PATH", str(model_path))
    from plantwatch.serving import handler as h

    h._BUNDLE = None
    resp = h.handler({"body": json.dumps({"window": [[1, 2, 3]]})})
    assert resp["statusCode"] == 400
    assert "error" in json.loads(resp["body"])
