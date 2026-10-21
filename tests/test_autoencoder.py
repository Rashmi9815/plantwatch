"""Autoencoder tests run only when the optional torch extra is installed."""

import pytest

pytest.importorskip("torch")

from plantwatch.features.dataset import build_windows  # noqa: E402
from plantwatch.ingest.synthetic import DEFAULT_SENSORS, generate_sensor_frame  # noqa: E402
from plantwatch.models.autoencoder import LSTMAutoencoder  # noqa: E402


def test_autoencoder_trains_and_scores():
    df = generate_sensor_frame(n_rows=1500, seed=5)
    ds = build_windows(df, DEFAULT_SENSORS, window=20)
    normal = ds.y == 0

    model = LSTMAutoencoder(epochs=2, hidden=16, latent=8).fit(ds.X_seq[normal])
    scores = model.score(ds.X_seq)

    assert scores.shape == (len(ds),)
    assert scores[ds.y == 1].mean() > scores[ds.y == 0].mean()
