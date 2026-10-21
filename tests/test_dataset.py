import numpy as np

from plantwatch.features.dataset import build_windows, time_split
from plantwatch.ingest.synthetic import DEFAULT_SENSORS, generate_sensor_frame


def test_window_shapes_align():
    df = generate_sensor_frame(n_rows=1000, seed=1)
    ds = build_windows(df, DEFAULT_SENSORS, window=30)
    n_windows = 1000 - 30 + 1
    assert len(ds) == n_windows
    assert ds.X_seq.shape == (n_windows, 30, len(DEFAULT_SENSORS))
    assert ds.X_feat.shape[0] == n_windows
    assert ds.y.shape == (n_windows,)
    assert len(ds.index) == n_windows


def test_labels_binary_and_present():
    df = generate_sensor_frame(n_rows=2000, seed=1)
    ds = build_windows(df, DEFAULT_SENSORS, window=30)
    assert set(np.unique(ds.y)).issubset({0, 1})
    assert ds.y.sum() > 0


def test_time_split_is_chronological_and_covers_all():
    df = generate_sensor_frame(n_rows=1000, seed=1)
    tr, va, te = time_split(df, train=0.6, val=0.2)
    assert len(tr) + len(va) + len(te) == 1000
    assert tr.index.max() <= va.index.min()
    assert va.index.max() <= te.index.min()
