import pandas as pd

from plantwatch.ingest.synthetic import DEFAULT_SENSORS, generate_sensor_frame


def test_shape_and_columns():
    df = generate_sensor_frame(n_rows=1000, seed=1)
    assert len(df) == 1000
    assert isinstance(df.index, pd.DatetimeIndex)
    for sensor in DEFAULT_SENSORS:
        assert sensor in df.columns
    assert "anomaly" in df.columns


def test_label_is_binary_with_some_anomalies():
    df = generate_sensor_frame(n_rows=1000, seed=1)
    assert df["anomaly"].isin([0, 1]).all()
    assert 0 < df["anomaly"].sum() < len(df)


def test_deterministic_for_seed():
    a = generate_sensor_frame(n_rows=500, seed=7)
    b = generate_sensor_frame(n_rows=500, seed=7)
    pd.testing.assert_frame_equal(a, b)
