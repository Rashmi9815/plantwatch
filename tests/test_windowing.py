import numpy as np
import pandas as pd

from plantwatch.features.windowing import clean, sliding_window_features


def _frame():
    index = pd.date_range("2024-01-01", periods=10, freq="1s", name="timestamp")
    return pd.DataFrame({"s1": [1.0] * 10, "s2": list(range(10))}, index=index, dtype=float)


def test_clean_fills_gaps():
    df = _frame()
    df.loc[df.index[3], "s1"] = np.nan
    out = clean(df, ["s1", "s2"])
    assert out["s1"].isna().sum() == 0


def test_clean_drops_duplicate_timestamps():
    df = _frame()
    dup = pd.concat([df, df.iloc[[5]]])
    out = clean(dup, ["s1", "s2"])
    assert out.index.is_unique


def test_window_on_constant_series():
    feats = sliding_window_features(_frame(), ["s1"], window=5)
    assert len(feats) == 10 - 5 + 1
    assert (feats["s1_mean"] == 1.0).all()
    assert (feats["s1_std"] == 0.0).all()
    assert (feats["s1_min"] == 1.0).all()
    assert (feats["s1_max"] == 1.0).all()


def test_window_feature_columns():
    feats = sliding_window_features(_frame(), ["s1", "s2"], window=3, stats=("mean", "std"))
    assert set(feats.columns) == {"s1_mean", "s1_std", "s2_mean", "s2_std"}
