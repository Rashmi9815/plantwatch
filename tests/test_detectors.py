"""The baseline and PCA detectors should separate faults from normal operation."""

import numpy as np

from plantwatch.evaluation.metrics import pr_auc
from plantwatch.features.dataset import build_windows
from plantwatch.ingest.synthetic import DEFAULT_SENSORS, generate_sensor_frame
from plantwatch.models.baseline import IsolationForestDetector
from plantwatch.models.reconstruction import PCAReconstructionDetector

import pytest


@pytest.fixture(scope="module")
def dataset():
    df = generate_sensor_frame(n_rows=6000, seed=11)
    return build_windows(df, DEFAULT_SENSORS, window=20)


@pytest.mark.parametrize(
    "detector_cls", [IsolationForestDetector, PCAReconstructionDetector]
)
def test_anomalies_score_higher_than_normal(dataset, detector_cls):
    normal = dataset.y == 0
    detector = detector_cls().fit(dataset.X_feat[normal])
    scores = detector.score(dataset.X_feat)
    assert scores[dataset.y == 1].mean() > scores[dataset.y == 0].mean()
    # Better than the base rate => the detector carries real signal.
    assert pr_auc(dataset.y, scores) > dataset.y.mean()


def test_score_shape(dataset):
    detector = IsolationForestDetector().fit(dataset.X_feat)
    scores = detector.score(dataset.X_feat)
    assert scores.shape == (len(dataset),)
    assert np.isfinite(scores).all()
