import numpy as np
import pandas as pd

from plantwatch.pipelines.retrain import decide_promotion, retrain, should_retrain


def test_promote_when_no_incumbent():
    d = decide_promotion(0.80, None)
    assert d.promoted is True
    assert "no incumbent" in d.reason


def test_keep_incumbent_when_candidate_not_better():
    d = decide_promotion(0.80, 0.82, min_improvement=0.01)
    assert d.promoted is False


def test_promote_when_candidate_beats_margin():
    d = decide_promotion(0.85, 0.80, min_improvement=0.01)
    assert d.promoted is True
    assert d.margin > 0


def test_below_margin_is_not_promoted():
    d = decide_promotion(0.803, 0.80, min_improvement=0.01)
    assert d.promoted is False


def test_retrain_promotes_first_model(tmp_path):
    out = tmp_path / "model.joblib"
    decision = retrain(
        model="pca", incumbent_path=str(out), out=str(out), n_rows=4000, window=20, seed=43
    )
    assert decision.promoted is True
    assert out.exists()  # the promoted bundle was written


def test_should_retrain_detects_shift():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"f": rng.normal(0, 1, 2000)})
    current = pd.DataFrame({"f": rng.normal(4, 1, 2000)})
    drifted, report = should_retrain(reference, current)
    assert drifted is True
    assert report["f"]["drifted"] is True
