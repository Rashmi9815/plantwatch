"""Retraining loop with a promote-on-improvement gate.

Closes the MLOps loop: train a fresh candidate on recent data, score it and the live
(incumbent) model on the same holdout, and promote the candidate **only if it beats the
incumbent by a margin**. A new model never ships just because a schedule fired.

Triggers (Phase 5 feeds this):
  * a schedule (EventBridge, daily), or
  * drift — :func:`should_retrain` flags when feature distributions have shifted.

In production the heavy training runs as a SageMaker job / Autopilot; the same decision
logic applies to whatever produced the candidate. ``lambda_handler`` is the entrypoint
EventBridge invokes.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from plantwatch._logging import get_logger
from plantwatch.evaluation.metrics import pr_auc
from plantwatch.evaluation.thresholds import quantile_threshold
from plantwatch.features.dataset import WindowedDataset, build_windows, time_split
from plantwatch.features.windowing import clean
from plantwatch.ingest.synthetic import DEFAULT_SENSORS, generate_sensor_frame
from plantwatch.monitoring.drift import SIGNIFICANT, drift_report
from plantwatch.pipelines.export import STATS, _make_detector
from plantwatch.serving.service import ModelBundle, load_bundle, save_bundle

log = get_logger("plantwatch.retrain")


@dataclass
class PromotionDecision:
    promoted: bool
    candidate_metric: float
    incumbent_metric: float | None
    margin: float
    reason: str
    report: dict = field(default_factory=dict)


def decide_promotion(
    candidate_metric: float,
    incumbent_metric: float | None,
    *,
    min_improvement: float = 0.01,
) -> PromotionDecision:
    if incumbent_metric is None or math.isnan(incumbent_metric):
        return PromotionDecision(
            True, candidate_metric, incumbent_metric, math.inf, "no incumbent; promoting candidate"
        )
    margin = candidate_metric - incumbent_metric
    promoted = margin >= min_improvement
    reason = (
        f"candidate +{margin:.4f} ≥ {min_improvement} → promote"
        if promoted
        else f"candidate {margin:+.4f} < {min_improvement} → keep incumbent"
    )
    return PromotionDecision(promoted, candidate_metric, incumbent_metric, margin, reason)


def _evaluate_bundle(bundle: ModelBundle, ds: WindowedDataset) -> float:
    features = ds.X_seq if bundle.sequence_model else ds.X_feat
    return pr_auc(ds.y, bundle.detector.score(features))


def should_retrain(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    *,
    threshold: float = SIGNIFICANT,
) -> tuple[bool, dict]:
    """Drift trigger: retrain if any feature's distribution has shifted."""
    report = drift_report(reference_features, current_features, threshold=threshold)
    return any(col["drifted"] for col in report.values()), report


def retrain(
    *,
    model: str = "pca",
    incumbent_path: str | None = "artifacts/model.joblib",
    out: str = "artifacts/model.joblib",
    n_rows: int = 20000,
    window: int = 30,
    seed: int = 43,
    quantile: float = 0.99,
    min_improvement: float = 0.01,
) -> PromotionDecision:
    sensors = list(DEFAULT_SENSORS)
    df = clean(generate_sensor_frame(n_rows=n_rows, seed=seed), sensors)
    train_df, val_df, holdout_df = time_split(df)
    train = build_windows(train_df, sensors, window=window, stats=STATS)
    val = build_windows(val_df, sensors, window=window, stats=STATS)
    holdout = build_windows(holdout_df, sensors, window=window, stats=STATS)

    detector, sequence_model = _make_detector(model)
    normal = train.y == 0
    detector.fit((train.X_seq if sequence_model else train.X_feat)[normal])
    val_scores = detector.score(val.X_seq if sequence_model else val.X_feat)
    threshold = quantile_threshold(val_scores[val.y == 0], q=quantile)

    candidate = ModelBundle(
        detector=detector,
        threshold=threshold,
        model_version=f"{detector.name}@w{window}",
        window=window,
        sensors=sensors,
        stats=STATS,
        sequence_model=sequence_model,
    )
    candidate_metric = _evaluate_bundle(candidate, holdout)

    incumbent_metric = None
    if incumbent_path and Path(incumbent_path).exists():
        incumbent_metric = _evaluate_bundle(load_bundle(incumbent_path), holdout)

    decision = decide_promotion(candidate_metric, incumbent_metric, min_improvement=min_improvement)
    log.info(
        "candidate PR-AUC=%.4f incumbent=%s | %s",
        candidate_metric,
        f"{incumbent_metric:.4f}" if incumbent_metric is not None else "none",
        decision.reason,
    )
    if decision.promoted:
        save_bundle(candidate, out)
        log.info("promoted -> %s", out)
    return decision


def lambda_handler(event=None, context=None) -> dict:
    """EventBridge entrypoint. Config comes from the environment."""
    model_path = os.environ.get("PLANTWATCH_MODEL_PATH", "model.joblib")
    decision = retrain(
        model=os.environ.get("PLANTWATCH_RETRAIN_MODEL", "pca"),
        incumbent_path=model_path,
        out=model_path,
    )
    return {
        "promoted": decision.promoted,
        "candidate_metric": decision.candidate_metric,
        "incumbent_metric": decision.incumbent_metric,
        "reason": decision.reason,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retrain and (maybe) promote a model.")
    parser.add_argument("--model", choices=["pca", "iforest", "lstm"], default="pca")
    parser.add_argument("--incumbent", default="artifacts/model.joblib")
    parser.add_argument("--out", default="artifacts/model.joblib")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    args = parser.parse_args(argv)
    decision = retrain(
        model=args.model,
        incumbent_path=args.incumbent,
        out=args.out,
        n_rows=args.rows,
        window=args.window,
        seed=args.seed,
        min_improvement=args.min_improvement,
    )
    print(decision.reason)


if __name__ == "__main__":
    main()
