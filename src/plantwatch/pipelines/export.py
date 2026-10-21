"""Train a single detector and save it as a servable model bundle.

    python -m plantwatch.pipelines.export --model pca --out artifacts/model.joblib

The bundle is what the Lambda loads. PCA is the default because it is accurate here and
has no heavy runtime dependency; ``--model lstm`` exports the autoencoder (needs torch).
"""

from __future__ import annotations

import argparse

from plantwatch._logging import get_logger
from plantwatch.evaluation.thresholds import quantile_threshold
from plantwatch.features.dataset import build_windows, time_split
from plantwatch.features.windowing import clean
from plantwatch.ingest.synthetic import DEFAULT_SENSORS, generate_sensor_frame
from plantwatch.models.baseline import IsolationForestDetector
from plantwatch.models.reconstruction import PCAReconstructionDetector
from plantwatch.serving.service import ModelBundle, save_bundle

log = get_logger("plantwatch.export")

STATS = ("mean", "std", "min", "max")


def _make_detector(model: str):
    if model == "pca":
        return PCAReconstructionDetector(), False
    if model == "iforest":
        return IsolationForestDetector(), False
    if model == "lstm":
        from plantwatch.models.autoencoder import LSTMAutoencoder

        return LSTMAutoencoder(), True
    raise ValueError(f"unknown model '{model}' (choose pca, iforest, lstm)")


def export_model(
    *,
    model: str = "pca",
    out: str = "artifacts/model.joblib",
    n_rows: int = 20000,
    window: int = 30,
    seed: int = 42,
    quantile: float = 0.99,
) -> tuple[str, float]:
    sensors = list(DEFAULT_SENSORS)
    df = clean(generate_sensor_frame(n_rows=n_rows, seed=seed), sensors)
    train_df, val_df, _ = time_split(df)
    train = build_windows(train_df, sensors, window=window, stats=STATS)
    val = build_windows(val_df, sensors, window=window, stats=STATS)

    detector, sequence_model = _make_detector(model)
    normal = train.y == 0
    detector.fit((train.X_seq if sequence_model else train.X_feat)[normal])

    val_scores = detector.score(val.X_seq if sequence_model else val.X_feat)
    threshold = quantile_threshold(val_scores[val.y == 0], q=quantile)

    bundle = ModelBundle(
        detector=detector,
        threshold=threshold,
        model_version=f"{detector.name}@w{window}",
        window=window,
        sensors=sensors,
        stats=STATS,
        sequence_model=sequence_model,
    )
    path = save_bundle(bundle, out)
    log.info("exported %s (threshold=%.4f) -> %s", bundle.model_version, threshold, path)
    return path, threshold


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a servable model bundle.")
    parser.add_argument("--model", choices=["pca", "iforest", "lstm"], default="pca")
    parser.add_argument("--out", default="artifacts/model.joblib")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantile", type=float, default=0.99)
    args = parser.parse_args(argv)
    export_model(
        model=args.model,
        out=args.out,
        n_rows=args.rows,
        window=args.window,
        seed=args.seed,
        quantile=args.quantile,
    )


if __name__ == "__main__":
    main()
