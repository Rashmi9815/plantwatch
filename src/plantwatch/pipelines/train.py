"""Train and evaluate the anomaly detectors end to end.

Pipeline:
  1. generate (or load) telemetry and clean it
  2. chronological train / val / test split (no shuffling)
  3. window each split independently (no leakage across the boundaries)
  4. fit each detector on **normal** training windows only
  5. pick a threshold from the normal validation scores (realistic, label-free)
  6. evaluate on the held-out test split and write a JSON metrics report

    python -m plantwatch.pipelines.train --rows 20000 --window 30
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from plantwatch._logging import get_logger
from plantwatch.evaluation.metrics import best_f1_threshold, classification_at_threshold, pr_auc
from plantwatch.evaluation.thresholds import quantile_threshold
from plantwatch.features.dataset import build_windows, time_split
from plantwatch.features.windowing import clean
from plantwatch.ingest.synthetic import DEFAULT_SENSORS, generate_sensor_frame
from plantwatch.models.baseline import IsolationForestDetector
from plantwatch.models.reconstruction import PCAReconstructionDetector

log = get_logger("plantwatch.train")


def _build_detectors(use_autoencoder: bool):
    detectors = [IsolationForestDetector(), PCAReconstructionDetector()]
    if use_autoencoder:
        try:
            from plantwatch.models.autoencoder import LSTMAutoencoder

            detectors.append(LSTMAutoencoder(epochs=20))
        except RuntimeError as exc:
            log.warning("skipping autoencoder: %s", exc)
    return detectors


def evaluate_detector(detector, train_ds, val_ds, test_ds, *, q: float) -> dict:
    sequence_model = detector.name == "lstm_autoencoder"
    train_X = train_ds.X_seq if sequence_model else train_ds.X_feat
    val_X = val_ds.X_seq if sequence_model else val_ds.X_feat
    test_X = test_ds.X_seq if sequence_model else test_ds.X_feat

    normal = train_ds.y == 0
    detector.fit(train_X[normal])

    val_scores = detector.score(val_X)
    threshold = quantile_threshold(val_scores[val_ds.y == 0], q=q)

    test_scores = detector.score(test_X)
    metrics = classification_at_threshold(test_ds.y, test_scores, threshold)
    metrics["pr_auc"] = pr_auc(test_ds.y, test_scores)
    metrics["best_f1_threshold_oracle"] = best_f1_threshold(test_ds.y, test_scores)
    return {"metrics": metrics, "threshold": threshold, "scores": test_scores}


def run(
    *,
    n_rows: int = 20000,
    window: int = 30,
    seed: int = 42,
    quantile: float = 0.99,
    use_autoencoder: bool = True,
    out_dir: str = "reports",
    db_url: str | None = None,
    machine_id: str = "SYNTH-1",
) -> dict:
    sensors = list(DEFAULT_SENSORS)
    df = clean(generate_sensor_frame(n_rows=n_rows, seed=seed), sensors)
    train_df, val_df, test_df = time_split(df)

    train_ds = build_windows(train_df, sensors, window=window)
    val_ds = build_windows(val_df, sensors, window=window)
    test_ds = build_windows(test_df, sensors, window=window)
    log.info(
        "windows — train:%d val:%d test:%d | test anomalies:%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        int(test_ds.y.sum()),
    )

    results = {}
    prediction_rows = []
    for detector in _build_detectors(use_autoencoder):
        log.info("fitting %s ...", detector.name)
        outcome = evaluate_detector(detector, train_ds, val_ds, test_ds, q=quantile)
        results[detector.name] = outcome["metrics"]
        if db_url:
            from plantwatch.db import repository as repo

            prediction_rows += repo.predictions_from_scores(
                machine_id,
                test_ds.index,
                outcome["scores"],
                threshold=outcome["threshold"],
                model_version=f"{detector.name}@w{window}",
            )

    if db_url:
        from plantwatch.db import repository as repo

        engine = repo.get_engine(db_url)
        repo.create_all(engine)
        repo.upsert_machine(engine, machine_id, "Synthetic line", location="simulator")
        n = repo.insert_predictions(engine, prediction_rows)
        log.info("persisted %d predictions to %s", n, db_url)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {"n_rows": n_rows, "window": window, "seed": seed, "quantile": quantile},
        "test_anomaly_rate": float(test_ds.y.mean()),
        "results": results,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_file = out_path / f"metrics_{stamp}.json"
    report_file.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", report_file)

    _print_table(report)
    return report


def _print_table(report: dict) -> None:
    rows = report["results"]
    print(f"\nTest anomaly rate: {report['test_anomaly_rate']:.3f}")
    header = f"{'model':<20}{'precision':>11}{'recall':>9}{'f1':>8}{'pr_auc':>9}"
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        pr = f"{m['pr_auc']:.3f}" if m["pr_auc"] == m["pr_auc"] else "n/a"
        print(f"{name:<20}{m['precision']:>11.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}{pr:>9}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate PlantWatch detectors.")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--no-autoencoder", action="store_true", help="skip the LSTM model")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument(
        "--db-url",
        default=None,
        help="persist predictions, e.g. sqlite:///plantwatch.db or postgresql+psycopg://...",
    )
    args = parser.parse_args(argv)
    run(
        n_rows=args.rows,
        window=args.window,
        seed=args.seed,
        quantile=args.quantile,
        use_autoencoder=not args.no_autoencoder,
        out_dir=args.out_dir,
        db_url=args.db_url,
    )


if __name__ == "__main__":
    main()
