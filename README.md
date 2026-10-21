# PlantWatch

Near-real-time **anomaly detection on industrial sensor streams**, built and operated as a
production **MLOps** system on AWS — the full lifecycle from ingestion to retraining.

A recycling/processing plant runs machines instrumented with sensors (motor current, bearing
temperature, vibration, throughput, line pressure). Unplanned failure halts the line. PlantWatch
learns each machine's *normal* signature, flags deviations early, and keeps itself healthy through
monitoring and scheduled retraining.

> This repository is built in six phases, each one demoable on its own.
> **Phases 1–2 complete:** data & local loop (ingestion, cleaning, windowing) and the models
> (Isolation Forest baseline, PCA + LSTM-autoencoder reconstruction challengers) with time-ordered
> validation, label-free threshold selection, and a JSON metrics report.

## Architecture (target)

```
DATA & TRAINING   simulator → S3 bronze → Lambda clean/window → RDS+S3 features → SageMaker → S3 registry
                                                                                                    │ deploy
SERVE & MONITOR   dashboard ← CloudWatch alarms ← RDS gold (scores) ← API (Lambda+Gateway) ←────────┘
                                   └── EventBridge → AutoML retrain ─→ SageMaker
```

Terraform provisions every resource; GitHub Actions lints, tests, and deploys. Develop against
**LocalStack** (free) for daily work; deploy to real AWS only for screenshots, then `terraform destroy`.

## Quickstart

Requires Python 3.11+. Docker is only needed for the LocalStack demo.

```bash
cd plantwatch
python -m venv .venv && source .venv/bin/activate
make install          # editable install + dev tools + git hooks
make test             # runs the unit tests — no AWS needed (S3 is mocked with moto)
```

Run the ingestion loop against a local S3:

```bash
make demo             # starts LocalStack and streams synthetic telemetry into it
# or, step by step:
make up               # start LocalStack
make replay           # stream 5,000 synthetic rows into s3://plantwatch-raw/sensors/
```

Point at a real SKAB CSV instead of synthetic data:

```bash
python -m plantwatch.ingest.replay --source csv --csv path/to/skab.csv --batch-size 200
```

## What's in Phase 1

| Module | Role |
| --- | --- |
| `ingest/synthetic.py` | Deterministic multivariate telemetry generator with labeled spike/drift/stuck faults |
| `ingest/replay.py` | Streams a frame to S3 in batches (LocalStack or real AWS) |
| `features/windowing.py` | Cleans gaps/duplicates and builds rolling mean/std/min/max features |
| `config.py` | Environment-driven settings (one switch for LocalStack vs AWS) |
| `db/schema.sql` | Starter Postgres schema for machines, features, predictions |

## Phase 2 — models

Train all detectors, pick thresholds, and write a metrics report:

```bash
python -m plantwatch.pipelines.train --rows 20000 --window 30
# lightweight only (skip the deep model):
python -m plantwatch.pipelines.train --no-autoencoder
```

Example output on synthetic data (your numbers will match for a fixed seed):

```
model                 precision   recall      f1   pr_auc
---------------------------------------------------------
isolation_forest          0.805    0.632   0.708    0.804
pca_reconstruction        0.962    0.817   0.883    0.874
```

A timestamped report lands in `reports/metrics_*.json`. The LSTM autoencoder needs the
optional deep extra:

```bash
pip install -e ".[deep]"   # installs PyTorch
```

| Module | Role |
| --- | --- |
| `models/baseline.py` | Isolation Forest on rolling-window features |
| `models/reconstruction.py` | PCA reconstruction-error detector (no deep deps) |
| `models/autoencoder.py` | LSTM autoencoder challenger (optional PyTorch) |
| `features/dataset.py` | Windowing + chronological train/val/test split |
| `evaluation/metrics.py` | Precision / recall / F1 / PR-AUC |
| `evaluation/thresholds.py` | Label-free quantile threshold from normal scores |
| `pipelines/train.py` | End-to-end train → threshold → evaluate → report |

## Development

```bash
make lint    # ruff static checks
make fmt     # auto-format and fix
make test    # pytest
```

Pre-commit runs ruff + hygiene hooks on every commit (`make install` sets it up).

## Roadmap

1. ~~Data & local loop~~ ✅
2. ~~Model: Isolation Forest baseline → PCA / LSTM autoencoder challenger~~ ✅
3. Data model & SQL on RDS ← next
4. Serving (Lambda + API Gateway) & Terraform IaC
5. CI/CD (GitHub Actions) & CloudWatch monitoring
6. Retraining loop (EventBridge + SageMaker Autopilot)
