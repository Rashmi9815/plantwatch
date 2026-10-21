"""CloudWatch metrics via the Embedded Metric Format (EMF).

The Lambda prints one EMF JSON line per request; CloudWatch Logs parses it and
publishes the metrics automatically — no extra API call, no added latency. We emit the
anomaly score and a 0/1 flag, dimensioned by machine and model version, so the alarms
and dashboard in ``infra/monitoring.tf`` have something to watch.
"""

from __future__ import annotations

import json
import sys
import time
from typing import TextIO

NAMESPACE = "PlantWatch"


def build_emf(result: dict, *, namespace: str = NAMESPACE) -> dict:
    return {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [["MachineId", "ModelVersion"]],
                    "Metrics": [
                        {"Name": "AnomalyScore", "Unit": "None"},
                        {"Name": "AnomalyFlag", "Unit": "Count"},
                    ],
                }
            ],
        },
        "MachineId": result.get("machine_id") or "unknown",
        "ModelVersion": result.get("model_version", "unknown"),
        "AnomalyScore": float(result.get("score", 0.0)),
        "AnomalyFlag": int(bool(result.get("is_anomaly"))),
    }


def emit(result: dict, *, namespace: str = NAMESPACE, stream: TextIO | None = None) -> None:
    print(json.dumps(build_emf(result, namespace=namespace)), file=stream or sys.stdout)
