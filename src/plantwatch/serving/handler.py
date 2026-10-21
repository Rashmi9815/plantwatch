"""AWS Lambda entrypoint behind API Gateway (HTTP API, payload format 2.0).

The model bundle is loaded once per cold start from ``PLANTWATCH_MODEL_PATH`` and cached.
All domain logic lives in :func:`score_payload`; this file only translates between the
API Gateway event/response envelope and that function.
"""

from __future__ import annotations

import json
import os

from plantwatch.serving.service import load_bundle, score_payload

_BUNDLE = None


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        path = os.environ.get("PLANTWATCH_MODEL_PATH", "model.joblib")
        _BUNDLE = load_bundle(path)
    return _BUNDLE


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event: dict, context=None) -> dict:
    try:
        raw = event.get("body") if isinstance(event, dict) else None
        payload = json.loads(raw) if isinstance(raw, str) else (raw or event or {})
        result = score_payload(payload, _bundle())
        return _response(200, result)
    except (ValueError, KeyError, TypeError) as exc:
        return _response(400, {"error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - last-resort guard so Lambda returns JSON
        return _response(500, {"error": f"internal error: {exc}"})
