"""Runtime configuration, read from the environment.

A single ``Settings`` object carries everything the pipeline needs to talk to S3.
Leaving ``AWS_ENDPOINT_URL`` unset targets real AWS; setting it to the LocalStack
endpoint (``http://localhost:4566``) keeps all development local and free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    aws_region: str = "us-east-1"
    aws_endpoint_url: str | None = None
    raw_bucket: str = "plantwatch-raw"
    raw_prefix: str = "sensors"

    @classmethod
    def from_env(cls) -> Settings:
        endpoint = os.getenv("AWS_ENDPOINT_URL") or None
        return cls(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            aws_endpoint_url=endpoint,
            raw_bucket=os.getenv("PLANTWATCH_RAW_BUCKET", "plantwatch-raw"),
            raw_prefix=os.getenv("PLANTWATCH_RAW_PREFIX", "sensors"),
        )
