"""Replay sensor telemetry into S3 as if it were arriving live.

This is the front door of the pipeline. It takes a frame (synthetic, or a SKAB-style
CSV), splits it into batches, and writes each batch as newline-delimited JSON to the
raw ("bronze") S3 bucket. Point it at LocalStack for free local runs, or at real AWS
by unsetting ``AWS_ENDPOINT_URL``.

    python -m plantwatch.ingest.replay --source synthetic --rows 5000 --batch-size 200
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from plantwatch._logging import get_logger
from plantwatch.config import Settings
from plantwatch.ingest.synthetic import generate_sensor_frame

log = get_logger("plantwatch.ingest")


def get_s3_client(settings: Settings):
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url,
    )


def ensure_bucket(client, bucket: str) -> None:
    """Create the bucket if it does not already exist (convenient for LocalStack)."""
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        log.info("created bucket s3://%s", bucket)


def batch_to_jsonl(batch: pd.DataFrame) -> str:
    """Serialize a batch to newline-delimited JSON, timestamps rendered as ISO strings."""
    records = batch.reset_index()
    for col in records.select_dtypes(include=["datetime64[ns]"]).columns:
        records[col] = records[col].astype(str)
    return records.to_json(orient="records", lines=True)


def stream_to_s3(
    df: pd.DataFrame,
    *,
    client,
    bucket: str,
    prefix: str = "sensors",
    batch_size: int = 100,
    interval: float = 0.0,
    limit: int | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Write ``df`` to S3 in batches. Returns the number of objects written."""
    if limit is not None:
        df = df.iloc[:limit]
    n_batches = 0
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        key = f"{prefix}/batch-{n_batches:05d}.jsonl"
        client.put_object(Bucket=bucket, Key=key, Body=batch_to_jsonl(batch).encode("utf-8"))
        n_batches += 1
        log.info("uploaded %d rows -> s3://%s/%s", len(batch), bucket, key)
        if interval:
            sleeper(interval)
    return n_batches


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for candidate in ("timestamp", "datetime", "time"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate])
            return df.set_index(candidate)
    return df


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay sensor telemetry into S3.")
    parser.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--csv", help="Path to a SKAB-style CSV (required when --source csv)")
    parser.add_argument("--rows", type=int, default=5000, help="rows to generate (synthetic)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.0, help="seconds between batches")
    parser.add_argument("--limit", type=int, default=None, help="cap total rows streamed")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = Settings.from_env()

    if args.source == "csv":
        if not args.csv:
            raise SystemExit("--csv is required when --source csv")
        df = _load_csv(args.csv)
    else:
        df = generate_sensor_frame(n_rows=args.rows, seed=args.seed)

    client = get_s3_client(settings)
    ensure_bucket(client, settings.raw_bucket)
    n = stream_to_s3(
        df,
        client=client,
        bucket=settings.raw_bucket,
        prefix=settings.raw_prefix,
        batch_size=args.batch_size,
        interval=args.interval,
        limit=args.limit,
    )
    log.info("done: %d batches -> s3://%s/%s", n, settings.raw_bucket, settings.raw_prefix)


if __name__ == "__main__":
    main()
