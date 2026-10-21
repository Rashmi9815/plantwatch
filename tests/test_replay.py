import boto3
from moto import mock_aws

from plantwatch.ingest.replay import batch_to_jsonl, ensure_bucket, stream_to_s3
from plantwatch.ingest.synthetic import generate_sensor_frame


@mock_aws
def test_stream_writes_expected_batch_count():
    client = boto3.client("s3", region_name="us-east-1")
    bucket = "plantwatch-raw"
    ensure_bucket(client, bucket)

    df = generate_sensor_frame(n_rows=250, seed=3)
    n = stream_to_s3(df, client=client, bucket=bucket, prefix="sensors", batch_size=100)

    assert n == 3  # 250 rows / 100 -> 3 objects
    listing = client.list_objects_v2(Bucket=bucket, Prefix="sensors")
    assert listing["KeyCount"] == 3


@mock_aws
def test_stream_respects_limit():
    client = boto3.client("s3", region_name="us-east-1")
    ensure_bucket(client, "plantwatch-raw")
    df = generate_sensor_frame(n_rows=250, seed=3)
    n = stream_to_s3(df, client=client, bucket="plantwatch-raw", batch_size=100, limit=120)
    assert n == 2


def test_batch_to_jsonl_one_line_per_row():
    df = generate_sensor_frame(n_rows=5, seed=3)
    text = batch_to_jsonl(df)
    assert len(text.strip().splitlines()) == 5
    assert "timestamp" in text  # index was serialized as a column
