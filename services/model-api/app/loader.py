import io

import boto3
import joblib
from botocore.config import Config


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"bad s3 uri, expected s3:// : {uri}")

    rest = uri.removeprefix("s3://")

    if "/" not in rest:
        raise ValueError(f"no key in s3 uri: {uri}")

    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"empty bucket or key in s3 uri: {uri}")
    return bucket, key


def make_s3(endpoint: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


def load_model(s3, uri: str):
    bucket, key = parse_s3_uri(uri)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return joblib.load(io.BytesIO(body))
