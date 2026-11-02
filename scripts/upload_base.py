import io
import os

import boto3
from botocore.config import Config

from datagen import generate

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    region_name="us-east-1",
    config=Config(s3={"addressing_style": "path"}),
)

existing = {b["Name"] for b in s3.list_buckets()["Buckets"]}
for bucket in ("datasets", "models"):
     if bucket not in existing:
        s3.create_bucket(Bucket=bucket)
        print("created bucket", bucket)

for key, df in [
    ("base/train.parquet", generate(5000, seed=42)),
    ("incoming/latest.parquet", generate(3000, shift=1.5, concept="v2", seed=7)),
]:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    s3.put_object(Bucket="datasets", Key=key, Body=buf.getvalue())
    print("uploaded", key, len(df), "rows")