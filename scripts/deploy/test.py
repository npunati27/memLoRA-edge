import boto3
from botocore import UNSIGNED
from botocore.client import Config

bucket = "memlora-adapters-525"
key = "adapters/carbon_footprint_field/README.md"

s3 = boto3.client(
    "s3",
    region_name="us-east-2",
    config=Config(signature_version=UNSIGNED),
)

obj = s3.get_object(Bucket=bucket, Key=key)
print(obj["Body"].read().decode("utf-8"))