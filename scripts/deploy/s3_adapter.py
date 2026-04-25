import os
import shutil
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from .config import (
    ADAPTER_PATH,
    S3_BUCKET,
    S3_REGION,
    S3_PREFIX_ROOT,
    EXPECTED_ADAPTER_FILES,
    logger,
)


def is_safe_rel_path(rel: str) -> bool:
    """Check if rel is a safe relative path (no absolute, no .., no empty segments)."""
    if os.path.isabs(rel):
        return False
    parts = rel.split(os.sep)
    for part in parts:
        if part in ("", ".", ".."):
            return False
    return True


_s3 = boto3.client(
    "s3",
    region_name=S3_REGION,
    config=Config(signature_version=UNSIGNED),
)

def list_adapters_from_s3():
    resp = _s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=f"{S3_PREFIX_ROOT}/",
        Delimiter="/",
    )
    prefixes = resp.get("CommonPrefixes", [])
    adapters = []
    for entry in prefixes:
        prefix = entry["Prefix"]
        name = prefix[len(f"{S3_PREFIX_ROOT}/"):-1]
        if name:
            adapters.append(name)
    return sorted(adapters)

def adapter_prefix(adapter_name: str) -> str:
    return f"{S3_PREFIX_ROOT}/{adapter_name}/"


def local_adapter_dir(adapter_name: str) -> Path:
    return Path(ADAPTER_PATH) / adapter_name


def adapter_exists_locally(adapter_name: str) -> bool:
    return local_adapter_dir(adapter_name).is_dir()


def list_adapter_objects(adapter_name: str):
    prefix = adapter_prefix(adapter_name)
    resp = _s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", []) if not obj["Key"].endswith("/")]


def adapter_exists_in_s3(adapter_name: str) -> bool:
    keys = list_adapter_objects(adapter_name)
    return len(keys) > 0


def download_adapter_from_s3(adapter_name: str) -> str:
    keys = list_adapter_objects(adapter_name)
    if not keys:
        raise FileNotFoundError(f"Adapter '{adapter_name}' not found in S3")

    target_dir = local_adapter_dir(adapter_name)
    tmp_dir = target_dir.parent / f".{adapter_name}.tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prefix = adapter_prefix(adapter_name)
    downloaded = set()

    for key in keys:
        rel = key[len(prefix):]
        if not is_safe_rel_path(rel):
            raise ValueError(f"Unsafe relative path in S3 key '{key}': '{rel}'")
        out_path = tmp_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _s3.download_file(S3_BUCKET, key, str(out_path))
        downloaded.add(rel)

    missing = EXPECTED_ADAPTER_FILES - downloaded
    if missing:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Adapter '{adapter_name}' missing required files after S3 download: {sorted(missing)}"
        )

    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_dir.rename(target_dir)

    logger.info(f"[s3] downloaded adapter={adapter_name} to {target_dir}")
    return str(target_dir)