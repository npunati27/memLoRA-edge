"""
Mock-only tier / S3 latency emulation (random bands, ms).

First time an adapter is promoted disk -> GPU on this node, we charge the S3 band
then record it in ``s3_fetched``. Later disk -> GPU hits use the local disk band
(cache re-warm). LRU cascades gpu -> cpu and cpu -> disk use CPU and disk bands.
"""

from __future__ import annotations

import os
import random
from typing import Any

from . import defaults as d


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _band(
    min_key: str,
    max_key: str,
    default_min: int,
    default_max: int,
) -> tuple[int, int]:
    lo = _env_int(min_key, default_min)
    hi = _env_int(max_key, default_max)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def mock_latency_bands() -> dict[str, tuple[int, int]]:
    return {
        "gpu": _band(
            "MEMLORA_MOCK_LATENCY_GPU_MIN_MS",
            "MEMLORA_MOCK_LATENCY_GPU_MAX_MS",
            d.DEFAULT_MOCK_LATENCY_GPU_MIN_MS,
            d.DEFAULT_MOCK_LATENCY_GPU_MAX_MS,
        ),
        "cpu": _band(
            "MEMLORA_MOCK_LATENCY_CPU_MIN_MS",
            "MEMLORA_MOCK_LATENCY_CPU_MAX_MS",
            d.DEFAULT_MOCK_LATENCY_CPU_MIN_MS,
            d.DEFAULT_MOCK_LATENCY_CPU_MAX_MS,
        ),
        "disk": _band(
            "MEMLORA_MOCK_LATENCY_DISK_MIN_MS",
            "MEMLORA_MOCK_LATENCY_DISK_MAX_MS",
            d.DEFAULT_MOCK_LATENCY_DISK_MIN_MS,
            d.DEFAULT_MOCK_LATENCY_DISK_MAX_MS,
        ),
        "s3": _band(
            "MEMLORA_MOCK_LATENCY_S3_MIN_MS",
            "MEMLORA_MOCK_LATENCY_S3_MAX_MS",
            d.DEFAULT_MOCK_LATENCY_S3_MIN_MS,
            d.DEFAULT_MOCK_LATENCY_S3_MAX_MS,
        ),
    }


def _sample(bands: dict[str, tuple[int, int]], key: str) -> int:
    lo, hi = bands[key]
    return random.randint(lo, hi)


def compute_mock_tier_delays(
    adapter_name: str | None,
    changes: list[tuple[str, str, str]],
    s3_fetched: set[str],
) -> tuple[int, list[dict[str, Any]]]:
    """
    Returns (total_delay_ms, detail_rows) for sleeps before mock inference body.
    Mutates ``s3_fetched`` when an S3 first-load path is taken.
    """
    if not adapter_name:
        return 0, []

    bands = mock_latency_bands()
    total_ms = 0
    details: list[dict[str, Any]] = []

    for adapter, old_tier, new_tier in changes:
        kind = f"{old_tier}->{new_tier}"
        delay_ms = 0
        label = kind

        if old_tier == "disk" and new_tier == "gpu":
            if adapter not in s3_fetched:
                delay_ms = _sample(bands, "s3")
                s3_fetched.add(adapter)
                label = "s3_first_load"
            else:
                delay_ms = _sample(bands, "disk")
                label = "disk_cache_to_gpu"
        elif old_tier == "cpu" and new_tier == "gpu":
            delay_ms = _sample(bands, "cpu")
            label = "cpu_to_gpu"
        elif old_tier == "gpu" and new_tier == "cpu":
            delay_ms = _sample(bands, "cpu")
            label = "gpu_to_cpu"
        elif old_tier == "cpu" and new_tier == "disk":
            delay_ms = _sample(bands, "disk")
            label = "cpu_to_disk"

        if delay_ms > 0:
            total_ms += delay_ms
            details.append({
                "adapter": adapter,
                "old_tier": old_tier,
                "new_tier": new_tier,
                "delay_ms": delay_ms,
                "kind": label,
            })

    # Already resident on GPU: LRU returns no transitions; charge a small GPU-path cost.
    if not changes:
        delay_ms = _sample(bands, "gpu")
        total_ms += delay_ms
        details.append({
            "adapter": adapter_name,
            "old_tier": "gpu",
            "new_tier": "gpu",
            "delay_ms": delay_ms,
            "kind": "gpu_resident_hit",
        })

    return total_ms, details
