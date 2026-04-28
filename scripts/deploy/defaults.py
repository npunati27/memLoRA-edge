"""
Editable default runtime settings for memLoRA deploy scripts.

Change values here when you want persistent local defaults without exporting
environment variables on every run. Env vars still override these values.
"""

from __future__ import annotations

# Mock inference (separate from tier/S3 emulation; often 0 when using banded delays)
DEFAULT_MOCK_INFERENCE_DELAY_MS = 800
DEFAULT_MOCK_INFERENCE_JITTER_MS = 100
DEFAULT_MOCK_RESPONSE_TEXT = (
    "Simulated memLoRA-edge completion (CPU mock). "
    "Tune values in scripts/deploy/defaults.py or MEMLORA_MOCK_* env vars."
)
DEFAULT_MOCK_MODEL_ID = "qwen-base"
DEFAULT_MOCK_LORA_CSV = "crop_alfalfa_health,pest_aphid,dairy_milk_quality"
DEFAULT_MOCK_SKIP_ADAPTER_CHECK = True
DEFAULT_MOCK_LOG_EXTRA_JSON = ""

# Mock tier latency bands (ms, inclusive). Random sample per transition.
# GPU: hot path; CPU: promote/evict to CPU tier; Disk: local re-load / spill;
# S3: first disk->GPU load on this node only (see mock_tier_latency).
DEFAULT_MOCK_LATENCY_GPU_MIN_MS = 1
DEFAULT_MOCK_LATENCY_GPU_MAX_MS = 5
DEFAULT_MOCK_LATENCY_CPU_MIN_MS = 5
DEFAULT_MOCK_LATENCY_CPU_MAX_MS = 15
DEFAULT_MOCK_LATENCY_DISK_MIN_MS = 200
DEFAULT_MOCK_LATENCY_DISK_MAX_MS = 600
DEFAULT_MOCK_LATENCY_S3_MIN_MS = 2000
DEFAULT_MOCK_LATENCY_S3_MAX_MS = 10000
