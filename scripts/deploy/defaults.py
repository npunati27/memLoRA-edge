"""
Editable default runtime settings for memLoRA deploy scripts.

Change values here when you want persistent local defaults without exporting
environment variables on every run. Env vars still override these values.
"""

from __future__ import annotations

# Mock inference defaults
DEFAULT_MOCK_INFERENCE_DELAY_MS = 120
DEFAULT_MOCK_INFERENCE_JITTER_MS = 40
DEFAULT_MOCK_RESPONSE_TEXT = (
    "Simulated memLoRA-edge completion (CPU mock). "
    "Tune values in scripts/deploy/defaults.py or MEMLORA_MOCK_* env vars."
)
DEFAULT_MOCK_MODEL_ID = "qwen-base"
DEFAULT_MOCK_LORA_CSV = "crop_alfalfa_health,pest_aphid,dairy_milk_quality"
DEFAULT_MOCK_SKIP_ADAPTER_CHECK = True
DEFAULT_MOCK_LOG_EXTRA_JSON = ""

# Tier transition emulation defaults (ms)
DEFAULT_DELAY_DISK_TO_GPU_MS = 0
DEFAULT_DELAY_CPU_TO_GPU_MS = 0
DEFAULT_DELAY_GPU_TO_CPU_MS = 0
DEFAULT_DELAY_CPU_TO_DISK_MS = 0
DEFAULT_DELAY_DISK_TO_CPU_MS = 0
DEFAULT_DELAY_GPU_TO_DISK_MS = 0
