"""
CPU-only mock server defaults. Used when ``MEMLORA_MOCK=1`` (see ``config.USE_MOCK_ENGINE``).

Run the same entrypoint as production: ``MEMLORA_MOCK=1 python -m scripts.deploy``.
Optional shorthand: ``python -m scripts.deploy.mock_main`` (sets the flag for you).

Override with environment variables (no code edits):

Examples:
  MEMLORA_MOCK_DELAY_MS=250
  MEMLORA_MOCK_RESPONSE='{"note":"demo"}'
  MEMLORA_MOCK_MODEL_ID=demo-base
  MEMLORA_MOCK_LORA_NAMES=crop_alfalfa_health,pest_aphid
"""

from __future__ import annotations

import json
import os
from .defaults import (
    DEFAULT_MOCK_INFERENCE_DELAY_MS,
    DEFAULT_MOCK_INFERENCE_JITTER_MS,
    DEFAULT_MOCK_RESPONSE_TEXT,
    DEFAULT_MOCK_MODEL_ID,
    DEFAULT_MOCK_LORA_CSV,
    DEFAULT_MOCK_SKIP_ADAPTER_CHECK,
    DEFAULT_MOCK_LOG_EXTRA_JSON,
)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Simulated inference time (wall clock, async sleep)
MOCK_INFERENCE_DELAY_MS = _env_int("MEMLORA_MOCK_DELAY_MS", DEFAULT_MOCK_INFERENCE_DELAY_MS)

# Optional extra jitter in ms (uniform 0..N); set 0 to disable
MOCK_INFERENCE_JITTER_MS = _env_int("MEMLORA_MOCK_JITTER_MS", DEFAULT_MOCK_INFERENCE_JITTER_MS)

# OpenAI-style assistant message content (string). For JSON-looking text, use quotes in shell.
MOCK_RESPONSE_TEXT = _env_str(
    "MEMLORA_MOCK_RESPONSE",
    DEFAULT_MOCK_RESPONSE_TEXT,
)

# Base model id used in /v1/models and in model field parsing (must match client requests)
MOCK_MODEL_ID = _env_str("MEMLORA_MOCK_MODEL_ID", DEFAULT_MOCK_MODEL_ID)

# Comma-separated adapter names advertised when ~/adapters is empty or unset
_DEFAULT_LORA_CSV = DEFAULT_MOCK_LORA_CSV


def mock_lora_names_from_env() -> list[str]:
    raw = _env_str("MEMLORA_MOCK_LORA_NAMES", _DEFAULT_LORA_CSV)
    return sorted({p.strip() for p in raw.split(",") if p.strip()})


# If "1", skip on-disk adapter directory checks (recommended for VMs without adapter files)
_default_skip_adapter_check = "1" if DEFAULT_MOCK_SKIP_ADAPTER_CHECK else "0"
MOCK_SKIP_ADAPTER_PATH_CHECK = _env_str("MEMLORA_MOCK_SKIP_ADAPTER_CHECK", _default_skip_adapter_check).lower() in (
    "1",
    "true",
    "yes",
)

# JSON log fields for mock inference (optional structured echo)
MOCK_LOG_EXTRA_JSON = _env_str("MEMLORA_MOCK_LOG_EXTRA_JSON", DEFAULT_MOCK_LOG_EXTRA_JSON)


def mock_log_extra() -> dict | None:
    s = MOCK_LOG_EXTRA_JSON.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"_raw": s}
