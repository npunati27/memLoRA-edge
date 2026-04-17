#!/usr/bin/env python3
"""Shim: same app as ``python -m scripts.deploy`` with ``MEMLORA_MOCK=1`` preset."""

import os

os.environ.setdefault("MEMLORA_MOCK", "1")

import uvicorn

from .config import SERVE_PORT, logger
from .deployment import app


def main():
    logger.info(
        f"Starting memLoRA API on port {SERVE_PORT} (MEMLORA_MOCK=1; "
        "mock tuning: scripts/deploy/mock_settings.py or MEMLORA_MOCK_* env)"
    )
    uvicorn.run(app, host="0.0.0.0", port=SERVE_PORT, loop="asyncio", log_level="warning")


if __name__ == "__main__":
    main()
