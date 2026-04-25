#!/usr/bin/env python3
import uvicorn
from .deployment import app
from .config import SERVE_PORT, logger

def main():
    logger.info(f"Starting on port {SERVE_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVE_PORT, loop="asyncio", log_level="warning")

if __name__ == "__main__":
    main()