#!/usr/bin/env python3

import time

import ray
import requests
from ray import serve

from .config import SERVE_PORT, logger
from .deployment import VLLMDeployment


def main():
    ray.init()
    logger.info(f"Cluster resources: {ray.cluster_resources()}")

    try:
        serve.start(http_options={"host": "0.0.0.0", "port": SERVE_PORT})
        serve.delete("memlora")
        time.sleep(2)
    except Exception:
        pass

    serve.shutdown()
    time.sleep(2)
    serve.start(http_options={"host": "0.0.0.0", "port": SERVE_PORT})
    handle = VLLMDeployment.bind()
    serve.run(handle, name="memlora", route_prefix="/", blocking=False)

    logger.info("Waiting for deployment to be ready...")
    for i in range(60):
        try:
            r = requests.get(f"http://localhost:{SERVE_PORT}/health", timeout=5)
            if r.status_code == 200:
                logger.info(f"Ready! {r.json()}")
                break
        except Exception:
            pass
        time.sleep(60)
        logger.info(f"  waiting... {i+1}/60 (~{(i+1)*10}s elapsed)")
    else:
        logger.error("Did not become ready in time.")
        exit(1)


if __name__ == "__main__":
    main()
