import os
import json
import logging
from .defaults import (
    DEFAULT_DELAY_DISK_TO_GPU_MS,
    DEFAULT_DELAY_CPU_TO_GPU_MS,
    DEFAULT_DELAY_GPU_TO_CPU_MS,
    DEFAULT_DELAY_CPU_TO_DISK_MS,
    DEFAULT_DELAY_DISK_TO_CPU_MS,
    DEFAULT_DELAY_GPU_TO_DISK_MS,
)

HOME = os.path.expanduser("~")
LOG_DIR = os.path.join(HOME, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ROUTING_MODE = os.getenv("ROUTING_MODE", "baseline").strip().lower()
# CPU mock: same HTTP app/routes/gossip as GPU; no vLLM. Set before importing deployment.
USE_MOCK_ENGINE = os.getenv("MEMLORA_MOCK", "").strip().lower() in ("1", "true", "yes")
TIER_RANK = {"gpu": 0, "cpu": 1, "disk": 2}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "deploy.log")),
    ],
)
logger = logging.getLogger("memlora")

MODEL_PATH = f"{HOME}/model_cache/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = f"{HOME}/adapters"
PEERS_FILE = os.path.expanduser("~/peers.json")

MAX_GPU_LORA = 3
MAX_CPU_LORA = 6
SERVE_PORT = 5000

RTT_MAX_MS = 50
MAX_QUEUE_LEN = 8
MEMORY_COST = {
    "gpu":  0.0,
    "cpu":  0.015,
    "disk": 1.0,
    "s3":   float("inf"),
}


def _env_nonneg_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


# Optional emulated latency per adapter tier transition (milliseconds).
# This is applied when LRU causes local tier changes during request handling.
TIER_TRANSITION_DELAY_MS = {
    ("disk", "gpu"): _env_nonneg_int("MEMLORA_DELAY_DISK_TO_GPU_MS", DEFAULT_DELAY_DISK_TO_GPU_MS),
    ("cpu", "gpu"): _env_nonneg_int("MEMLORA_DELAY_CPU_TO_GPU_MS", DEFAULT_DELAY_CPU_TO_GPU_MS),
    ("gpu", "cpu"): _env_nonneg_int("MEMLORA_DELAY_GPU_TO_CPU_MS", DEFAULT_DELAY_GPU_TO_CPU_MS),
    ("cpu", "disk"): _env_nonneg_int("MEMLORA_DELAY_CPU_TO_DISK_MS", DEFAULT_DELAY_CPU_TO_DISK_MS),
    ("disk", "cpu"): _env_nonneg_int("MEMLORA_DELAY_DISK_TO_CPU_MS", DEFAULT_DELAY_DISK_TO_CPU_MS),
    ("gpu", "disk"): _env_nonneg_int("MEMLORA_DELAY_GPU_TO_DISK_MS", DEFAULT_DELAY_GPU_TO_DISK_MS),
}


def load_peer_config():
    with open(PEERS_FILE) as f:
        cfg = json.load(f)
    return cfg["my_ip"], cfg["peers"]


def get_lora_names():
    return sorted([
        name for name in os.listdir(ADAPTER_PATH)
        if os.path.isdir(os.path.join(ADAPTER_PATH, name))
    ])
