import os
import json
import logging

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


def load_peer_config():
    with open(PEERS_FILE) as f:
        cfg = json.load(f)
    return cfg["my_ip"], cfg["peers"]


def get_lora_names():
    return sorted([
        name for name in os.listdir(ADAPTER_PATH)
        if os.path.isdir(os.path.join(ADAPTER_PATH, name))
    ])
