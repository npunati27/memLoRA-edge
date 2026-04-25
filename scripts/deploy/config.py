import os
import json
import logging

HOME = os.path.expanduser("~")
LOG_DIR = os.path.join(HOME, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ROUTING_MODE = os.getenv("ROUTING_MODE", "baseline").strip().lower()
TIER_RANK = {"gpu": 0, "cpu": 1, "disk": 2, "s3": 3}

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

S3_BUCKET = os.getenv("S3_BUCKET", "memlora-adapters-525")
S3_REGION = os.getenv("S3_REGION", "us-east-2")
S3_PREFIX_ROOT = os.getenv("S3_PREFIX_ROOT", "adapters")
USE_S3_ADAPTERS = os.getenv("USE_S3_ADAPTERS", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
EXPECTED_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
}

def load_peer_config():
    with open(PEERS_FILE) as f:
        cfg = json.load(f)
    return cfg["my_ip"], cfg["peers"]


def get_lora_names():
    """Get list of available LoRA adapter names."""
    if USE_S3_ADAPTERS:
        try:
            from .s3_adapter import list_adapters_from_s3
            adapters = list_adapters_from_s3()
            logger.info(f"[config] Loaded {len(adapters)} adapters from S3")
            return adapters
        except Exception as e:
            logger.warning(f"[config] S3 adapter loading failed, falling back to local: {e}")

    if not os.path.isdir(ADAPTER_PATH):
        return []
    
    return sorted([
        name for name in os.listdir(ADAPTER_PATH)
        if os.path.isdir(os.path.join(ADAPTER_PATH, name))
    ])
