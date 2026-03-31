#!/usr/bin/env python3

import os, time, requests, uuid, asyncio, json
from collections import OrderedDict
import ray
from ray import serve
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse

HOME         = os.path.expanduser("~")
MODEL_PATH   = f"{HOME}/model_cache/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = f"{HOME}/adapters"
PEERS_FILE   = os.path.expanduser("~/peers.json")

MAX_GPU_LORA = 3
MAX_CPU_LORA = 6

def load_peer_config():
    with open(PEERS_FILE) as f:
        cfg = json.load(f)
    return cfg["my_ip"], cfg["peers"]

def get_lora_names():
    return sorted([
        name for name in os.listdir(ADAPTER_PATH)
        if os.path.isdir(os.path.join(ADAPTER_PATH, name))
    ])

app = FastAPI()

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1, "num_cpus": 8},
    max_ongoing_requests=8,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        self.my_ip, self.peer_ips = load_peer_config()
        self.lora_names = get_lora_names()
        self.model_id   = "qwen-base"
        self._ongoing   = 0 

        # DS that is a lru tracker for this nodes gpu cpu state


        # DS to keep track of peer states 

        print(f"[vllm] Node: {self.my_ip}")
        print(f"[vllm] Peers: {[p for p in self.peer_ips if p != self.my_ip]}")
        print(f"[vllm] Loading model: {MODEL_PATH}")
        print(f"[vllm] LoRA adapters ({len(self.lora_names)}): {self.lora_names}")

        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            enable_lora=True,
            max_loras=MAX_GPU_LORA,
            max_cpu_loras=MAX_CPU_LORA,
            max_lora_rank=8,
            gpu_memory_utilization=0.6,
            max_model_len=512,
            dtype="float16",
            tensor_parallel_size=1,
            trust_remote_code=False,
            disable_log_stats=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print(f"[vllm] Engine ready.")


    # gossip endpoint for state sharing between nodes 
    @app.post("/internal/gossip")
    async def receive_gossip(self, request: Request):
        pass

    # inference endpoint 
    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        pass

    # list models and adapters
    @app.get("/v1/models")
    async def list_models(self):
        models = [{"id": self.model_id, "object": "model"}]
        for name in self.lora_names:
            models.append({"id": f"{self.model_id}/{name}", "object": "model"})
        return JSONResponse({"object": "list", "data": models})

    # health check endpoint
    @app.get("/health")
    async def health(self):
        return JSONResponse({
            "status":   "ok",
        })


if __name__ == "__main__":
    ray.init()

    print(f"Cluster resources: {ray.cluster_resources()}")

    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    handle = VLLMDeployment.bind()
    serve.run(handle, name="memlora", blocking=False)

    print("\nWaiting for deployment to be ready...")
    for i in range(60):
        try:
            r = requests.get("http://localhost:8000/health", timeout=5)
            if r.status_code == 200:
                print(f"\nReady! {r.json()}")
                break
        except Exception:
            pass
        time.sleep(10)
        print(f"  waiting... {i+1}/60 (~{(i+1)*10}s elapsed)")
    else:
        print("Did not become ready in time.")
        exit(1)