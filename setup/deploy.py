#!/usr/bin/env python3
# =============================================================================
# deploy.py - Run on node0 ONLY after setup.sh on all nodes
# Usage: python3 deploy.py
# =============================================================================

import os
import time
import requests
import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

# ── Init Ray ──────────────────────────────────────────────────────────────────
ray.init(address="auto")

print(f"\nCluster resources: {ray.cluster_resources()}")

alive_nodes = [n for n in ray.nodes() if n["Alive"]]
print(f"Nodes alive: {len(alive_nodes)}/4")

print("\nNode IPs (should ALL be 10.10.x.x):")
for n in alive_nodes:
    print(f"  - {n['NodeManagerAddress']}")

# ── Start Serve ───────────────────────────────────────────────────────────────
serve.start(
    http_options={
        "host": "0.0.0.0",  # accessible via SSH tunnel
        "port": 8000,
    }
)

# ── Paths (FIXED) ─────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")

MODEL_PATH = f"{HOME}/model_cache/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = f"{HOME}/adapters"

print(f"\nModel path: {MODEL_PATH}")
print(f"Adapter path: {ADAPTER_PATH}")

# ── LLM Config ────────────────────────────────────────────────────────────────
# llm_config = LLMConfig(
#     model_loading_config=dict(
#         model_id="qwen-base",
#         model_source=MODEL_PATH,
#     ),
#     lora_config=dict(
#         dynamic_lora_loading_path=ADAPTER_PATH,

#         # VERY IMPORTANT: forces constant eviction + cold loads
#         max_num_adapters_per_replica=3,
#     ),
#     engine_kwargs=dict(
#         enable_lora=True,
#         max_loras=3,          # GPU slots (tight)
#         max_cpu_loras=6,      # CPU cache (also limited)
#         max_lora_rank=8,
#         gpu_memory_utilization=0.6,
#         max_model_len=512,
#         dtype="float16",
#     ),
#     deployment_config=dict(
#         num_replicas=4,

#         # Prevent overload (keeps behavior stable for experiment)
#         max_ongoing_requests=4,

#         # CRITICAL: ensures 1 replica per node (since 1 GPU per node)
#         ray_actor_options={
#             "num_gpus": 1,
#         },
#     ),
# )

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-base",
        model_source=MODEL_PATH,
    ),
    # NO lora_config here
    engine_kwargs=dict(
        enable_lora=True,
        max_loras=3,
        max_cpu_loras=6,
        max_lora_rank=8,
        gpu_memory_utilization=0.6,
        max_model_len=512,
        dtype="float16",
        # Pass adapters directly as LoRA modules
        lora_modules=[
            f"{name}={ADAPTER_PATH}/{name}"
            for name in os.listdir(ADAPTER_PATH)
            if os.path.isdir(f"{ADAPTER_PATH}/{name}")
        ],
    ),
    deployment_config=dict(
        num_replicas=4,
        max_ongoing_requests=4,
        ray_actor_options={
            "num_gpus": 1,
        },
    ),
)

# ── Deploy ────────────────────────────────────────────────────────────────────
print("\nDeploying application...")

app = build_openai_app({"llm_configs": [llm_config]})

serve.run(app, name="memlora", blocking=False)

# ── Wait for readiness ─────────────────────────────────────────────────────────
print("Waiting for deployment to be ready...")

ready = False
for i in range(30):
    try:
        r = requests.get("http://localhost:8000/v1/models", timeout=5)
        if r.status_code == 200:
            print("✅ Deployment READY!")
            ready = True
            break
    except Exception:
        pass

    time.sleep(5)
    print(f"  still waiting... ({i+1}/30)")

if not ready:
    print("❌ Deployment did not become ready in time.")
    exit(1)

# ── Instructions ──────────────────────────────────────────────────────────────
print("""
============================================================
🚀 Deployment running

Test locally (on node0):

curl http://localhost:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model":"qwen-base/crop_corn_disease",
    "messages":[{"role":"user","content":"Describe corn disease."}],
    "max_tokens":20
  }'

From your laptop (recommended):
ssh -L 8000:10.10.1.1:8000 <user>@<node0-public-ip>

Then open:
http://localhost:8000

============================================================
""")
