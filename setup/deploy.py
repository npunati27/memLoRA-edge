#!/usr/bin/env python3
# =============================================================================
# deploy.py - Run on node0 ONLY after setup.sh on all nodes
# Usage: python3 deploy.py
# =============================================================================
import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
import requests, time

ray.init(address="auto")

print(f"Cluster resources: {ray.cluster_resources()}")
print(f"Nodes alive: {len([n for n in ray.nodes() if n['Alive']])}/4")

serve.start(http_options={"host": "0.0.0.0", "port": 8000})

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-base",
        model_source="/root/model_cache/Qwen2.5-0.5B-Instruct",
    ),
    lora_config=dict(
        dynamic_lora_loading_path="/root/adapters",
        # With 25 adapters and only 3 VRAM slots:
        # ~22 adapters are always cold on any given node
        # This creates constant eviction pressure
        max_num_adapters_per_replica=3,
    ),
    engine_kwargs=dict(
        enable_lora=True,
        max_loras=3,          # VRAM slots — deliberately tight
        max_cpu_loras=6,      # CPU RAM slots — also limited
        max_lora_rank=8,
        gpu_memory_utilization=0.6,
        max_model_len=512,
        dtype="float16",
    ),
    deployment_config=dict(
        num_replicas=4,
        max_ongoing_requests=4,
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, name="memlora", blocking=False)

print("Waiting for deployment to be ready...")
for _ in range(30):
    try:
        r = requests.get("http://localhost:8000/v1/models", timeout=5)
        if r.status_code == 200:
            print("Ready!")
            break
    except:
        pass
    time.sleep(10)
    print("  still waiting...")

print("""
Deployment running.

Test:
  curl http://localhost:8000/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"qwen-base/crop_corn_disease",
         "messages":[{"role":"user","content":"Describe corn disease."}],
         "max_tokens":20}'

Run load test:
  python3 load_test.py
""")