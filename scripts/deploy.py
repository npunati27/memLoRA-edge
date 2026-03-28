#!/usr/bin/env python3
# Requires: vllm==0.8.5, ray[serve,llm]==2.47.1, torch==2.5.1, transformers==4.51.3

import os, time, requests
import ray
from ray import serve
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse

HOME         = os.path.expanduser("~")
MODEL_PATH   = f"{HOME}/model_cache/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = f"{HOME}/adapters"

def get_lora_modules():
    """Return list of 'name=path' strings for every adapter directory."""
    modules = []
    for name in sorted(os.listdir(ADAPTER_PATH)):
        path = os.path.join(ADAPTER_PATH, name)
        if os.path.isdir(path):
            modules.append(f"{name}={path}")
    return modules

app = FastAPI()

@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 8,
    },
    max_ongoing_requests=8,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        lora_modules = get_lora_modules()
        print(f"[vllm] Loading model: {MODEL_PATH}")
        print(f"[vllm] LoRA adapters ({len(lora_modules)}): {[m.split('=')[0] for m in lora_modules]}")

        # vllm 0.8.5 AsyncEngineArgs — lora_modules is valid here
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            enable_lora=True,
            lora_modules=lora_modules,       # "name=path" strings, valid in 0.8.5
            max_loras=3,                     # VRAM slots
            max_cpu_loras=6,                 # CPU RAM slots
            max_lora_rank=8,
            gpu_memory_utilization=0.6,
            max_model_len=512,
            dtype="float16",
            tensor_parallel_size=1,
            trust_remote_code=False,
            disable_log_requests=True,       # valid in 0.8.5
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.model_id = "qwen-base"
        self.lora_names = [m.split("=")[0] for m in lora_modules]
        print(f"[vllm] Engine ready with {len(self.lora_names)} adapters.")

    def _get_chat_handler(self):
        """Lazily build OpenAIServingChat (needs the running event loop)."""
        if not hasattr(self, "_chat_handler"):
            from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
            from vllm.entrypoints.openai.serving_engine import LoRAModulePath

            lora_module_paths = [
                LoRAModulePath(name=n, path=f"{ADAPTER_PATH}/{n}")
                for n in self.lora_names
            ]
            self._chat_handler = OpenAIServingChat(
                engine=self.engine,
                served_model_names=[self.model_id],
                response_role="assistant",
                lora_modules=lora_module_paths,
                prompt_adapters=None,
                request_logger=None,
                chat_template=None,
            )
        return self._chat_handler

    @app.get("/v1/models")
    async def list_models(self):
        models = [{"id": self.model_id, "object": "model"}]
        for name in self.lora_names:
            models.append({"id": f"{self.model_id}/{name}", "object": "model"})
        return JSONResponse({"object": "list", "data": models})

    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
        from vllm.lora.request import LoRARequest

        body = await request.json()

        # Parse model field: "qwen-base/crop_corn_disease" -> pick adapter
        model = body.get("model", self.model_id)
        lora_request = None

        if "/" in model:
            base, adapter_name = model.split("/", 1)
            if adapter_name in self.lora_names:
                lora_request = LoRARequest(
                    lora_name=adapter_name,
                    lora_int_id=abs(hash(adapter_name)) % (2**31),
                    lora_local_path=f"{ADAPTER_PATH}/{adapter_name}",
                )
                body["model"] = base

        req = ChatCompletionRequest(**body)

        if lora_request:
            req.model = f"{self.model_id}/{lora_request.lora_name}"

        handler = self._get_chat_handler()
        raw_body = await request.body()
        generator = await handler.create_chat_completion(req, raw_body)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.dict(), status_code=generator.code)

        if req.stream:
            from starlette.responses import StreamingResponse
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            return JSONResponse(content=generator.dict())

    @app.post("/v1/load_lora_adapter")
    async def load_lora(self, request: Request):
        """Dynamically register a new adapter name (sidecar use)."""
        body = await request.json()
        name = body.get("lora_name")
        if name and name not in self.lora_names:
            self.lora_names.append(name)
        return JSONResponse({"status": "ok", "lora_name": name})

    @app.get("/health")
    async def health(self):
        return JSONResponse({"status": "ok", "adapters": self.lora_names})


if __name__ == "__main__":
    ray.init(address="auto")

    print(f"Cluster resources: {ray.cluster_resources()}")
    alive = [n for n in ray.nodes() if n["Alive"]]
    print(f"Nodes alive: {len(alive)}/4")
    print("Node IPs:")
    for n in alive:
        print(f"  - {n['NodeManagerAddress']}")

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
        print("Did not become ready in time. Check logs with: ray logs --tail 100")
        exit(1)

    print("""
Test:
  curl http://localhost:8000/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"qwen-base/crop_corn_disease",
         "messages":[{"role":"user","content":"Describe corn disease."}],
         "max_tokens":20}'

SSH tunnel from laptop:
  ssh -L 8000:10.10.1.1:8000 <user>@<node0-public-ip>
""")