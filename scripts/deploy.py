#!/usr/bin/env python3

import os, time, requests, uuid
import ray
from ray import serve
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse

HOME         = os.path.expanduser("~")
MODEL_PATH   = f"{HOME}/model_cache/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = f"{HOME}/adapters"

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

        self.lora_names = get_lora_names()
        self.model_id   = "qwen-base"

        print(f"[vllm] Loading model: {MODEL_PATH}")
        print(f"[vllm] LoRA adapters ({len(self.lora_names)}): {self.lora_names}")

        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            enable_lora=True,
            max_loras=3,
            max_cpu_loras=6,
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

    @app.get("/v1/models")
    async def list_models(self):
        models = [{"id": self.model_id, "object": "model"}]
        for name in self.lora_names:
            models.append({"id": f"{self.model_id}/{name}", "object": "model"})
        return JSONResponse({"object": "list", "data": models})

    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import SamplingParams

        body        = await request.json()
        model       = body.get("model", self.model_id)
        messages    = body.get("messages", [])
        max_tokens  = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)

        lora_request = None
        if "/" in model:
            _, adapter_name = model.split("/", 1)
            if adapter_name not in self.lora_names:
                return JSONResponse(
                    {"error": f"Unknown adapter: {adapter_name}"},
                    status_code=400
                )
            lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=abs(hash(adapter_name)) % (2**31),
                lora_local_path=os.path.join(ADAPTER_PATH, adapter_name),
            )

        tokenizer = await self.engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        request_id  = f"chatcmpl-{uuid.uuid4().hex}"
        final_output = None
        async for output in self.engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
        ):
            final_output = output

        text = final_output.outputs[0].text

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": final_output.outputs[0].finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
            }
        })

    @app.get("/health")
    async def health(self):
        return JSONResponse({"status": "ok", "adapters": self.lora_names})


if __name__ == "__main__":
    ray.init(address="auto")

    alive = [n for n in ray.nodes() if n["Alive"]]
    print(f"Cluster resources: {ray.cluster_resources()}")
    print(f"Nodes alive: {len(alive)}/4")

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
