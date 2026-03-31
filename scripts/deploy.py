#!/usr/bin/env python3

import os, time, requests, uuid, asyncio, json
from collections import OrderedDict
import ray
from ray import serve
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

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

        # TODO: LRU tracker for local GPU/CPU adapter state
        # TODO: peer state cache

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

    # Routing Logic 

    def _choose_target_node(self, adapter_name: str, source_ip: str) -> str:
        # TODO: use peer state + LRU tracker to pick best node
        return self.my_ip

    # Request Parsing Functions

    def _extract_client_ip(self, request: Request, body: dict) -> str:
        if body.get("_client_ip"):
            return body["_client_ip"]
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    #extract sender IP, which may be different from client IP if request is forwarded from another node
    def _extract_sender_ip(self, request: Request, body: dict) -> str:
        if body.get("_sender_ip"):
            return body["_sender_ip"]
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    # Parse the 'model' field to extract base model and optional adapter, validating the format
    def _parse_model_and_adapter(self, model_name: str):
        if not model_name:
            raise ValueError("Missing 'model' field")
        if model_name == self.model_id:
            return self.model_id, None
        prefix = f"{self.model_id}/"
        if model_name.startswith(prefix):
            adapter_name = model_name[len(prefix):]
            if not adapter_name:
                raise ValueError("Missing adapter name after model prefix")
            return self.model_id, adapter_name
        raise ValueError(f"Unsupported model '{model_name}'")

    # Extract prompt text from either 'messages' or 'prompt' field, supporting both chat and completion formats
    def _extract_prompt(self, body: dict) -> dict:
        if "messages" in body:
            messages = body["messages"]
            if not isinstance(messages, list):
                raise ValueError("'messages' must be a list")
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    prompt_parts.append(content if isinstance(content, str) else str(content))
            return {"messages": messages, "prompt": "\n".join(prompt_parts).strip()}
        if "prompt" in body:
            prompt = body["prompt"]
            return {"messages": None, "prompt": prompt if isinstance(prompt, str) else str(prompt)}
        raise ValueError("Request must include either 'messages' or 'prompt'")

    #extract and parse the inference request, returning a dict with all relevant info
    async def _parse_inference_request(self, request: Request) -> dict:
        body = await request.json()
        model_name = body.get("model")
        base_model, adapter_name = self._parse_model_and_adapter(model_name)
        prompt_info = self._extract_prompt(body)
        return {
            "request_id":   body.get("request_id", str(uuid.uuid4())),
            "model":        model_name,
            "base_model":   base_model,
            "adapter_name": adapter_name,
            "prompt":       prompt_info["prompt"],
            "messages":     prompt_info["messages"],
            "client_ip":    self._extract_client_ip(request, body),
            "sender_ip":    self._extract_sender_ip(request, body),
            "forward_path": body.get("_forward_path", []),
            "raw_body":     body,
        }

    # Inference Handling Functions

    # serving the request locally
    async def _serve_local_chat_request(self, parsed: dict) -> JSONResponse:
        body         = parsed["raw_body"]
        adapter_name = parsed["adapter_name"]

        sampling_params = SamplingParams(
            max_tokens=body.get("max_tokens", 64),
            temperature=body.get("temperature", 0.0),
            top_p=body.get("top_p", 1.0),
        )

        lora_request = None
        if adapter_name is not None:
            lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=abs(hash(adapter_name)) % (2**31),
                lora_local_path=os.path.join(ADAPTER_PATH, adapter_name),
            )

        self._ongoing += 1
        try:
            final_output = None
            async for output in self.engine.generate(
                parsed["prompt"],
                sampling_params,
                request_id=parsed["request_id"],
                lora_request=lora_request,
            ):
                final_output = output

            if final_output is None or not final_output.outputs:
                return JSONResponse({"error": "No output generated"}, status_code=500)

            return JSONResponse({
                "id":           parsed["request_id"],
                "object":       "chat.completion",
                "model":        parsed["model"],
                "choices": [{
                    "index":         0,
                    "message":       {"role": "assistant", "content": final_output.outputs[0].text},
                    "finish_reason": final_output.outputs[0].finish_reason,
                }],
                "served_by":    self.my_ip,
                "adapter_name": adapter_name,
            })
        except Exception as e:
            return JSONResponse({"error": f"Inference failed: {str(e)}"}, status_code=500)
        finally:
            self._ongoing -= 1

    # Forward the request to another node and return its response, handling any network errors gracefully
    async def _forward_chat_request(self, target_ip: str, body: dict) -> JSONResponse:
        import aiohttp
        url = f"http://{target_ip}:8000/internal/chat/completions"
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    url, json=body,
                    timeout=aiohttp.ClientTimeout(total=180),
                )
                return JSONResponse(content=await resp.json(), status_code=resp.status)
        except Exception as e:
            return JSONResponse({"error": f"Forward to {target_ip} failed: {str(e)}"}, status_code=502)

    # internal endpoint for gossip messages from other nodes about their state.
    @app.post("/internal/gossip")
    async def receive_gossip(self, request: Request):
        # TODO: parse and store peer state
        body = await request.json()
        return JSONResponse({"ok": True})

    # internal endpoint for forwarded requests from other nodes in the cluster. 
    @app.post("/internal/chat/completions")
    async def internal_chat_completions(self, request: Request):
        try:
            parsed = await self._parse_inference_request(request)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return await self._serve_local_chat_request(parsed)

    # main public endpoint for chat completions
    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        try:
            parsed = await self._parse_inference_request(request)
        except (ValueError, Exception) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        target_ip = self._choose_target_node(parsed["adapter_name"], parsed["client_ip"])

        if target_ip == self.my_ip:
            return await self._serve_local_chat_request(parsed)

        body = dict(parsed["raw_body"])
        body["_client_ip"]    = parsed["client_ip"]
        body["_sender_ip"]    = self.my_ip
        body["_forward_path"] = parsed["forward_path"] + [self.my_ip]
        return await self._forward_chat_request(target_ip, body)

    # get the models 
    @app.get("/v1/models")
    async def list_models(self):
        models = [{"id": self.model_id, "object": "model"}]
        for name in self.lora_names:
            models.append({"id": f"{self.model_id}/{name}", "object": "model"})
        return JSONResponse({"object": "list", "data": models})

    @app.get("/health")
    async def health(self):
        return JSONResponse({"status": "ok", "node": self.my_ip, "ongoing": self._ongoing})


if __name__ == "__main__":
    ray.init()

    print(f"Cluster resources: {ray.cluster_resources()}")

    #delete old deployment if it exists. 
    try:
        serve.start()
        serve.delete("memlora")
        time.sleep(2)
    except Exception:
        pass

    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    handle = VLLMDeployment.bind()
    serve.run(handle, name="memlora", route_prefix="/", blocking=False)

    print("\nWaiting for deployment to be ready...")
    for i in range(60):
        try:
            r = requests.get("http://localhost:8000/health", timeout=5)
            if r.status_code == 200:
                print(f"\nReady! {r.json()}")
                break
        except Exception:
            pass
        time.sleep(60)
        print(f"  waiting... {i+1}/60 (~{(i+1)*10}s elapsed)")
    else:
        print("Did not become ready in time.")
        exit(1)