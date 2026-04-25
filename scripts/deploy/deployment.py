import os
import time
import asyncio
import traceback
import shutil
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from .config import (
    LOG_DIR, MODEL_PATH, ADAPTER_PATH,
    MAX_GPU_LORA, MAX_CPU_LORA, SERVE_PORT,
    ROUTING_MODE, logger,
    load_peer_config, get_lora_names,
)
from .metrics import MetricsLogger
from .lru import LRUMixin
from .routing import RoutingMixin
from .gossip import GossipMixin
from .parsing import ParsingMixin
from .inference import InferenceMixin


class MemLoRAEngine(LRUMixin, RoutingMixin, GossipMixin, ParsingMixin, InferenceMixin):
    def __init__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        self.my_ip, self.peer_ips = load_peer_config()
        self.lora_names = get_lora_names()
        self.model_id = "qwen-base"
        self._ongoing = 0

        self.metrics = MetricsLogger(self.my_ip)

        self._peer_queue_lengths: dict[str, int] = {
            ip: 0 for ip in self.peer_ips if ip != self.my_ip
        }
        self._peer_queue_timestamps: dict[str, float] = {
            ip: 0.0 for ip in self.peer_ips if ip != self.my_ip
        }
        self._peer_adapter_state: dict[str, dict[str, set]] = {
            name: {"gpu": set(), "cpu": set(), "disk": set()}
            for name in self.lora_names
        }
        for name in self.lora_names:
            self._peer_adapter_state[name]["disk"].add(self.my_ip)
        self._adapter_state_timestamps: dict[tuple, float] = {}

        self._gossip_task = None
        self._gossip_running = False
        self._aiohttp_session = None

        self._local_gpu_lru: OrderedDict[str, None] = OrderedDict()
        self._local_cpu_lru: OrderedDict[str, None] = OrderedDict()

        logger.info(f"[vllm] Node: {self.my_ip}")
        logger.info(f"[vllm] Peers: {[p for p in self.peer_ips if p != self.my_ip]}")
        logger.info(f"[vllm] Loading model: {MODEL_PATH}")
        logger.info(f"[vllm] LoRA adapters ({len(self.lora_names)}): {self.lora_names}")

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
        logger.info("[vllm] Engine ready.")

    def _get_local_disk_adapters(self) -> list[str]:
        return [
            name for name in self.lora_names
            if name not in self._local_gpu_lru
            and name not in self._local_cpu_lru
            and os.path.isdir(os.path.join(ADAPTER_PATH, name))
        ]

    async def _stop_gossip_loop(self):
        if self._gossip_task and not self._gossip_task.done():
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
        self._gossip_running = False

        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()


engine_node: MemLoRAEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_node
    engine_node = MemLoRAEngine()
    engine_node._start_gossip_loop()
    yield
    await engine_node._stop_gossip_loop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Internal endpoints ────────────────────────────────────────────────────────

@app.post("/internal/gossip")
async def receive_gossip(request: Request):
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    body = await request.json()
    msg_type = body.get("type")
    if msg_type == "queue_length":
        engine_node._handle_queue_gossip(body)
    elif msg_type == "adapter_state":
        engine_node._handle_adapter_state_gossip(body)
    return JSONResponse({"ok": True})


@app.get("/internal/queue")
async def get_queue_length():
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    return JSONResponse({
        "node": engine_node.my_ip,
        "queue_len": engine_node._ongoing,
        "ts": time.time(),
    })


@app.post("/internal/chat/completions")
async def internal_chat_completions(request: Request):
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    logger.info(
        f"[endpoint] /internal/chat/completions from "
        f"{request.client.host if request.client else 'unknown'}"
    )
    try:
        parsed = await engine_node._parse_inference_request(request)
    except ValueError as e:
        logger.error(f"[endpoint] /internal/chat/completions parse error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)
    return await engine_node._serve_local_chat_request(parsed)


# ── Public endpoints ──────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    e2e_start = time.perf_counter()
    logger.info(
        f"[endpoint] /v1/chat/completions from "
        f"{request.client.host if request.client else 'unknown'}"
    )

    try:
        parsed = await engine_node._parse_inference_request(request)
    except Exception as e:
        logger.error(f"[endpoint] parse error: {e}\n{traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=400)

    request_id = parsed["request_id"]
    adapter_name = parsed["adapter_name"]

    routing_start = time.perf_counter()
    if ROUTING_MODE == "memory":
        target_ip = engine_node._choose_target_node_memory(adapter_name, parsed["client_ip"])
    else:
        target_ip = engine_node._choose_target_node_baseline(adapter_name, parsed["client_ip"])
    routing_time_ms = (time.perf_counter() - routing_start) * 1000

    is_local = target_ip == engine_node.my_ip
    reason = "local" if is_local else "forwarded"
    logger.info(
        f"[routing] request_id={request_id} adapter={adapter_name} "
        f"target={target_ip} reason={reason}"
    )
    engine_node.metrics.log(
        "routing_decision",
        request_id=request_id,
        adapter=adapter_name,
        target_node=target_ip,
        reason=reason,
        decision_time_ms=routing_time_ms,
    )

    if is_local:
        response = await engine_node._serve_local_chat_request(parsed)
    else:
        body = dict(parsed["raw_body"])
        body["_client_ip"] = parsed["client_ip"]
        body["_sender_ip"] = engine_node.my_ip
        body["_forward_path"] = parsed["forward_path"] + [engine_node.my_ip]
        response = await engine_node._forward_chat_request(target_ip, body, request_id)

    e2e_time_ms = (time.perf_counter() - e2e_start) * 1000
    logger.info(
        f"[e2e] request_id={request_id} status={response.status_code} "
        f"total_ms={e2e_time_ms:.1f} forwarded={not is_local} served_by={target_ip}"
    )
    engine_node.metrics.log(
        "e2e_latency",
        request_id=request_id,
        total_ms=e2e_time_ms,
        was_forwarded=not is_local,
        served_by=target_ip,
    )
    return response


@app.get("/v1/models")
async def list_models():
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    models = [{"id": engine_node.model_id, "object": "model"}]
    for name in engine_node.lora_names:
        models.append({"id": f"{engine_node.model_id}/{name}", "object": "model"})
    return JSONResponse({"object": "list", "data": models})


@app.get("/health")
async def health():
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    if engine_node._gossip_task is None and not engine_node._gossip_running:
        engine_node._gossip_running = True
        engine_node._gossip_task = asyncio.create_task(engine_node._gossip_queue_loop())
        logger.info("[gossip] Started gossip loop lazily on first health check")
    return JSONResponse({
        "status": "ok", "node": engine_node.my_ip, "ongoing": engine_node._ongoing,
    })


# ── Debug endpoints ───────────────────────────────────────────────────────────

@app.get("/internal/debug/state")
async def debug_state():
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    return JSONResponse({
        "node": engine_node.my_ip,
        "local_queue": engine_node._ongoing,
        "peer_queues": engine_node._peer_queue_lengths,
        "peer_timestamps": engine_node._peer_queue_timestamps,
        "local_adapters": {
            "gpu": list(engine_node._local_gpu_lru.keys()),
            "cpu": list(engine_node._local_cpu_lru.keys()),
            "disk": engine_node._get_local_disk_adapters(),
        },
        "adapter_state": {
            adapter: {tier: list(nodes) for tier, nodes in tiers.items()}
            for adapter, tiers in engine_node._peer_adapter_state.items()
        },
        "gossip_running": engine_node._gossip_running,
        "gossip_task_active": (
            engine_node._gossip_task is not None and not engine_node._gossip_task.done()
        ),
    })


@app.post("/internal/debug/reset_cache")
async def reset_cache(request: Request):
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    import aiohttp
    data = await request.json()
    adapters = data.get("adapters", [])
    fanout = data.get("fanout", False)
    
    reset_results = {}
    for adapter in adapters:
        if adapter in engine_node._peer_adapter_state:
            # Clear from all tiers
            for tier in ["gpu", "cpu", "disk", "s3"]:
                engine_node._peer_adapter_state[adapter][tier].clear()
            # S3 remains empty
        # Clear local caches
        engine_node._local_gpu_lru.pop(adapter, None)
        engine_node._local_cpu_lru.pop(adapter, None)
        adapter_path = os.path.join(ADAPTER_PATH, adapter)
        if os.path.isdir(adapter_path):
            shutil.rmtree(adapter_path)
            reset_results[adapter] = "cleared"
        else:
            reset_results[adapter] = "not_present"
    
    if fanout:
        # Send to peers
        async def reset_peer(peer_ip):
            try:
                session = await engine_node._ensure_session()
                async with session.post(
                    f"http://{peer_ip}:{SERVE_PORT}/internal/debug/reset_cache",
                    json={"adapters": adapters, "fanout": False},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return await resp.json()
            except Exception as e:
                return {"error": str(e)}
        
        tasks = [reset_peer(ip) for ip in engine_node.peer_ips if ip != engine_node.my_ip]
        peer_results = await asyncio.gather(*tasks, return_exceptions=True)
        reset_results["peers"] = peer_results
    
    return JSONResponse({"node": engine_node.my_ip, "reset_results": reset_results})


@app.get("/internal/logs")
async def get_logs(lines: int = 80):
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    log_path = os.path.join(LOG_DIR, "deploy.log")
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return JSONResponse({"node": engine_node.my_ip, "lines": [l.rstrip() for l in tail]})
    except FileNotFoundError:
        return JSONResponse({"node": engine_node.my_ip, "lines": []})


@app.get("/internal/cluster")
async def cluster_state():
    if engine_node is None:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    import aiohttp

    nodes = {}
    log_lines = []
    log_path = os.path.join(LOG_DIR, "deploy.log")
    try:
        with open(log_path, "r") as f:
            raw = f.readlines()
        log_lines = [l.rstrip() for l in raw[-80:]]
    except Exception:
        pass

    nodes[engine_node.my_ip] = {
        "node": engine_node.my_ip,
        "status": "ok",
        "local_queue": engine_node._ongoing,
        "peer_queues": dict(engine_node._peer_queue_lengths),
        "peer_timestamps": dict(engine_node._peer_queue_timestamps),
        "local_adapters": {
            "gpu": list(engine_node._local_gpu_lru.keys()),
            "cpu": list(engine_node._local_cpu_lru.keys()),
            "disk": engine_node._get_local_disk_adapters(),
        },
        "adapter_state": {
            a: {t: list(n) for t, n in tiers.items()}
            for a, tiers in engine_node._peer_adapter_state.items()
        },
        "gossip_running": engine_node._gossip_running,
        "gossip_task_active": (
            engine_node._gossip_task is not None and not engine_node._gossip_task.done()
        ),
        "logs": log_lines,
    }

    async def fetch_peer(peer_ip):
        try:
            session = await engine_node._ensure_session()
            async with session.get(
                f"http://{peer_ip}:{SERVE_PORT}/internal/debug/state",
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                state = await resp.json()
            async with session.get(
                f"http://{peer_ip}:{SERVE_PORT}/internal/logs?lines=80",
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                logs_data = await resp.json()
            state["status"] = "ok"
            state["logs"] = logs_data.get("lines", [])
            return peer_ip, state
        except Exception as e:
            return peer_ip, {
                "node": peer_ip, "status": "unreachable",
                "local_queue": None, "peer_queues": {},
                "local_adapters": {"gpu": [], "cpu": []},
                "gossip_running": False, "gossip_task_active": False,
                "adapter_state": {}, "logs": [], "error": str(e),
            }

    tasks = [fetch_peer(ip) for ip in engine_node.peer_ips if ip != engine_node.my_ip]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, tuple):
            ip, data = result
            nodes[ip] = data

    return JSONResponse({"ts": time.time(), "source_node": engine_node.my_ip, "nodes": nodes})