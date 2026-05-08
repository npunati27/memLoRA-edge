from __future__ import annotations

import os
import time
import asyncio
import traceback
import shutil
import random
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from .bloom import BloomFilter
from .config import (
    LOG_DIR, MODEL_PATH, ADAPTER_PATH,
    MAX_GPU_LORA, MAX_CPU_LORA, SERVE_PORT,
    FORWARD_MAX_ATTEMPTS, FORWARD_FAILURE_COOLDOWN_S, FORWARD_RETRY_TIMEOUT_S,
    FORWARD_TIMEOUT_S,
    ROUTING_MODE, USE_MOCK_ENGINE, logger,
    load_peer_config, get_lora_names, TIER_RANK,
)
from .metrics import MetricsLogger
from .lru import LRUMixin
from .routing import RoutingMixin
from .gossip import GossipMixin
from .parsing import ParsingMixin
from .inference import InferenceMixin
from .probe import ProbeMixin


class MemLoRAEngine(LRUMixin, RoutingMixin, GossipMixin, ParsingMixin, InferenceMixin, ProbeMixin):
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
        self._peer_presence_blooms: dict[str, BloomFilter] = {}
        self._sync_all_peer_presence_blooms()

        self._gossip_task = None
        self._gossip_running = False
        self._aiohttp_session = None

        self._local_gpu_lru: OrderedDict[str, None] = OrderedDict()
        self._local_cpu_lru: OrderedDict[str, None] = OrderedDict()

        self._measured_rtt: dict[str, float] = {
            ip: 50.0
            for ip in self.peer_ips if ip != self.my_ip
        }
        self._probe_failures: dict[str, int] = {
            ip: 0
            for ip in self.peer_ips if ip != self.my_ip
        }
        self._forward_failure_deadlines: dict[str, float] = {}

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


def create_app(engine_class: Callable[[], Any] | None = None) -> FastAPI:
    """Build the FastAPI app. Routes are identical for GPU and mock; only the engine differs.

    If ``engine_class`` is omitted, ``MEMLORA_MOCK=1`` selects :class:`MockMemLoRAEngine`;
    otherwise :class:`MemLoRAEngine` (vLLM). Pass a class explicitly to override env (tests).
    """
    if engine_class is None:
        if USE_MOCK_ENGINE:
            from .mock_engine import MockMemLoRAEngine

            engine_class = MockMemLoRAEngine
        else:
            engine_class = MemLoRAEngine

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine_class()
        app.state.engine._start_gossip_loop()
        app.state.engine._start_probe_loop()
        yield
        await app.state.engine._stop_gossip_loop()
        await app.state.engine._stop_probe_loop()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    def _prune_forward_cooldowns(e):
        now = time.time()
        stale = [ip for ip, until in e._forward_failure_deadlines.items() if until <= now]
        for ip in stale:
            del e._forward_failure_deadlines[ip]

    def _mark_forward_failure(e, target_ip: str, reason: str):
        until = time.time() + FORWARD_FAILURE_COOLDOWN_S
        e._forward_failure_deadlines[target_ip] = until
        logger.warning(
            f"[fallback] target={target_ip} cooled_down_for_s={FORWARD_FAILURE_COOLDOWN_S} "
            f"reason={reason}"
        )

    def _pick_target_with_exclusions(e, adapter_name: str, source_ip: str, excluded: set[str]) -> str:
        _prune_forward_cooldowns(e)
        blocked = set(excluded)
        blocked.update(e._forward_failure_deadlines.keys())
        all_nodes = [e.my_ip] + [ip for ip in e.peer_ips if ip != e.my_ip]
        candidates = [ip for ip in all_nodes if ip not in blocked]
        if not candidates:
            return e.my_ip

        if ROUTING_MODE == "cost":
            costs = {node: e._compute_cost(node, adapter_name) for node in candidates}
            reachable = {node: c for node, c in costs.items() if c != float("inf")}
            if reachable:
                best_cost = min(reachable.values())
                best_nodes = [n for n, c in reachable.items() if c == best_cost]
                return random.choice(best_nodes)
            return e.my_ip if e.my_ip in candidates else random.choice(candidates)

        if ROUTING_MODE == "memory":
            queues = e._get_known_queue_lengths()
            ranked = []
            for node in candidates:
                tier = e._get_node_tier(node, adapter_name)
                ranked.append((node, TIER_RANK[tier], queues.get(node, 0)))
            best_rank = min(rank for _, rank, _ in ranked)
            tier_best = [row for row in ranked if row[1] == best_rank]
            best_q = min(q for _, _, q in tier_best)
            queue_best = [node for node, _, q in tier_best if q == best_q]
            return random.choice(queue_best)

        if len(candidates) == 1:
            return candidates[0]
        sampled = random.sample(candidates, 2)
        queues = e._get_known_queue_lengths()
        n1, n2 = sampled
        q1 = queues.get(n1, 0)
        q2 = queues.get(n2, 0)
        if q1 < q2:
            return n1
        if q2 < q1:
            return n2
        return random.choice([n1, n2])

    @app.post("/internal/gossip")
    async def receive_gossip(request: Request):
        e = request.app.state.engine
        body = await request.json()
        msg_type = body.get("type")
        if msg_type == "queue_length":
            e._handle_queue_gossip(body)
        elif msg_type == "adapter_state":
            e._handle_adapter_state_gossip(body)
        return JSONResponse({"ok": True})

    @app.get("/internal/queue")
    async def get_queue_length(request: Request):
        e = request.app.state.engine
        return JSONResponse({
            "node": e.my_ip,
            "queue_len": e._ongoing,
            "ts": time.time(),
        })

    @app.post("/internal/chat/completions")
    async def internal_chat_completions(request: Request):
        e = request.app.state.engine
        logger.info(
            f"[endpoint] /internal/chat/completions from "
            f"{request.client.host if request.client else 'unknown'}"
        )
        try:
            parsed = await e._parse_inference_request(request)
        except ValueError as err:
            logger.error(f"[endpoint] /internal/chat/completions parse error: {err}")
            return JSONResponse({"error": str(err)}, status_code=400)
        return await e._serve_local_chat_request(parsed)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        e = request.app.state.engine
        e2e_start = time.perf_counter()
        logger.info(
            f"[endpoint] /v1/chat/completions from "
            f"{request.client.host if request.client else 'unknown'}"
        )

        try:
            parsed = await e._parse_inference_request(request)
        except Exception as err:
            logger.error(f"[endpoint] parse error: {err}\n{traceback.format_exc()}")
            return JSONResponse({"error": str(err)}, status_code=400)

        request_id = parsed["request_id"]
        adapter_name = parsed["adapter_name"]

        excluded_nodes: set[str] = set()
        attempts_used = 0
        final_target_ip = e.my_ip
        fallback_used = False
        response = None
        last_failure = ""

        while attempts_used < FORWARD_MAX_ATTEMPTS:
            attempts_used += 1
            routing_start = time.perf_counter()
            target_ip = _pick_target_with_exclusions(
                e, adapter_name, parsed["client_ip"], excluded_nodes
            )
            routing_time_ms = (time.perf_counter() - routing_start) * 1000

            is_local = target_ip == e.my_ip
            reason = "local" if is_local else "forwarded"
            logger.info(
                f"[routing] request_id={request_id} adapter={adapter_name} "
                f"target={target_ip} reason={reason} attempt={attempts_used}"
            )
            e.metrics.log(
                "routing_decision",
                request_id=request_id,
                adapter=adapter_name,
                target_node=target_ip,
                reason=reason,
                decision_time_ms=routing_time_ms,
                attempt=attempts_used,
                excluded_nodes=sorted(excluded_nodes),
            )

            final_target_ip = target_ip
            if is_local:
                response = await e._serve_local_chat_request(parsed)
                break

            body = dict(parsed["raw_body"])
            body["_client_ip"] = parsed["client_ip"]
            body["_sender_ip"] = e.my_ip
            body["_forward_path"] = parsed["forward_path"] + [e.my_ip]
            body["_fallback_hop"] = attempts_used - 1
            attempt_timeout_s = (
                FORWARD_TIMEOUT_S if attempts_used == 1 else FORWARD_RETRY_TIMEOUT_S
            )
            response = await e._forward_chat_request(
                target_ip,
                body,
                request_id,
                timeout_s=attempt_timeout_s,
            )
            if response.status_code == 200:
                break

            fallback_used = True
            last_failure = f"forward_status_{response.status_code}"
            excluded_nodes.add(target_ip)
            _mark_forward_failure(e, target_ip, last_failure)

            e.metrics.log(
                "forward_fallback",
                request_id=request_id,
                failed_target=target_ip,
                failure=last_failure,
                attempt=attempts_used,
                timeout_s=attempt_timeout_s,
            )

        if response is None:
            response = JSONResponse(
                {"error": "No routing candidates available"},
                status_code=502,
            )

        e2e_time_ms = (time.perf_counter() - e2e_start) * 1000
        logger.info(
            f"[e2e] request_id={request_id} status={response.status_code} "
            f"total_ms={e2e_time_ms:.1f} forwarded={final_target_ip != e.my_ip} "
            f"served_by={final_target_ip} attempts={attempts_used} fallback_used={fallback_used}"
        )
        e.metrics.log(
            "e2e_latency",
            request_id=request_id,
            total_ms=e2e_time_ms,
            was_forwarded=final_target_ip != e.my_ip,
            served_by=final_target_ip,
            attempts=attempts_used,
            fallback_used=fallback_used,
            final_status=response.status_code,
            last_failure=last_failure,
        )
        return response

    @app.get("/v1/models")
    async def list_models(request: Request):
        e = request.app.state.engine
        models = [{"id": e.model_id, "object": "model"}]
        for name in e.lora_names:
            models.append({"id": f"{e.model_id}/{name}", "object": "model"})
        return JSONResponse({"object": "list", "data": models})

    @app.get("/health")
    async def health(request: Request):
        e = request.app.state.engine
        if e._gossip_task is None and not e._gossip_running:
            e._gossip_running = True
            e._gossip_task = asyncio.create_task(e._gossip_queue_loop())
            logger.info("[gossip] Started gossip loop lazily on first health check")
        return JSONResponse({
            "status": "ok", "node": e.my_ip, "ongoing": e._ongoing,
        })

    @app.get("/internal/debug/state")
    async def debug_state(request: Request):
        e = request.app.state.engine
        return JSONResponse({
            "node": e.my_ip,
            "local_queue": e._ongoing,
            "peer_queues": e._peer_queue_lengths,
            "peer_timestamps": e._peer_queue_timestamps,
            "local_adapters": {
                "gpu": list(e._local_gpu_lru.keys()),
                "cpu": list(e._local_cpu_lru.keys()),
                "disk": e._get_local_disk_adapters(),
            },
            "adapter_state": {
                adapter: {tier: list(nodes) for tier, nodes in tiers.items()}
                for adapter, tiers in e._peer_adapter_state.items()
            },
            "gossip_running": e._gossip_running,
            "gossip_task_active": (
                e._gossip_task is not None and not e._gossip_task.done()
            ),
            "peer_presence_blooms": {
                ip: bf.wire_shape()
                for ip, bf in getattr(e, "_peer_presence_blooms", {}).items()
            },
            "outbound_presence_bloom": (
                e._outbound_presence_bloom.wire_shape()
                if getattr(e, "_outbound_presence_bloom", None) is not None
                else None
            ),
        })

    @app.post("/internal/debug/reset_cache")
    async def reset_cache(request: Request):
        import aiohttp

        e = request.app.state.engine
        data = await request.json()
        adapters = data.get("adapters", [])
        fanout = data.get("fanout", False)

        reset_results = {}
        for adapter in adapters:
            if adapter in e._peer_adapter_state:
                for tier in ["gpu", "cpu", "disk"]:
                    e._peer_adapter_state[adapter][tier].clear()
            e._local_gpu_lru.pop(adapter, None)
            e._local_cpu_lru.pop(adapter, None)
            adapter_path = os.path.join(ADAPTER_PATH, adapter)
            if os.path.isdir(adapter_path):
                shutil.rmtree(adapter_path)
                reset_results[adapter] = "cleared"
            else:
                reset_results[adapter] = "not_present"

        e._sync_all_peer_presence_blooms()

        if fanout:
            async def reset_peer(peer_ip: str):
                try:
                    session = await e._ensure_session()
                    async with session.post(
                        f"http://{peer_ip}:{SERVE_PORT}/internal/debug/reset_cache",
                        json={"adapters": adapters, "fanout": False},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        return await resp.json()
                except Exception as err:
                    return {"error": str(err)}

            tasks = [reset_peer(ip) for ip in e.peer_ips if ip != e.my_ip]
            peer_results = await asyncio.gather(*tasks, return_exceptions=True)
            reset_results["peers"] = peer_results

        return JSONResponse({"node": e.my_ip, "reset_results": reset_results})

    @app.get("/internal/ping")
    async def ping():
        return JSONResponse({"ok": True})

    @app.get("/internal/logs")
    async def get_logs(request: Request, lines: int = 80):
        e = request.app.state.engine
        log_path = os.path.join(LOG_DIR, "deploy.log")
        try:
            with open(log_path, "r") as f:
                all_lines = f.readlines()
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return JSONResponse({"node": e.my_ip, "lines": [l.rstrip() for l in tail]})
        except FileNotFoundError:
            return JSONResponse({"node": e.my_ip, "lines": []})

    @app.get("/internal/cluster")
    async def cluster_state(request: Request):
        import aiohttp

        e = request.app.state.engine
        nodes = {}
        log_lines = []
        log_path = os.path.join(LOG_DIR, "deploy.log")
        try:
            with open(log_path, "r") as f:
                raw = f.readlines()
            log_lines = [l.rstrip() for l in raw[-80:]]
        except Exception:
            pass

        nodes[e.my_ip] = {
            "node": e.my_ip,
            "status": "ok",
            "local_queue": e._ongoing,
            "peer_queues": dict(e._peer_queue_lengths),
            "peer_timestamps": dict(e._peer_queue_timestamps),
            "local_adapters": {
                "gpu": list(e._local_gpu_lru.keys()),
                "cpu": list(e._local_cpu_lru.keys()),
                "disk": e._get_local_disk_adapters(),
            },
            "adapter_state": {
                a: {t: list(n) for t, n in tiers.items()}
                for a, tiers in e._peer_adapter_state.items()
            },
            "gossip_running": e._gossip_running,
            "gossip_task_active": (
                e._gossip_task is not None and not e._gossip_task.done()
            ),
            "peer_presence_blooms": {
                ip: bf.wire_shape()
                for ip, bf in getattr(e, "_peer_presence_blooms", {}).items()
            },
            "outbound_presence_bloom": (
                e._outbound_presence_bloom.wire_shape()
                if getattr(e, "_outbound_presence_bloom", None) is not None
                else None
            ),
            "logs": log_lines,
        }

        async def fetch_peer(peer_ip):
            try:
                session = await e._ensure_session()
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
            except Exception as err:
                return peer_ip, {
                    "node": peer_ip, "status": "unreachable",
                    "local_queue": None, "peer_queues": {},
                    "local_adapters": {"gpu": [], "cpu": []},
                    "gossip_running": False, "gossip_task_active": False,
                    "adapter_state": {}, "logs": [], "error": str(err),
                }

        tasks = [fetch_peer(ip) for ip in e.peer_ips if ip != e.my_ip]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, tuple):
                ip, data = result
                nodes[ip] = data

        return JSONResponse({"ts": time.time(), "source_node": e.my_ip, "nodes": nodes})

    return app


app = create_app()
