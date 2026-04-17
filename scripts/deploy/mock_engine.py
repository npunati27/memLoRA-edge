"""CPU-only engine: same routing/gossip/LRU as production, fake local inference."""

from __future__ import annotations

import asyncio
import random
import time
import traceback
from collections import OrderedDict

from starlette.responses import JSONResponse

from .config import logger, load_peer_config, get_lora_names
from .metrics import MetricsLogger
from .lru import LRUMixin
from .routing import RoutingMixin
from .gossip import GossipMixin
from .parsing import ParsingMixin
from . import mock_settings as ms


class MockInferenceMixin:
    """Local fake generation: sleep + fixed body (no vLLM)."""

    async def _serve_local_chat_request(self, parsed: dict) -> JSONResponse:
        body = parsed["raw_body"]
        adapter_name = parsed["adapter_name"]
        request_id = parsed["request_id"]

        logger.info(
            f"[mock-inference] START request_id={request_id} "
            f"adapter={adapter_name} ongoing={self._ongoing}"
        )

        if adapter_name is not None and not ms.MOCK_SKIP_ADAPTER_PATH_CHECK:
            import os
            from .config import ADAPTER_PATH

            lora_path = os.path.join(ADAPTER_PATH, adapter_name)
            if not os.path.isdir(lora_path):
                logger.error(
                    f"[mock-inference] request_id={request_id} "
                    f"adapter path not found: {lora_path}"
                )
                return JSONResponse(
                    {"error": f"Adapter '{adapter_name}' not found at {lora_path}"},
                    status_code=400,
                )

        tier_before = None
        if adapter_name is not None:
            tier_before = self._get_local_tier(adapter_name)
            changes = self._track_local_adapter(adapter_name)
            for adapter, old_tier, new_tier in changes:
                asyncio.create_task(self._broadcast_state_change(adapter, old_tier, new_tier))
                logger.info(
                    f"[mock-lru] request_id={request_id} adapter={adapter} "
                    f"{old_tier}->{new_tier} "
                    f"gpu={list(self._local_gpu_lru.keys())} "
                    f"cpu={list(self._local_cpu_lru.keys())}"
                )

        self._ongoing += 1
        inf_start = time.perf_counter()
        tokens_generated = 0
        try:
            delay_ms = ms.MOCK_INFERENCE_DELAY_MS
            if ms.MOCK_INFERENCE_JITTER_MS > 0:
                delay_ms += random.randint(0, ms.MOCK_INFERENCE_JITTER_MS)
            await asyncio.sleep(delay_ms / 1000.0)

            content = ms.MOCK_RESPONSE_TEXT
            extra = ms.mock_log_extra()
            if extra:
                content = f"{content}\n\n{extra}"

            tokens_generated = max(1, len(content) // 4)

            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "model": parsed["model"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "served_by": self.my_ip,
                "adapter_name": adapter_name,
                "tier_before": tier_before,
                "mock": True,
                "mock_delay_ms": delay_ms,
            })
        except Exception as e:
            logger.error(
                f"[mock-inference] FAILED request_id={request_id} "
                f"adapter={adapter_name} error={e}\n{traceback.format_exc()}"
            )
            return JSONResponse(
                {"error": f"Mock inference failed: {str(e)}"}, status_code=500
            )
        finally:
            inf_time_ms = (time.perf_counter() - inf_start) * 1000
            self.metrics.log(
                "inference_latency",
                request_id=request_id,
                adapter=adapter_name,
                latency_ms=inf_time_ms,
                tokens=tokens_generated,
                tier_before=tier_before,
                mock=True,
            )
            self._ongoing -= 1

    async def _forward_chat_request(self, target_ip: str, body: dict,
                                    request_id: str = None) -> JSONResponse:
        """Forward to another node (same HTTP contract as production)."""
        import aiohttp

        from .config import SERVE_PORT

        url = f"http://{target_ip}:{SERVE_PORT}/internal/chat/completions"
        logger.info(f"[mock-forward] START request_id={request_id} target={target_ip}")
        fwd_start = time.perf_counter()
        success = False
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    url, json=body,
                    timeout=aiohttp.ClientTimeout(total=180),
                )
                success = resp.status == 200
                resp_body = await resp.json()
                if not success:
                    logger.error(
                        f"[mock-forward] request_id={request_id} target={target_ip} "
                        f"returned status={resp.status} body={resp_body}"
                    )
                else:
                    logger.info(
                        f"[mock-forward] OK request_id={request_id} target={target_ip}"
                    )
                return JSONResponse(content=resp_body, status_code=resp.status)
        except Exception as e:
            logger.error(
                f"[mock-forward] FAILED request_id={request_id} target={target_ip} "
                f"error={e}\n{traceback.format_exc()}"
            )
            return JSONResponse(
                {"error": f"Forward to {target_ip} failed: {str(e)}"},
                status_code=502,
            )
        finally:
            fwd_time_ms = (time.perf_counter() - fwd_start) * 1000
            self.metrics.log(
                "forward_latency",
                request_id=request_id,
                target_node=target_ip,
                latency_ms=fwd_time_ms,
                success=success,
            )


class MockMemLoRAEngine(
    LRUMixin, RoutingMixin, GossipMixin, ParsingMixin, MockInferenceMixin
):
    """Same cluster behavior as MemLoRAEngine, without loading vLLM or GPU."""

    def __init__(self):
        self.my_ip, self.peer_ips = load_peer_config()
        disk_loras = get_lora_names()
        self.lora_names = disk_loras if disk_loras else ms.mock_lora_names_from_env()
        self.model_id = ms.MOCK_MODEL_ID
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

        logger.info(f"[mock] Node: {self.my_ip}")
        logger.info(f"[mock] Peers: {[p for p in self.peer_ips if p != self.my_ip]}")
        logger.info(
            f"[mock] Adapters ({len(self.lora_names)}): {self.lora_names} "
            f"(delay_ms≈{ms.MOCK_INFERENCE_DELAY_MS}+{ms.MOCK_INFERENCE_JITTER_MS} jitter)"
        )

    async def _stop_gossip_loop(self):
        if self._gossip_task and not self._gossip_task.done():
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
        self._gossip_running = False
