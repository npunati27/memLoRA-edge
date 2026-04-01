#!/usr/bin/env python3

import os, time, requests, uuid, asyncio, json, random, logging, traceback
from collections import OrderedDict
import ray
from ray import serve
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from vllm import SamplingParams
from vllm.lora.request import LoRARequest


HOME         = os.path.expanduser("~")
LOG_DIR      = os.path.join(HOME, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "deploy.log")),
    ],
)
logger = logging.getLogger("memlora")
MODEL_PATH   = f"{HOME}/model_cache/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = f"{HOME}/adapters"
PEERS_FILE   = os.path.expanduser("~/peers.json")

MAX_GPU_LORA = 3
MAX_CPU_LORA = 6
SERVE_PORT   = 5000

def load_peer_config():
    with open(PEERS_FILE) as f:
        cfg = json.load(f)
    return cfg["my_ip"], cfg["peers"]

def get_lora_names():
    return sorted([
        name for name in os.listdir(ADAPTER_PATH)
        if os.path.isdir(os.path.join(ADAPTER_PATH, name))
    ])


class MetricsLogger:
    """Structured JSON lines logger for timing and routing metrics."""
    
    def __init__(self, node_ip: str, log_dir: str = "~/logs"):
        self.node_ip = node_ip
        log_dir_expanded = os.path.expanduser(log_dir)
        os.makedirs(log_dir_expanded, exist_ok=True)
        self.log_path = os.path.join(log_dir_expanded, f"metrics_{node_ip}.jsonl")
        self._file = open(self.log_path, "a")
    
    def log(self, event_type: str, **kwargs):
        record = {
            "ts": time.time(),
            "node": self.node_ip,
            "event": event_type,
            **kwargs
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
    
    def close(self):
        if self._file:
            self._file.close()


from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

        self.metrics = MetricsLogger(self.my_ip)

        # Peer queue length tracking (for both gossip and RTT approaches)
        self._peer_queue_lengths: dict[str, int] = {
            ip: 0 for ip in self.peer_ips if ip != self.my_ip
        }
        self._peer_queue_timestamps: dict[str, float] = {
            ip: 0.0 for ip in self.peer_ips if ip != self.my_ip
        }

        # Peer adapter state cache: {adapter_name: {tier: set(node_ips)}}
        self._peer_adapter_state: dict[str, dict[str, set]] = {}
        # Timestamps for adapter state per (adapter, node) to reject out-of-order messages
        self._adapter_state_timestamps: dict[tuple[str, str], float] = {}

        # Gossip loop will be started after engine is ready
        self._gossip_task = None
        self._gossip_running = False
        self._aiohttp_session = None

        # TODO: LRU tracker for local GPU/CPU adapter state

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
        logger.info(f"[vllm] Engine ready.")

        # Start gossip loop now that engine is ready
        self._start_gossip_loop()


    def _get_known_queue_lengths(self) -> dict[str, int]:
        queues = {self.my_ip: self._ongoing}

        for ip in self.peer_ips:
            if ip != self.my_ip:
                queues[ip] = self._peer_queue_lengths.get(ip, 0)

        return queues

    # Routing Logic 

    async def _choose_target_node(self, adapter_name: str, source_ip: str) -> str:
        # TODO: use peer state + LRU tracker to pick best node
        all_nodes = [self.my_ip] + [ip for ip in self.peer_ips if ip != self.my_ip]

        if len(all_nodes) == 1:
            return self.my_ip

        sampled = random.sample(all_nodes, 2)
        queues = self._get_known_queue_lengths()

        n1, n2 = sampled
        q1 = queues.get(n1, 0)
        q2 = queues.get(n2, 0)

        if q1 < q2:
            chosen = n1
        elif q2 < q1:
            chosen = n2
        else:
            chosen = random.choice([n1, n2])

        self.metrics.log(
            "baseline_p2c_choice",
            adapter_name=adapter_name,
            source_ip=source_ip,
            sampled_nodes=sampled,
            sampled_queue_lengths={
                n1: q1,
                n2: q2,
            },
            chosen_node=chosen,
        )

        return chosen

    # Gossip Lifecycle Methods

    def _start_gossip_loop(self):
        """Start the background gossip loop. Safe to call from __init__ after engine ready."""
        if self._gossip_task is None:
            try:
                loop = asyncio.get_event_loop()
                self._gossip_task = loop.create_task(self._gossip_queue_loop())
                self._gossip_running = True
                logger.info(f"[gossip] Started gossip loop for {self.my_ip}")
            except RuntimeError:
                self._gossip_running = False
                logger.warning(f"[gossip] No event loop available, gossip will start on first request")

    async def _ensure_session(self):
        """Lazily create and return the shared aiohttp session."""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            import aiohttp
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=2)
            )
        return self._aiohttp_session

    # Queue Length Gossip Methods

    async def _gossip_queue_loop(self):
        """Background task that broadcasts local queue length to all peers every 150ms."""
        await asyncio.sleep(1)
        logger.info(f"[gossip] Gossip loop active, broadcasting to {len([p for p in self.peer_ips if p != self.my_ip])} peers")
        
        while self._gossip_running:
            try:
                await self._broadcast_queue_length()
            except Exception as e:
                logger.error(f"[gossip] Broadcast error: {e}")
            await asyncio.sleep(0.15)

    async def _broadcast_queue_length(self):
        """Send current queue length to all peers."""
        msg = {
            "type": "queue_length",
            "node": self.my_ip,
            "queue_len": self._ongoing,
            "ts": time.time()
        }
        await self._broadcast_to_peers(msg)

    def _handle_queue_gossip(self, body: dict):
        """Process incoming queue length gossip from a peer."""
        node = body.get("node")
        queue_len = body.get("queue_len", 0)
        ts = body.get("ts", 0)
        
        if node and node != self.my_ip and node in self._peer_queue_lengths:
            if ts > self._peer_queue_timestamps.get(node, 0):
                self._peer_queue_lengths[node] = queue_len
                self._peer_queue_timestamps[node] = ts

    # Broadcast Helper Methods

    async def _broadcast_to_peers(self, msg: dict):
        """Send a message to all peers concurrently."""
        tasks = []
        for peer in self.peer_ips:
            if peer != self.my_ip:
                tasks.append(self._send_gossip(peer, msg))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_gossip(self, peer_ip: str, msg: dict):
        """Send a gossip message to a single peer."""
        url = f"http://{peer_ip}:{SERVE_PORT}/internal/gossip"
        try:
            session = await self._ensure_session()
            async with session.post(url, json=msg) as resp:
                await resp.read()
        except Exception:
            pass

    # Adapter State Broadcast Methods

    async def _broadcast_state_change(self, adapter_name: str, old_tier: str, new_tier: str):
        """Broadcast an adapter tier change to all peers immediately."""
        msg = {
            "type": "adapter_state",
            "node": self.my_ip,
            "adapter": adapter_name,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "ts": time.time()
        }
        await self._broadcast_to_peers(msg)

    def _handle_adapter_state_gossip(self, body: dict):
        """Process incoming adapter state change from a peer."""
        node = body.get("node")
        adapter = body.get("adapter")
        old_tier = body.get("old_tier")
        new_tier = body.get("new_tier")
        ts = body.get("ts", 0)
        
        if not all([node, adapter, new_tier]) or node == self.my_ip:
            return
        
        key = (adapter, node)
        if ts <= self._adapter_state_timestamps.get(key, 0):
            return
        self._adapter_state_timestamps[key] = ts
        
        if adapter not in self._peer_adapter_state:
            self._peer_adapter_state[adapter] = {
                "gpu": set(),
                "cpu": set(),
                "disk": set()
            }
        
        tiers = self._peer_adapter_state[adapter]
        # Remove node from ALL tiers first to prevent multi-tier corruption
        for tier_set in tiers.values():
            tier_set.discard(node)
        if new_tier in tiers:
            tiers[new_tier].add(node)

    # RTT-Based Queue Query Methods

    async def _query_peer_queue(self, peer_ip: str) -> int:
        """Query a peer's queue length directly via HTTP (RTT approach)."""
        import aiohttp
        url = f"http://{peer_ip}:{SERVE_PORT}/internal/queue"
        try:
            session = await self._ensure_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=1)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    queue_len = data.get("queue_len", 0)
                    ts = data.get("ts", time.time())
                    self._peer_queue_lengths[peer_ip] = queue_len
                    self._peer_queue_timestamps[peer_ip] = ts
                    return queue_len
        except Exception as e:
            pass
        return self._peer_queue_lengths.get(peer_ip, 0)

    async def _query_all_peer_queues(self) -> dict[str, int]:
        """Query all peers' queue lengths concurrently."""
        tasks = {}
        for peer in self.peer_ips:
            if peer != self.my_ip:
                tasks[peer] = self._query_peer_queue(peer)
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            return {ip: (r if isinstance(r, int) else 0) for ip, r in zip(tasks.keys(), results)}
        return {}

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
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"[parse] Failed to parse request body as JSON: {e}")
            raise ValueError(f"Invalid JSON body: {e}")
        
        model_name = body.get("model")
        base_model, adapter_name = self._parse_model_and_adapter(model_name)
        prompt_info = self._extract_prompt(body)
        request_id = body.get("request_id", str(uuid.uuid4()))
        
        logger.info(f"[parse] request_id={request_id} model={model_name} adapter={adapter_name}")
        
        return {
            "request_id":   request_id,
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
        request_id   = parsed["request_id"]

        logger.info(f"[inference] START request_id={request_id} adapter={adapter_name} ongoing={self._ongoing}")

        sampling_params = SamplingParams(
            max_tokens=body.get("max_tokens", 64),
            temperature=body.get("temperature", 0.0),
            top_p=body.get("top_p", 1.0),
        )

        lora_request = None
        if adapter_name is not None:
            lora_path = os.path.join(ADAPTER_PATH, adapter_name)
            if not os.path.isdir(lora_path):
                logger.error(f"[inference] request_id={request_id} adapter path not found: {lora_path}")
                return JSONResponse({"error": f"Adapter '{adapter_name}' not found at {lora_path}"}, status_code=400)
            lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=abs(hash(adapter_name)) % (2**31),
                lora_local_path=lora_path,
            )

        self._ongoing += 1
        inf_start = time.perf_counter()
        tokens_generated = 0
        try:
            final_output = None
            async for output in self.engine.generate(
                parsed["prompt"],
                sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            ):
                final_output = output

            if final_output is None or not final_output.outputs:
                logger.error(f"[inference] request_id={request_id} engine returned no output (final_output={final_output})")
                return JSONResponse({"error": "No output generated"}, status_code=500)

            tokens_generated = len(final_output.outputs[0].token_ids) if hasattr(final_output.outputs[0], 'token_ids') else 0
            logger.info(f"[inference] OK request_id={request_id} tokens={tokens_generated}")

            return JSONResponse({
                "id":           request_id,
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
            logger.error(f"[inference] FAILED request_id={request_id} adapter={adapter_name} error={e}\n{traceback.format_exc()}")
            return JSONResponse({"error": f"Inference failed: {str(e)}"}, status_code=500)
        finally:
            inf_time_ms = (time.perf_counter() - inf_start) * 1000
            self.metrics.log(
                "inference_latency",
                request_id=request_id,
                adapter=adapter_name,
                latency_ms=inf_time_ms,
                tokens=tokens_generated
            )
            self._ongoing -= 1

    # Forward the request to another node and return its response, handling any network errors gracefully
    async def _forward_chat_request(self, target_ip: str, body: dict, request_id: str = None) -> JSONResponse:
        import aiohttp
        url = f"http://{target_ip}:{SERVE_PORT}/internal/chat/completions"
        logger.info(f"[forward] START request_id={request_id} target={target_ip}")
        fwd_start = time.perf_counter()
        success = False
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    url, json=body,
                    timeout=aiohttp.ClientTimeout(total=180),
                )
                success = (resp.status == 200)
                resp_body = await resp.json()
                if not success:
                    logger.error(f"[forward] request_id={request_id} target={target_ip} returned status={resp.status} body={resp_body}")
                else:
                    logger.info(f"[forward] OK request_id={request_id} target={target_ip} status={resp.status}")
                return JSONResponse(content=resp_body, status_code=resp.status)
        except Exception as e:
            logger.error(f"[forward] FAILED request_id={request_id} target={target_ip} error={e}\n{traceback.format_exc()}")
            return JSONResponse({"error": f"Forward to {target_ip} failed: {str(e)}"}, status_code=502)
        finally:
            fwd_time_ms = (time.perf_counter() - fwd_start) * 1000
            self.metrics.log(
                "forward_latency",
                request_id=request_id,
                target_node=target_ip,
                latency_ms=fwd_time_ms,
                success=success
            )

    # internal endpoint for gossip messages from other nodes about their state.
    @app.post("/internal/gossip")
    async def receive_gossip(self, request: Request):
        body = await request.json()
        msg_type = body.get("type")
        
        if msg_type == "queue_length":
            self._handle_queue_gossip(body)
        elif msg_type == "adapter_state":
            self._handle_adapter_state_gossip(body)
        
        return JSONResponse({"ok": True})

    # internal endpoint for RTT-based queue length queries
    @app.get("/internal/queue")
    async def get_queue_length(self):
        return JSONResponse({
            "node": self.my_ip,
            "queue_len": self._ongoing,
            "ts": time.time()
        })

    # internal endpoint for forwarded requests from other nodes in the cluster. 
    @app.post("/internal/chat/completions")
    async def internal_chat_completions(self, request: Request):
        logger.info(f"[endpoint] /internal/chat/completions from {request.client.host if request.client else 'unknown'}")
        try:
            parsed = await self._parse_inference_request(request)
        except ValueError as e:
            logger.error(f"[endpoint] /internal/chat/completions parse error: {e}")
            return JSONResponse({"error": str(e)}, status_code=400)
        return await self._serve_local_chat_request(parsed)

    # main public endpoint for chat completions
    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        e2e_start = time.perf_counter()
        logger.info(f"[endpoint] /v1/chat/completions from {request.client.host if request.client else 'unknown'}")
        
        try:
            parsed = await self._parse_inference_request(request)
        except (ValueError, Exception) as e:
            logger.error(f"[endpoint] /v1/chat/completions parse error: {e}\n{traceback.format_exc()}")
            return JSONResponse({"error": str(e)}, status_code=400)

        request_id = parsed["request_id"]
        adapter_name = parsed["adapter_name"]

        routing_start = time.perf_counter()
        target_ip = await self._choose_target_node(adapter_name, parsed["client_ip"])
        routing_time_ms = (time.perf_counter() - routing_start) * 1000
        
        is_local = (target_ip == self.my_ip)
        reason = "local" if is_local else "forwarded"
        logger.info(f"[routing] request_id={request_id} adapter={adapter_name} target={target_ip} reason={reason}")
        self.metrics.log(
            "routing_decision",
            request_id=request_id,
            adapter=adapter_name,
            target_node=target_ip,
            reason=reason,
            decision_time_ms=routing_time_ms
        )

        if is_local:
            response = await self._serve_local_chat_request(parsed)
        else:
            body = dict(parsed["raw_body"])
            body["_client_ip"]    = parsed["client_ip"]
            body["_sender_ip"]    = self.my_ip
            body["_forward_path"] = parsed["forward_path"] + [self.my_ip]
            response = await self._forward_chat_request(target_ip, body, request_id)

        e2e_time_ms = (time.perf_counter() - e2e_start) * 1000
        logger.info(f"[e2e] request_id={request_id} status={response.status_code} total_ms={e2e_time_ms:.1f} forwarded={not is_local} served_by={target_ip}")
        self.metrics.log(
            "e2e_latency",
            request_id=request_id,
            total_ms=e2e_time_ms,
            was_forwarded=not is_local,
            served_by=target_ip
        )

        return response

    # get the models 
    @app.get("/v1/models")
    async def list_models(self):
        models = [{"id": self.model_id, "object": "model"}]
        for name in self.lora_names:
            models.append({"id": f"{self.model_id}/{name}", "object": "model"})
        return JSONResponse({"object": "list", "data": models})

    @app.get("/health")
    async def health(self):
        # Ensure gossip is running (lazy start if __init__ couldn't start it)
        if self._gossip_task is None and not self._gossip_running:
            self._gossip_running = True
            self._gossip_task = asyncio.create_task(self._gossip_queue_loop())
            logger.info(f"[gossip] Started gossip loop lazily on first health check")
        
        return JSONResponse({"status": "ok", "node": self.my_ip, "ongoing": self._ongoing})

    @app.get("/internal/debug/state")
    async def debug_state(self):
        """Debug endpoint to view current gossip state."""
        return JSONResponse({
            "node": self.my_ip,
            "local_queue": self._ongoing,
            "peer_queues": self._peer_queue_lengths,
            "peer_timestamps": self._peer_queue_timestamps,
            "adapter_state": {
                adapter: {tier: list(nodes) for tier, nodes in tiers.items()}
                for adapter, tiers in self._peer_adapter_state.items()
            },
            "gossip_running": self._gossip_running,
            "gossip_task_active": self._gossip_task is not None and not self._gossip_task.done()
        })

    @app.get("/internal/logs")
    async def get_logs(self, lines: int = 80):
        """Return tail of deploy.log."""
        log_path = os.path.join(LOG_DIR, "deploy.log")
        try:
            with open(log_path, "r") as f:
                all_lines = f.readlines()
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return JSONResponse({"node": self.my_ip, "lines": [l.rstrip() for l in tail]})
        except FileNotFoundError:
            return JSONResponse({"node": self.my_ip, "lines": []})

    @app.get("/internal/cluster")
    async def cluster_state(self):
        """Aggregate debug state and logs from this node and all peers."""
        import aiohttp

        nodes = {}

        # This node
        log_lines = []
        log_path = os.path.join(LOG_DIR, "deploy.log")
        try:
            with open(log_path, "r") as f:
                raw = f.readlines()
            log_lines = [l.rstrip() for l in raw[-80:]]
        except Exception:
            pass

        nodes[self.my_ip] = {
            "node": self.my_ip,
            "status": "ok",
            "local_queue": self._ongoing,
            "peer_queues": dict(self._peer_queue_lengths),
            "peer_timestamps": dict(self._peer_queue_timestamps),
            "adapter_state": {
                a: {t: list(n) for t, n in tiers.items()}
                for a, tiers in self._peer_adapter_state.items()
            },
            "gossip_running": self._gossip_running,
            "gossip_task_active": self._gossip_task is not None and not self._gossip_task.done(),
            "logs": log_lines,
        }

        async def fetch_peer(peer_ip):
            try:
                session = await self._ensure_session()
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
                    "node": peer_ip, "status": f"unreachable",
                    "local_queue": None, "peer_queues": {},
                    "gossip_running": False, "gossip_task_active": False,
                    "adapter_state": {}, "logs": [],
                    "error": str(e),
                }

        tasks = [fetch_peer(ip) for ip in self.peer_ips if ip != self.my_ip]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, tuple):
                ip, data = result
                nodes[ip] = data

        return JSONResponse({"ts": time.time(), "source_node": self.my_ip, "nodes": nodes})


if __name__ == "__main__":
    ray.init()

    logger.info(f"Cluster resources: {ray.cluster_resources()}")

    #delete old deployment if it exists. 
    try:
        serve.start(http_options={"host": "0.0.0.0", "port": SERVE_PORT})
        serve.delete("memlora")
        time.sleep(2)
    except Exception:
        pass

    serve.shutdown()
    time.sleep(2)
    serve.start(http_options={"host": "0.0.0.0", "port": SERVE_PORT})
    handle = VLLMDeployment.bind()
    serve.run(handle, name="memlora", route_prefix="/", blocking=False)

    logger.info("Waiting for deployment to be ready...")
    for i in range(60):
        try:
            r = requests.get(f"http://localhost:{SERVE_PORT}/health", timeout=5)
            if r.status_code == 200:
                logger.info(f"Ready! {r.json()}")
                break
        except Exception:
            pass
        time.sleep(60)
        logger.info(f"  waiting... {i+1}/60 (~{(i+1)*10}s elapsed)")
    else:
        logger.error("Did not become ready in time.")
        exit(1)