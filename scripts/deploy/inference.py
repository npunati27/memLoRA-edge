import os
import time
import asyncio
import traceback

from starlette.responses import JSONResponse

from .config import ADAPTER_PATH, SERVE_PORT, logger, USE_S3_ADAPTERS
from .s3_adapter import download_adapter_from_s3

class InferenceMixin:
    """Local inference execution and request forwarding."""

    async def _serve_local_chat_request(self, parsed: dict) -> JSONResponse:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        body = parsed["raw_body"]
        adapter_name = parsed["adapter_name"]
        request_id = parsed["request_id"]

        logger.info(
            f"[inference] START request_id={request_id} "
            f"adapter={adapter_name} ongoing={self._ongoing}"
        )

        sampling_params = SamplingParams(
            max_tokens=body.get("max_tokens", 64),
            temperature=body.get("temperature", 0.0),
            top_p=body.get("top_p", 1.0),
        )

        lora_request = None
        tier_before = None
        adapter_source = "none"
        adapter_load_ms = 0.0
        if adapter_name is not None:
            tier_before = self._get_local_tier(adapter_name)
            lora_path = os.path.join(ADAPTER_PATH, adapter_name)
            load_start = time.perf_counter()
            if not hasattr(self, '_adapter_download_locks'):
                self._adapter_download_locks = {}
            lock = self._adapter_download_locks.setdefault(adapter_name, asyncio.Lock())
            async with lock:
                if os.path.isdir(lora_path):
                    adapter_source = "local"
                elif USE_S3_ADAPTERS:
                    try:
                        logger.info(
                            f"[inference] request_id={request_id} "
                            f"Adapter not cached locally, downloading from S3: {adapter_name}"
                        )
                        lora_path = await asyncio.to_thread(download_adapter_from_s3, adapter_name)
                        adapter_source = "s3"
                        logger.info(
                            f"[inference] request_id={request_id} "
                            f"Successfully downloaded adapter: {adapter_name}"
                        )
                    except FileNotFoundError as e:
                        logger.error(
                            f"[inference] request_id={request_id} "
                            f"Adapter not found in S3: {adapter_name} error={e}"
                        )
                        return JSONResponse(
                            {"error": f"Adapter '{adapter_name}' not found locally or in S3"},
                            status_code=404,
                        )
                    except Exception as e:
                        logger.error(
                            f"[inference] request_id={request_id} "
                            f"Failed to download adapter from S3: {e}"
                        )
                        return JSONResponse(
                            {"error": f"Failed to load adapter '{adapter_name}' from S3: {str(e)}"},
                            status_code=500,
                        )
                else:
                    logger.error(
                        f"[inference] request_id={request_id} "
                        f"adapter path not found and S3 disabled: {lora_path}"
                    )
                    return JSONResponse(
                        {"error": f"Adapter '{adapter_name}' not found at {lora_path}"},
                        status_code=404,
                    )
            adapter_load_ms = (time.perf_counter() - load_start) * 1000
            self.metrics.log(
                "adapter_load",
                request_id=request_id,
                adapter=adapter_name,
                source=adapter_source,
                latency_ms=adapter_load_ms,
            )
            lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=abs(hash(adapter_name)) % (2**31),
                lora_local_path=lora_path,
            )

        if adapter_name is not None:
            changes = self._track_local_adapter(adapter_name)
            for adapter, old_tier, new_tier in changes:
                asyncio.create_task(
                    self._broadcast_state_change(adapter, old_tier, new_tier)
                )
                logger.info(
                    f"[lru] request_id={request_id} adapter={adapter} "
                    f"{old_tier}->{new_tier} "
                    f"gpu={list(self._local_gpu_lru.keys())} "
                    f"cpu={list(self._local_cpu_lru.keys())}"
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
                logger.error(
                    f"[inference] request_id={request_id} engine returned no output "
                    f"(final_output={final_output})"
                )
                return JSONResponse({"error": "No output generated"}, status_code=500)

            tokens_generated = (
                len(final_output.outputs[0].token_ids)
                if hasattr(final_output.outputs[0], "token_ids")
                else 0
            )
            logger.info(
                f"[inference] OK request_id={request_id} tokens={tokens_generated}"
            )

            return JSONResponse({
                "id":           request_id,
                "object":       "chat.completion",
                "model":        parsed["model"],
                "choices": [{
                    "index":         0,
                    "message": {
                        "role": "assistant",
                        "content": final_output.outputs[0].text,
                    },
                    "finish_reason": final_output.outputs[0].finish_reason,
                }],
                "served_by":    self.my_ip,
                "adapter_name": adapter_name,
                "tier_before":  tier_before,
                "adapter_source": adapter_source,
                "adapter_load_ms": adapter_load_ms,
            })
        except Exception as e:
            logger.error(
                f"[inference] FAILED request_id={request_id} "
                f"adapter={adapter_name} error={e}\n{traceback.format_exc()}"
            )
            return JSONResponse(
                {"error": f"Inference failed: {str(e)}"}, status_code=500
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
            )
            self._ongoing -= 1

    async def _forward_chat_request(self, target_ip: str, body: dict,
                                    request_id: str = None) -> JSONResponse:
        """Forward the request to another node, handling network errors gracefully."""
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
                success = resp.status == 200
                resp_body = await resp.json()
                if not success:
                    logger.error(
                        f"[forward] request_id={request_id} target={target_ip} "
                        f"returned status={resp.status} body={resp_body}"
                    )
                else:
                    logger.info(
                        f"[forward] OK request_id={request_id} target={target_ip}"
                    )
                return JSONResponse(content=resp_body, status_code=resp.status)
        except Exception as e:
            logger.error(
                f"[forward] FAILED request_id={request_id} target={target_ip} "
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
