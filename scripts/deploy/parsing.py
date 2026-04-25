import uuid
import re

from starlette.requests import Request

from .config import logger


class ParsingMixin:
    """Request parsing: extract client/sender IP, model/adapter, prompt."""

    def _extract_client_ip(self, request: Request, body: dict) -> str:
        if body.get("_client_ip"):
            return body["_client_ip"]
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    def _extract_sender_ip(self, request: Request, body: dict) -> str:
        """Extract sender IP (may differ from client IP if forwarded from another node)."""
        if body.get("_sender_ip"):
            return body["_sender_ip"]
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    def _parse_model_and_adapter(self, model_name: str):
        """Parse 'model' field into (base_model, adapter_name | None)."""
        if not model_name:
            raise ValueError("Missing 'model' field")
        if model_name == self.model_id:
            return self.model_id, None
        prefix = f"{self.model_id}/"
        if model_name.startswith(prefix):
            adapter_name = model_name[len(prefix):]
            if not adapter_name:
                raise ValueError("Missing adapter name after model prefix")
            if not re.match(r'^[A-Za-z0-9._-]+$', adapter_name):
                raise ValueError(f"Invalid adapter name: {adapter_name}")
            return self.model_id, adapter_name
        raise ValueError(f"Unsupported model '{model_name}'")

    def _extract_prompt(self, body: dict) -> dict:
        """Extract prompt text from 'messages' or 'prompt' field (chat + completion formats)."""
        if "messages" in body:
            messages = body["messages"]
            if not isinstance(messages, list):
                raise ValueError("'messages' must be a list")
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    prompt_parts.append(
                        content if isinstance(content, str) else str(content)
                    )
            return {"messages": messages, "prompt": "\n".join(prompt_parts).strip()}
        if "prompt" in body:
            prompt = body["prompt"]
            return {
                "messages": None,
                "prompt": prompt if isinstance(prompt, str) else str(prompt),
            }
        raise ValueError("Request must include either 'messages' or 'prompt'")

    async def _parse_inference_request(self, request: Request) -> dict:
        """Parse and validate the full inference request."""
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"[parse] Failed to parse request body as JSON: {e}")
            raise ValueError(f"Invalid JSON body: {e}")

        model_name = body.get("model")
        base_model, adapter_name = self._parse_model_and_adapter(model_name)
        prompt_info = self._extract_prompt(body)
        request_id = body.get("request_id", str(uuid.uuid4()))

        logger.info(
            f"[parse] request_id={request_id} model={model_name} adapter={adapter_name}"
        )

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
