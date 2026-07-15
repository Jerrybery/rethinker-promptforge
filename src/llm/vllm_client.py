"""OpenAI-compatible vLLM client with retry and image support."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from loguru import logger

from rethinker_promptforge.config import load_config


class VLLMClient:
    """Thin client for a local vLLM OpenAI-compatible chat endpoint.

    Configuration is read from ``configs/models.yaml`` under the ``vllm`` key.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
    ) -> None:
        if config_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            config_path = repo_root / "configs" / "models.yaml"
        cfg = load_config(config_path).get("vllm", {})
        self.base_url = str(cfg.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = cfg.get("api_key") or "EMPTY"
        self.model_id = cfg.get("model_id", "openvla/openvla-7b")
        self.temperature = cfg.get("temperature", 0.0)
        self.top_p = cfg.get("top_p", 1.0)
        self.max_tokens = cfg.get("max_tokens", 512)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode an RGB numpy image as a base64 PNG data URL."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        from PIL import Image

        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _build_messages(
        self, messages: list[dict], images: list[np.ndarray] | None
    ) -> list[dict[str, Any]]:
        """Merge conversation messages with optional image content."""
        request_messages: list[dict[str, Any]] = [dict(m) for m in messages]
        if not images:
            return request_messages
        image_content = [
            {"type": "image_url", "image_url": {"url": self._encode_image(img)}}
            for img in images
        ]
        last_user_idx: int | None = None
        for i in range(len(request_messages) - 1, -1, -1):
            if request_messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            user_msg = request_messages[last_user_idx]
            user_content = user_msg.get("content", "")
            if isinstance(user_content, str):
                new_content = image_content + [{"type": "text", "text": user_content}]
            elif isinstance(user_content, list):
                new_content = image_content + list(user_content)
            else:
                new_content = image_content
            request_messages[last_user_idx] = {**user_msg, "content": new_content}
        else:
            request_messages.append({"role": "user", "content": image_content})
        return request_messages

    def _request_payload(
        self, messages: list[dict], images: list[np.ndarray] | None
    ) -> dict[str, Any]:
        """Build the OpenAI-compatible chat completion payload."""
        return {
            "model": self.model_id,
            "messages": self._build_messages(messages, images),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    def chat(self, messages: list[dict], images: list[np.ndarray] | None = None) -> str:
        """Send a chat request and return the assistant's text content.

        Retries on network errors and malformed responses with exponential
        backoff (``base_delay * 2 ** attempt`` capped at ``max_delay``).
        """
        payload = self._request_payload(messages, images)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                content_text = data["choices"][0]["message"].get("content", "")
                if not isinstance(content_text, str):
                    raise ValueError(f"Unexpected content type: {type(content_text)}")
                return content_text
            except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
                last_exception = exc
                logger.warning(
                    "vLLM request failed (attempt {}/{}): {}",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
        raise RuntimeError(
            f"vLLM request failed after {self.max_retries + 1} attempts"
        ) from last_exception
