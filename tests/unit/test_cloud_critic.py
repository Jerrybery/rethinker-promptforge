"""Unit tests for the cloud VLM critic client."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
import responses

from llm.cloud_critic import CloudVLMClient
from llm.vllm_client import VLLMClient

CLOUD_URL = "http://cloud.example.com/v1"


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
vllm:
  model_id: local-model
  base_url: http://localhost:8000/v1
  api_key: null
  temperature: 0.0
  max_tokens: 512

cloud_critic:
  model_id: cloud-vlm-1
  base_url: http://cloud.example.com/v1
  api_key: test-key-123
  temperature: 0.2
  max_tokens: 300
""",
        encoding="utf-8",
    )
    return path


def _completion_response(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "cloud-vlm-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def test_init_reads_cloud_critic_section(config_path: Path) -> None:
    client = CloudVLMClient(config_path=config_path)
    assert client.model_id == "cloud-vlm-1"
    assert client.base_url == CLOUD_URL
    assert client.api_key == "test-key-123"
    assert client.temperature == 0.2
    assert client.max_tokens == 300


def test_init_missing_model_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
cloud_critic:
  model_id: null
  base_url: http://cloud.example.com/v1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model_id"):
        CloudVLMClient(config_path=path)


def test_vllm_client_still_reads_vllm_section(config_path: Path) -> None:
    """Regression guard: the shared config-section machinery must not break
    the local vLLM client."""
    client = VLLMClient(config_path=config_path)
    assert client.model_id == "local-model"
    assert client.base_url == "http://localhost:8000/v1"


@responses.activate
def test_chat_text_only(config_path: Path) -> None:
    client = CloudVLMClient(config_path=config_path, max_retries=0)
    responses.post(
        f"{CLOUD_URL}/chat/completions",
        json=_completion_response("episode looks bad"),
    )
    result = client.chat([{"role": "user", "content": "critique this"}])
    assert result == "episode looks bad"
    payload = json.loads(responses.calls[0].request.body)
    assert payload["model"] == "cloud-vlm-1"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 300
    auth = responses.calls[0].request.headers["Authorization"]
    assert auth == "Bearer test-key-123"


@responses.activate
def test_chat_with_images_sends_data_urls(config_path: Path) -> None:
    client = CloudVLMClient(config_path=config_path, max_retries=0)
    responses.post(
        f"{CLOUD_URL}/chat/completions",
        json=_completion_response("saw frames"),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    result = client.chat(
        [{"role": "user", "content": "describe"}],
        images=[image, image],
    )
    assert result == "saw frames"
    payload = json.loads(responses.calls[0].request.body)
    content = payload["messages"][0]["content"]
    image_parts = [c for c in content if c["type"] == "image_url"]
    assert len(image_parts) == 2
    url = image_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",")[1])
    assert decoded[:4] == b"\x89PNG"


@responses.activate
def test_chat_retry_on_network_error(config_path: Path) -> None:
    client = CloudVLMClient(
        config_path=config_path, max_retries=1, base_delay=0.0, max_delay=0.0
    )
    responses.add(
        responses.POST,
        f"{CLOUD_URL}/chat/completions",
        body=requests.ConnectionError("boom"),
    )
    responses.add(
        responses.POST,
        f"{CLOUD_URL}/chat/completions",
        json=_completion_response("recovered"),
    )
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result == "recovered"
    assert len(responses.calls) == 2


@responses.activate
def test_chat_retry_exhausted_raises(config_path: Path) -> None:
    client = CloudVLMClient(
        config_path=config_path, max_retries=1, base_delay=0.0, max_delay=0.0
    )
    responses.add(
        responses.POST,
        f"{CLOUD_URL}/chat/completions",
        body=requests.ConnectionError("boom"),
    )
    responses.add(
        responses.POST,
        f"{CLOUD_URL}/chat/completions",
        body=requests.Timeout("timeout"),
    )
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(responses.calls) == 2
