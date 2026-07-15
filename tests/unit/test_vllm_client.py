"""Unit tests for the vLLM client."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
import responses

from llm.vllm_client import VLLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_URL = "http://localhost:8000/v1"


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
vllm:
  model_id: test-model
  base_url: http://localhost:8000/v1
  api_key: null
  temperature: 0.1
  top_p: 0.9
  max_tokens: 64
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def client(config_path: Path) -> VLLMClient:
    return VLLMClient(config_path=config_path, max_retries=0)


def _completion_response(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@responses.activate
def test_chat_text_only(client: VLLMClient) -> None:
    responses.post(
        f"{MODELS_URL}/chat/completions",
        json=_completion_response("hello back"),
    )
    result = client.chat([{"role": "user", "content": "hello"}])
    assert result == "hello back"
    assert len(responses.calls) == 1
    payload = json.loads(responses.calls[0].request.body)
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 64
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


@responses.activate
def test_chat_with_images(client: VLLMClient) -> None:
    responses.post(
        f"{MODELS_URL}/chat/completions",
        json=_completion_response("saw it"),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    result = client.chat(
        [{"role": "user", "content": "describe this"}],
        images=[image],
    )
    assert result == "saw it"
    payload = json.loads(responses.calls[0].request.body)
    content = payload["messages"][0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",")[1])
    assert decoded[:4] == b"\x89PNG"


@responses.activate
def test_chat_api_key_defaults_to_empty(config_path: Path) -> None:
    client = VLLMClient(config_path=config_path, max_retries=0)
    responses.post(
        f"{MODELS_URL}/chat/completions",
        json=_completion_response("ok"),
    )
    client.chat([{"role": "user", "content": "hi"}])
    auth = responses.calls[0].request.headers["Authorization"]
    assert auth == "Bearer EMPTY"


@responses.activate
def test_chat_retry_on_network_error(config_path: Path) -> None:
    client = VLLMClient(
        config_path=config_path,
        max_retries=2,
        base_delay=0.0,
        max_delay=0.0,
    )
    responses.add(
        responses.POST,
        f"{MODELS_URL}/chat/completions",
        body=requests.ConnectionError("boom"),
    )
    responses.add(
        responses.POST,
        f"{MODELS_URL}/chat/completions",
        json=_completion_response("recovered"),
    )
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result == "recovered"
    assert len(responses.calls) == 2


@responses.activate
def test_chat_retry_exhausted_raises(config_path: Path) -> None:
    client = VLLMClient(
        config_path=config_path,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
    )
    responses.add(
        responses.POST,
        f"{MODELS_URL}/chat/completions",
        body=requests.ConnectionError("boom"),
    )
    responses.add(
        responses.POST,
        f"{MODELS_URL}/chat/completions",
        body=requests.Timeout("timeout"),
    )
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(responses.calls) == 2


@responses.activate
def test_chat_malformed_response_retries(config_path: Path) -> None:
    client = VLLMClient(
        config_path=config_path,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
    )
    responses.add(
        responses.POST,
        f"{MODELS_URL}/chat/completions",
        json={"choices": []},
    )
    responses.add(
        responses.POST,
        f"{MODELS_URL}/chat/completions",
        json=_completion_response("fixed"),
    )
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result == "fixed"


def test_encode_image_grayscale() -> None:
    client = VLLMClient(config_path=REPO_ROOT / "configs" / "models.yaml")
    gray = np.zeros((8, 8), dtype=np.uint8)
    url = client._encode_image(gray)
    assert url.startswith("data:image/png;base64,")


def test_encode_image_float_normalizes() -> None:
    client = VLLMClient(config_path=REPO_ROOT / "configs" / "models.yaml")
    image = np.ones((2, 2, 3), dtype=np.float32)
    url = client._encode_image(image)
    assert url.startswith("data:image/png;base64,")
