"""Unit tests for the vLLM client."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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


def test_client_logs_configuration(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_logger = MagicMock()
    monkeypatch.setattr("llm.vllm_client.logger", mock_logger)
    client = VLLMClient(config_path=config_path)
    mock_logger.info.assert_called_once()
    log_args = mock_logger.info.call_args.args
    assert client.model_id in log_args
    assert client.base_url in log_args
    assert client.temperature in log_args
    assert client.top_p in log_args
    assert client.max_tokens in log_args



class _FakeHTTPError(__import__("requests").RequestException):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = type("Resp", (), {"status_code": status_code})()


def _client_for_ratelimit():
    from llm.vllm_client import VLLMClient

    c = VLLMClient.__new__(VLLMClient)
    c.model_id = "m"
    c.base_url = "http://x"
    c.api_key = None
    c.temperature = 0.0
    c.top_p = 1.0
    c.max_tokens = 16
    c.max_retries = 3
    c.base_delay = 0.01
    c.max_delay = 0.02
    c.reasoning_effort = None
    return c


def test_rate_limit_uses_long_backoff_then_succeeds(monkeypatch) -> None:
    import llm.vllm_client as vc

    client = _client_for_ratelimit()
    calls = {"posts": 0}
    sleeps: list[float] = []

    def fake_post(*a, **kw):
        calls["posts"] += 1
        if calls["posts"] <= 2:
            raise _FakeHTTPError(429)
        return type(
            "Resp",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "choices": [{"message": {"content": "ok"}}]
                },
            },
        )()

    monkeypatch.setattr(vc.requests, "post", fake_post)
    monkeypatch.setattr(vc.time, "sleep", lambda d: sleeps.append(d))

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert sleeps == [30.0, 60.0]
    assert calls["posts"] == 3


def test_rate_limit_exhausts_extended_budget(monkeypatch) -> None:
    import llm.vllm_client as vc
    import pytest

    client = _client_for_ratelimit()
    sleeps: list[float] = []
    monkeypatch.setattr(
        vc.requests, "post",
        lambda *a, **kw: (_ for _ in ()).throw(_FakeHTTPError(403)),
    )
    monkeypatch.setattr(vc.time, "sleep", lambda d: sleeps.append(d))

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        client.chat([{"role": "user", "content": "hi"}])
    assert sleeps == [30.0, 60.0, 120.0, 240.0]


def test_non_rate_limit_error_keeps_short_backoff(monkeypatch) -> None:
    import llm.vllm_client as vc
    import pytest
    import requests as real_requests

    client = _client_for_ratelimit()
    sleeps: list[float] = []

    def fake_post(*a, **kw):
        raise real_requests.ConnectionError("boom")

    monkeypatch.setattr(vc.requests, "post", fake_post)
    monkeypatch.setattr(vc.time, "sleep", lambda d: sleeps.append(d))

    with pytest.raises(RuntimeError, match="after 4 attempts"):
        client.chat([{"role": "user", "content": "hi"}])
    assert all(d <= 0.02 for d in sleeps)
    assert len(sleeps) == 3
