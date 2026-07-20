"""Minimal cloud VLM client for the forge video-stage critic.

Points the shared :class:`VLLMClient` machinery (OpenAI-compatible chat
completions, base64 image data URLs, retry with exponential backoff) at the
``cloud_critic`` section of ``configs/models.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from llm.vllm_client import VLLMClient


class CloudVLMClient(VLLMClient):
    """OpenAI-compatible cloud VLM client (e.g. a hosted VLM chat endpoint).

    Reads endpoint/model/params from the ``cloud_critic`` section of the
    models config. Unlike the local endpoint, ``model_id`` is mandatory:
    the cloud provider must know which model to route the request to.

    Raises:
        ValueError: if ``cloud_critic.model_id`` is not configured.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
    ) -> None:
        super().__init__(
            config_path=config_path,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            config_section="cloud_critic",
        )
        if not self.model_id:
            raise ValueError(
                "cloud_critic.model_id is not configured; set it in "
                "configs/models.yaml to the cloud critic model identifier"
            )
