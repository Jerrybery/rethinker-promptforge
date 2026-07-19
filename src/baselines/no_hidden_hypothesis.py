"""No-hidden-hypothesis baseline: Rethinker without the hypothesis field.

Uses a prompt variant identical to the full v1 Rethinker prompt except that
no ``hidden_hypothesis`` field is requested or produced. Implemented as a
thin subclass of ``RethinkerAgent``; only the prompt version differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from llm.vllm_client import VLLMClient
from rethinker.agent import RethinkerAgent

if TYPE_CHECKING:
    from pathlib import Path

NO_HYPOTHESIS_PROMPT_VERSION = "v1_nohh"


class NoHiddenHypothesisRethinker(RethinkerAgent):
    """Rethinker variant whose prompt omits the hidden-hypothesis field."""

    def __init__(
        self,
        vllm_client: VLLMClient | None = None,
        config_path: "str | Path | None" = None,
    ) -> None:
        super().__init__(
            vllm_client=vllm_client,
            prompt_version=NO_HYPOTHESIS_PROMPT_VERSION,
            config_path=config_path,
        )
        logger.info("NoHiddenHypothesisRethinker initialized (no hidden_hypothesis)")
