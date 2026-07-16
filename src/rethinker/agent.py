"""Rethinker agent: high-level mission reasoning from vision and memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from common.schema import DetectedObject, Feedback, MissionType, RethinkerOutput
from llm.parser import extract_json
from llm.vllm_client import VLLMClient
from rethinker.memory import RethinkerMemory
from rethinker.prompts.registry import PromptRegistry
from rethinker.schema import RethinkerContext

if TYPE_CHECKING:
    from pathlib import Path


class RethinkerAgent:
    """High-level reasoning agent that selects the next mission.

    The agent consumes an RGB image, DINO detections, a task goal, prior
    memory, and feedback. It calls a VLM and parses the response into a
    ``RethinkerOutput``. It never emits low-level control.
    """

    def __init__(
        self,
        vllm_client: VLLMClient | None = None,
        prompt_version: str = "v0",
        config_path: "str | Path | None" = None,
    ) -> None:
        self.vllm_client = vllm_client or VLLMClient(config_path=config_path)
        self.prompt_version = prompt_version
        self._system_template, self._user_template = PromptRegistry.load(prompt_version)
        logger.info(
            "RethinkerAgent initialized: prompt_version={}", prompt_version
        )

    def act(
        self,
        task_goal: str,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
        memory: RethinkerMemory | None = None,
        previous_feedback: Feedback | None = None,
    ) -> RethinkerOutput:
        """Run one Rethinker reasoning step.

        Args:
            task_goal: natural-language instruction.
            rgb_image: current RGB image as a numpy array.
            detections: list of ``DetectedObject`` from DINO.
            memory: optional ``RethinkerMemory`` summary.
            previous_feedback: optional ``Feedback`` from the last step.

        Returns:
            A validated ``RethinkerOutput``.

        Raises:
            ValueError: if the model response cannot be parsed into a valid
                ``RethinkerOutput`` or violates the high-level-only contract.
        """
        context = RethinkerContext(
            task_goal=task_goal,
            rgb_image=rgb_image,
            detections=list(detections),
            memory_summary=(
                memory.summarize(k=2) if memory is not None else "No prior rounds."
            ),
            previous_feedback=previous_feedback,
        )

        user_prompt = self._render_user(context)
        messages = [
            {"role": "system", "content": self._system_template},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug(
            "RethinkerAgent.act calling VLLM for task_goal={}", task_goal
        )
        raw_response = self.vllm_client.chat(messages, images=[rgb_image])
        logger.debug("RethinkerAgent raw response: {}", raw_response)

        try:
            output = extract_json(raw_response, RethinkerOutput)
        except ValueError as exc:
            logger.warning(
                "RethinkerAgent failed to parse response for task_goal={}: {}",
                task_goal,
                exc,
            )
            raise ValueError(
                f"RethinkerAgent could not parse model response: {exc}"
            ) from exc

        return output

    def _render_user(self, context: RethinkerContext) -> str:
        """Substitute context fields into the user prompt template."""
        feedback_text = (
            context.previous_feedback.model_dump_json()
            if context.previous_feedback is not None
            else "None"
        )
        return (
            self._user_template.replace("{{task_goal}}", context.task_goal)
            .replace("{{detections}}", context.detection_summary)
            .replace("{{memory_summary}}", context.memory_summary)
            .replace("{{previous_feedback}}", feedback_text)
        )

    @staticmethod
    def output_schema_description() -> dict[str, Any]:
        """Return a JSON-serializable description of ``RethinkerOutput``."""
        return {
            "mission_type": "PICK_AND_PLACE | PICK_ONLY | MOVE_ASIDE | REOBSERVE | STOP",
            "reasoning": "string (required, non-empty)",
            "target_object": "string | null",
            "target_container": "string | null",
            "arm_hint": "left | right | both | null",
        }
