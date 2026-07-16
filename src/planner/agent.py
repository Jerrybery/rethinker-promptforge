"""Planner agent: semantic target-label planning from Rethinker analysis."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from common.schema import Feedback, MissionType, PlannerOutput, RethinkerOutput
from llm.parser import extract_json
from llm.vllm_client import VLLMClient
from planner.memory import PlannerMemory
from planner.prompts.registry import PromptRegistry
from planner.schema import PlannerContext

if TYPE_CHECKING:
    from pathlib import Path


class PlannerAgent:
    """Semantic planner that maps Rethinker missions to target labels.

    The agent consumes a RethinkerOutput, a DINO label set, an action
    library, prior planner memory, and feedback. It calls a VLM and parses
    the response into a PlannerOutput. It never emits low-level control
    or receives raw images.
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
            "PlannerAgent initialized: prompt_version={}", prompt_version
        )

    def act(
        self,
        rethinker_output: RethinkerOutput,
        dino_labels: list[str],
        action_library: list[str] | None = None,
        memory: PlannerMemory | None = None,
        previous_feedback: Feedback | None = None,
    ) -> PlannerOutput:
        """Run one Planner reasoning step.

        Args:
            rethinker_output: high-level mission analysis from the Rethinker.
            dino_labels: list of DINO label strings available in the scene.
            action_library: optional list of available primitive actions.
            memory: optional PlannerMemory summary.
            previous_feedback: optional Feedback from the last step.

        Returns:
            A validated PlannerOutput whose pick and place labels
            are members of *dino_labels* (pick may be `"none"` when the
            mission is `STOP`).

        Raises:
            ValueError: if the model response cannot be parsed into a valid
                PlannerOutput, if mission is not a valid
                MissionType, or if pick/place are not in the
                provided label set.
        """
        context = PlannerContext(
            rethinker_output=rethinker_output,
            dino_labels=list(dino_labels),
            action_library=list(action_library or []),
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
            "PlannerAgent.act calling VLLM for mission={}",
            rethinker_output.mission_type.value,
        )
        raw_response = self.vllm_client.chat(messages)
        logger.debug("PlannerAgent raw response: {}", raw_response)

        try:
            output = extract_json(raw_response, PlannerOutput)
        except ValueError as exc:
            logger.warning(
                "PlannerAgent failed to parse response for mission={}: {}",
                rethinker_output.mission_type.value,
                exc,
            )
            raise ValueError(
                f"PlannerAgent could not parse model response: {exc}"
            ) from exc

        self._validate_plan(output, set(dino_labels))
        return output

    def _render_user(self, context: PlannerContext) -> str:
        """Substitute context fields into the user prompt template."""
        kwargs = context.to_prompt_kwargs()
        text = self._user_template
        for key, value in kwargs.items():
            text = text.replace(f"{{{{{key}}}}}", value)
        return text

    def _validate_plan(
        self,
        output: PlannerOutput,
        label_set: set[str],
    ) -> None:
        """Ensure the parsed plan respects mission and label constraints."""
        stop_with_no_pick = (
            output.mission is MissionType.STOP and output.pick == "none"
        )
        if not stop_with_no_pick and output.pick not in label_set:
            raise ValueError(
                f"Pick label '{output.pick}' is not in the DINO label set"
            )

        if output.place is not None and output.place not in label_set:
            raise ValueError(
                f"Place label '{output.place}' is not in the DINO label set"
            )

    @staticmethod
    def output_schema_description() -> dict[str, Any]:
        """Return a JSON-serializable description of PlannerOutput."""
        return {
            "plan_id": "string (required, unique identifier)",
            "mission": "PICK_AND_PLACE | PICK_ONLY | MOVE_ASIDE | REOBSERVE | STOP",
            "pick": "string (required, must be a DINO label; STOP may use 'none')",
            "place": "string | null (must be a DINO label when not null)",
        }
