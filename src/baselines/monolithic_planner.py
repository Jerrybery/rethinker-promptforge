"""Monolithic planner baseline: one agent for reasoning and planning.

This baseline collapses the Rethinker -> Planner chain into a single
vision-language call that performs scene understanding, mission selection,
and pick/place target selection directly from the raw RGB image. It plugs
into ``ClosedLoopRunner`` unmodified: the monolithic agent fills the
``"rethinker"`` slot and a paired ``PassThroughPlanner`` translates its
stored decision into a ``PlannerOutput`` for the ``"planner"`` slot.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from baselines.prompts.registry import PromptRegistry
from baselines.schema import MonolithicDecision
from common.schema import (
    DetectedObject,
    Feedback,
    MissionType,
    PlannerOutput,
    RethinkerOutput,
)
from llm.parser import extract_json
from llm.vllm_client import VLLMClient
from rethinker.schema import RethinkerContext

if TYPE_CHECKING:
    from pathlib import Path


class MonolithicPlannerAgent:
    """Single-agent baseline doing scene understanding and planning at once.

    Consumes an RGB image, DINO detections, a task goal, and optional
    feedback in one prompt; parses the response into a ``MonolithicDecision``;
    and returns the semantic part as a ``RethinkerOutput`` so it can occupy
    the runner's ``"rethinker"`` slot. The full decision is kept on
    ``last_decision`` for the paired ``PassThroughPlanner``. Cross-round
    memory is accepted for runner compatibility but intentionally unused.
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
        self._last_decision: MonolithicDecision | None = None
        logger.info(
            "MonolithicPlannerAgent initialized: prompt_version={}", prompt_version
        )

    @property
    def last_decision(self) -> MonolithicDecision | None:
        """Return the most recent monolithic decision, if any."""
        return self._last_decision

    def act(
        self,
        task_goal: str,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
        memory: Any | None = None,
        previous_feedback: Feedback | None = None,
    ) -> RethinkerOutput:
        """Run one monolithic reasoning + planning step.

        Args:
            task_goal: natural-language instruction.
            rgb_image: current RGB image as a numpy array.
            detections: list of ``DetectedObject`` from DINO.
            memory: accepted for runner compatibility; unused by design.
            previous_feedback: optional ``Feedback`` from the last step.

        Returns:
            A validated ``RethinkerOutput`` carrying the semantic part of the
            monolithic decision. The full decision is stored on
            ``last_decision``.

        Raises:
            ValueError: if the model response cannot be parsed into a valid
                ``MonolithicDecision``.
        """
        context = RethinkerContext(
            task_goal=task_goal,
            rgb_image=rgb_image,
            detections=list(detections),
            previous_feedback=previous_feedback,
        )
        feedback_text = (
            context.previous_feedback.model_dump_json()
            if context.previous_feedback is not None
            else "None"
        )
        user_prompt = (
            self._user_template.replace("{{task_goal}}", context.task_goal)
            .replace("{{detections}}", context.detection_summary)
            .replace("{{previous_feedback}}", feedback_text)
        )
        messages = [
            {"role": "system", "content": self._system_template},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug(
            "MonolithicPlannerAgent.act calling VLLM for task_goal={}", task_goal
        )
        raw_response = self.vllm_client.chat(messages, images=[rgb_image])
        logger.debug("MonolithicPlannerAgent raw response: {}", raw_response)

        try:
            decision = extract_json(raw_response, MonolithicDecision)
        except ValueError as exc:
            logger.warning(
                "MonolithicPlannerAgent failed to parse response for task_goal={}: {}",
                task_goal,
                exc,
            )
            raise ValueError(
                f"MonolithicPlannerAgent could not parse model response: {exc}"
            ) from exc

        self._last_decision = decision
        return decision.to_rethinker_output()


class PassThroughPlanner:
    """Translate the paired monolithic agent's decision into a PlannerOutput.

    Makes no LLM call of its own; the single monolithic call already produced
    both the semantic decision and the plan. Memory and feedback are accepted
    for runner compatibility but unused.
    """

    def __init__(self, monolithic: MonolithicPlannerAgent) -> None:
        self.monolithic = monolithic

    def act(
        self,
        rethinker_output: RethinkerOutput,
        dino_labels: list[str],
        action_library: list[str] | None = None,
        memory: Any | None = None,
        previous_feedback: Feedback | None = None,
    ) -> PlannerOutput:
        """Return the plan stored by the paired monolithic agent.

        Raises:
            ValueError: if the monolithic agent has not run yet, if the given
                ``rethinker_output`` does not match the stored decision, or if
                pick/place labels are not in the DINO label set.
        """
        decision = self.monolithic.last_decision
        if decision is None:
            raise ValueError(
                "PassThroughPlanner has no monolithic decision; "
                "the monolithic agent must act first"
            )
        if decision.to_rethinker_output() != rethinker_output:
            raise ValueError(
                "rethinker_output does not match the stored monolithic decision"
            )
        plan = decision.to_planner_output(plan_id=f"mono-{uuid.uuid4().hex[:8]}")
        self._validate_plan(plan, set(dino_labels))
        return plan

    @staticmethod
    def _validate_plan(output: PlannerOutput, label_set: set[str]) -> None:
        """Ensure the plan respects mission and label constraints."""
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


def build_monolithic_agents(
    vllm_client: VLLMClient | None = None,
    prompt_version: str = "v0",
    config_path: "str | Path | None" = None,
) -> dict[str, Any]:
    """Build the monolithic baseline's ``"rethinker"``/``"planner"`` pair.

    The caller supplies the ``"executor"`` entry and passes the dict to
    ``ClosedLoopRunner``.
    """
    monolithic = MonolithicPlannerAgent(
        vllm_client=vllm_client,
        prompt_version=prompt_version,
        config_path=config_path,
    )
    return {
        "rethinker": monolithic,
        "planner": PassThroughPlanner(monolithic),
    }
