"""Forge planner agent: runs the real PlannerAgent on SimEnv observations.

The forge optimizes the *planner* prompt, so this agent calls the same
prompt templates and produces the identical ``PlannerOutput`` schema as the
real system — a forged prompt is drop-in replaceable. The only difference
is the observation source: instead of a ``RethinkerOutput`` from the vision
pipeline, the agent receives the SimEnv obs dict (state description +
detections + optional image) and synthesizes a ``RethinkerOutput``-shaped
semantic summary from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from common.schema import Feedback, PlannerOutput, RethinkerOutput, MissionType
from forge.memory import ForgePlannerMemory
from planner.agent import PlannerAgent

if TYPE_CHECKING:
    from pathlib import Path

    from llm.vllm_client import VLLMClient


def obs_to_rethinker_output(obs: dict[str, Any]) -> RethinkerOutput:
    """Convert a SimEnv obs dict into a synthetic RethinkerOutput.

    The mission type is taken from the task metadata; the reasoning field
    carries a synthesized text description of the scene (instruction, robot
    state, detections) standing in for the Rethinker's semantic analysis.

    Raises:
        ValueError: if the obs dict lacks required keys or the task mission
            type is not a valid :class:`MissionType`.
    """
    try:
        task = obs["task"]
        state = obs["state"]
        detections = obs["detections"]
        mission_type = MissionType(task["mission_type"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SimEnv observation: {exc}") from exc

    return RethinkerOutput(
        mission_type=mission_type,
        reasoning=_describe_scene(obs, task, state, detections),
    )


def _describe_scene(
    obs: dict[str, Any],
    task: dict[str, Any],
    state: dict[str, Any],
    detections: list[dict[str, Any]],
) -> str:
    """Render the obs dict as a deterministic text scene description."""
    lines = [
        f"Simulated scene observation (task {task['id']}, "
        f"step {obs.get('step_index', 0)}).",
        f"Instruction: {task['instruction']}",
    ]
    pose = state.get("pose")
    if pose:
        lines.append(
            "Robot state: end-effector position "
            f"({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}); "
            f"gripper opening {state.get('gripper', 0.0):.2f}."
        )
    if detections:
        lines.append("Detected objects:")
        for det in detections:
            lines.append(
                f"- {det['label']} (bbox={list(det['bbox'])}, "
                f"confidence={det['confidence']:.2f})"
            )
    else:
        lines.append("Detected objects: none.")
    return "\n".join(lines)


class _ImageInjectingClient:
    """Delegating client that attaches one image to every chat call."""

    def __init__(self, client: Any, image: np.ndarray) -> None:
        self._client = client
        self._image = image

    def chat(
        self, messages: list[dict], images: list[np.ndarray] | None = None
    ) -> str:
        merged = [self._image, *list(images or [])]
        return self._client.chat(messages, images=merged)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class ForgePlannerAgent:
    """Composition wrapper adapting :class:`PlannerAgent` to SimEnv obs.

    Args:
        planner: Optional pre-built :class:`PlannerAgent` to wrap. When
            omitted, one is constructed from the remaining arguments.
        vllm_client: VLLM client for the wrapped planner (mockable).
        prompt_version: Prompt registry version for the planner templates.
        config_path: Optional VLLM config path forwarded to the planner.
    """

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        vllm_client: "VLLMClient | None" = None,
        prompt_version: str = "v0",
        config_path: "str | Path | None" = None,
    ) -> None:
        self._planner = planner or PlannerAgent(
            vllm_client=vllm_client,
            prompt_version=prompt_version,
            config_path=config_path,
        )

    @property
    def planner(self) -> PlannerAgent:
        """The wrapped real planner agent."""
        return self._planner

    @property
    def prompt_version(self) -> str:
        """The prompt registry version currently in use."""
        return self._planner.prompt_version

    def set_prompt_version(self, version: str) -> None:
        """Swap candidate prompt templates without changing the interface.

        Rebuilds the wrapped planner around the same VLLM client, loading
        the templates registered under ``version``. This is the hook the
        forge loop uses to evaluate candidate prompts.
        """
        logger.info("ForgePlannerAgent swapping prompt version to {}", version)
        self._planner = PlannerAgent(
            vllm_client=self._planner.vllm_client,
            prompt_version=version,
        )

    def act_from_obs(
        self,
        obs: dict[str, Any],
        action_library: list[str] | None = None,
        memory: ForgePlannerMemory | None = None,
        previous_feedback: Feedback | None = None,
        include_image: bool = False,
    ) -> PlannerOutput:
        """Run one planner step from a SimEnv observation dict.

        Converts ``obs`` into a synthetic :class:`RethinkerOutput` plus the
        DINO label set, then delegates to the real
        :meth:`PlannerAgent.act`, so the returned :class:`PlannerOutput`
        obeys exactly the same schema and validation as the real system.

        Args:
            obs: SimEnv observation dict (see ``forge.env`` for the shape).
            action_library: optional list of available primitive actions.
            memory: optional :class:`ForgePlannerMemory` with prior rounds.
            previous_feedback: optional feedback from the last step.
            include_image: when True, attach ``obs["image"]`` to the VLLM
                call. Defaults to False, matching the real planner, which
                never receives raw images.

        Returns:
            A validated :class:`PlannerOutput`.

        Raises:
            ValueError: if the obs dict is malformed, the model response
                cannot be parsed, or pick/place labels are not in the obs
                detection label set.
        """
        rethinker_output = obs_to_rethinker_output(obs)
        dino_labels = [str(det["label"]) for det in obs["detections"]]

        if include_image and obs.get("image") is not None:
            planner = self._planner
            original = planner.vllm_client
            planner.vllm_client = _ImageInjectingClient(
                original, np.asarray(obs["image"])
            )
            try:
                return planner.act(
                    rethinker_output=rethinker_output,
                    dino_labels=dino_labels,
                    action_library=action_library,
                    memory=memory,
                    previous_feedback=previous_feedback,
                )
            finally:
                planner.vllm_client = original

        return self._planner.act(
            rethinker_output=rethinker_output,
            dino_labels=dino_labels,
            action_library=action_library,
            memory=memory,
            previous_feedback=previous_feedback,
        )
