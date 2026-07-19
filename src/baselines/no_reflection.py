"""No-reflection baseline: drop feedback and cross-round memory.

Thin wrappers around the existing Rethinker and Planner agents. Each round
the wrapper (a) withholds the previous round's feedback so no failure
re-analysis happens and (b) withholds the shared memory so the planner only
sees the latest observation. The underlying agent logic is reused unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from common.schema import (
    DetectedObject,
    Feedback,
    PlannerOutput,
    RethinkerOutput,
)
from planner.agent import PlannerAgent
from rethinker.agent import RethinkerAgent


class NoReflectionRethinker:
    """Wrap a ``RethinkerAgent``, dropping feedback and memory each round."""

    def __init__(self, agent: RethinkerAgent) -> None:
        self.agent = agent
        logger.info("NoReflectionRethinker wrapping {}", type(agent).__name__)

    def act(
        self,
        task_goal: str,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
        memory: Any | None = None,
        previous_feedback: Feedback | None = None,
    ) -> RethinkerOutput:
        """Delegate to the wrapped agent with memory and feedback removed."""
        return self.agent.act(
            task_goal=task_goal,
            rgb_image=rgb_image,
            detections=detections,
            memory=None,
            previous_feedback=None,
        )


class NoReflectionPlanner:
    """Wrap a ``PlannerAgent``, dropping feedback and memory each round."""

    def __init__(self, agent: PlannerAgent) -> None:
        self.agent = agent
        logger.info("NoReflectionPlanner wrapping {}", type(agent).__name__)

    def act(
        self,
        rethinker_output: RethinkerOutput,
        dino_labels: list[str],
        action_library: list[str] | None = None,
        memory: Any | None = None,
        previous_feedback: Feedback | None = None,
    ) -> PlannerOutput:
        """Delegate to the wrapped agent with memory and feedback removed."""
        return self.agent.act(
            rethinker_output=rethinker_output,
            dino_labels=dino_labels,
            action_library=action_library,
            memory=None,
            previous_feedback=None,
        )


def build_no_reflection_agents(
    rethinker: RethinkerAgent,
    planner: PlannerAgent,
) -> dict[str, Any]:
    """Wrap full-system agents into the no-reflection baseline pair.

    The caller supplies the ``"executor"`` entry and passes the dict to
    ``ClosedLoopRunner``.
    """
    return {
        "rethinker": NoReflectionRethinker(rethinker),
        "planner": NoReflectionPlanner(planner),
    }
