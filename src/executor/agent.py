"""Executor agent: structured dispatch from PlannerOutput to primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from common.schema import DetectedObject, Feedback, MissionType, PlannerOutput
from executor.primitives import PrimitiveLibrary, PrimitiveResult
from executor.schema import ExecutorAgentOutput
from perception.dino_client import DINOClient
from robot.interface import RobotInterface


class ExecutorAgent:
    """Structured executor that consumes PlannerOutput and dispatches primitives.

    The agent performs no high-level reasoning. It resolves target labels
    against DINO detections, flags lookup failures / multiple matches /
    low-confidence detections, and forwards each mission to the
    ``PrimitiveLibrary``. The returned ``ExecutorAgentOutput`` carries the
    execution success/failure, risk flag, and natural-language feedback.

    Args:
        primitives: symbolic primitive library from Task 1.8.
        dino: object detector exposing ``detect(image) -> list[DetectedObject]``.
        robot: robot facade (currently held only for interface parity; the
            primitives own the robot interactions).
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        primitives: PrimitiveLibrary,
        dino: DINOClient,
        robot: RobotInterface,
    ) -> None:
        self.primitives = primitives
        self.dino = dino
        self.robot = robot
        self._step_index = 0

    def act(
        self,
        planner_output: PlannerOutput,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
    ) -> ExecutorAgentOutput:
        """Execute one PlannerOutput step.

        Args:
            planner_output: semantic plan produced by the Planner.
            rgb: current RGB image used for target label lookup.
            depth: optional depth image (reserved for future use).

        Returns:
            ``ExecutorAgentOutput`` with success/failure/risk status and
            natural-language feedback.
        """
        step = self._step_index
        self._step_index += 1

        mission = planner_output.mission
        pick_label = planner_output.pick
        place_label = planner_output.place

        status_parts: list[str] = []
        risk = False

        # Resolve the target label for manipulation missions.
        if mission in {
            MissionType.PICK_AND_PLACE,
            MissionType.PICK_ONLY,
            MissionType.MOVE_ASIDE,
        }:
            detection, detections = self._resolve_target(pick_label, rgb)
            if detection is None:
                msg = f"target {pick_label!r} not detected"
                logger.warning("ExecutorAgent {}: {}", step, msg)
                return ExecutorAgentOutput(
                    step_index=step,
                    success=False,
                    risk=True,
                    status=msg,
                    feedback=Feedback(success=False, error_message=msg),
                )

            matches = [d for d in detections if d.label.lower() == pick_label.lower()]
            if len(matches) > 1:
                risk = True
                status_parts.append(
                    f"multiple matches for {pick_label!r}; using best confidence"
                )
            if detection.confidence < self.DEFAULT_CONFIDENCE_THRESHOLD:
                risk = True
                status_parts.append(
                    f"low confidence {detection.confidence:.2f} for {pick_label!r}"
                )

        result: PrimitiveResult | None = None

        if mission is MissionType.PICK_AND_PLACE:
            pick_result = self.primitives.pick(pick_label)
            if not pick_result.success:
                return self._finalize(
                    step, pick_result, risk, status_parts, failed_phase="pick"
                )
            result = self.primitives.place(place_label)
        elif mission is MissionType.PICK_ONLY:
            result = self.primitives.pick(pick_label)
        elif mission is MissionType.MOVE_ASIDE:
            result = self.primitives.move_aside(pick_label)
        elif mission is MissionType.REOBSERVE:
            result = self.primitives.reobserve()
        elif mission is MissionType.STOP:
            result = self.primitives.stop()
        else:
            msg = f"unsupported mission {mission.value!r}"
            logger.warning("ExecutorAgent {}: {}", step, msg)
            return ExecutorAgentOutput(
                step_index=step,
                success=False,
                risk=True,
                status=msg,
                feedback=Feedback(success=False, error_message=msg),
            )

        return self._finalize(step, result, risk, status_parts)

    def _resolve_target(
        self,
        label: str,
        rgb: np.ndarray,
    ) -> tuple[DetectedObject | None, list[DetectedObject]]:
        """Find the best detection matching ``label`` (case-insensitive)."""
        detections = self.dino.detect(rgb)
        matches = [d for d in detections if d.label.lower() == label.lower()]
        if not matches:
            return None, detections
        best = max(matches, key=lambda d: d.confidence)
        return best, detections

    def _finalize(
        self,
        step_index: int,
        result: PrimitiveResult,
        risk: bool,
        status_parts: list[str],
        failed_phase: str | None = None,
    ) -> ExecutorAgentOutput:
        """Build an ``ExecutorAgentOutput`` from a primitive result."""
        if failed_phase is not None:
            success = False
            status = f"{failed_phase} failed: {result.status}"
            feedback = Feedback(success=False, error_message=status)
        else:
            success = result.success
            status = result.status
            feedback = Feedback(
                success=result.success,
                observation=result.status,
                error_message=None if result.success else result.status,
            )

        if status_parts:
            status = "; ".join(status_parts + [status])

        if not result.success:
            risk = True

        return ExecutorAgentOutput(
            step_index=step_index,
            success=success,
            risk=risk,
            status=status,
            feedback=feedback,
        )
