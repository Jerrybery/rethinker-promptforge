"""Symbolic action primitives for robot execution.

Primitives are label-based wrappers around perception (DINO) and the low-level
``RobotInterface``. Grasp and place poses are resolved symbolically: the
primitive queries detection results and delegates 3-D pose estimation to a
stubbed module that will later be backed by AnyGrasp or an equivalent grasp
pose estimator. No concrete grasp/place coordinates are emitted by the Planner
or Rethinker.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from common.schema import DetectedObject
from perception.dino_client import DINOClient
from robot.interface import RobotInterface
from robot.state import Pose, RobotState


class PrimitiveResult:
    """Outcome of a single primitive execution.

    Attributes:
        success: Whether the primitive completed successfully.
        status: Human-readable status message.
        data: Optional payload (detections, resolved poses, etc.).
    """

    def __init__(
        self,
        success: bool,
        status: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.success = success
        self.status = status
        self.data = data or {}

    def __repr__(self) -> str:
        return f"PrimitiveResult(success={self.success}, status={self.status!r})"


class PrimitiveLibrary:
    """Collection of symbolic manipulation primitives.

    The library is constructed with a ``RobotInterface`` and a ``DINOClient``
    (or any object exposing ``detect(image) -> list[DetectedObject]``). Each
    primitive method consumes semantic labels and returns a ``PrimitiveResult``.
    """

    def __init__(self, robot: RobotInterface, dino: DINOClient) -> None:
        self.robot = robot
        self.dino = dino

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _detect(
        self,
        label: str | None = None,
    ) -> tuple[DetectedObject | None, list[DetectedObject]]:
        """Run detection on the current camera image.

        Returns the detection matching ``label`` (case-insensitive) and the
        full detection list.
        """
        state = self.robot.read_state()
        detections = self.dino.detect(state.camera_image)
        if label is None:
            return None, detections

        lowered = label.lower()
        for det in detections:
            if det.label.lower() == lowered:
                return det, detections
        return None, detections

    def _resolve_grasp_pose(self, detection: DetectedObject, state: RobotState) -> Pose:
        """Resolve a grasp pose from a DINO detection.

        TODO: replace this stub with AnyGrasp (or equivalent) once the grasp
        pose estimator is integrated. The current implementation returns a
        safe placeholder pose so that primitives can be exercised in mock mode.
        """
        logger.warning(
            "Grasp pose resolution is stubbed for label={}; "
            "integrate AnyGrasp here.",
            detection.label,
        )
        return Pose(position=[0.5, 0.0, 0.3], orientation=[0.0, 0.0, 0.0, 1.0])

    def _resolve_place_pose(
        self,
        detection: DetectedObject | None,
        state: RobotState,
    ) -> Pose:
        """Resolve a place pose from an optional target detection.

        TODO: integrate target-affordance / placement pose estimation.
        """
        logger.warning(
            "Place pose resolution is stubbed for target={}; "
            "integrate placement pose estimator here.",
            detection.label if detection else None,
        )
        return Pose(position=[0.5, 0.1, 0.3], orientation=[0.0, 0.0, 0.0, 1.0])

    def _resolve_aside_pose(
        self,
        detection: DetectedObject | None,
        state: RobotState,
    ) -> Pose:
        """Resolve a collision-free pose for moving an object aside.

        TODO: integrate motion planning / obstacle-aware aside pose selection.
        """
        logger.warning(
            "Aside pose resolution is stubbed for label={}; "
            "integrate motion planner here.",
            detection.label if detection else None,
        )
        return Pose(position=[0.4, -0.2, 0.3], orientation=[0.0, 0.0, 0.0, 1.0])

    # ------------------------------------------------------------------ #
    # Public primitives
    # ------------------------------------------------------------------ #

    def pick(self, label: str, arm_tag: str = "right") -> PrimitiveResult:
        """Symbolically pick the object named ``label``.

        Sequence: approach, grasp, close gripper. Grasp pose is resolved from
        DINO detection via a stub that will later call AnyGrasp.
        """
        detection, detections = self._detect(label)
        if detection is None:
            return PrimitiveResult(
                success=False,
                status=f"object {label!r} not detected",
                data={"detections": detections},
            )

        state = self.robot.read_state(arm_tag=arm_tag)
        grasp_pose = self._resolve_grasp_pose(detection, state)

        self.robot.gripper(open=True, arm_tag=arm_tag)
        self.robot.move_to(grasp_pose, arm_tag=arm_tag)
        self.robot.gripper(open=False, arm_tag=arm_tag)

        return PrimitiveResult(
            success=True,
            status=f"picked {label!r}",
            data={"detection": detection, "grasp_pose": grasp_pose},
        )

    def place(self, target_label: str | None = None, arm_tag: str = "right") -> PrimitiveResult:
        """Symbolically place the currently held object.

        If ``target_label`` is provided, the primitive attempts to locate the
        target with DINO and uses it to resolve a place pose (stub).
        """
        detection, detections = self._detect(target_label)
        state = self.robot.read_state(arm_tag=arm_tag)
        place_pose = self._resolve_place_pose(detection, state)

        self.robot.move_to(place_pose, arm_tag=arm_tag)
        self.robot.gripper(open=True, arm_tag=arm_tag)

        return PrimitiveResult(
            success=True,
            status=f"placed at {target_label!r}" if target_label else "placed at current pose",
            data={"detection": detection, "place_pose": place_pose},
        )

    def move_aside(self, label: str | None = None, arm_tag: str = "right") -> PrimitiveResult:
        """Move an object (or the held object) to a safe aside location."""
        detection, detections = self._detect(label)
        state = self.robot.read_state(arm_tag=arm_tag)
        aside_pose = self._resolve_aside_pose(detection, state)

        self.robot.move_to(aside_pose, arm_tag=arm_tag)

        return PrimitiveResult(
            success=True,
            status=f"moved {label!r} aside" if label else "moved held object aside",
            data={"detection": detection, "aside_pose": aside_pose},
        )

    def reobserve(self) -> PrimitiveResult:
        """Refresh visual observations and return detections."""
        state = self.robot.read_state()
        detections = self.dino.detect(state.camera_image)
        return PrimitiveResult(
            success=True,
            status="reobserved scene",
            data={"state": state, "detections": detections},
        )

    def stop(self) -> PrimitiveResult:
        """Halt execution and report a clean stop."""
        return PrimitiveResult(success=True, status="stopped")
