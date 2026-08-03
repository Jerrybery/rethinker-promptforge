"""Symbolic action primitives for robot execution.

Primitives are label-based wrappers around perception (DINO) and the low-level
``RobotInterface``. Grasp and place poses are resolved symbolically: the
primitive queries detection results and delegates 3-D pose estimation to a
stubbed module that will later be backed by AnyGrasp or an equivalent grasp
pose estimator. No concrete grasp/place coordinates are emitted by the Planner
or Rethinker.

When the task metadata provides an ``object_actors`` mapping (semantic label
-> env actor attribute) and the robot backend exposes the RoboTwin
actor-level skills (``grasp_object``/``place_object_at``/``lift``), the
manipulation primitives execute real actor-level motions instead of the
stub pose sequence.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from common.schema import DetectedObject
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

    The library is constructed with a ``RobotInterface`` and any object
    exposing ``detect(image) -> list[DetectedObject]`` (a ``DINOClient`` or
    an ``OracleDetector``). Each primitive method consumes semantic labels
    and returns a ``PrimitiveResult``.

    Args:
        robot: High-level robot facade.
        dino: Detector used for label-based detection.
        object_actors: Optional mapping of semantic label -> env actor
            attribute name. When set and the backend exposes actor-level
            skills, pick/place/move_aside execute real RoboTwin motions.
        place_offsets: Optional per-place-label ``[dx, dy, dz]`` offset (or
            ``{"offset": [...], "mirror_with_arm": bool}``) applied when
            placing beside a target actor.
        grasp_contact_point_id: When true, grasps pass the RoboTwin
            contact_point_id convention (right=0, left=2) first, matching
            tasks like ``place_empty_cup`` whose play_once grasps that way.
    """

    def __init__(
        self,
        robot: RobotInterface,
        dino: Any,
        object_actors: dict[str, str] | None = None,
        place_offsets: dict[str, Any] | None = None,
        grasp_contact_point_id: bool = False,
    ) -> None:
        self.robot = robot
        self.dino = dino
        self.object_actors = dict(object_actors or {})
        self.place_offsets = dict(place_offsets or {})
        self.grasp_contact_point_id = bool(grasp_contact_point_id)

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

    def _actor_backend(self) -> Any | None:
        """Return the robot backend when it exposes actor-level skills."""
        backend = getattr(self.robot, "_backend", None)
        if backend is not None and callable(getattr(backend, "grasp_object", None)):
            return backend
        return None

    def _place_offset(self, target_label: str | None) -> tuple[list[float], bool]:
        """Return ``(offset, mirror_with_arm)`` for ``target_label``."""
        spec = self.place_offsets.get(target_label)
        if spec is None:
            return [0.0, 0.0, 0.0], False
        if isinstance(spec, dict):
            return (
                [float(v) for v in spec.get("offset", [0.0, 0.0, 0.0])],
                bool(spec.get("mirror_with_arm", False)),
            )
        return [float(v) for v in spec], False

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

        With an ``object_actors`` mapping and an actor-level backend this
        runs the real RoboTwin grasp sequence (grasp + lift). Otherwise it
        falls back to the stub sequence: approach, grasp, close gripper.
        """
        attr = self.object_actors.get(label)
        backend = self._actor_backend()
        if attr is not None and backend is not None:
            try:
                grasp = backend.grasp_object(
                    attr,
                    arm_tag="auto",
                    contact_point_id=self.grasp_contact_point_id,
                )
                if not grasp.get("success"):
                    return PrimitiveResult(
                        success=False,
                        status=f"grasp of {label!r} failed",
                        data={"grasp": grasp},
                    )
                lift = backend.lift(arm_tag=grasp.get("arm_tag"), dz=0.08)
                if not lift.get("carried", True):
                    # Gripper closed on nothing (or the object slipped):
                    # retry the grasp with the other contact-point mode.
                    logger.warning(
                        "pick {!r}: grasp did not hold; retrying with "
                        "alternate contact mode",
                        label,
                    )
                    grasp = backend.grasp_object(
                        attr,
                        arm_tag="auto",
                        contact_point_id=not self.grasp_contact_point_id,
                    )
                    if not grasp.get("success"):
                        return PrimitiveResult(
                            success=False,
                            status=f"grasp of {label!r} failed",
                            data={"grasp": grasp},
                        )
                    lift = backend.lift(arm_tag=grasp.get("arm_tag"), dz=0.08)
                if not lift.get("success"):
                    return PrimitiveResult(
                        success=False,
                        status=f"grasp of {label!r} failed to hold",
                        data={"grasp": grasp, "lift": lift},
                    )
            except Exception as exc:
                return PrimitiveResult(
                    success=False,
                    status=f"pick of {label!r} raised: {exc}",
                )
            return PrimitiveResult(
                success=True,
                status=f"picked {label!r}",
                data={"grasp": grasp, "lift": lift},
            )

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

        With an ``object_actors`` mapping and an actor-level backend this
        runs the real RoboTwin place skill: onto the target's functional
        point when it has one, else beside the target (or back at the grasp
        spot when ``target_label`` is None/unmapped) plus the configured
        ``place_offsets`` entry. Otherwise the stub sequence runs.
        """
        backend = self._actor_backend()
        if backend is not None and self.object_actors:
            target_attr = (
                self.object_actors.get(target_label) if target_label else None
            )
            offset, mirror = self._place_offset(target_label)
            try:
                result = backend.place_object_at(
                    target_attr_name=target_attr,
                    offset=offset,
                    mirror_offset_with_arm=mirror,
                )
            except Exception as exc:
                return PrimitiveResult(
                    success=False,
                    status=f"place at {target_label!r} raised: {exc}",
                )
            return PrimitiveResult(
                success=bool(result.get("success")),
                status=(
                    f"placed at {target_label!r}"
                    if target_label
                    else "placed at current pose"
                ),
                data={"place": result},
            )

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
        """Move an object (or the held object) to a safe aside location.

        With an ``object_actors`` mapping and an actor-level backend this
        grasps the object, lifts it, and places it back on the table with a
        small lateral shift. Otherwise the stub sequence runs.
        """
        attr = self.object_actors.get(label) if label else None
        backend = self._actor_backend()
        if attr is not None and backend is not None:
            try:
                grasp = backend.grasp_object(
                    attr,
                    arm_tag="auto",
                    contact_point_id=self.grasp_contact_point_id,
                )
                if not grasp.get("success"):
                    return PrimitiveResult(
                        success=False,
                        status=f"grasp of {label!r} failed",
                        data={"grasp": grasp},
                    )
                lift = backend.lift(arm_tag=grasp.get("arm_tag"), dz=0.08)
                if not lift.get("carried", True):
                    logger.warning(
                        "move_aside {!r}: grasp did not hold; retrying with "
                        "alternate contact mode",
                        label,
                    )
                    grasp = backend.grasp_object(
                        attr,
                        arm_tag="auto",
                        contact_point_id=not self.grasp_contact_point_id,
                    )
                    if not grasp.get("success"):
                        return PrimitiveResult(
                            success=False,
                            status=f"grasp of {label!r} failed",
                            data={"grasp": grasp},
                        )
                    lift = backend.lift(arm_tag=grasp.get("arm_tag"), dz=0.08)
                if not lift.get("success"):
                    return PrimitiveResult(
                        success=False,
                        status=f"grasp of {label!r} failed to hold",
                        data={"grasp": grasp, "lift": lift},
                    )
                if callable(getattr(backend, "place_aside", None)):
                    result = backend.place_aside()
                else:
                    result = backend.place_object_at(offset=[0.0, -0.2, 0.0])
            except Exception as exc:
                return PrimitiveResult(
                    success=False,
                    status=f"move_aside of {label!r} raised: {exc}",
                )
            return PrimitiveResult(
                success=bool(result.get("success")),
                status=f"moved {label!r} aside",
                data={"grasp": grasp, "place": result},
            )

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
        self.robot.stop()
        return PrimitiveResult(success=True, status="stopped")
