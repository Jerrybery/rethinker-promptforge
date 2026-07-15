"""Shared Pydantic schemas for Rethinker + EmbodiedPromptForge."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MissionType(str, Enum):
    """High-level mission categories the Rethinker can select."""

    PICK_AND_PLACE = "PICK_AND_PLACE"
    PICK_ONLY = "PICK_ONLY"
    MOVE_ASIDE = "MOVE_ASIDE"
    REOBSERVE = "REOBSERVE"
    STOP = "STOP"


class RethinkerOutput(BaseModel):
    """High-level mission decision produced by the Rethinker.

    Rethinker reasons about what to do next but must not emit low-level
    joint-angle commands (those belong to the Executor).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mission_type: MissionType
    reasoning: str = Field(..., min_length=1)
    target_object: Optional[str] = None
    target_container: Optional[str] = None
    arm_hint: Optional[Literal["left", "right", "both"]] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_joint_angles(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for key in values:
            lowered = key.lower()
            if "joint" in lowered and "angle" in lowered:
                raise ValueError(
                    f"RethinkerOutput cannot contain joint angles (got field: {key})"
                )
        return values


class PlannerOutput(BaseModel):
    """Semantic motion plan produced by the Planner.

    The Planner must not emit concrete grasp/place 3D coordinates;
    those are resolved later by perception/execution modules.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(..., min_length=1)
    trajectory_name: Optional[str] = None
    waypoints: List[str] = Field(default_factory=list)
    approach_type: Optional[str] = None
    gripper_action: Optional[Literal["open", "close", "hold"]] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_grasp_place_coordinates(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for key in values:
            lowered = key.lower()
            if ("grasp" in lowered or "place" in lowered) and any(
                token in lowered for token in ("coord", "position", "point", "pose")
            ):
                raise ValueError(
                    f"PlannerOutput cannot contain grasp/place coordinates (got field: {key})"
                )
        return values


class ExecutorOutput(BaseModel):
    """Low-level robot command produced by the Executor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: int = Field(..., ge=0)
    joint_angles: List[float]
    gripper_state: float = Field(..., ge=0.0, le=1.0)
    timestamp: Optional[float] = None


class TaskUnit(BaseModel):
    """A single manipulation task description."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)
    mission_type: MissionType
    objects: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class Feedback(BaseModel):
    """Observation / outcome feedback for a single step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    observation: Optional[str] = None
    error_message: Optional[str] = None
    reward: Optional[float] = None


class EpisodeStep(BaseModel):
    """One step within an episode, tying together task, plans, and feedback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: int = Field(..., ge=0)
    task: TaskUnit
    rethinker_output: RethinkerOutput
    planner_output: Optional[PlannerOutput] = None
    executor_output: Optional[ExecutorOutput] = None
    feedback: Optional[Feedback] = None


class Episode(BaseModel):
    """A full episode composed of one or more steps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    steps: List[EpisodeStep] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class DetectedObject(BaseModel):
    """A single object detection result.

    Coordinates are expressed as top-left and bottom-right corners in
    pixel space (x1, y1, x2, y2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(..., min_length=1)
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bbox_is_ordered(self) -> "DetectedObject":
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"bbox must satisfy x2 > x1 and y2 > y1, got {self.bbox}"
            )
        return self
