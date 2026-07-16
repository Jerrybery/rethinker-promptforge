"""Rethinker-specific input and context schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from common.schema import DetectedObject, Feedback


class RethinkerContext(BaseModel):
    """Scene and task context supplied to the Rethinker agent.

    The Rethinker consumes this information to decide the next high-level
    mission. It must not receive low-level control targets such as grasp or
    place points.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_goal: str = Field(..., min_length=1)
    rgb_image: Optional[object] = Field(default=None, exclude=True)
    detections: List[DetectedObject] = Field(default_factory=list)
    memory_summary: str = Field(default="No prior rounds.")
    previous_feedback: Optional[Feedback] = None

    @property
    def detection_summary(self) -> str:
        """Return a concise text summary of DINO detections."""
        if not self.detections:
            return "No objects detected."
        lines = []
        for det in self.detections:
            x1, y1, x2, y2 = det.bbox
            lines.append(
                f"- {det.label}: bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
                f"confidence={det.confidence:.2f}"
            )
        return "\n".join(lines)
