"""Planner-specific input and context schemas."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from common.schema import Feedback, MissionType, RethinkerOutput


class PlannerContext(BaseModel):
    """Semantic context supplied to the Planner agent.

    The Planner consumes Rethinker semantic analysis, a DINO label set, an
    action library, optional memory, and optional feedback. It must not
    receive raw images or low-level control targets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rethinker_output: RethinkerOutput
    dino_labels: List[str] = Field(default_factory=list)
    action_library: List[str] = Field(default_factory=list)
    memory_summary: str = Field(default="No prior rounds.")
    previous_feedback: Optional[Feedback] = None

    @property
    def rethinker_summary(self) -> str:
        """Return a concise text summary of the Rethinker analysis."""
        ro = self.rethinker_output
        parts = [
            f"mission_type: {ro.mission_type.value}",
            f"reasoning: {ro.reasoning}",
        ]
        if ro.target_object is not None:
            parts.append(f"target_object: {ro.target_object}")
        if ro.target_container is not None:
            parts.append(f"target_container: {ro.target_container}")
        if ro.arm_hint is not None:
            parts.append(f"arm_hint: {ro.arm_hint}")
        if ro.hidden_hypothesis is not None:
            parts.append(f"hidden_hypothesis: {ro.hidden_hypothesis}")
        if ro.risk_note is not None:
            parts.append(f"risk_note: {ro.risk_note}")
        return "\n".join(parts)

    @property
    def label_list_text(self) -> str:
        """Return the DINO label set as a bulleted list."""
        if not self.dino_labels:
            return "No labels available."
        return "\n".join(f"- {label}" for label in self.dino_labels)

    @property
    def action_library_text(self) -> str:
        """Return the action library as a bulleted list."""
        if not self.action_library:
            return "No actions available."
        return "\n".join(f"- {action}" for action in self.action_library)

    def to_prompt_kwargs(self) -> dict[str, Any]:
        """Return the substitution mapping for user prompt templates."""
        feedback_text = (
            self.previous_feedback.model_dump_json()
            if self.previous_feedback is not None
            else "None"
        )
        return {
            "rethinker_output": self.rethinker_summary,
            "dino_labels": self.label_list_text,
            "action_library": self.action_library_text,
            "memory_summary": self.memory_summary,
            "previous_feedback": feedback_text,
        }
