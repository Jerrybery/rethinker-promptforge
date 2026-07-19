"""Baseline-specific schemas."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from common.schema import MissionType, PlannerOutput, RethinkerOutput


class MonolithicDecision(BaseModel):
    """Combined semantic decision and pick/place plan from one monolithic call.

    The monolithic baseline collapses the Rethinker and Planner into a single
    agent, so its raw model output carries both the high-level semantic
    analysis (mirroring ``RethinkerOutput``) and the target-label plan
    (mirroring ``PlannerOutput``). It must not emit low-level control or
    grasp/place coordinates; targets are DINO labels only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mission_type: MissionType
    reasoning: str = Field(..., min_length=1)
    target_object: Optional[str] = None
    target_container: Optional[str] = None
    arm_hint: Optional[Literal["left", "right", "both"]] = None
    hidden_hypothesis: Optional[str] = None
    risk_note: Optional[str] = None
    pick: str = Field(..., min_length=1)
    place: Optional[str] = None

    def to_rethinker_output(self) -> RethinkerOutput:
        """Return the semantic part as a ``RethinkerOutput``."""
        return RethinkerOutput(
            mission_type=self.mission_type,
            reasoning=self.reasoning,
            target_object=self.target_object,
            target_container=self.target_container,
            arm_hint=self.arm_hint,
            hidden_hypothesis=self.hidden_hypothesis,
            risk_note=self.risk_note,
        )

    def to_planner_output(self, plan_id: str) -> PlannerOutput:
        """Return the plan part as a ``PlannerOutput`` with the given id."""
        return PlannerOutput(
            plan_id=plan_id,
            mission=self.mission_type,
            pick=self.pick,
            place=self.place,
        )
