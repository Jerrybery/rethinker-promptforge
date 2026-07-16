"""Planner memory module."""

from __future__ import annotations

from common._memory_base import _Memory, _Round
from common.schema import PlannerOutput


class PlannerMemory(_Memory[PlannerOutput]):
    """Fixed-size memory for Planner rounds.

    Stores ``PlannerOutput`` plans per round together with the query,
    scene token, and optional feedback. Older entries are summarized
    deterministically by ``summarize(k)``.
    """

    def _compress(self, record: _Round[PlannerOutput]) -> str:
        fb = "with_feedback" if record.feedback is not None else "no_feedback"
        answer = record.answer
        return (
            f"Round {record.round}: "
            f"{answer.__class__.__name__} "
            f"plan_id={answer.plan_id} "
            f"trajectory={answer.trajectory_name or 'N/A'} "
            f"gripper={answer.gripper_action or 'N/A'} "
            f"[{fb}]"
        )
