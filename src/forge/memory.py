"""Forge planner memory: PlannerOutput memory for simulated rounds."""

from __future__ import annotations

from common._memory_base import _Round
from common.schema import PlannerOutput
from planner.memory import PlannerMemory


class ForgePlannerMemory(PlannerMemory):
    """Fixed-size memory for forge planner rounds.

    Same storage / summarize / serialize pattern as :class:`PlannerMemory`
    (fixed-size deque, deterministic ``summarize(k)``, JSON serialization).
    The compressed view additionally records the mission and pick/place
    labels so a candidate prompt sees which symbolic decisions were taken
    in earlier sim rounds.
    """

    def _compress(self, record: _Round[PlannerOutput]) -> str:
        fb = "with_feedback" if record.feedback is not None else "no_feedback"
        answer = record.answer
        return (
            f"Round {record.round}: "
            f"{answer.__class__.__name__} "
            f"plan_id={answer.plan_id} "
            f"mission={answer.mission.value} "
            f"pick={answer.pick} "
            f"place={answer.place or 'N/A'} "
            f"[{fb}]"
        )
