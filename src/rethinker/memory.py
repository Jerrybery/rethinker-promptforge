"""Rethinker memory module."""

from __future__ import annotations

from common._memory_base import _Memory, _Round
from common.schema import RethinkerOutput


class RethinkerMemory(_Memory[RethinkerOutput]):
    """Fixed-size memory for Rethinker rounds.

    Stores ``RethinkerOutput`` decisions per round together with the query,
    scene token, and optional feedback. Older entries are summarized
    deterministically by ``summarize(k)``.
    """

    def _compress(self, record: _Round[RethinkerOutput]) -> str:
        fb = "with_feedback" if record.feedback is not None else "no_feedback"
        return (
            f"Round {record.round}: "
            f"{record.answer.__class__.__name__} "
            f"mission={record.answer.mission_type.value} "
            f"[{fb}]"
        )
