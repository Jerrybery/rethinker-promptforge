"""Executor memory module."""

from __future__ import annotations

from common._memory_base import _Memory, _Round
from common.schema import ExecutorOutput


class ExecutorMemory(_Memory[ExecutorOutput]):
    """Fixed-size memory for Executor rounds.

    Stores ``ExecutorOutput`` low-level commands per round together with the
    query, scene token, and optional feedback. Older entries are summarized
    deterministically by ``summarize(k)``.
    """

    def _compress(self, record: _Round[ExecutorOutput]) -> str:
        fb = "with_feedback" if record.feedback is not None else "no_feedback"
        return (
            f"Round {record.round}: "
            f"{record.answer.__class__.__name__} "
            f"step={record.answer.step_index} "
            f"gripper={record.answer.gripper_state} "
            f"[{fb}]"
        )
