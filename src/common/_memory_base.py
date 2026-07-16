"""Private generic base for Rethinker and Planner memory modules."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Sequence, TypeVar

from common.schema import BaseModel, Feedback

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class _Round(Generic[OutputT]):
    """Immutable record of a single memory round."""

    round: int
    scene_token: str
    query: str
    answer: OutputT
    feedback: Feedback | None = None


class RoundView(Generic[OutputT], Protocol):
    """Public read-only view of a single stored round."""

    round: int
    scene_token: str
    query: str
    answer: OutputT
    feedback: Feedback | None


class _Memory(Generic[OutputT]):
    """Fixed-size, deterministic memory for model rounds.

    Keeps the most recent ``capacity`` full rounds. Older entries are evicted
    automatically by the underlying deque. ``summarize(k)`` returns a text
    representation where the last ``k`` rounds are shown in full and earlier
    rounds are compressed into a short deterministic summary.
    """

    def __init__(self, capacity: int = 100) -> None:
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError(f"capacity must be a positive integer, got {capacity}")
        self._capacity = capacity
        self._history: deque[_Round[OutputT]] = deque(maxlen=self._capacity)

    def append(
        self,
        round: int,
        scene_token: str,
        query: str,
        answer: OutputT,
        feedback: Feedback | None = None,
    ) -> None:
        """Store a new immutable round record."""
        record = _Round(
            round=int(round),
            scene_token=scene_token,
            query=query,
            answer=answer,
            feedback=feedback,
        )
        self._history.append(record)

    def summarize(self, k: int) -> str:
        """Return a deterministic text summary of the stored rounds.

        The most recent ``k`` rounds are rendered in full; all older rounds
        are compressed into a compact summary block. If ``k`` is zero, every
        round is compressed. If ``k`` is greater than or equal to the number
        of stored rounds, every round is rendered in full.
        """
        k = max(0, int(k))
        history = tuple(self._history)
        if not history:
            return "No rounds recorded."
        if k >= len(history):
            return self._render_full(history)

        if k == 0:
            older = history
            recent = ()
        else:
            older = history[:-k]
            recent = history[-k:]
        lines = ["Compressed older rounds:", ""]
        lines.extend(self._compress(rec) for rec in older)
        if recent:
            lines.extend(("", f"Recent {len(recent)} round(s) in full:", ""))
            lines.extend(self._render_record(rec) for rec in recent)
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize the full stored history as JSON-compatible dicts."""
        return [self._record_to_dict(rec) for rec in self._history]

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the full stored history to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self._history)

    @property
    def rounds(self) -> Sequence[RoundView[OutputT]]:
        """Immutable view of the stored rounds, oldest first."""
        return tuple(self._history)

    def _render_full(self, history: Sequence[_Round[OutputT]]) -> str:
        lines = [f"All {len(history)} round(s) in full:", ""]
        lines.extend(self._render_record(rec) for rec in history)
        return "\n".join(lines)

    def _render_record(self, record: _Round[OutputT]) -> str:
        parts = [
            f"Round {record.round} ({record.answer.__class__.__name__})",
            f"  scene_token: {record.scene_token}",
            f"  query: {record.query}",
            f"  answer: {record.answer.model_dump_json()}",
        ]
        if record.feedback is not None:
            parts.append(f"  feedback: {record.feedback.model_dump_json()}")
        return "\n".join(parts)

    def _compress(self, record: _Round[OutputT]) -> str:
        fb = "with_feedback" if record.feedback is not None else "no_feedback"
        return (
            f"Round {record.round}: "
            f"{record.answer.__class__.__name__} "
            f"[{fb}]"
        )

    def _record_to_dict(self, record: _Round[OutputT]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "round": record.round,
            "scene_token": record.scene_token,
            "query": record.query,
            "answer": record.answer.model_dump(mode="json"),
        }
        if record.feedback is not None:
            payload["feedback"] = record.feedback.model_dump(mode="json")
        return payload
