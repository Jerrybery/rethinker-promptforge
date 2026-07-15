"""Rethinker memory module."""

from __future__ import annotations

from common._memory_base import _Memory
from common.schema import RethinkerOutput


class RethinkerMemory(_Memory[RethinkerOutput]):
    """Fixed-size memory for Rethinker rounds.

    Stores ``RethinkerOutput`` decisions per round together with the query,
    scene token, and optional feedback. Older entries are summarized
    deterministically by ``summarize(k)``.
    """
