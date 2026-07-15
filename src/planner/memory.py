"""Planner memory module."""

from __future__ import annotations

from common._memory_base import _Memory
from common.schema import PlannerOutput


class PlannerMemory(_Memory[PlannerOutput]):
    """Fixed-size memory for Planner rounds.

    Stores ``PlannerOutput`` plans per round together with the query,
    scene token, and optional feedback. Older entries are summarized
    deterministically by ``summarize(k)``.
    """
