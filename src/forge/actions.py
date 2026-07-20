"""Mapping from planner-level outputs to simulation-level actions.

The forge planner emits :class:`common.schema.PlannerOutput` (prompt-level
symbolic decisions); :class:`forge.env.SimEnv` consumes
:class:`forge.env.SimAction`. This module is the single pure mapping point
between the two so the forge loop stays free of field-plumbing.
"""

from __future__ import annotations

from common.schema import MissionType, PlannerOutput
from forge.env import SimAction

# Planner convention: STOP (and REOBSERVE) may use the sentinel pick "none".
_PICK_NONE = "none"


def planner_output_to_sim_action(output: PlannerOutput) -> SimAction:
    """Convert a validated :class:`PlannerOutput` into a :class:`SimAction`.

    Field mapping is direct: ``mission -> mission``, ``pick -> target`` (the
    sentinel ``"none"`` maps to ``None``), ``place -> place_target``. ``arm``
    keeps the :class:`SimAction` default since ``PlannerOutput`` carries no
    arm hint.

    Raises:
        ValueError: if the planner output cannot form a valid
            :class:`SimAction` (e.g. a manipulation mission with pick
            ``"none"``, which leaves ``target`` unset).
    """
    if output.mission is MissionType.STOP:
        return SimAction(mission=MissionType.STOP)

    target = None if output.pick == _PICK_NONE else output.pick
    try:
        return SimAction(
            mission=output.mission,
            target=target,
            place_target=output.place,
        )
    except ValueError as exc:
        raise ValueError(
            f"PlannerOutput {output.plan_id!r} cannot map to a SimAction: {exc}"
        ) from exc
