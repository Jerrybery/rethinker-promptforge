"""Unit tests for the PlannerOutput -> SimAction mapping helper."""

from __future__ import annotations

import pytest

from common.schema import MissionType, PlannerOutput
from forge.actions import planner_output_to_sim_action
from forge.env import SimAction


def _output(
    mission: MissionType, pick: str, place: str | None = None
) -> PlannerOutput:
    return PlannerOutput(plan_id="p-1", mission=mission, pick=pick, place=place)


def test_pick_and_place_maps_fields() -> None:
    action = planner_output_to_sim_action(
        _output(MissionType.PICK_AND_PLACE, "mug", "saucer")
    )
    assert isinstance(action, SimAction)
    assert action.mission is MissionType.PICK_AND_PLACE
    assert action.target == "mug"
    assert action.place_target == "saucer"
    assert action.arm == "right"


def test_pick_only_maps_without_place() -> None:
    action = planner_output_to_sim_action(_output(MissionType.PICK_ONLY, "mug"))
    assert action.mission is MissionType.PICK_ONLY
    assert action.target == "mug"
    assert action.place_target is None


def test_move_aside_maps_target() -> None:
    action = planner_output_to_sim_action(_output(MissionType.MOVE_ASIDE, "cloth"))
    assert action.mission is MissionType.MOVE_ASIDE
    assert action.target == "cloth"


def test_reobserve_maps_without_target() -> None:
    action = planner_output_to_sim_action(_output(MissionType.REOBSERVE, "none"))
    assert action.mission is MissionType.REOBSERVE
    assert action.target is None


def test_stop_with_pick_none_maps_to_stop() -> None:
    action = planner_output_to_sim_action(_output(MissionType.STOP, "none"))
    assert action.mission is MissionType.STOP
    assert action.target is None


def test_manipulation_mission_with_pick_none_raises() -> None:
    for mission in (
        MissionType.PICK_AND_PLACE,
        MissionType.PICK_ONLY,
        MissionType.MOVE_ASIDE,
    ):
        with pytest.raises(ValueError, match="SimAction"):
            planner_output_to_sim_action(_output(mission, "none"))
