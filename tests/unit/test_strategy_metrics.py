"""Unit tests for semantic strategy metrics."""

from __future__ import annotations

from typing import Any

from common.schema import (
    Episode,
    EpisodeStep,
    Feedback,
    MissionType,
    PlannerOutput,
    RethinkerOutput,
)
from forge.strategy_metrics import (
    aggregate_strategy_metrics,
    episode_strategy_metrics,
)
from tasks.schema import TaskDefinition


def _task(metadata: dict[str, Any] | None = None) -> TaskDefinition:
    return TaskDefinition(
        id="t1",
        instruction="do the thing",
        mission_type=MissionType.PICK_AND_PLACE,
        objects=["can", "pot", "occluder"],
        metadata=metadata or {},
    )


def _output(mission: MissionType, pick: str, place: str | None = None) -> PlannerOutput:
    return PlannerOutput(plan_id=f"p-{mission.value}-{pick}", mission=mission, pick=pick, place=place)


def _episode(
    outputs: list[PlannerOutput],
    violations: int = 0,
    success: bool = False,
) -> Episode:
    task_unit = _task()
    steps = [
        EpisodeStep(
            step_index=i,
            task=task_unit,
            rethinker_output=RethinkerOutput(mission_type=o.mission, reasoning="r"),
            planner_output=o,
            feedback=Feedback(success=False, observation=""),
        )
        for i, o in enumerate(outputs)
    ]
    return Episode(
        id="ep1",
        task_id="t1",
        steps=steps,
        metadata={"success": success, "presence_violations": violations},
    )


OCCLUSION_META = {"hidden_by": {"can": ["occluder"]}}


def test_move_aside_first_true_when_occluder_cleared_first() -> None:
    ep = _episode(
        [
            _output(MissionType.MOVE_ASIDE, "occluder"),
            _output(MissionType.PICK_AND_PLACE, "can", "pot"),
        ]
    )
    m = episode_strategy_metrics(ep, _task(OCCLUSION_META))
    assert m["move_aside_first"] is True
    assert m["action_count"] == 2
    assert m["presence_violations"] == 0


def test_move_aside_first_false_when_target_manipulated_first() -> None:
    ep = _episode(
        [
            _output(MissionType.PICK_AND_PLACE, "can", "pot"),
            _output(MissionType.MOVE_ASIDE, "occluder"),
        ]
    )
    m = episode_strategy_metrics(ep, _task(OCCLUSION_META))
    assert m["move_aside_first"] is False


def test_move_aside_first_none_for_non_occlusion_task() -> None:
    ep = _episode([_output(MissionType.PICK_AND_PLACE, "can", "pot")])
    m = episode_strategy_metrics(ep, _task())
    assert m["move_aside_first"] is None


def test_move_aside_first_none_when_hidden_target_never_touched() -> None:
    ep = _episode([_output(MissionType.MOVE_ASIDE, "occluder"), _output(MissionType.STOP, "none")])
    m = episode_strategy_metrics(ep, _task(OCCLUSION_META))
    assert m["move_aside_first"] is None


def test_move_aside_of_unrelated_object_does_not_count() -> None:
    ep = _episode(
        [
            _output(MissionType.MOVE_ASIDE, "pot"),
            _output(MissionType.PICK_AND_PLACE, "can", "pot"),
        ]
    )
    m = episode_strategy_metrics(ep, _task(OCCLUSION_META))
    assert m["move_aside_first"] is False


def test_violations_read_from_metadata() -> None:
    ep = _episode([_output(MissionType.PICK_AND_PLACE, "can", "pot")], violations=2)
    m = episode_strategy_metrics(ep, _task())
    assert m["presence_violations"] == 2


def test_aggregate_rates() -> None:
    per = [
        {"action_count": 2, "presence_violations": 1, "move_aside_first": True},
        {"action_count": 4, "presence_violations": 1, "move_aside_first": False},
        {"action_count": 4, "presence_violations": 0, "move_aside_first": None},
    ]
    agg = aggregate_strategy_metrics(per)
    assert agg["move_aside_first_rate"] == 0.5
    assert agg["presence_violations"] == 2
    assert abs(agg["presence_violation_rate"] - 2 / 10) < 1e-9
    assert abs(agg["mean_action_count"] - 10 / 3) < 1e-9


def test_aggregate_no_applicable_move_aside() -> None:
    agg = aggregate_strategy_metrics(
        [{"action_count": 1, "presence_violations": 0, "move_aside_first": None}]
    )
    assert agg["move_aside_first_rate"] is None
    assert agg["presence_violation_rate"] == 0.0
