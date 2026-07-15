"""Unit tests for PlannerMemory."""

from __future__ import annotations

import json
from typing import Optional

import pytest

from common.schema import Feedback, PlannerOutput
from planner.memory import PlannerMemory


@pytest.fixture
def plan() -> PlannerOutput:
    return PlannerOutput(
        plan_id="plan-001",
        trajectory_name="top_down",
        waypoints=["approach", "grasp", "place"],
        gripper_action="open",
    )


@pytest.fixture
def feedback() -> Feedback:
    return Feedback(success=False, observation="missed", reward=-1.0)


def _append_rounds(
    mem: PlannerMemory,
    count: int,
    plan: PlannerOutput,
    last_feedback: Optional[Feedback] = None,
) -> None:
    for i in range(count):
        mem.append(
            round=i,
            scene_token=f"scene-{i:03d}",
            query=f"plan query {i}",
            answer=plan,
            feedback=last_feedback if i == count - 1 else None,
        )


def test_append_and_length(plan: PlannerOutput) -> None:
    mem = PlannerMemory(capacity=10)
    _append_rounds(mem, 3, plan)
    assert len(mem) == 3
    assert mem.rounds[1].query == "plan query 1"


def test_capacity_truncation(plan: PlannerOutput) -> None:
    mem = PlannerMemory(capacity=2)
    _append_rounds(mem, 5, plan)
    assert len(mem) == 2
    assert mem.rounds[0].round == 3
    assert mem.rounds[-1].round == 4


def test_summarize_compression_and_recent(plan: PlannerOutput, feedback: Feedback) -> None:
    mem = PlannerMemory()
    _append_rounds(mem, 6, plan, last_feedback=feedback)
    text = mem.summarize(k=3)
    assert "Compressed older rounds:" in text
    assert "Recent 3 round(s) in full:" in text
    assert text.count("PlannerOutput") == 6
    assert text.count("scene_token:") == 3
    assert "feedback:" in text


def test_summarize_k_zero(plan: PlannerOutput) -> None:
    mem = PlannerMemory()
    _append_rounds(mem, 3, plan)
    text = mem.summarize(k=0)
    assert "Recent" not in text
    assert "Compressed older rounds:" in text
    assert text.count("Round") == 3


def test_summarize_deterministic(plan: PlannerOutput) -> None:
    mem1 = PlannerMemory()
    mem2 = PlannerMemory()
    _append_rounds(mem1, 6, plan)
    _append_rounds(mem2, 6, plan)
    assert mem1.summarize(k=3) == mem2.summarize(k=3)


def test_to_dict_and_json(plan: PlannerOutput, feedback: Feedback) -> None:
    mem = PlannerMemory()
    mem.append(
        round=9,
        scene_token="scene-y",
        query="next plan?",
        answer=plan,
        feedback=feedback,
    )
    data = mem.to_dict()
    assert data[0]["answer"]["plan_id"] == "plan-001"
    assert data[0]["answer"]["gripper_action"] == "open"
    assert data[0]["feedback"]["reward"] == -1.0

    parsed = json.loads(mem.to_json(indent=2))
    assert parsed == data


def test_rounds_view_is_immutable(plan: PlannerOutput) -> None:
    mem = PlannerMemory()
    _append_rounds(mem, 2, plan)
    rounds = mem.rounds
    assert len(rounds) == 2
    with pytest.raises(TypeError):
        rounds[0] = rounds[1]  # type: ignore[index]
