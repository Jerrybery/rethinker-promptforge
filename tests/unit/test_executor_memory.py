"""Unit tests for ExecutorMemory."""

from __future__ import annotations

import json

import pytest

from common.schema import ExecutorOutput, Feedback
from executor.memory import ExecutorMemory


@pytest.fixture
def command() -> ExecutorOutput:
    return ExecutorOutput(
        step_index=0,
        joint_angles=[0.0, -0.5, 1.0, -1.5, 0.5, 0.0],
        gripper_state=0.75,
    )


@pytest.fixture
def feedback() -> Feedback:
    return Feedback(success=True, observation="executed", reward=0.5)


def _append_rounds(
    mem: ExecutorMemory,
    count: int,
    command: ExecutorOutput,
    last_feedback: Feedback | None = None,
) -> None:
    for i in range(count):
        mem.append(
            round=i,
            scene_token=f"scene-{i:03d}",
            query=f"execute query {i}",
            answer=command,
            feedback=last_feedback if i == count - 1 else None,
        )


def test_append_and_length(command: ExecutorOutput) -> None:
    mem = ExecutorMemory(capacity=10)
    _append_rounds(mem, 3, command)
    assert len(mem) == 3
    assert mem.rounds[1].query == "execute query 1"


def test_capacity_truncation(command: ExecutorOutput) -> None:
    mem = ExecutorMemory(capacity=2)
    _append_rounds(mem, 5, command)
    assert len(mem) == 2
    assert mem.rounds[0].round == 3
    assert mem.rounds[-1].round == 4


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity must be a positive integer"):
        ExecutorMemory(capacity=0)
    with pytest.raises(ValueError, match="capacity must be a positive integer"):
        ExecutorMemory(capacity=-1)


def test_summarize_compression_and_recent(command: ExecutorOutput, feedback: Feedback) -> None:
    mem = ExecutorMemory()
    _append_rounds(mem, 6, command, last_feedback=feedback)
    text = mem.summarize(k=3)
    assert "Compressed older rounds:" in text
    assert "Recent 3 round(s) in full:" in text
    assert text.count("ExecutorOutput") == 6
    assert text.count("scene_token:") == 3
    assert "feedback:" in text


def test_summarize_k_zero(command: ExecutorOutput) -> None:
    mem = ExecutorMemory()
    _append_rounds(mem, 3, command)
    text = mem.summarize(k=0)
    assert "Recent" not in text
    assert "Compressed older rounds:" in text
    assert text.count("Round") == 3


def test_summarize_deterministic(command: ExecutorOutput) -> None:
    mem1 = ExecutorMemory()
    mem2 = ExecutorMemory()
    _append_rounds(mem1, 6, command)
    _append_rounds(mem2, 6, command)
    assert mem1.summarize(k=3) == mem2.summarize(k=3)


def test_to_dict_and_json(command: ExecutorOutput, feedback: Feedback) -> None:
    mem = ExecutorMemory()
    mem.append(
        round=2,
        scene_token="scene-z",
        query="execute?",
        answer=command,
        feedback=feedback,
    )
    data = mem.to_dict()
    assert data[0]["answer"]["step_index"] == 0
    assert data[0]["answer"]["gripper_state"] == 0.75
    assert data[0]["feedback"]["reward"] == 0.5

    parsed = json.loads(mem.to_json(indent=2))
    assert parsed == data


def test_rounds_view_is_immutable(command: ExecutorOutput) -> None:
    mem = ExecutorMemory()
    _append_rounds(mem, 2, command)
    rounds = mem.rounds
    assert len(rounds) == 2
    with pytest.raises(TypeError):
        rounds[0] = rounds[1]  # type: ignore[index]
