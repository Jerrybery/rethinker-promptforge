"""Unit tests for RethinkerMemory."""

from __future__ import annotations

import json

import pytest

from common.schema import Feedback, MissionType, RethinkerOutput
from rethinker.memory import RethinkerMemory


@pytest.fixture
def answer() -> RethinkerOutput:
    return RethinkerOutput(
        mission_type=MissionType.PICK_AND_PLACE,
        reasoning="Target visible.",
        target_object="mug",
        target_container="saucer",
        arm_hint="right",
    )


@pytest.fixture
def feedback() -> Feedback:
    return Feedback(success=True, observation="grasped", reward=1.0)


def _append_rounds(mem: RethinkerMemory, count: int, answer: RethinkerOutput) -> None:
    for i in range(count):
        mem.append(
            round=i,
            scene_token=f"scene-{i:03d}",
            query=f"query {i}",
            answer=answer,
            feedback=Feedback(success=(i % 2 == 0)) if i % 2 == 0 else None,
        )


def test_append_and_length(answer: RethinkerOutput) -> None:
    mem = RethinkerMemory(capacity=10)
    _append_rounds(mem, 3, answer)
    assert len(mem) == 3
    assert mem.rounds[0].round == 0
    assert mem.rounds[-1].round == 2


def test_append_without_feedback(answer: RethinkerOutput) -> None:
    mem = RethinkerMemory()
    mem.append(round=0, scene_token="s0", query="q0", answer=answer)
    assert len(mem) == 1
    assert mem.rounds[0].feedback is None


def test_capacity_truncation(answer: RethinkerOutput) -> None:
    mem = RethinkerMemory(capacity=4)
    _append_rounds(mem, 6, answer)
    assert len(mem) == 4
    assert mem.rounds[0].round == 2
    assert mem.rounds[-1].round == 5


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity must be a positive integer"):
        RethinkerMemory(capacity=0)
    with pytest.raises(ValueError, match="capacity must be a positive integer"):
        RethinkerMemory(capacity=-2)


def test_summarize_all_full_when_k_greater_than_length(answer: RethinkerOutput) -> None:
    mem = RethinkerMemory()
    _append_rounds(mem, 3, answer)
    text = mem.summarize(k=10)
    assert "All 3 round(s) in full" in text
    assert "scene-000" in text
    assert "scene-002" in text


def test_summarize_compresses_older_rounds(answer: RethinkerOutput) -> None:
    mem = RethinkerMemory()
    _append_rounds(mem, 5, answer)
    text = mem.summarize(k=2)
    assert "Compressed older rounds:" in text
    assert "Recent 2 round(s) in full:" in text
    assert "Round 0: RethinkerOutput" in text
    assert "Round 1: RethinkerOutput" in text
    assert "Round 2: RethinkerOutput" in text
    assert "Round 3 (RethinkerOutput)" in text
    assert "Round 4 (RethinkerOutput)" in text
    assert text.count("scene_token:") == 2


def test_summarize_k_zero_compresses_everything(answer: RethinkerOutput) -> None:
    mem = RethinkerMemory()
    _append_rounds(mem, 4, answer)
    text = mem.summarize(k=0)
    assert "Compressed older rounds:" in text
    assert "Recent" not in text
    assert "scene_token:" not in text
    assert text.count("Round") == 4


def test_summarize_empty() -> None:
    mem = RethinkerMemory()
    assert mem.summarize(k=3) == "No rounds recorded."


def test_summarize_deterministic(answer: RethinkerOutput) -> None:
    mem1 = RethinkerMemory()
    mem2 = RethinkerMemory()
    _append_rounds(mem1, 5, answer)
    _append_rounds(mem2, 5, answer)
    assert mem1.summarize(k=2) == mem2.summarize(k=2)


def test_summarize_truncation_deterministic(answer: RethinkerOutput) -> None:
    mem1 = RethinkerMemory(capacity=3)
    mem2 = RethinkerMemory(capacity=3)
    _append_rounds(mem1, 5, answer)
    _append_rounds(mem2, 5, answer)
    summary = mem1.summarize(k=2)
    assert summary == mem2.summarize(k=2)
    assert "Round 2: RethinkerOutput" in summary
    assert "Round 3 (RethinkerOutput)" in summary
    assert "Round 4 (RethinkerOutput)" in summary
    assert "Round 0" not in summary


def test_to_dict_and_json(answer: RethinkerOutput, feedback: Feedback) -> None:
    mem = RethinkerMemory()
    mem.append(
        round=7,
        scene_token="scene-x",
        query="what next?",
        answer=answer,
        feedback=feedback,
    )
    data = mem.to_dict()
    assert len(data) == 1
    assert data[0]["round"] == 7
    assert data[0]["scene_token"] == "scene-x"
    assert data[0]["answer"]["mission_type"] == "PICK_AND_PLACE"
    assert data[0]["feedback"]["success"] is True

    dumped = mem.to_json()
    parsed = json.loads(dumped)
    assert parsed == data
