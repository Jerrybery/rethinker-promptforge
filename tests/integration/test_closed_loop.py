"""Integration test for the closed-loop runner with mocked agents and robot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from common.schema import (
    DetectedObject,
    EpisodeStep,
    Feedback,
    MissionType,
    PlannerOutput,
    RethinkerOutput,
    TaskUnit,
)
from executor.schema import ExecutorAgentOutput
from perception.dino_client import DINOClient
from rethinker.runner import ClosedLoopRunner
from robot.interface import RobotInterface

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIG_PATH = REPO_ROOT / "configs" / "robot.yaml"


class FakeDINO:
    """Deterministic stand-in for DINOClient that returns canned detections."""

    mode = "fake"

    def __init__(self, detections: list[DetectedObject]) -> None:
        self._detections = list(detections)

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        return list(self._detections)


class _SequenceAgent:
    """Generic agent that returns a fixed sequence of outputs."""

    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = outputs
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def act(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        output = self._outputs[self._index]
        self._index = min(self._index + 1, len(self._outputs) - 1)
        return output


@pytest.fixture
def task() -> TaskUnit:
    return TaskUnit(
        id="task-int-001",
        instruction="pick the mug and place it on the saucer",
        mission_type=MissionType.PICK_AND_PLACE,
        objects=["mug", "saucer"],
    )


@pytest.fixture
def robot() -> RobotInterface:
    return RobotInterface(config_path=ROBOT_CONFIG_PATH, mock=True)


@pytest.fixture
def dino() -> FakeDINO:
    return FakeDINO(
        [
            DetectedObject(
                label="mug",
                bbox=[100.0, 100.0, 200.0, 200.0],
                confidence=0.95,
            ),
            DetectedObject(
                label="saucer",
                bbox=[250.0, 250.0, 350.0, 350.0],
                confidence=0.92,
            ),
        ]
    )


@pytest.fixture
def agents() -> dict[str, Any]:
    rethinker_outputs = [
        RethinkerOutput(
            mission_type=MissionType.PICK_AND_PLACE,
            reasoning="Mug and saucer are visible; pick mug and place on saucer.",
            target_object="mug",
            target_container="saucer",
            arm_hint="right",
        ),
        RethinkerOutput(
            mission_type=MissionType.STOP,
            reasoning="Task completed successfully.",
        ),
    ]
    planner_outputs = [
        PlannerOutput(
            plan_id="plan-0",
            mission=MissionType.PICK_AND_PLACE,
            pick="mug",
            place="saucer",
        ),
        PlannerOutput(
            plan_id="plan-1",
            mission=MissionType.STOP,
            pick="none",
        ),
    ]
    executor_outputs = [
        ExecutorAgentOutput(
            step_index=0,
            success=True,
            status="picked mug and placed on saucer",
            feedback=Feedback(success=True, observation="placed cleanly"),
        ),
        ExecutorAgentOutput(
            step_index=1,
            success=True,
            status="stopped",
            feedback=Feedback(success=True, observation="stopped cleanly"),
        ),
    ]
    return {
        "rethinker": _SequenceAgent(rethinker_outputs),
        "planner": _SequenceAgent(planner_outputs),
        "executor": _SequenceAgent(executor_outputs),
    }


def test_closed_loop_runner_full_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: TaskUnit,
    robot: RobotInterface,
    dino: FakeDINO,
    agents: dict[str, Any],
) -> None:
    """A full episode should execute, log, and update memories without LLMs or real hardware."""
    monkeypatch.chdir(tmp_path)

    runner = ClosedLoopRunner(
        task=task,
        config_path=ROBOT_CONFIG_PATH,
        agents=agents,
        robot=robot,
        dino=dino,
        max_rounds=5,
    )
    episode = runner.run()

    assert episode.task_id == task.id
    assert len(episode.steps) == 2
    assert all(isinstance(step, EpisodeStep) for step in episode.steps)

    first, second = episode.steps
    assert first.step_index == 0
    assert first.rethinker_output.mission_type == MissionType.PICK_AND_PLACE
    assert first.planner_output is not None
    assert first.planner_output.pick == "mug"
    assert first.executor_output is not None
    assert first.feedback is not None
    assert first.feedback.success is True

    assert second.step_index == 1
    assert second.rethinker_output.mission_type == MissionType.STOP
    assert second.feedback is not None
    assert second.feedback.success is True

    # Memories should record one entry per step.
    assert len(runner.rethinker_memory) == 2
    assert len(runner.planner_memory) == 2
    assert len(runner.executor_memory) == 2

    # Logger should have written metadata plus one line per step.
    assert runner.log_path.exists()
    lines = runner.log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 3
    events = [json.loads(line) for line in lines]
    assert events[0]["event"] == "metadata"
    step_events = [event for event in events if event["event"] == "step"]
    assert len(step_events) == 2
    assert step_events[0]["step_index"] == 0
    assert step_events[1]["step_index"] == 1


def test_closed_loop_runner_uses_dino_detections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: TaskUnit,
    robot: RobotInterface,
    agents: dict[str, Any],
) -> None:
    """The runner should pass DINO labels (not raw images) to the planner."""
    monkeypatch.chdir(tmp_path)

    # Keep the same rethinker/planner/executor fixtures but swap in a DINO
    # client that only reports the mock_object label.  We only assert that the
    # planner received the label set derived from DINO, which confirms the
    # perception-to-planner boundary is wired correctly.
    mock_dino = DINOClient(config_path=ROBOT_CONFIG_PATH, mode="mock")

    runner = ClosedLoopRunner(
        task=task,
        config_path=ROBOT_CONFIG_PATH,
        agents=agents,
        robot=robot,
        dino=mock_dino,
        max_rounds=5,
    )
    runner.run()

    planner_calls = agents["planner"].calls
    assert len(planner_calls) >= 1
    labels = planner_calls[0].get("dino_labels", [])
    assert "mock_object" in labels


def test_closed_loop_runner_failure_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: TaskUnit,
    robot: RobotInterface,
    dino: FakeDINO,
) -> None:
    """A failed executor step should terminate the loop early."""
    monkeypatch.chdir(tmp_path)

    failing_agents = {
        "rethinker": _SequenceAgent(
            [
                RethinkerOutput(
                    mission_type=MissionType.PICK_ONLY,
                    reasoning="Pick the mug.",
                    target_object="mug",
                )
            ]
        ),
        "planner": _SequenceAgent(
            [
                PlannerOutput(
                    plan_id="plan-fail",
                    mission=MissionType.PICK_ONLY,
                    pick="mug",
                )
            ]
        ),
        "executor": _SequenceAgent(
            [
                ExecutorAgentOutput(
                    step_index=0,
                    success=False,
                    status="gripper failed",
                    feedback=Feedback(success=False, error_message="gripper failed"),
                )
            ]
        ),
    }

    runner = ClosedLoopRunner(
        task=task,
        config_path=ROBOT_CONFIG_PATH,
        agents=failing_agents,
        robot=robot,
        dino=dino,
        max_rounds=10,
    )
    episode = runner.run()

    assert len(episode.steps) == 1
    assert episode.steps[0].feedback is not None
    assert episode.steps[0].feedback.success is False


def test_closed_loop_runner_max_rounds_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: TaskUnit,
    robot: RobotInterface,
    dino: FakeDINO,
) -> None:
    """The runner should not exceed max_rounds even if the agent never stops."""
    monkeypatch.chdir(tmp_path)

    nonstop_agents = {
        "rethinker": _SequenceAgent(
            [
                RethinkerOutput(
                    mission_type=MissionType.REOBSERVE,
                    reasoning="Keep observing.",
                )
            ]
        ),
        "planner": _SequenceAgent(
            [
                PlannerOutput(
                    plan_id="plan-watch",
                    mission=MissionType.REOBSERVE,
                    pick="none",
                )
            ]
        ),
        "executor": _SequenceAgent(
            [
                ExecutorAgentOutput(
                    step_index=0,
                    success=True,
                    status="reobserved",
                    feedback=Feedback(success=True, observation="reobserved"),
                )
            ]
        ),
    }

    runner = ClosedLoopRunner(
        task=task,
        config_path=ROBOT_CONFIG_PATH,
        agents=nonstop_agents,
        robot=robot,
        dino=dino,
        max_rounds=3,
    )
    episode = runner.run()

    assert len(episode.steps) == 3
