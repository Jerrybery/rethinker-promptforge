"""Integration tests: every ablation baseline runs through ClosedLoopRunner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from baselines.monolithic_planner import build_monolithic_agents
from baselines.no_hidden_hypothesis import NoHiddenHypothesisRethinker
from baselines.no_reflection import build_no_reflection_agents
from common.schema import DetectedObject, Feedback, MissionType, TaskUnit
from executor.schema import ExecutorAgentOutput
from planner.agent import PlannerAgent
from rethinker.agent import RethinkerAgent
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


class _StubExecutor:
    """Executor stub that always succeeds."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, **kwargs: Any) -> ExecutorAgentOutput:
        out = ExecutorAgentOutput(
            step_index=self.calls,
            success=True,
            status="ok",
            feedback=Feedback(success=True, observation="ok"),
        )
        self.calls += 1
        return out


def _make_client(*responses: str) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock(side_effect=list(responses))
    return client


@pytest.fixture
def task() -> TaskUnit:
    return TaskUnit(
        id="task-baseline-001",
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
                label="mug", bbox=[100.0, 100.0, 200.0, 200.0], confidence=0.95
            ),
            DetectedObject(
                label="saucer", bbox=[250.0, 250.0, 350.0, 350.0], confidence=0.92
            ),
        ]
    )


def _rethinker_payload(mission: str) -> dict:
    payload: dict[str, Any] = {
        "mission_type": mission,
        "reasoning": "Round decision.",
    }
    if mission != "STOP":
        payload.update(
            {
                "target_object": "mug",
                "target_container": "saucer",
                "hidden_hypothesis": "Mug handle may be occluded.",
                "risk_note": "Approach carefully.",
            }
        )
    return payload


def _planner_payload(mission: str) -> dict:
    if mission == "STOP":
        return {"plan_id": "plan-stop", "mission": "STOP", "pick": "none"}
    return {
        "plan_id": "plan-act",
        "mission": mission,
        "pick": "mug",
        "place": "saucer",
    }


def _monolithic_payload(mission: str) -> dict:
    payload = _rethinker_payload(mission)
    if mission == "STOP":
        payload.update({"pick": "none", "place": None})
    else:
        payload.update({"pick": "mug", "place": "saucer"})
    return payload


def test_monolithic_baseline_runs_through_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: TaskUnit,
    robot: RobotInterface,
    dino: FakeDINO,
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _make_client(
        json.dumps(_monolithic_payload("PICK_AND_PLACE")),
        json.dumps(_monolithic_payload("STOP")),
    )
    agents = build_monolithic_agents(vllm_client=client)
    agents["executor"] = _StubExecutor()

    runner = ClosedLoopRunner(
        task=task,
        config_path=ROBOT_CONFIG_PATH,
        agents=agents,
        robot=robot,
        dino=dino,
        max_rounds=5,
    )
    episode = runner.run()

    assert len(episode.steps) == 2
    first = episode.steps[0]
    assert first.planner_output is not None
    assert first.planner_output.pick == "mug"
    assert first.planner_output.place == "saucer"
    assert episode.metadata is not None
    assert episode.metadata["termination_reason"] == "stop"


def test_no_hidden_hypothesis_baseline_runs_through_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: TaskUnit,
    robot: RobotInterface,
    dino: FakeDINO,
) -> None:
    monkeypatch.chdir(tmp_path)
    r_client = _make_client(
        json.dumps(_rethinker_payload("PICK_AND_PLACE")),
        json.dumps(_rethinker_payload("STOP")),
    )
    p_client = _make_client(
        json.dumps(_planner_payload("PICK_AND_PLACE")),
        json.dumps(_planner_payload("STOP")),
    )
    agents = {
        "rethinker": NoHiddenHypothesisRethinker(vllm_client=r_client),
        "planner": PlannerAgent(vllm_client=p_client),
        "executor": _StubExecutor(),
    }

    runner = ClosedLoopRunner(
        task=task,
        config_path=ROBOT_CONFIG_PATH,
        agents=agents,
        robot=robot,
        dino=dino,
        max_rounds=5,
    )
    episode = runner.run()

    assert len(episode.steps) == 2
    assert episode.metadata is not None
    assert episode.metadata["termination_reason"] == "stop"


def test_no_reflection_baseline_runs_through_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: TaskUnit,
    robot: RobotInterface,
    dino: FakeDINO,
) -> None:
    monkeypatch.chdir(tmp_path)
    r_client = _make_client(
        json.dumps(_rethinker_payload("PICK_AND_PLACE")),
        json.dumps(_rethinker_payload("STOP")),
    )
    p_client = _make_client(
        json.dumps(_planner_payload("PICK_AND_PLACE")),
        json.dumps(_planner_payload("STOP")),
    )
    agents = build_no_reflection_agents(
        rethinker=RethinkerAgent(vllm_client=r_client),
        planner=PlannerAgent(vllm_client=p_client),
    )
    agents["executor"] = _StubExecutor()

    runner = ClosedLoopRunner(
        task=task,
        config_path=ROBOT_CONFIG_PATH,
        agents=agents,
        robot=robot,
        dino=dino,
        max_rounds=5,
    )
    episode = runner.run()

    assert len(episode.steps) == 2
    assert episode.metadata is not None
    assert episode.metadata["termination_reason"] == "stop"

    # Second-round prompts must show no memory and no feedback even though the
    # runner carried a successful round and passed both to the wrappers.
    second_r_text = r_client.chat.call_args_list[1].args[0][1]["content"]
    assert "No prior rounds." in second_r_text
    assert '"success": true' not in second_r_text
    second_p_text = p_client.chat.call_args_list[1].args[0][1]["content"]
    assert "No prior rounds." in second_p_text
