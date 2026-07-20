"""Unit tests for the forge planner agent and memory."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from common.schema import Feedback, MissionType, PlannerOutput
from forge.memory import ForgePlannerMemory
from forge.planner_agent import ForgePlannerAgent, obs_to_rethinker_output
from planner.memory import PlannerMemory
from planner.prompts.registry import PromptRegistry


@pytest.fixture
def obs() -> dict[str, Any]:
    return {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "state": {
            "pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            "gripper": 0.5,
            "timestamp": 123.0,
        },
        "detections": [
            {"label": "mug", "bbox": [0, 0, 10, 10], "confidence": 0.9},
            {"label": "saucer", "bbox": [20, 20, 30, 30], "confidence": 0.8},
        ],
        "task": {
            "id": "task-001",
            "instruction": "Pick the mug and place it on the saucer.",
            "mission_type": "PICK_AND_PLACE",
        },
        "step_index": 0,
    }


@pytest.fixture
def valid_output() -> PlannerOutput:
    return PlannerOutput(
        plan_id="plan-001",
        mission=MissionType.PICK_AND_PLACE,
        pick="mug",
        place="saucer",
    )


def _make_agent(mock_response: str) -> ForgePlannerAgent:
    client = MagicMock()
    client.chat = MagicMock(return_value=mock_response)
    return ForgePlannerAgent(vllm_client=client)


# --------------------------------------------------------------------- #
# obs -> RethinkerOutput adapter
# --------------------------------------------------------------------- #


def test_obs_to_rethinker_output(obs: dict[str, Any]) -> None:
    ro = obs_to_rethinker_output(obs)
    assert ro.mission_type is MissionType.PICK_AND_PLACE
    assert "Pick the mug and place it on the saucer." in ro.reasoning
    assert "task-001" in ro.reasoning
    assert "mug" in ro.reasoning
    assert "saucer" in ro.reasoning
    assert "gripper" in ro.reasoning


def test_obs_to_rethinker_output_missing_task() -> None:
    with pytest.raises(ValueError, match="Invalid SimEnv observation"):
        obs_to_rethinker_output({"detections": []})


def test_obs_to_rethinker_output_bad_mission(obs: dict[str, Any]) -> None:
    bad = dict(obs)
    bad["task"] = dict(obs["task"], mission_type="NOT_A_MISSION")
    with pytest.raises(ValueError, match="Invalid SimEnv observation"):
        obs_to_rethinker_output(bad)


# --------------------------------------------------------------------- #
# act_from_obs
# --------------------------------------------------------------------- #


def test_act_from_obs_valid(
    obs: dict[str, Any], valid_output: PlannerOutput
) -> None:
    agent = _make_agent(json.dumps(valid_output.model_dump()))
    output = agent.act_from_obs(obs)
    assert output == valid_output
    client = agent.planner.vllm_client
    assert client.chat.call_count == 1
    messages = client.chat.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_text = messages[1]["content"]
    assert "mug" in user_text
    assert "saucer" in user_text
    assert "PICK_AND_PLACE" in user_text
    # Default is text-only, identical to the real planner call.
    assert client.chat.call_args.kwargs == {}


def test_act_from_obs_includes_image_when_requested(
    obs: dict[str, Any], valid_output: PlannerOutput
) -> None:
    agent = _make_agent(json.dumps(valid_output.model_dump()))
    output = agent.act_from_obs(obs, include_image=True)
    assert output == valid_output
    client = agent.planner.vllm_client
    images = client.chat.call_args.kwargs["images"]
    assert len(images) == 1
    assert isinstance(images[0], np.ndarray)


def test_act_from_obs_rejects_unparseable(obs: dict[str, Any]) -> None:
    agent = _make_agent("this is not json")
    with pytest.raises(ValueError, match="could not parse"):
        agent.act_from_obs(obs)


def test_act_from_obs_rejects_invalid_mission(obs: dict[str, Any]) -> None:
    bad = {
        "plan_id": "plan-bad",
        "mission": "INVALID_MISSION",
        "pick": "mug",
        "place": "saucer",
    }
    agent = _make_agent(json.dumps(bad))
    with pytest.raises(ValueError, match="mission"):
        agent.act_from_obs(obs)


def test_act_from_obs_rejects_invalid_pick_label(obs: dict[str, Any]) -> None:
    bad = {
        "plan_id": "plan-bad",
        "mission": "PICK_AND_PLACE",
        "pick": "cup",
        "place": "saucer",
    }
    agent = _make_agent(json.dumps(bad))
    with pytest.raises(ValueError, match="Pick label"):
        agent.act_from_obs(obs)


def test_act_from_obs_rejects_invalid_place_label(obs: dict[str, Any]) -> None:
    bad = {
        "plan_id": "plan-bad",
        "mission": "PICK_AND_PLACE",
        "pick": "mug",
        "place": "plate",
    }
    agent = _make_agent(json.dumps(bad))
    with pytest.raises(ValueError, match="Place label"):
        agent.act_from_obs(obs)


def test_act_from_obs_allows_stop_with_none_pick(obs: dict[str, Any]) -> None:
    payload = {
        "plan_id": "plan-stop",
        "mission": "STOP",
        "pick": "none",
        "place": None,
    }
    agent = _make_agent(json.dumps(payload))
    output = agent.act_from_obs(obs)
    assert output.mission is MissionType.STOP
    assert output.pick == "none"


def test_act_from_obs_with_memory_and_feedback(
    obs: dict[str, Any], valid_output: PlannerOutput
) -> None:
    memory = ForgePlannerMemory(capacity=2)
    memory.append(
        round=0,
        scene_token="sim:task-001:step:0",
        query="q0",
        answer=valid_output,
        feedback=Feedback(success=False, observation="missed placement"),
    )
    agent = _make_agent(json.dumps(valid_output.model_dump()))
    output = agent.act_from_obs(
        obs,
        memory=memory,
        previous_feedback=Feedback(success=False, observation="missed"),
    )
    assert output == valid_output
    user_text = agent.planner.vllm_client.chat.call_args.args[0][1]["content"]
    assert "missed" in user_text
    assert "plan-001" in user_text


# --------------------------------------------------------------------- #
# Prompt version selection
# --------------------------------------------------------------------- #


def test_prompt_version_default_v0(obs: dict[str, Any]) -> None:
    agent = _make_agent(
        json.dumps({"plan_id": "p", "mission": "STOP", "pick": "none"})
    )
    assert agent.prompt_version == "v0"
    assert "{{rethinker_output}}" in agent.planner._user_template


def test_prompt_version_selectable_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_load(version: str) -> tuple[str, str]:
        seen.append(version)
        return f"system-{version}", f"user-{version}"

    monkeypatch.setattr(PromptRegistry, "load", staticmethod(fake_load))
    client = MagicMock()
    client.chat = MagicMock(
        return_value=json.dumps({"plan_id": "p", "mission": "STOP", "pick": "none"})
    )
    agent = ForgePlannerAgent(vllm_client=client, prompt_version="v9")
    assert agent.prompt_version == "v9"
    assert seen == ["v9"]
    assert agent.planner._system_template == "system-v9"


def test_set_prompt_version_swaps_templates(
    monkeypatch: pytest.MonkeyPatch, obs: dict[str, Any]
) -> None:
    def fake_load(version: str) -> tuple[str, str]:
        return f"system-{version}", f"user-{version}"

    monkeypatch.setattr(PromptRegistry, "load", staticmethod(fake_load))
    agent = _make_agent(
        json.dumps({"plan_id": "p", "mission": "STOP", "pick": "none"})
    )
    client = agent.planner.vllm_client
    agent.set_prompt_version("v7")
    assert agent.prompt_version == "v7"
    assert agent.planner._system_template == "system-v7"
    # The underlying VLLM client is preserved across prompt swaps.
    assert agent.planner.vllm_client is client
    agent.act_from_obs(obs)
    system_text = client.chat.call_args.args[0][0]["content"]
    assert system_text == "system-v7"


# --------------------------------------------------------------------- #
# ForgePlannerMemory: parity with the existing memory pattern
# --------------------------------------------------------------------- #


def test_forge_memory_is_planner_memory() -> None:
    assert isinstance(ForgePlannerMemory(), PlannerMemory)


def test_forge_memory_summarize_empty() -> None:
    memory = ForgePlannerMemory()
    assert memory.summarize(k=2) == PlannerMemory().summarize(k=2)


def test_forge_memory_append_summarize(valid_output: PlannerOutput) -> None:
    memory = ForgePlannerMemory(capacity=10)
    memory.append(
        round=0,
        scene_token="sim:task-001:step:0",
        query="q0",
        answer=valid_output,
    )
    memory.append(
        round=1,
        scene_token="sim:task-001:step:1",
        query="q1",
        answer=valid_output,
        feedback=Feedback(success=True, reward=1.0),
    )
    assert len(memory) == 2
    text = memory.summarize(k=1)
    assert "Compressed older rounds:" in text
    assert "mission=PICK_AND_PLACE" in text
    assert "pick=mug" in text
    assert "place=saucer" in text
    assert "Recent 1 round(s) in full:" in text


def test_forge_memory_capacity_eviction(valid_output: PlannerOutput) -> None:
    memory = ForgePlannerMemory(capacity=2)
    for i in range(3):
        memory.append(
            round=i,
            scene_token=f"sim:task-001:step:{i}",
            query=f"q{i}",
            answer=valid_output,
        )
    assert len(memory) == 2
    rounds = memory.rounds
    assert rounds[0].round == 1
    assert rounds[1].round == 2


def test_forge_memory_serialization_parity(valid_output: PlannerOutput) -> None:
    feedback = Feedback(success=False, observation="missed", reward=0.0)
    forge_memory = ForgePlannerMemory()
    planner_memory = PlannerMemory()
    for mem in (forge_memory, planner_memory):
        mem.append(
            round=0,
            scene_token="sim:task-001:step:0",
            query="q0",
            answer=valid_output,
            feedback=feedback,
        )
    assert forge_memory.to_dict() == planner_memory.to_dict()
    assert forge_memory.to_json() == planner_memory.to_json()
    record = forge_memory.to_dict()[0]
    assert record["round"] == 0
    assert record["scene_token"] == "sim:task-001:step:0"
    assert record["answer"]["plan_id"] == "plan-001"
    assert record["feedback"]["observation"] == "missed"


def test_forge_memory_rejects_bad_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ForgePlannerMemory(capacity=0)
