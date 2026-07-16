"""Unit tests for PlannerAgent."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from common.schema import Feedback, MissionType, PlannerOutput, RethinkerOutput
from planner.agent import PlannerAgent
from planner.memory import PlannerMemory
from planner.prompts.registry import PromptRegistry


@pytest.fixture
def rethinker_output() -> RethinkerOutput:
    return RethinkerOutput(
        mission_type=MissionType.PICK_AND_PLACE,
        reasoning="Mug and saucer are visible; pick the mug and place it on the saucer.",
        target_object="mug",
        target_container="saucer",
        arm_hint="right",
    )


@pytest.fixture
def dino_labels() -> list[str]:
    return ["mug", "saucer", "spoon"]


@pytest.fixture
def action_library() -> list[str]:
    return ["pick", "place", "move_aside"]


@pytest.fixture
def valid_output() -> PlannerOutput:
    return PlannerOutput(
        plan_id="plan-001",
        mission=MissionType.PICK_AND_PLACE,
        pick="mug",
        place="saucer",
    )


def _make_agent(mock_response: str) -> PlannerAgent:
    client = MagicMock()
    client.chat = MagicMock(return_value=mock_response)
    return PlannerAgent(vllm_client=client)


def test_act_valid_response(
    rethinker_output: RethinkerOutput,
    dino_labels: list[str],
    action_library: list[str],
    valid_output: PlannerOutput,
) -> None:
    response = json.dumps(valid_output.model_dump())
    agent = _make_agent(response)
    output = agent.act(
        rethinker_output=rethinker_output,
        dino_labels=dino_labels,
        action_library=action_library,
    )
    assert output == valid_output
    assert agent.vllm_client.chat.call_count == 1
    messages = agent.vllm_client.chat.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_text = messages[1]["content"]
    assert "mug" in user_text
    assert "saucer" in user_text
    assert "PICK_AND_PLACE" in user_text
    # Planner must not receive raw images.
    assert agent.vllm_client.chat.call_args.kwargs == {}


def test_act_with_memory_and_feedback(
    rethinker_output: RethinkerOutput,
    dino_labels: list[str],
    valid_output: PlannerOutput,
) -> None:
    memory = PlannerMemory(capacity=2)
    memory.append(
        round=0,
        scene_token="scene-000",
        query="q0",
        answer=valid_output,
        feedback=Feedback(success=False, observation="missed placement"),
    )
    response = json.dumps(valid_output.model_dump())
    agent = _make_agent(response)
    output = agent.act(
        rethinker_output=rethinker_output,
        dino_labels=dino_labels,
        memory=memory,
        previous_feedback=Feedback(success=False, observation="missed"),
    )
    assert output == valid_output
    user_text = agent.vllm_client.chat.call_args.args[0][1]["content"]
    assert "Compressed older rounds:" in user_text or "All" in user_text or "Recent" in user_text
    assert "missed" in user_text


def test_act_invalid_json_raises(
    rethinker_output: RethinkerOutput,
    dino_labels: list[str],
) -> None:
    agent = _make_agent("this is not json")
    with pytest.raises(ValueError, match="could not parse"):
        agent.act(
            rethinker_output=rethinker_output,
            dino_labels=dino_labels,
        )


def test_act_rejects_invalid_mission(
    rethinker_output: RethinkerOutput,
    dino_labels: list[str],
) -> None:
    bad = {
        "plan_id": "plan-bad",
        "mission": "INVALID_MISSION",
        "pick": "mug",
        "place": "saucer",
    }
    agent = _make_agent(json.dumps(bad))
    with pytest.raises(ValueError, match="mission"):
        agent.act(
            rethinker_output=rethinker_output,
            dino_labels=dino_labels,
        )


def test_act_rejects_invalid_pick_label(
    rethinker_output: RethinkerOutput,
    dino_labels: list[str],
) -> None:
    bad = {
        "plan_id": "plan-bad",
        "mission": "PICK_AND_PLACE",
        "pick": "cup",
        "place": "saucer",
    }
    agent = _make_agent(json.dumps(bad))
    with pytest.raises(ValueError, match="Pick label"):
        agent.act(
            rethinker_output=rethinker_output,
            dino_labels=dino_labels,
        )


def test_act_rejects_invalid_place_label(
    rethinker_output: RethinkerOutput,
    dino_labels: list[str],
) -> None:
    bad = {
        "plan_id": "plan-bad",
        "mission": "PICK_AND_PLACE",
        "pick": "mug",
        "place": "plate",
    }
    agent = _make_agent(json.dumps(bad))
    with pytest.raises(ValueError, match="Place label"):
        agent.act(
            rethinker_output=rethinker_output,
            dino_labels=dino_labels,
        )


def test_act_rejects_grasp_coordinates(
    rethinker_output: RethinkerOutput,
    dino_labels: list[str],
) -> None:
    bad = {
        "plan_id": "plan-bad",
        "mission": "PICK_AND_PLACE",
        "pick": "mug",
        "place": "saucer",
        "grasp_point": [0.12, 0.34, 0.56],
    }
    agent = _make_agent(json.dumps(bad))
    with pytest.raises(ValueError):
        agent.act(
            rethinker_output=rethinker_output,
            dino_labels=dino_labels,
        )


def test_act_allows_null_place(
    rethinker_output: RethinkerOutput,
    dino_labels: list[str],
) -> None:
    ro = rethinker_output.model_copy(update={"mission_type": MissionType.PICK_ONLY})
    payload = {
        "plan_id": "plan-002",
        "mission": "PICK_ONLY",
        "pick": "mug",
        "place": None,
    }
    agent = _make_agent(json.dumps(payload))
    output = agent.act(
        rethinker_output=ro,
        dino_labels=dino_labels,
    )
    assert output.mission == MissionType.PICK_ONLY
    assert output.pick == "mug"
    assert output.place is None


def test_registry_loads_v0() -> None:
    system, user = PromptRegistry.load("v0")
    assert "Planner" in system
    assert "{{rethinker_output}}" in user


def test_registry_versions() -> None:
    assert "v0" in PromptRegistry.versions()


def test_agent_uses_registry_defaults() -> None:
    client = MagicMock()
    client.chat = MagicMock(
        return_value=json.dumps({
            "plan_id": "plan-003",
            "mission": "STOP",
            "pick": "mug",
        })
    )
    agent = PlannerAgent(vllm_client=client)
    assert agent.prompt_version == "v0"
    assert "{{rethinker_output}}" in agent._user_template
