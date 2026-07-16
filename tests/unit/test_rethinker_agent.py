"""Unit tests for RethinkerAgent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from common.schema import DetectedObject, Feedback, MissionType, RethinkerOutput
from rethinker.agent import RethinkerAgent
from rethinker.memory import RethinkerMemory
from rethinker.prompts.registry import PromptRegistry


@pytest.fixture
def rgb_image() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def detections() -> list[DetectedObject]:
    return [
        DetectedObject(label="mug", bbox=[10.0, 20.0, 40.0, 50.0], confidence=0.92),
        DetectedObject(label="saucer", bbox=[5.0, 5.0, 30.0, 30.0], confidence=0.88),
    ]


@pytest.fixture
def valid_output() -> RethinkerOutput:
    return RethinkerOutput(
        mission_type=MissionType.PICK_AND_PLACE,
        reasoning="Mug and saucer are visible; pick the mug and place it on the saucer.",
        target_object="mug",
        target_container="saucer",
        arm_hint="right",
    )


def _make_agent(mock_response: str) -> RethinkerAgent:
    client = MagicMock()
    client.chat = MagicMock(return_value=mock_response)
    return RethinkerAgent(vllm_client=client)


def test_act_valid_response(
    rgb_image: np.ndarray,
    detections: list[DetectedObject],
    valid_output: RethinkerOutput,
) -> None:
    response = json.dumps(valid_output.model_dump())
    agent = _make_agent(response)
    output = agent.act(
        task_goal="Put the mug on the saucer.",
        rgb_image=rgb_image,
        detections=detections,
    )
    assert output == valid_output
    assert agent.vllm_client.chat.call_count == 1
    messages = agent.vllm_client.chat.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Put the mug on the saucer." in messages[1]["content"]
    assert agent.vllm_client.chat.call_args.kwargs["images"] == [rgb_image]


def test_act_with_memory_and_feedback(
    rgb_image: np.ndarray,
    detections: list[DetectedObject],
    valid_output: RethinkerOutput,
) -> None:
    memory = RethinkerMemory(capacity=2)
    memory.append(
        round=0,
        scene_token="scene-000",
        query="q0",
        answer=valid_output,
        feedback=Feedback(success=False, observation="missed grasp"),
    )
    response = json.dumps(valid_output.model_dump())
    agent = _make_agent(response)
    output = agent.act(
        task_goal="Put the mug on the saucer.",
        rgb_image=rgb_image,
        detections=detections,
        memory=memory,
        previous_feedback=Feedback(success=False, observation="missed"),
    )
    assert output == valid_output
    user_text = agent.vllm_client.chat.call_args.args[0][1]["content"]
    assert "Recent" in user_text or "All" in user_text or "Compressed" in user_text
    assert "missed" in user_text


def test_act_invalid_json_raises(rgb_image: np.ndarray) -> None:
    agent = _make_agent("this is not json")
    with pytest.raises(ValueError, match="could not parse"):
        agent.act(
            task_goal="Do something.",
            rgb_image=rgb_image,
            detections=[],
        )


def test_act_rejects_low_level_joint_angles(
    rgb_image: np.ndarray,
    detections: list[DetectedObject],
) -> None:
    bad = {
        "mission_type": "PICK_AND_PLACE",
        "reasoning": "Moving joints to grasp.",
        "target_object": "mug",
        "joint_target_angles": [0.1, 0.2, 0.3],
    }
    agent = _make_agent(json.dumps(bad))
    with pytest.raises(ValueError, match="joint"):
        agent.act(
            task_goal="Put the mug on the saucer.",
            rgb_image=rgb_image,
            detections=detections,
        )


def test_act_rejects_grasp_coordinates(
    rgb_image: np.ndarray,
    detections: list[DetectedObject],
) -> None:
    bad = {
        "mission_type": "PICK_AND_PLACE",
        "reasoning": "Grasping at this point.",
        "target_object": "mug",
        "grasp_point": [0.12, 0.34, 0.56],
    }
    agent = _make_agent(json.dumps(bad))
    # Extra keys are forbidden by RethinkerOutput, so this must fail.
    with pytest.raises(ValueError):
        agent.act(
            task_goal="Put the mug on the saucer.",
            rgb_image=rgb_image,
            detections=detections,
        )


def test_act_handles_empty_detections(
    rgb_image: np.ndarray,
    valid_output: RethinkerOutput,
) -> None:
    valid_output = valid_output.model_copy(
        update={"mission_type": MissionType.REOBSERVE}
    )
    agent = _make_agent(json.dumps(valid_output.model_dump()))
    output = agent.act(
        task_goal="Find the mug.",
        rgb_image=rgb_image,
        detections=[],
    )
    assert output.mission_type == MissionType.REOBSERVE


def test_registry_loads_v0() -> None:
    system, user = PromptRegistry.load("v0")
    assert "Rethinker" in system
    assert "{{task_goal}}" in user


def test_registry_versions() -> None:
    assert "v0" in PromptRegistry.versions()


def test_agent_uses_registry_defaults() -> None:
    client = MagicMock()
    client.chat = MagicMock(return_value=json.dumps({
        "mission_type": "STOP",
        "reasoning": "Done.",
    }))
    agent = RethinkerAgent(vllm_client=client)
    assert agent.prompt_version == "v0"
    assert "{{task_goal}}" in agent._user_template
