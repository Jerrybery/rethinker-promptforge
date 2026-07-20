"""Integration test: planner-driven forge loop with episode recording.

Runs 2-3 steps of ``reset -> act_from_obs -> planner_output_to_sim_action ->
env.step`` against a fake duck-typed RoboTwin env (same approach as
``tests/unit/test_forge_env.py``) and a mocked VLLM client. No real
Sapien/LLM. Verifies the planner runs in the sim loop and the recorder
captures frames plus keyframe metadata matching the steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from forge.actions import planner_output_to_sim_action
from forge.env import SimEnv
from forge.planner_agent import ForgePlannerAgent
from forge.recorder import EpisodeRecorder, EpisodeRecording


class FakeRobot:
    """Minimal stand-in for the RoboTwin robot object."""

    def __init__(self) -> None:
        self._gripper_val = 1.0

    def get_right_gripper_val(self) -> float:
        return self._gripper_val

    def get_left_gripper_val(self) -> float:
        return self._gripper_val


class FakeRoboTwinEnv:
    """Duck-typed fake matching the RoboTwin base task API used by SimEnv."""

    def __init__(self, success: bool = False) -> None:
        self.robot = FakeRobot()
        self._success = success
        self.stopped = False

    def get_obs(self) -> dict[str, Any]:
        rgb = np.full((8, 8, 3), 17, dtype=np.uint8)
        return {"observation": {"head_camera": {"rgb": rgb}}}

    def get_arm_pose(self, arm_tag: str = "right") -> list[float]:
        return [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]

    def move_to_pose(self, arm_tag: str, pose: list[float]) -> list[str]:
        return ["action"]

    def open_gripper(self, arm_tag: str) -> list[str]:
        self.robot._gripper_val = 1.0
        return ["action"]

    def close_gripper(self, arm_tag: str) -> list[str]:
        self.robot._gripper_val = 0.0
        return ["action"]

    def move(self, actions: list[Any]) -> bool:
        return True

    def check_success(self) -> bool:
        return self._success

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


TASK_CONFIG: dict[str, Any] = {
    "id": "loop-pick",
    "instruction": "Pick up the mock object.",
    "mission_type": "PICK_ONLY",
    "objects": ["mock_object"],
    "initial_scene": {
        "task_name": "fake_task",
        "embodiment": ["aloha-agilex"],
        "seed": 0,
        "render_freq": 0,
        "save_data": False,
        "collect_data": False,
    },
    "metadata": {
        "robottwin_task_name": "fake_task",
        "robottwin_task_config": "demo_clean",
    },
}


def make_factory(fake_env: FakeRoboTwinEnv):
    def factory(
        task_name: str,
        task_config_name: str = "demo_clean",
        repo_root: Any = None,
        seed: int = 0,
        render_freq: int = 0,
        overrides: dict[str, Any] | None = None,
    ) -> FakeRoboTwinEnv:
        return fake_env

    return factory


def _mock_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)
    return client


def _run_episode(
    env: SimEnv,
    agent: ForgePlannerAgent,
    recorder: EpisodeRecorder,
    max_steps: int,
) -> int:
    obs = env.reset(TASK_CONFIG)
    recorder.add_frame(obs["image"])
    steps = 0
    done = False
    while not done and steps < max_steps:
        output = agent.act_from_obs(obs)
        action = planner_output_to_sim_action(output)
        recorder.mark_event(
            obs["step_index"],
            "decision",
            f"{output.mission.value} pick={output.pick} place={output.place}",
        )
        obs, reward, done, info = env.step(action)
        recorder.add_frame(obs["image"])
        if reward == 1.0:
            recorder.mark_event(info["step_index"], "success", "task succeeded")
        elif not info["primitive_success"]:
            recorder.mark_event(
                info["step_index"], "failure", info["primitive_status"]
            )
        steps += 1
    return steps


def test_planner_loop_records_episode(tmp_path: Path) -> None:
    fake = FakeRoboTwinEnv()
    env = SimEnv(env_factory=make_factory(fake))
    # Mock DINO returns label "mock_object"; the planner must pick from it.
    pick = json.dumps({"plan_id": "p-1", "mission": "PICK_ONLY", "pick": "mock_object"})
    agent = ForgePlannerAgent(vllm_client=_mock_client([pick, pick, pick]))

    recorder = EpisodeRecorder()
    recorder.start_episode("loop-ep", tmp_path, fps=5.0)
    steps = _run_episode(env, agent, recorder, max_steps=3)
    recording = recorder.finish()

    assert steps == 3
    # Frames: reset + one per step, matching the env frame buffer.
    assert recording.frame_count == len(env.render()) == 4

    video_path = Path(recording.video_path)
    assert video_path.exists()
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    cap.release()

    decisions = [kf for kf in recording.keyframes if kf.kind == "decision"]
    assert [kf.step_index for kf in decisions] == [0, 1, 2]
    assert [kf.frame_index for kf in decisions] == [0, 1, 2]
    assert all("PICK_ONLY" in kf.detail for kf in decisions)
    # First/last frames always marked.
    assert recording.keyframes[0].kind == "start"
    assert recording.keyframes[-1].kind == "end"

    # Metadata JSON round-trips for the critic.
    loaded = EpisodeRecording(
        **json.loads(Path(recording.metadata_path).read_text())
    )
    assert loaded == recording


def test_planner_loop_stop_ends_episode(tmp_path: Path) -> None:
    fake = FakeRoboTwinEnv()
    env = SimEnv(env_factory=make_factory(fake))
    pick = json.dumps({"plan_id": "p-1", "mission": "PICK_ONLY", "pick": "mock_object"})
    stop = json.dumps({"plan_id": "p-2", "mission": "STOP", "pick": "none"})
    agent = ForgePlannerAgent(vllm_client=_mock_client([pick, stop, pick]))

    recorder = EpisodeRecorder()
    recorder.start_episode("loop-stop", tmp_path, fps=5.0)
    steps = _run_episode(env, agent, recorder, max_steps=5)
    recording = recorder.finish()

    assert steps == 2
    assert fake.stopped is True
    assert recording.frame_count == 3
    decisions = [kf for kf in recording.keyframes if kf.kind == "decision"]
    assert any("STOP" in kf.detail for kf in decisions)


def test_planner_loop_failure_events_recorded(tmp_path: Path) -> None:
    fake = FakeRoboTwinEnv()
    env = SimEnv(env_factory=make_factory(fake))
    # One planner step, then a manual risk annotation mid-episode (as the
    # forge loop would when the rethinker flags an occlusion hypothesis).
    pick = json.dumps({"plan_id": "p-1", "mission": "PICK_ONLY", "pick": "mock_object"})
    agent = ForgePlannerAgent(vllm_client=_mock_client([pick, pick]))

    recorder = EpisodeRecorder()
    recorder.start_episode("loop-risk", tmp_path, fps=5.0)
    obs = env.reset(TASK_CONFIG)
    recorder.add_frame(obs["image"])
    output = agent.act_from_obs(obs)
    action = planner_output_to_sim_action(output)
    obs, reward, done, info = env.step(action)
    recorder.add_frame(obs["image"])
    recorder.mark_event(info["step_index"], "risk", "target possibly occluded")
    recording = recorder.finish()

    risks = [kf for kf in recording.keyframes if kf.kind == "risk"]
    assert len(risks) == 1
    assert risks[0].step_index == 1
    assert risks[0].frame_index == 1
