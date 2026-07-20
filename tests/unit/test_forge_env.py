"""Unit tests for the forge simulation environment wrapper.

All tests use a fake duck-typed RoboTwin environment injected via
``env_factory`` so no Sapien/GPU is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from common.schema import MissionType
from forge.env import SimAction, SimEnv
from forge.loader import load_forge_tasks, occlusion_sources
from tasks.schema import TaskDefinition


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

    def __init__(self, success: bool = False, with_check_success: bool = True) -> None:
        self.robot = FakeRobot()
        self._success = success
        self.move_calls = 0
        self.stopped = False
        self.close_env_calls = 0
        if not with_check_success:
            # Remove the method so SimEnv must fall back gracefully.
            self.check_success = None  # shadow class method with non-callable

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
        self.move_calls += 1
        return True

    def check_success(self) -> bool:
        return self._success

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def close_env(self, clear_cache: bool = False) -> None:
        self.close_env_calls += 1


TASK_CONFIG: dict[str, Any] = {
    "id": "unit-pick",
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


def make_factory(fake_env: FakeRoboTwinEnv, captured: dict[str, Any] | None = None):
    """Return an env_factory that records its kwargs and yields ``fake_env``."""

    def factory(
        task_name: str,
        task_config_name: str = "demo_clean",
        repo_root: Any = None,
        seed: int = 0,
        render_freq: int = 0,
        overrides: dict[str, Any] | None = None,
    ) -> FakeRoboTwinEnv:
        if captured is not None:
            captured.update(
                task_name=task_name,
                task_config_name=task_config_name,
                seed=seed,
                render_freq=render_freq,
                overrides=overrides,
            )
        return fake_env

    return factory


@pytest.fixture
def fake_env() -> FakeRoboTwinEnv:
    return FakeRoboTwinEnv()


@pytest.fixture
def sim_env(fake_env: FakeRoboTwinEnv) -> SimEnv:
    return SimEnv(env_factory=make_factory(fake_env))


# --------------------------------------------------------------------- #
# reset / obs contract
# --------------------------------------------------------------------- #


def test_reset_returns_obs_contract(sim_env: SimEnv) -> None:
    obs = sim_env.reset(TASK_CONFIG)
    assert isinstance(obs, dict)
    assert set(obs) >= {"image", "state", "detections", "task", "step_index"}

    image = obs["image"]
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.size > 0

    state = obs["state"]
    assert len(state["pose"]) == 7
    assert 0.0 <= state["gripper"] <= 1.0
    assert isinstance(state["timestamp"], float)

    assert isinstance(obs["detections"], list)
    for det in obs["detections"]:
        assert set(det) >= {"label", "bbox", "confidence"}

    assert obs["task"]["id"] == "unit-pick"
    assert obs["task"]["mission_type"] == MissionType.PICK_ONLY.value
    assert obs["step_index"] == 0


def test_reset_accepts_task_definition(sim_env: SimEnv) -> None:
    task = TaskDefinition(**TASK_CONFIG)
    obs = sim_env.reset(task)
    assert obs["task"]["id"] == "unit-pick"


def test_reset_captures_initial_frame(sim_env: SimEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    frames = sim_env.render()
    assert len(frames) == 1
    assert frames[0].size > 0


def test_reset_passes_scene_overrides_to_factory(fake_env: FakeRoboTwinEnv) -> None:
    captured: dict[str, Any] = {}
    env = SimEnv(env_factory=make_factory(fake_env, captured))
    env.reset(TASK_CONFIG)
    assert captured["task_name"] == "fake_task"
    assert captured["task_config_name"] == "demo_clean"
    assert captured["seed"] == 0
    assert captured["overrides"]["embodiment"] == ["aloha-agilex"]
    assert captured["overrides"]["save_data"] is False
    # task_name/seed/render_freq are promoted to explicit kwargs, not overrides.
    assert "task_name" not in captured["overrides"]
    assert "seed" not in captured["overrides"]


def test_reset_without_initial_scene_raises(sim_env: SimEnv) -> None:
    bad = {k: v for k, v in TASK_CONFIG.items() if k != "initial_scene"}
    with pytest.raises(ValueError, match="initial_scene"):
        sim_env.reset(bad)


# --------------------------------------------------------------------- #
# step
# --------------------------------------------------------------------- #


def test_step_returns_gym_tuple(sim_env: SimEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    obs, reward, done, info = sim_env.step(
        {"mission": "PICK_ONLY", "target": "mock_object"}
    )
    assert isinstance(obs, dict)
    assert obs["step_index"] == 1
    assert reward == 0.0
    assert done is False
    assert info["success"] is False
    assert info["primitive_success"] is True
    assert info["task_id"] == "unit-pick"
    assert info["truncated"] is False


def test_step_accepts_sim_action_model(sim_env: SimEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    action = SimAction(mission=MissionType.PICK_ONLY, target="mock_object")
    obs, reward, done, info = sim_env.step(action)
    assert info["primitive_success"] is True


def test_step_appends_frame_synchronized(sim_env: SimEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    sim_env.step({"mission": "PICK_ONLY", "target": "mock_object"})
    sim_env.step({"mission": "REOBSERVE"})
    frames = sim_env.render()
    assert len(frames) == 3  # reset + 2 steps
    assert all(f.size > 0 for f in frames)


def test_step_success_gives_reward_and_done() -> None:
    fake = FakeRoboTwinEnv(success=True)
    env = SimEnv(env_factory=make_factory(fake))
    env.reset(TASK_CONFIG)
    obs, reward, done, info = env.step(
        {"mission": "PICK_ONLY", "target": "mock_object"}
    )
    assert reward == 1.0
    assert done is True
    assert info["success"] is True


def test_step_stop_action_ends_episode(sim_env: SimEnv, fake_env: FakeRoboTwinEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    obs, reward, done, info = sim_env.step({"mission": "STOP"})
    assert done is True
    assert fake_env.stopped is True


def test_step_without_check_success_reports_unknown() -> None:
    fake = FakeRoboTwinEnv(with_check_success=False)
    env = SimEnv(env_factory=make_factory(fake))
    env.reset(TASK_CONFIG)
    obs, reward, done, info = env.step(
        {"mission": "PICK_ONLY", "target": "mock_object"}
    )
    assert reward == 0.0
    assert info["success"] is None
    assert done is False


def test_step_move_aside_dispatches(sim_env: SimEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    obs, reward, done, info = sim_env.step(
        {"mission": "MOVE_ASIDE", "target": "mock_object"}
    )
    assert info["primitive_success"] is True


def test_step_failed_primitive_still_returns_obs(sim_env: SimEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    obs, reward, done, info = sim_env.step(
        {"mission": "PICK_ONLY", "target": "nonexistent_object"}
    )
    assert info["primitive_success"] is False
    assert "nonexistent_object" in info["primitive_status"]
    assert reward == 0.0
    assert obs["step_index"] == 1


def test_step_before_reset_raises(sim_env: SimEnv) -> None:
    with pytest.raises(RuntimeError, match="reset"):
        sim_env.step({"mission": "STOP"})


def test_step_after_done_raises(sim_env: SimEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    sim_env.step({"mission": "STOP"})
    with pytest.raises(RuntimeError, match="done|reset"):
        sim_env.step({"mission": "REOBSERVE"})


def test_step_truncates_at_max_rounds() -> None:
    fake = FakeRoboTwinEnv(success=False)
    env = SimEnv(env_factory=make_factory(fake))
    config = dict(TASK_CONFIG)
    config["metadata"] = {**TASK_CONFIG["metadata"], "max_rounds": 1}
    env.reset(config)
    obs, reward, done, info = env.step({"mission": "REOBSERVE"})
    assert done is True
    assert info["truncated"] is True
    assert info["success"] is False


def test_action_missing_target_rejected() -> None:
    with pytest.raises(ValueError):
        SimAction(mission=MissionType.PICK_ONLY)


def test_render_before_reset_returns_empty(sim_env: SimEnv) -> None:
    assert sim_env.render() == []


def test_rerender_clears_frames(sim_env: SimEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    sim_env.step({"mission": "REOBSERVE"})
    sim_env.reset(TASK_CONFIG)
    assert len(sim_env.render()) == 1


# --------------------------------------------------------------------- #
# close / env release
# --------------------------------------------------------------------- #


def test_close_releases_wrapped_env(sim_env: SimEnv, fake_env: FakeRoboTwinEnv) -> None:
    sim_env.reset(TASK_CONFIG)
    sim_env.close()
    assert fake_env.close_env_calls == 1
    assert sim_env.task is None
    assert sim_env.render() == []
    with pytest.raises(RuntimeError, match="reset"):
        sim_env.step({"mission": "STOP"})


def test_close_is_idempotent_and_safe_before_reset(sim_env: SimEnv, fake_env: FakeRoboTwinEnv) -> None:
    sim_env.close()  # no reset yet: must not raise
    sim_env.reset(TASK_CONFIG)
    sim_env.close()
    sim_env.close()
    assert fake_env.close_env_calls == 1


def test_reset_closes_previous_env() -> None:
    envs = [FakeRoboTwinEnv(), FakeRoboTwinEnv()]
    calls = {"n": 0}

    def factory(*args: Any, **kwargs: Any) -> FakeRoboTwinEnv:
        env = envs[calls["n"]]
        calls["n"] += 1
        return env

    sim_env = SimEnv(env_factory=factory)
    sim_env.reset(TASK_CONFIG)
    sim_env.reset(TASK_CONFIG)
    assert envs[0].close_env_calls == 1
    assert envs[1].close_env_calls == 0
    assert calls["n"] == 2


def test_reset_with_invalid_config_keeps_previous_env_open() -> None:
    fake = FakeRoboTwinEnv()
    sim_env = SimEnv(env_factory=make_factory(fake))
    sim_env.reset(TASK_CONFIG)
    bad = {k: v for k, v in TASK_CONFIG.items() if k != "initial_scene"}
    with pytest.raises(ValueError, match="initial_scene"):
        sim_env.reset(bad)
    assert fake.close_env_calls == 0


# --------------------------------------------------------------------- #
# occlusion variant loader
# --------------------------------------------------------------------- #


def _write_catalogue(tmp_path: Path, tasks: list[dict[str, Any]]) -> Path:
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump({"tasks": tasks}), encoding="utf-8")
    return path


def test_loader_round_trips_plain_catalogue(tmp_path: Path) -> None:
    path = _write_catalogue(tmp_path, [TASK_CONFIG])
    tasks = load_forge_tasks(path)
    assert len(tasks) == 1
    assert tasks[0].id == "unit-pick"
    assert occlusion_sources(tasks[0]) == []


def test_loader_accepts_occlusion_variants(tmp_path: Path) -> None:
    variant = dict(TASK_CONFIG)
    variant["id"] = "unit-pick-occluded"
    variant["metadata"] = {
        **TASK_CONFIG["metadata"],
        "occlusion_sources": [
            "cloth fully covers the target object",
            "distractor tools partially occlude the target",
        ],
    }
    path = _write_catalogue(tmp_path, [variant])
    tasks = load_forge_tasks(path)
    assert occlusion_sources(tasks[0]) == [
        "cloth fully covers the target object",
        "distractor tools partially occlude the target",
    ]


def test_loader_rejects_malformed_occlusion_sources(tmp_path: Path) -> None:
    variant = dict(TASK_CONFIG)
    variant["id"] = "unit-pick-bad-occlusion"
    variant["metadata"] = {**TASK_CONFIG["metadata"], "occlusion_sources": "a cloth"}
    path = _write_catalogue(tmp_path, [variant])
    with pytest.raises(ValueError, match="occlusion_sources"):
        load_forge_tasks(path)


def test_loader_loads_repo_hello_catalogue() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tasks = load_forge_tasks(repo_root / "data" / "tasks" / "hello_tasks.yaml")
    assert len(tasks) >= 2
    assert all(occlusion_sources(t) == [] for t in tasks)
