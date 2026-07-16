"""Unit tests for symbolic action primitives.

All tests use the mock ``RobotInterface`` and the mock DINO client so no
hardware or simulator is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from executor.primitives import PrimitiveLibrary, PrimitiveResult
from perception.dino_client import DINOClient
from robot.interface import RoboTwinBackend, RobotInterface


@pytest.fixture
def robot() -> RobotInterface:
    return RobotInterface(mock=True)


@pytest.fixture
def dino() -> DINOClient:
    return DINOClient(mode="mock")


@pytest.fixture
def library(robot: RobotInterface, dino: DINOClient) -> PrimitiveLibrary:
    return PrimitiveLibrary(robot=robot, dino=dino)


def test_pick_runs_without_error(library: PrimitiveLibrary) -> None:
    result = library.pick("mock_object")
    assert isinstance(result, PrimitiveResult)
    assert result.success is True
    assert "mock_object" in result.status


def test_place_runs_without_error(library: PrimitiveLibrary) -> None:
    result = library.place("mock_object")
    assert isinstance(result, PrimitiveResult)
    assert result.success is True


def test_move_aside_runs_without_error(library: PrimitiveLibrary) -> None:
    result = library.move_aside("mock_object")
    assert isinstance(result, PrimitiveResult)
    assert result.success is True


def test_reobserve_runs_without_error(library: PrimitiveLibrary) -> None:
    result = library.reobserve()
    assert isinstance(result, PrimitiveResult)
    assert result.success is True
    assert "detections" in result.data


def test_stop_runs_without_error(library: PrimitiveLibrary) -> None:
    result = library.stop()
    assert isinstance(result, PrimitiveResult)
    assert result.success is True
    assert result.status == "stopped"


def test_pick_missing_object_returns_failure(library: PrimitiveLibrary) -> None:
    result = library.pick("nonexistent_object")
    assert isinstance(result, PrimitiveResult)
    assert result.success is False
    assert "nonexistent_object" in result.status


def test_robot_interface_mock_state_shape() -> None:
    interface = RobotInterface(mock=True)
    state = interface.read_state()
    assert state.camera_image.ndim == 3
    assert state.camera_image.shape[2] == 3
    assert 0.0 <= state.gripper <= 1.0


def test_robot_interface_mock_move_and_gripper_update_state() -> None:
    interface = RobotInterface(mock=True)
    target = [0.55, 0.05, 0.35, 0.0, 0.0, 0.0, 1.0]

    move_result = interface.move_to(target)
    assert move_result["success"] is True

    gripper_result = interface.gripper(open=False)
    assert gripper_result["success"] is True

    state = interface.read_state()
    assert list(state.pose.to_list()) == pytest.approx(target)
    assert state.gripper == 0.0


def test_stop_invokes_backend(library: PrimitiveLibrary) -> None:
    calls: list[None] = []
    library.robot._backend.stop = lambda: calls.append(None)
    result = library.stop()
    assert isinstance(result, PrimitiveResult)
    assert result.success is True
    assert result.status == "stopped"
    assert len(calls) == 1


def test_place_without_target_label(library: PrimitiveLibrary) -> None:
    result = library.place()
    assert isinstance(result, PrimitiveResult)
    assert result.success is True
    assert result.status == "placed at current pose"


def test_robot_interface_mock_reset_returns_home() -> None:
    interface = RobotInterface(mock=True)
    interface.move_to([0.55, 0.05, 0.35, 0.0, 0.0, 0.0, 1.0])
    interface.gripper(open=False)

    interface.reset()

    state = interface.read_state()
    assert list(state.pose.to_list()) == pytest.approx([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert state.gripper == 1.0


class _FakeRobot:
    def get_right_gripper_val(self) -> float:
        return 0.5


class _FakeEnv:
    def __init__(self, robot: _FakeRobot | None = None, obs: dict | None = None):
        if robot is not None:
            self.robot = robot
        self._obs = obs

    def get_obs(self) -> dict:
        if self._obs is not None:
            return self._obs
        return {
            "observation": {
                "head_camera": {
                    "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
                }
            }
        }

    def get_arm_pose(self, arm_tag: str = "right") -> list[float]:
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "env_factory,expected_pattern",
    [
        (lambda: object(), "missing required method 'get_obs'"),
        (
            lambda: _FakeEnv(obs={}),
            "missing the expected 'observation' key",
        ),
        (
            lambda: _FakeEnv(obs={"observation": {}}),
            "missing the expected 'head_camera' key",
        ),
        (
            lambda: _FakeEnv(obs={"observation": {"head_camera": {}}}),
            "missing the expected 'rgb' key",
        ),
        (
            lambda: type("NoArmPose", (), {"robot": _FakeRobot(), "get_obs": _FakeEnv().get_obs})(),
            "missing required method 'get_arm_pose'",
        ),
        (
            lambda: type("NoRobot", (), {"get_obs": _FakeEnv().get_obs, "get_arm_pose": _FakeEnv().get_arm_pose})(),
            "missing required attribute 'robot'",
        ),
        (
            lambda: _FakeEnv(robot=object()),
            "missing required method 'get_right_gripper_val'",
        ),
    ],
)
def test_robotwin_backend_read_state_handles_missing_env_attributes(
    env_factory: Any,
    expected_pattern: str,
) -> None:
    backend = RoboTwinBackend(env=env_factory())
    with pytest.raises(RuntimeError, match=expected_pattern):
        backend.read_state()
