"""Unit tests for symbolic action primitives.

All tests use the mock ``RobotInterface`` and the mock DINO client so no
hardware or simulator is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from executor.primitives import PrimitiveLibrary, PrimitiveResult
from perception.dino_client import DINOClient
from robot.interface import RobotInterface


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
