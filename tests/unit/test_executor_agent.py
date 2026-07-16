"""Unit tests for ExecutorAgent.

All dependencies are mocked so the tests run without hardware, simulators,
or real object detectors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from common.schema import DetectedObject, ExecutorOutput, Feedback, MissionType, PlannerOutput
from executor.agent import ExecutorAgent
from executor.primitives import PrimitiveLibrary, PrimitiveResult
from executor.schema import ExecutorAgentOutput
from perception.dino_client import DINOClient
from robot.interface import RobotInterface


@pytest.fixture
def rgb() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_primitives() -> MagicMock:
    return MagicMock(spec=PrimitiveLibrary)


@pytest.fixture
def mock_dino() -> MagicMock:
    return MagicMock(spec=DINOClient)


@pytest.fixture
def mock_robot() -> MagicMock:
    return MagicMock(spec=RobotInterface)


@pytest.fixture
def agent(
    mock_primitives: MagicMock,
    mock_dino: MagicMock,
    mock_robot: MagicMock,
) -> ExecutorAgent:
    return ExecutorAgent(
        primitives=mock_primitives,
        dino=mock_dino,
        robot=mock_robot,
    )


def _plan(mission: MissionType, pick: str = "mug", place: str | None = None) -> PlannerOutput:
    return PlannerOutput(
        plan_id="plan-001",
        mission=mission,
        pick=pick,
        place=place,
    )


def _det(
    label: str = "mug",
    confidence: float = 0.95,
    bbox: list[float] | None = None,
) -> DetectedObject:
    return DetectedObject(
        label=label,
        bbox=bbox or [100.0, 100.0, 200.0, 200.0],
        confidence=confidence,
    )


class TestMissionDispatch:
    def test_pick_and_place_success(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [_det()]
        mock_primitives.pick.return_value = PrimitiveResult(
            success=True, status="picked mug"
        )
        mock_primitives.place.return_value = PrimitiveResult(
            success=True, status="placed mug on saucer"
        )

        output = agent.act(_plan(MissionType.PICK_AND_PLACE, place="saucer"), rgb)

        assert isinstance(output, ExecutorOutput)
        assert output.success is True
        assert output.risk is False
        assert output.feedback is not None
        assert output.feedback.success is True
        mock_primitives.pick.assert_called_once_with("mug")
        mock_primitives.place.assert_called_once_with("saucer")

    def test_pick_only_success(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [_det()]
        mock_primitives.pick.return_value = PrimitiveResult(
            success=True, status="picked mug"
        )

        output = agent.act(_plan(MissionType.PICK_ONLY), rgb)

        assert output.success is True
        assert output.risk is False
        mock_primitives.pick.assert_called_once_with("mug")
        mock_primitives.place.assert_not_called()

    def test_move_aside_success(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [_det()]
        mock_primitives.move_aside.return_value = PrimitiveResult(
            success=True, status="moved mug aside"
        )

        output = agent.act(_plan(MissionType.MOVE_ASIDE), rgb)

        assert output.success is True
        assert output.risk is False
        mock_primitives.move_aside.assert_called_once_with("mug")

    def test_reobserve_success(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_primitives.reobserve.return_value = PrimitiveResult(
            success=True, status="reobserved scene"
        )

        output = agent.act(_plan(MissionType.REOBSERVE, pick="none"), rgb)

        assert output.success is True
        assert output.risk is False
        mock_primitives.reobserve.assert_called_once()

    def test_stop_success(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_primitives.stop.return_value = PrimitiveResult(
            success=True, status="stopped"
        )

        output = agent.act(_plan(MissionType.STOP, pick="none"), rgb)

        assert output.success is True
        assert output.risk is False
        assert output.status == "stopped"
        mock_primitives.stop.assert_called_once()


class TestTargetLookup:
    def test_missing_target_returns_failure(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = []

        output = agent.act(_plan(MissionType.PICK_ONLY), rgb)

        assert output.success is False
        assert output.risk is True
        assert output.feedback is not None
        assert output.feedback.success is False
        assert "mug" in (output.feedback.error_message or "")
        mock_primitives.pick.assert_not_called()

    def test_multiple_matches_set_risk(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [
            _det(confidence=0.9),
            _det(confidence=0.7, bbox=[250.0, 250.0, 350.0, 350.0]),
        ]
        mock_primitives.pick.return_value = PrimitiveResult(
            success=True, status="picked mug"
        )

        output = agent.act(_plan(MissionType.PICK_ONLY), rgb)

        assert output.success is True
        assert output.risk is True
        assert "multiple matches" in (output.status or "").lower()
        mock_primitives.pick.assert_called_once()

    def test_low_confidence_sets_risk(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [_det(confidence=0.3)]
        mock_primitives.pick.return_value = PrimitiveResult(
            success=True, status="picked mug"
        )

        output = agent.act(_plan(MissionType.PICK_ONLY), rgb)

        assert output.success is True
        assert output.risk is True
        assert "low confidence" in (output.status or "").lower()


class TestPrimitiveFailures:
    def test_pick_failure_stops_place(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [_det()]
        mock_primitives.pick.return_value = PrimitiveResult(
            success=False, status="object mug not detected"
        )

        output = agent.act(_plan(MissionType.PICK_AND_PLACE, place="saucer"), rgb)

        assert output.success is False
        assert output.risk is True
        assert "pick failed" in (output.status or "").lower()
        mock_primitives.pick.assert_called_once()
        mock_primitives.place.assert_not_called()

    def test_place_failure_returns_failure(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [_det()]
        mock_primitives.pick.return_value = PrimitiveResult(
            success=True, status="picked mug"
        )
        mock_primitives.place.return_value = PrimitiveResult(
            success=False, status="no free placement pose"
        )

        output = agent.act(_plan(MissionType.PICK_AND_PLACE, place="saucer"), rgb)

        assert output.success is False
        assert output.risk is True
        assert output.feedback is not None
        assert output.feedback.success is False


class TestMisc:
    def test_step_index_increments(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [_det()]
        mock_primitives.pick.return_value = PrimitiveResult(
            success=True, status="picked mug"
        )

        first = agent.act(_plan(MissionType.PICK_ONLY), rgb)
        second = agent.act(_plan(MissionType.PICK_ONLY), rgb)

        assert first.step_index == 0
        assert second.step_index == 1

    def test_depth_parameter_accepted(
        self,
        agent: ExecutorAgent,
        mock_primitives: MagicMock,
        mock_dino: MagicMock,
        rgb: np.ndarray,
    ) -> None:
        mock_dino.detect.return_value = [_det()]
        mock_primitives.pick.return_value = PrimitiveResult(
            success=True, status="picked mug"
        )

        depth = np.zeros((480, 640), dtype=np.float32)
        output = agent.act(_plan(MissionType.PICK_ONLY), rgb, depth=depth)

        assert output.success is True
