"""Unit tests for scripts/evaluate_real.py ``_build_robot_for_task``.

The heavy RoboTwin env/robot construction is monkeypatched; only the
argument marshalling (scene -> make_robottwin_env kwargs/overrides) is
exercised.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import robot.interface
import robot.robottwin_env
from tasks.schema import TaskDefinition

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_real.py"
_spec = importlib.util.spec_from_file_location("evaluate_real", _MODULE_PATH)
evaluate_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_real)


def _make_task(initial_scene: dict[str, Any]) -> TaskDefinition:
    return TaskDefinition(
        id="unit-real-task",
        instruction="Do the thing.",
        mission_type="PICK_ONLY",
        objects=["obj"],
        initial_scene=initial_scene,
    )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured_kwargs: dict[str, Any] = {}

    def fake_make_robottwin_env(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(
        robot.robottwin_env, "make_robottwin_env", fake_make_robottwin_env
    )
    monkeypatch.setattr(
        robot.interface, "RoboTwinBackend", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        evaluate_real, "RobotInterface", lambda **kwargs: ("robot", kwargs)
    )
    return captured_kwargs


def test_build_robot_for_task_filters_scene_kwarg_keys(captured: dict[str, Any]) -> None:
    task = _make_task(
        {
            "task_name": "fake_task",
            "seed": 7,
            "render_freq": 5,
            "embodiment": ["aloha-agilex"],
            "save_data": False,
        }
    )
    args = SimpleNamespace(config_path="unused.yaml")
    evaluate_real._build_robot_for_task(task, args)

    overrides = captured["overrides"]
    # Promoted to explicit kwargs (matching SimEnv), never forwarded as
    # overrides.
    assert "task_name" not in overrides
    assert "seed" not in overrides
    assert "render_freq" not in overrides
    # Genuine overrides still pass through.
    assert overrides["embodiment"] == ["aloha-agilex"]
    assert overrides["save_data"] is False
    assert captured["task_name"] == "fake_task"
    assert captured["seed"] == 7
