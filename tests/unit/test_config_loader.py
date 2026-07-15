"""Unit tests for the YAML config loader."""

from pathlib import Path

import pytest

from rethinker_promptforge.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_models_config() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "models.yaml")
    for key in ("vllm", "dino", "anygrasp", "cloud_critic"):
        assert key in cfg, f"missing top-level key: {key}"
    assert "model_id" in cfg["vllm"]
    assert "device" in cfg["dino"]
    assert "checkpoint_path" in cfg["anygrasp"]
    assert "temperature" in cfg["cloud_critic"]


def test_load_robot_config() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "robot.yaml")
    for key in ("robot", "camera", "workspace_bounds", "safety_limits"):
        assert key in cfg, f"missing top-level key: {key}"
    assert "model" in cfg["robot"]
    assert "gripper" in cfg["robot"]
    assert "intrinsic" in cfg["camera"]
    assert "extrinsic" in cfg["camera"]
    bounds = cfg["workspace_bounds"]
    for axis in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        assert axis in bounds, f"missing workspace bound: {axis}"
    limits = cfg["safety_limits"]
    for key in ("max_linear_velocity", "max_angular_velocity", "max_gripper_width"):
        assert key in limits, f"missing safety limit: {key}"


def test_load_config_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(REPO_ROOT / "configs" / "nonexistent.yaml")
