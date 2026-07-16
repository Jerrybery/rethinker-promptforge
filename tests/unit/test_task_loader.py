"""Unit tests for the task catalogue loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from common.schema import MissionType
from tasks.loader import get_task_by_id, load_task_definitions
from tasks.schema import TaskDefinition


REPO_ROOT = Path(__file__).resolve().parents[2]
HELLO_TASKS_PATH = REPO_ROOT / "data" / "tasks" / "hello_tasks.yaml"


def test_load_hello_tasks_yaml() -> None:
    """The bundled hello_tasks.yaml should load and validate."""
    tasks = load_task_definitions(HELLO_TASKS_PATH)
    assert len(tasks) >= 1
    assert all(isinstance(t, TaskDefinition) for t in tasks)


def test_task_id_lookup() -> None:
    """get_task_by_id returns the matching task."""
    tasks = load_task_definitions(HELLO_TASKS_PATH)
    task = get_task_by_id(tasks, "hello-place-a2b-right")
    assert task.id == "hello-place-a2b-right"
    assert task.mission_type == MissionType.PICK_AND_PLACE


def test_task_id_lookup_missing_raises() -> None:
    """get_task_by_id raises ValueError for unknown ids."""
    tasks = load_task_definitions(HELLO_TASKS_PATH)
    with pytest.raises(ValueError, match="not found"):
        get_task_by_id(tasks, "does-not-exist")


def test_missing_file_raises() -> None:
    """load_task_definitions raises FileNotFoundError for missing catalogues."""
    with pytest.raises(FileNotFoundError):
        load_task_definitions(REPO_ROOT / "data" / "tasks" / "missing.yaml")


def test_yaml_parsing_and_validation(tmp_path: Path) -> None:
    """A minimal valid catalogue is parsed into TaskDefinition objects."""
    catalogue = tmp_path / "tasks.yaml"
    catalogue.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "task-001",
                        "instruction": "pick the red block",
                        "mission_type": "PICK_ONLY",
                        "objects": ["red_block"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = load_task_definitions(catalogue)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "task-001"
    assert task.instruction == "pick the red block"
    assert task.mission_type == MissionType.PICK_ONLY
    assert task.objects == ["red_block"]


def test_mission_type_string_coercion(tmp_path: Path) -> None:
    """Lower-case mission_type strings are coerced to MissionType enums."""
    catalogue = tmp_path / "tasks.yaml"
    catalogue.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "task-coerce",
                        "instruction": "move aside",
                        "mission_type": "move_aside",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = load_task_definitions(catalogue)
    assert tasks[0].mission_type == MissionType.MOVE_ASIDE


def test_invalid_task_raises(tmp_path: Path) -> None:
    """Tasks failing Pydantic validation produce a clear ValueError."""
    catalogue = tmp_path / "tasks.yaml"
    catalogue.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "bad-task",
                        "instruction": "",
                        "mission_type": "PICK_ONLY",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="failed validation"):
        load_task_definitions(catalogue)


def test_empty_catalogue_raises(tmp_path: Path) -> None:
    """An empty YAML file must raise ValueError."""
    catalogue = tmp_path / "empty.yaml"
    catalogue.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_task_definitions(catalogue)


def test_malformed_catalogue_raises(tmp_path: Path) -> None:
    """A catalogue that is neither a list nor has a 'tasks' key must raise."""
    catalogue = tmp_path / "bad.yaml"
    catalogue.write_text(yaml.safe_dump({"not_tasks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain"):
        load_task_definitions(catalogue)


def test_optional_fields_loaded() -> None:
    """initial_scene, success_criteria, and metadata are loaded when present."""
    tasks = load_task_definitions(HELLO_TASKS_PATH)
    task = get_task_by_id(tasks, "hello-place-a2b-right")
    assert task.initial_scene is not None
    assert task.initial_scene.get("task_name") == "place_a2b_right"
    assert task.success_criteria is not None
    assert len(task.success_criteria) > 0
    assert task.metadata is not None
    assert task.metadata.get("robottwin_task_name") == "place_a2b_right"
