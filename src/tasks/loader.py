"""YAML task catalogue loader.

Loads one or more task catalogues and returns validated
:class:`tasks.schema.TaskDefinition` objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from common.schema import MissionType
from tasks.schema import TaskDefinition


def _coerce_mission_type(value: Any) -> Any:
    """Coerce string mission types into ``MissionType`` enums.

    Leaves already-valid ``MissionType`` values untouched and lets Pydantic
    reject anything else during validation.
    """
    if isinstance(value, MissionType):
        return value
    if isinstance(value, str):
        return MissionType(value.upper())
    return value


def _normalize_task(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw task dict before Pydantic validation."""
    normalized = dict(raw)
    if "mission_type" in normalized:
        normalized["mission_type"] = _coerce_mission_type(normalized["mission_type"])
    return normalized


def load_task_definitions(path: str | Path) -> list[TaskDefinition]:
    """Load task definitions from a YAML catalogue.

    The file must contain either a top-level ``tasks`` list or be a list of
    task mappings directly.

    Args:
        path: Path to the YAML catalogue.

    Returns:
        A list of validated :class:`TaskDefinition` objects.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is malformed or a task fails validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Task catalogue not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Task catalogue {file_path} contains malformed YAML") from exc

    if data is None:
        raise ValueError(f"Task catalogue {file_path} is empty")

    raw_tasks: list[dict[str, Any]]
    if isinstance(data, dict) and "tasks" in data:
        raw_tasks = data["tasks"]
    elif isinstance(data, list):
        raw_tasks = data
    else:
        raise ValueError(
            f"Task catalogue {file_path} must contain a top-level 'tasks' list "
            "or be a YAML list of tasks"
        )

    if not isinstance(raw_tasks, list):
        raise ValueError(
            f"Task catalogue {file_path} 'tasks' field must be a list"
        )

    definitions: list[TaskDefinition] = []
    for idx, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            raise ValueError(
                f"Task at index {idx} in {file_path} is not a mapping"
            )
        try:
            definitions.append(TaskDefinition(**_normalize_task(raw)))
        except ValidationError as exc:
            raise ValueError(
                f"Task at index {idx} in {file_path} failed validation: {exc}"
            ) from exc

    return definitions


def get_task_by_id(tasks: list[TaskDefinition], task_id: str) -> TaskDefinition:
    """Return the task whose ``id`` matches ``task_id``.

    Args:
        tasks: List of loaded task definitions.
        task_id: Task identifier to look up.

    Returns:
        The matching :class:`TaskDefinition`.

    Raises:
        ValueError: If no task with the given id exists.
    """
    for task in tasks:
        if task.id == task_id:
            return task
    raise ValueError(f"Task with id {task_id!r} not found")
