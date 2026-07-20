"""Task catalogue loader with occlusion-variant support.

Wraps :func:`tasks.loader.load_task_definitions` and additionally validates
the optional occlusion metadata that forge task variants carry under
``metadata.occlusion_sources`` (a list of human-readable occlusion
descriptions, e.g. "cloth fully covers the target object").

Occlusion variants are *mapped* onto existing RoboTwin tasks via
``metadata.robottwin_task_name`` / ``metadata.robottwin_task_config``;
spawning novel occluder objects inside RoboTwin is intentionally out of
scope here.
"""

from __future__ import annotations

from pathlib import Path

from tasks.loader import load_task_definitions
from tasks.schema import TaskDefinition


def _validate_occlusion_metadata(task: TaskDefinition, source: str) -> None:
    """Validate ``metadata.occlusion_sources`` for a single task.

    Raises:
        ValueError: If ``occlusion_sources`` is present but is not a list of
            non-empty strings.
    """
    metadata = task.metadata or {}
    if "occlusion_sources" not in metadata:
        return
    sources = metadata["occlusion_sources"]
    if not isinstance(sources, list) or not all(
        isinstance(item, str) and item.strip() for item in sources
    ):
        raise ValueError(
            f"Task {task.id!r} in {source}: metadata.occlusion_sources must be "
            f"a list of non-empty strings, got {sources!r}"
        )


def load_forge_tasks(path: str | Path) -> list[TaskDefinition]:
    """Load a task catalogue, validating forge occlusion metadata.

    Accepts the same YAML shape as :func:`tasks.loader.load_task_definitions`.
    Entries may add ``metadata.occlusion_sources`` (list of strings) to
    describe occlusion variants; any other ``metadata`` fields pass through
    unchanged.

    Raises:
        ValueError: If occlusion metadata is malformed (or the underlying
            catalogue fails validation).
    """
    tasks = load_task_definitions(path)
    for task in tasks:
        _validate_occlusion_metadata(task, str(path))
    return tasks


def occlusion_sources(task: TaskDefinition) -> list[str]:
    """Return the declared occlusion sources for ``task`` (empty if none)."""
    metadata = task.metadata or {}
    sources = metadata.get("occlusion_sources") or []
    return list(sources)
