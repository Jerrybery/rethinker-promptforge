"""Unit tests for the real-robot evaluation task catalogue."""

from __future__ import annotations

from pathlib import Path

from tasks.loader import get_task_by_id, load_task_definitions
from tasks.schema import TaskDefinition


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_TASKS_PATH = REPO_ROOT / "data" / "tasks" / "real_tasks.yaml"

EXPECTED_TASK_IDS = {
    "clear_cluttered_plate_easy",
    "get_hidden_pliers",
    "fruit_put_pot",
    "cookie_put_pot",
}

# Tasks whose evaluation explicitly involves occlusion handling.
OCCLUDED_TASK_IDS = {"get_hidden_pliers", "fruit_put_pot", "cookie_put_pot"}


def test_real_tasks_catalogue_loads() -> None:
    """The bundled real_tasks.yaml should load and validate."""
    tasks = load_task_definitions(REAL_TASKS_PATH)
    assert len(tasks) == 4
    assert all(isinstance(t, TaskDefinition) for t in tasks)


def test_real_tasks_exact_ids() -> None:
    """The catalogue must contain exactly the 4 real evaluation task ids."""
    tasks = load_task_definitions(REAL_TASKS_PATH)
    assert {t.id for t in tasks} == EXPECTED_TASK_IDS


def test_real_tasks_lookup_by_id() -> None:
    """get_task_by_id resolves each of the 4 real task ids."""
    tasks = load_task_definitions(REAL_TASKS_PATH)
    for task_id in EXPECTED_TASK_IDS:
        task = get_task_by_id(tasks, task_id)
        assert task.id == task_id


def test_real_tasks_have_success_criteria() -> None:
    """Every real task defines non-empty success criteria."""
    tasks = load_task_definitions(REAL_TASKS_PATH)
    for task in tasks:
        assert task.success_criteria, f"{task.id} missing success_criteria"


def test_real_tasks_have_max_rounds() -> None:
    """Every real task defines a positive metadata.max_rounds."""
    tasks = load_task_definitions(REAL_TASKS_PATH)
    for task in tasks:
        assert task.metadata is not None, f"{task.id} missing metadata"
        max_rounds = task.metadata.get("max_rounds")
        assert isinstance(max_rounds, int) and max_rounds > 0, (
            f"{task.id} missing positive metadata.max_rounds"
        )


def test_real_tasks_occlusion_info() -> None:
    """Occlusion sources are recorded; occluded tasks list at least one."""
    tasks = load_task_definitions(REAL_TASKS_PATH)
    for task in tasks:
        assert task.metadata is not None
        assert "occlusion_sources" in task.metadata, (
            f"{task.id} missing metadata.occlusion_sources"
        )
        if task.id in OCCLUDED_TASK_IDS:
            assert len(task.metadata["occlusion_sources"]) > 0, (
                f"{task.id} should list at least one occlusion source"
            )


def test_real_tasks_have_sim_mapping() -> None:
    """Every real task maps to a RoboTwin sim task name for reproduction."""
    tasks = load_task_definitions(REAL_TASKS_PATH)
    for task in tasks:
        assert task.metadata is not None
        assert task.metadata.get("robottwin_task_name"), (
            f"{task.id} missing metadata.robottwin_task_name"
        )
        assert task.initial_scene is not None
        assert task.initial_scene.get("task_name"), (
            f"{task.id} missing initial_scene.task_name"
        )
