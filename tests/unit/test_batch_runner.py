"""Unit tests for the batch evaluation runner and result persistence."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from common.schema import (
    Episode,
    EpisodeStep,
    ExecutorOutput,
    Feedback,
    MissionType,
    RethinkerOutput,
)
from evaluation.batch import (
    BatchResult,
    MethodTaskSummary,
    TrialResult,
    make_run_dir,
    run_batch,
    write_batch_results,
)
from tasks.schema import TaskDefinition


def _task(task_id: str = "task-a", criteria: list[str] | None = None) -> TaskDefinition:
    return TaskDefinition(
        id=task_id,
        instruction="mock instruction",
        mission_type=MissionType.PICK_AND_PLACE,
        objects=["object"],
        success_criteria=criteria,
    )


def _step(step_index: int, task: TaskDefinition) -> EpisodeStep:
    return EpisodeStep(
        step_index=step_index,
        task=task,
        rethinker_output=RethinkerOutput(
            mission_type=MissionType.PICK_AND_PLACE,
            reasoning="mock reasoning",
        ),
        executor_output=ExecutorOutput(
            step_index=step_index,
            joint_angles=[0.0],
            gripper_state=0.5,
        ),
        feedback=Feedback(success=True),
    )


def _episode(
    episode_id: str,
    task: TaskDefinition,
    *,
    steps: int = 1,
    termination_reason: str = "stop",
) -> Episode:
    return Episode(
        id=episode_id,
        task_id=task.id,
        steps=[_step(i, task) for i in range(steps)],
        metadata={"termination_reason": termination_reason},
    )


def _counter_clock(start: float = 0.0, step: float = 0.5):
    ticks = iter(start + i * step for i in range(100000))
    return lambda: next(ticks)


def _simple_producer(task: TaskDefinition, trial_index: int) -> Episode:
    return _episode(f"{task.id}-t{trial_index}", task, steps=trial_index + 1)


class TestRunBatch:
    def test_covers_all_method_task_trial_combinations(self) -> None:
        tasks = [_task("t1"), _task("t2")]
        methods = {"full": _simple_producer, "monolithic": _simple_producer}
        result = run_batch(methods, tasks, trials=3, clock=_counter_clock())
        assert len(result.trials) == 2 * 2 * 3
        combos = {(t.method, t.task_id, t.trial_index) for t in result.trials}
        assert len(combos) == 12
        assert len(result.summaries) == 4

    def test_driver_runtime_recorded_per_trial(self) -> None:
        result = run_batch(
            {"full": _simple_producer},
            [_task("t1")],
            trials=2,
            clock=_counter_clock(0.0, 0.5),
        )
        runtimes = [t.evaluation.runtime_seconds for t in result.trials]
        assert runtimes == [0.5, 0.5]

    def test_episode_metadata_runtime_takes_precedence(self) -> None:
        def produce(task: TaskDefinition, trial_index: int) -> Episode:
            episode = _episode("e", task)
            return episode.model_copy(
                update={
                    "metadata": {**(episode.metadata or {}), "runtime_seconds": 42.0}
                }
            )

        result = run_batch({"full": produce}, [_task()], trials=1, clock=_counter_clock())
        assert result.trials[0].evaluation.runtime_seconds == 42.0

    def test_summaries_aggregate_metrics(self) -> None:
        task = _task("t1", criteria=["episode stopped"])

        def produce(t: TaskDefinition, i: int) -> Episode:
            termination = "stop" if i % 2 == 0 else "failure"
            return _episode(f"e{i}", t, steps=i + 1, termination_reason=termination)

        result = run_batch({"full": produce}, [task], trials=4, clock=_counter_clock())
        assert len(result.summaries) == 1
        summary = result.summaries[0]
        assert summary.method == "full"
        assert summary.task_id == "t1"
        assert summary.trials == 4
        assert summary.success_count == 2
        assert summary.failure_count == 2
        assert summary.success_rate == 0.5
        assert summary.average_steps == 2.5
        assert summary.min_steps == 1
        assert summary.max_steps == 4
        assert summary.risky_actions == 0
        # steps per trial: 1,2,3,4 -> reflections 0+1+2+3
        assert summary.reflections == 6
        assert summary.average_runtime_seconds == 0.5
        assert summary.min_runtime_seconds == 0.5
        assert summary.max_runtime_seconds == 0.5
        assert summary.termination_reason_counts == {"stop": 2, "failure": 2}

    def test_rejects_zero_trials(self) -> None:
        with pytest.raises(ValueError, match="trials"):
            run_batch({"full": _simple_producer}, [_task()], trials=0)

    def test_rejects_empty_methods(self) -> None:
        with pytest.raises(ValueError, match="methods"):
            run_batch({}, [_task()], trials=1)


class TestWriteBatchResults:
    def _result(self) -> BatchResult:
        task = _task("t1", criteria=["episode stopped"])
        return run_batch(
            {"full": _simple_producer, "monolithic": _simple_producer},
            [task],
            trials=2,
            clock=_counter_clock(),
        )

    def test_writes_per_trial_json_and_summaries(self, tmp_path: Path) -> None:
        result = self._result()
        write_batch_results(result, tmp_path)
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "summary.csv").exists()
        trial_files = sorted((tmp_path / "trials").glob("*.json"))
        assert len(trial_files) == 4
        data = json.loads(trial_files[0].read_text(encoding="utf-8"))
        assert data["method"] in {"full", "monolithic"}
        assert data["task_id"] == "t1"
        assert "evaluation" in data

    def test_summary_json_contains_one_entry_per_method_task(
        self, tmp_path: Path
    ) -> None:
        result = self._result()
        write_batch_results(result, tmp_path)
        payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        summaries = payload["summaries"]
        assert len(summaries) == 2
        assert {s["method"] for s in summaries} == {"full", "monolithic"}
        for key in (
            "success_rate",
            "failure_count",
            "risky_actions",
            "reflections",
            "average_runtime_seconds",
        ):
            assert key in summaries[0]

    def test_summary_csv_columns_and_rows(self, tmp_path: Path) -> None:
        result = self._result()
        write_batch_results(result, tmp_path)
        with (tmp_path / "summary.csv").open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = set(reader.fieldnames or [])
        expected_columns = {
            "method",
            "task_id",
            "trials",
            "success_count",
            "failure_count",
            "success_rate",
            "average_steps",
            "min_steps",
            "max_steps",
            "risky_actions",
            "reflections",
            "average_runtime_seconds",
            "min_runtime_seconds",
            "max_runtime_seconds",
        }
        assert expected_columns <= fieldnames
        assert len(rows) == 2
        row = next(r for r in rows if r["method"] == "full")
        assert row["task_id"] == "t1"
        assert int(row["trials"]) == 2
        assert float(row["success_rate"]) == 1.0
        assert int(row["failure_count"]) == 0

    def test_make_run_dir_uses_timestamp_format(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(
            tmp_path,
            now=datetime(2026, 7, 20, 12, 34, 56, tzinfo=timezone.utc),
        )
        assert run_dir == tmp_path / "20260720_123456"
        assert run_dir.is_dir()
