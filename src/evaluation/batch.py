"""Batch evaluation runner over (method, task, trials) with persistence.

A "method" is a named agent-stack factory (e.g. ``full``, ``monolithic``,
``no_hidden_hypothesis``, ``no_reflection``). For unit-testability the runner
accepts injected episode-producer callables; the real wiring to
``ClosedLoopRunner`` lives behind a thin adapter in
``scripts/evaluate_real.py`` so no live robot/sim is required here.

Per-trial wall-clock runtime is measured by the driver through an injectable
``clock`` (monotonic seconds). If the episode already carries a runtime
(``metadata["runtime_seconds"]`` or derivable executor timestamps), that
value takes precedence over the driver measurement.

Results are persisted to a run directory (conventionally
``results/real/YYYYMMDD_HHMMSS/``):

- ``trials/{method}__{task_id}__trial-{NNN}.json`` — one file per trial.
- ``summary.json`` — aggregate :class:`MethodTaskSummary` entries.
- ``summary.csv`` — one row per (method, task) with every metric.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from common.schema import Episode
from evaluation.harness import (
    EpisodeEvaluation,
    SuccessChecker,
    evaluate_episode,
)
from tasks.schema import TaskDefinition


EpisodeProducer = Callable[[TaskDefinition, int], Episode]
"""Produce one episode for (task, trial_index). May run a live closed loop."""

SUMMARY_CSV_COLUMNS = [
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
]


class TrialResult(BaseModel):
    """Evaluation of a single (method, task, trial) episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    trial_index: int = Field(..., ge=0)
    evaluation: EpisodeEvaluation


class MethodTaskSummary(BaseModel):
    """Aggregate metrics for one (method, task) pair across trials."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    trials: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    failure_count: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    average_steps: float = Field(...)
    min_steps: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=0)
    termination_reason_counts: dict[str, int] = Field(default_factory=dict)
    risky_actions: int = Field(default=0, ge=0)
    reflections: int = Field(default=0, ge=0)
    average_runtime_seconds: float | None = Field(default=None, ge=0.0)
    min_runtime_seconds: float | None = Field(default=None, ge=0.0)
    max_runtime_seconds: float | None = Field(default=None, ge=0.0)


class BatchResult(BaseModel):
    """Full batch output: per-trial evaluations plus (method, task) summaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trials: list[TrialResult] = Field(default_factory=list)
    summaries: list[MethodTaskSummary] = Field(default_factory=list)


def _summarize(trials: Sequence[TrialResult]) -> list[MethodTaskSummary]:
    """Aggregate trial results into one summary per (method, task) pair."""
    groups: dict[tuple[str, str], list[TrialResult]] = {}
    for trial in trials:
        groups.setdefault((trial.method, trial.task_id), []).append(trial)

    summaries: list[MethodTaskSummary] = []
    for (method, task_id), group in groups.items():
        evaluations = [trial.evaluation for trial in group]
        total = len(evaluations)
        success_count = sum(1 for e in evaluations if e.success)
        step_counts = [e.steps for e in evaluations]
        runtimes = [
            e.runtime_seconds for e in evaluations if e.runtime_seconds is not None
        ]
        termination_counts = Counter(e.termination_reason for e in evaluations)
        summaries.append(
            MethodTaskSummary(
                method=method,
                task_id=task_id,
                trials=total,
                success_count=success_count,
                failure_count=total - success_count,
                success_rate=success_count / total if total else 0.0,
                average_steps=sum(step_counts) / total if total else 0.0,
                min_steps=min(step_counts) if step_counts else 0,
                max_steps=max(step_counts) if step_counts else 0,
                termination_reason_counts=dict(termination_counts),
                risky_actions=sum(e.risky_actions for e in evaluations),
                reflections=sum(e.reflections for e in evaluations),
                average_runtime_seconds=(
                    sum(runtimes) / len(runtimes) if runtimes else None
                ),
                min_runtime_seconds=min(runtimes) if runtimes else None,
                max_runtime_seconds=max(runtimes) if runtimes else None,
            )
        )
    return summaries


def run_batch(
    methods: Mapping[str, EpisodeProducer],
    tasks: Sequence[TaskDefinition],
    trials: int = 1,
    checker: SuccessChecker | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BatchResult:
    """Run ``trials`` episodes per (method, task) pair and evaluate each.

    Args:
        methods: Mapping of method name to an episode-producer callable.
        tasks: Task definitions to evaluate.
        trials: Number of trials per (method, task) pair; must be >= 1.
        checker: Optional success-criterion checker override.
        clock: Monotonic clock (seconds) used to measure per-trial runtime.

    Returns:
        A :class:`BatchResult` with per-trial evaluations and summaries.
    """
    if not methods:
        raise ValueError("methods must contain at least one method")
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")

    trial_results: list[TrialResult] = []
    for method_name, producer in methods.items():
        for task in tasks:
            for trial_index in range(trials):
                start = clock()
                episode = producer(task, trial_index)
                elapsed = clock() - start
                evaluation = evaluate_episode(episode, task, checker)
                if evaluation.runtime_seconds is None:
                    evaluation = evaluation.model_copy(
                        update={"runtime_seconds": float(elapsed)}
                    )
                trial_results.append(
                    TrialResult(
                        method=method_name,
                        task_id=task.id,
                        trial_index=trial_index,
                        evaluation=evaluation,
                    )
                )
                logger.info(
                    "Batch trial: method={} task={} trial={} success={} steps={}",
                    method_name,
                    task.id,
                    trial_index,
                    evaluation.success,
                    evaluation.steps,
                )

    return BatchResult(trials=trial_results, summaries=_summarize(trial_results))


def make_run_dir(base_dir: str | Path, now: datetime | None = None) -> Path:
    """Create and return ``base_dir/YYYYMMDD_HHMMSS`` (UTC)."""
    now = now or datetime.now(timezone.utc)
    run_dir = Path(base_dir) / now.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _trial_filename(trial: TrialResult) -> str:
    return f"{trial.method}__{trial.task_id}__trial-{trial.trial_index:03d}.json"


def write_batch_results(result: BatchResult, out_dir: str | Path) -> dict[str, Path]:
    """Persist per-trial JSON, aggregate summary.json, and summary.csv.

    Returns a mapping of artifact names to the paths written.
    """
    out_path = Path(out_dir)
    trials_dir = out_path / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    for trial in result.trials:
        trial_path = trials_dir / _trial_filename(trial)
        trial_path.write_text(trial.model_dump_json(indent=2), encoding="utf-8")

    summary_json_path = out_path / "summary.json"
    summary_json_path.write_text(
        BatchResult(trials=[], summaries=result.summaries).model_dump_json(indent=2),
        encoding="utf-8",
    )

    summary_csv_path = out_path / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for summary in result.summaries:
            row = summary.model_dump()
            writer.writerow(
                {
                    column: ("" if row[column] is None else row[column])
                    for column in SUMMARY_CSV_COLUMNS
                }
            )

    logger.info(
        "Batch results written to {} ({} trials, {} summaries)",
        out_path,
        len(result.trials),
        len(result.summaries),
    )
    return {
        "run_dir": out_path,
        "trials_dir": trials_dir,
        "summary_json": summary_json_path,
        "summary_csv": summary_csv_path,
    }
