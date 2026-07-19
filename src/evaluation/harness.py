"""Lightweight evaluation harness for task episodes.

Metric definitions
------------------
- ``success``: all of the task's success criteria pass (see
  :class:`KeywordSuccessChecker`); episodes with unknown task ids or failed
  criteria count as failures.
- ``failures``: aggregate count of episodes where ``success`` is False, with
  the per-reason breakdown in ``termination_reason_counts``.
- ``risky_actions``: per-episode count of steps whose ``executor_output``
  carries a truthy ``risk`` flag. The shared
  :class:`common.schema.ExecutorOutput` has no such field; the flag exists on
  :class:`executor.schema.ExecutorAgentOutput`, so the check is done via
  ``getattr(executor_output, "risk", False)`` and plain ``ExecutorOutput``
  steps always count as non-risky.
- ``reflections``: per-episode count of feedback-driven re-analysis rounds.
  A step counts as one reflection event iff it is not the first step of the
  episode and the immediately preceding step carries a non-None ``feedback``
  object — i.e. the Rethinker was invoked with the previous round's feedback
  and produced a new analysis in response to it. (For runner-produced
  episodes, where every executed round records feedback, this equals
  ``steps - 1``.)
- ``runtime_seconds``: wall-clock duration of the episode. Resolution order:
  (1) ``episode.metadata["runtime_seconds"]`` when numeric (recorded by the
  runner / batch driver), (2) ``max - min`` of ``executor_output.timestamp``
  across steps when at least two timestamps exist, (3) ``None`` when not
  derivable. Aggregates (mean/min/max) are computed over episodes with a
  known runtime only.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from common.schema import Episode, ExecutorOutput, MissionType
from tasks.schema import TaskDefinition


TERMINATION_REASON_UNKNOWN = "unknown"
TERMINATION_REASON_STOP = "stop"
GRIPPER_OPEN_THRESHOLD = 0.5


class CriterionResult(BaseModel):
    """Outcome of evaluating a single success criterion against an episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion: str = Field(..., min_length=1)
    passed: bool
    reason: str = Field(default="")


class EpisodeEvaluation(BaseModel):
    """Evaluation result for one episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    episode_id: str = Field(..., min_length=1)
    success: bool
    steps: int = Field(..., ge=0)
    termination_reason: str = Field(default=TERMINATION_REASON_UNKNOWN)
    criteria: list[CriterionResult] = Field(default_factory=list)
    risky_actions: int = Field(default=0, ge=0)
    reflections: int = Field(default=0, ge=0)
    runtime_seconds: float | None = Field(default=None, ge=0.0)


class EvaluationResult(BaseModel):
    """Aggregate evaluation result across a set of episodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    average_steps: float = Field(...)
    min_steps: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=0)
    termination_reason_counts: dict[str, int] = Field(default_factory=dict)
    per_episode: list[EpisodeEvaluation] = Field(default_factory=list)
    failure_count: int = Field(default=0, ge=0)
    risky_action_count: int = Field(default=0, ge=0)
    reflection_count: int = Field(default=0, ge=0)
    average_runtime_seconds: float | None = Field(default=None, ge=0.0)
    min_runtime_seconds: float | None = Field(default=None, ge=0.0)
    max_runtime_seconds: float | None = Field(default=None, ge=0.0)


@runtime_checkable
class SuccessChecker(Protocol):
    """Protocol for pluggable success-criterion evaluators."""

    def check(self, criterion: str, episode: Episode, task: TaskDefinition) -> bool:
        """Return True if the episode satisfies the criterion for the task."""
        ...


class KeywordSuccessChecker:
    """Deterministic keyword-based success criterion evaluator.

    Interprets free-form English success criteria by looking for known
    keywords. Unrecognized criteria log a warning and evaluate to False.
    """

    def check(self, criterion: str, episode: Episode, task: TaskDefinition) -> bool:
        lowered = criterion.lower()

        if any(keyword in lowered for keyword in ("stop", "stopped")):
            return self._termination_reason_is(episode, TERMINATION_REASON_STOP)

        if any(keyword in lowered for keyword in ("success", "successful")):
            return self._all_feedbacks_successful(episode)

        if any(keyword in lowered for keyword in ("grasp", "grasped", "picked")):
            return self._terminated_after_mission(
                episode, task, {MissionType.PICK_ONLY, MissionType.PICK_AND_PLACE}
            )

        if any(keyword in lowered for keyword in ("place", "placed")):
            return self._terminated_after_mission(
                episode, task, {MissionType.PICK_AND_PLACE}
            )

        if any(keyword in lowered for keyword in ("gripper", "open")):
            return self._final_gripper_open(episode)

        logger.warning("Unrecognized success criterion: {!r}", criterion)
        return False

    @staticmethod
    def _termination_reason_is(episode: Episode, reason: str) -> bool:
        metadata = episode.metadata or {}
        return metadata.get("termination_reason", TERMINATION_REASON_UNKNOWN) == reason

    @staticmethod
    def _all_feedbacks_successful(episode: Episode) -> bool:
        for step in episode.steps:
            feedback = step.feedback
            if feedback is not None and not feedback.success:
                return False
        return True

    @staticmethod
    def _terminated_after_mission(
        episode: Episode,
        task: TaskDefinition,
        allowed_missions: set[MissionType],
    ) -> bool:
        if task.mission_type not in allowed_missions:
            return False
        return KeywordSuccessChecker._termination_reason_is(
            episode, TERMINATION_REASON_STOP
        )

    @staticmethod
    def _final_gripper_open(episode: Episode) -> bool:
        for step in reversed(episode.steps):
            executor = step.executor_output
            if isinstance(executor, ExecutorOutput):
                return executor.gripper_state > GRIPPER_OPEN_THRESHOLD
        return False


def _count_risky_actions(episode: Episode) -> int:
    """Count steps whose executor output carries a truthy risk flag."""
    count = 0
    for step in episode.steps:
        executor = step.executor_output
        if executor is not None and bool(getattr(executor, "risk", False)):
            count += 1
    return count


def _count_reflections(episode: Episode) -> int:
    """Count feedback-driven re-analysis rounds (see module docstring)."""
    count = 0
    for previous, _current in zip(episode.steps, episode.steps[1:]):
        if previous.feedback is not None:
            count += 1
    return count


def _episode_runtime_seconds(episode: Episode) -> float | None:
    """Derive episode wall-clock runtime, or None when not derivable."""
    metadata = episode.metadata or {}
    runtime = metadata.get("runtime_seconds")
    if isinstance(runtime, (int, float)) and not isinstance(runtime, bool):
        return float(runtime)
    timestamps = [
        step.executor_output.timestamp
        for step in episode.steps
        if step.executor_output is not None
        and step.executor_output.timestamp is not None
    ]
    if len(timestamps) >= 2:
        return float(max(timestamps) - min(timestamps))
    return None


def evaluate_episode(
    episode: Episode,
    task: TaskDefinition,
    checker: SuccessChecker | None = None,
) -> EpisodeEvaluation:
    """Evaluate a single episode against its task's success criteria."""
    checker = checker or KeywordSuccessChecker()
    criteria: list[CriterionResult] = []
    success = True

    for criterion in task.success_criteria or []:
        passed = checker.check(criterion, episode, task)
        reason = "passed" if passed else "failed"
        criteria.append(
            CriterionResult(criterion=criterion, passed=passed, reason=reason)
        )
        if not passed:
            success = False

    return EpisodeEvaluation(
        task_id=episode.task_id,
        episode_id=episode.id,
        success=success,
        steps=len(episode.steps),
        termination_reason=(episode.metadata or {}).get(
            "termination_reason", TERMINATION_REASON_UNKNOWN
        ),
        criteria=criteria,
        risky_actions=_count_risky_actions(episode),
        reflections=_count_reflections(episode),
        runtime_seconds=_episode_runtime_seconds(episode),
    )


def evaluate_tasks(
    tasks: list[TaskDefinition],
    episodes: list[Episode],
    checker: SuccessChecker | None = None,
) -> EvaluationResult:
    """Evaluate all episodes against the supplied task definitions.

    Episodes are matched to tasks by ``task_id``. Episodes whose task_id is
    missing from ``tasks`` are still included in the aggregate result (marked
    as failures) and emit a warning.
    """
    task_map = {task.id: task for task in tasks}
    per_episode: list[EpisodeEvaluation] = []
    termination_reason_counts: Counter[str] = Counter()

    for episode in episodes:
        task = task_map.get(episode.task_id)
        if task is None:
            logger.warning(
                "Episode {!r} references unknown task_id {!r}",
                episode.id,
                episode.task_id,
            )
            evaluation = EpisodeEvaluation(
                task_id=episode.task_id,
                episode_id=episode.id,
                success=False,
                steps=len(episode.steps),
                termination_reason=(episode.metadata or {}).get(
                    "termination_reason", TERMINATION_REASON_UNKNOWN
                ),
                criteria=[],
                risky_actions=_count_risky_actions(episode),
                reflections=_count_reflections(episode),
                runtime_seconds=_episode_runtime_seconds(episode),
            )
        else:
            evaluation = evaluate_episode(episode, task, checker)

        per_episode.append(evaluation)
        termination_reason_counts[evaluation.termination_reason] += 1

    total = len(per_episode)
    success_count = sum(1 for evaluation in per_episode if evaluation.success)
    success_rate = success_count / total if total else 0.0
    step_counts = [evaluation.steps for evaluation in per_episode]
    average_steps = sum(step_counts) / total if total else 0.0
    min_steps = min(step_counts) if step_counts else 0
    max_steps = max(step_counts) if step_counts else 0
    runtimes = [
        evaluation.runtime_seconds
        for evaluation in per_episode
        if evaluation.runtime_seconds is not None
    ]

    return EvaluationResult(
        total=total,
        success_count=success_count,
        success_rate=success_rate,
        average_steps=average_steps,
        min_steps=min_steps,
        max_steps=max_steps,
        termination_reason_counts=dict(termination_reason_counts),
        per_episode=per_episode,
        failure_count=total - success_count,
        risky_action_count=sum(e.risky_actions for e in per_episode),
        reflection_count=sum(e.reflections for e in per_episode),
        average_runtime_seconds=(sum(runtimes) / len(runtimes) if runtimes else None),
        min_runtime_seconds=min(runtimes) if runtimes else None,
        max_runtime_seconds=max(runtimes) if runtimes else None,
    )
