"""Lightweight evaluation harness for task episodes."""

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

    return EvaluationResult(
        total=total,
        success_count=success_count,
        success_rate=success_rate,
        average_steps=average_steps,
        min_steps=min_steps,
        max_steps=max_steps,
        termination_reason_counts=dict(termination_reason_counts),
        per_episode=per_episode,
    )
