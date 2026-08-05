"""Validation gate for candidate forge prompts (Task 3.7).

The validator rolls a candidate prompt out on a held-out task set, computes
task-success metrics via the evaluation harness (plus optional stage/video
metrics via the forge critic), and accepts the candidate only on STRICT
improvement over the current best prompt recorded in the registry. Every
outcome is recorded via :meth:`ForgePromptRegistry.record_validation`.

Accept composite (exact definition)
-----------------------------------
The composite is the lexicographic key ``(success_rate, -average_steps)``
over the held-out task set:

- primary: task success rate (higher is better);
- tie-break: average steps per SUCCESSFUL episode (fewer is better;
  episodes that failed — including zero-step error rollouts — are excluded
  so a candidate cannot game the metric by failing fast, and a candidate
  with no successes gets the worst-case ``max_rounds``).

A candidate is accepted via one of two channels:

- **channel A (``"success"``)**: its key is STRICTLY greater than the
  incumbent best's key, i.e. ``success_rate`` is strictly higher, or equal
  with strictly fewer average steps. An exact tie is rejected.
- **channel B (``"semantic"``)**: ``success_rate`` within 0.05 of the
  baseline AND at least one semantic rate (presence-violation rate or
  decoy-pick rate) strictly lower with the other not higher. Binary
  success is dominated by physical execution noise; the semantic channel
  lets genuine strategy improvements through on a tied noise floor.

The scalar ``ValidationResult.composite`` (and the ``"composite"`` metric
written to the registry) is the primary component, ``success_rate``.
``ValidationResult.accepted_via`` (and the reason text recorded in the
registry detail) records which channel accepted.

Baseline
--------
The incumbent best's composite is read from its recorded validation metrics
(``"success_rate"`` / ``"average_steps"`` keys). If the best has no recorded
metrics for those keys (e.g. accepted before this metric set existed), it is
re-evaluated on the same held-out tasks with the same ``rollout_fn``; the
re-evaluated composite is used in-memory only (the registry refuses
re-validation of non-candidates). When no incumbent best exists at all, the
candidate is accepted as the first champion.

Critic usage is optional per call: with ``use_critic=True`` the caller
injects ``critic_fn(episode, task) -> CriticResult`` (the 3.8 runner wires
this to :class:`forge.critic.VideoStageCritic` plus its own episode
recordings). Without it, stage/video metrics are simply absent (``None``)
and no cloud calls happen.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from common.schema import Episode, EpisodeStep, Feedback
from evaluation.harness import evaluate_episode
from forge.actions import planner_output_to_sim_action
from forge.critic import CriticResult, StageScores
from forge.memory import ForgePlannerMemory
from forge.planner_agent import ForgePlannerAgent, obs_to_rethinker_output
from forge.strategy_metrics import (
    aggregate_strategy_metrics,
    episode_strategy_metrics,
)
from forge.registry import ForgePromptRegistry, PromptVersion
from tasks.schema import TaskDefinition

# Registry metric keys (contract with 3.8 runner / optimizer reporting).
SUCCESS_RATE_METRIC = "success_rate"
AVERAGE_STEPS_METRIC = "average_steps"
COMPOSITE_METRIC = "composite"
NUM_TASKS_METRIC = "num_tasks"
MEAN_VIDEO_SCORE_METRIC = "mean_video_score"

TERMINATION_STOP = "stop"

# Adaptive gate: candidates are first screened with 2 rollouts per task;
# only competitive ones earn the full validation budget. A candidate this
# far below the incumbent at screening can only win on noise anyway.
_SCREENING_MARGIN = 0.125
_SCREENING_ROUNDS = 2

# Semantic channel (channel B) success-rate tolerance: a candidate this
# close on binary success may still be accepted on semantic rates.
_SEMANTIC_RATE_TOLERANCE = 0.05
TERMINATION_MAX_ROUNDS = "max_rounds"

RolloutFn = Callable[[str, TaskDefinition], Episode]
CriticFn = Callable[[Episode, TaskDefinition], CriticResult]


class TaskValidationMetrics(BaseModel):
    """Per-task metrics for one validation rollout.

    ``stage_scores`` (means across the critic's per-episode evaluations) and
    ``video_score`` (mean of the three dims of the critic's ``"episode"``
    stage evaluation) are ``None`` when no critic ran or the critic filtered
    the episode without evaluations. ``move_aside_first`` is the per-episode
    semantic strategy flag (None when not applicable); see
    :mod:`forge.strategy_metrics`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    episode_id: str = Field(..., min_length=1)
    success: bool
    steps: int = Field(..., ge=0)
    stage_scores: StageScores | None = None
    video_score: float | None = Field(default=None, ge=0.0, le=1.0)
    move_aside_first: bool | None = None
    presence_violations: int = Field(default=0, ge=0)
    action_count: int = Field(default=0, ge=0)
    decoy_picks: int = Field(default=0, ge=0)


class ValidationResult(BaseModel):
    """Aggregate validation outcome for one candidate prompt.

    ``composite`` is the primary accept metric (``success_rate``);
    ``baseline_composite`` / ``baseline_average_steps`` describe the
    incumbent the candidate was compared against (``None`` when no incumbent
    existed).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version_id: str = Field(..., min_length=1)
    target_agent: str = Field(..., min_length=1)
    per_task: list[TaskValidationMetrics] = Field(default_factory=list)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    average_steps: float = Field(..., ge=0.0)
    mean_video_score: float | None = Field(default=None, ge=0.0, le=1.0)
    composite: float = Field(..., ge=0.0, le=1.0)
    baseline_composite: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_average_steps: float | None = Field(default=None, ge=0.0)
    move_aside_first_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_violation_rate: float = Field(default=0.0, ge=0.0)
    decoy_pick_rate: float = Field(default=0.0, ge=0.0)
    mean_action_count: float = Field(default=0.0, ge=0.0)
    accepted: bool
    reason: str
    accepted_via: str | None = None


class RolloutSuccessChecker:
    """Harness ``SuccessChecker`` reading the rollout-recorded success flag.

    Forge task success criteria are free-form English the keyword checker
    cannot interpret; ``rollout_episode`` instead records the env's
    ground-truth ``check_success`` verdict under ``metadata["success"]`` and
    this checker maps every criterion onto that flag.
    """

    def check(self, criterion: str, episode: Episode, task: TaskDefinition) -> bool:
        return bool((episode.metadata or {}).get("success", False))


def rollout_episode(
    env: Any,
    planner: ForgePlannerAgent,
    task: TaskDefinition | dict[str, Any],
    max_rounds: int,
) -> Episode:
    """Roll one episode of ``planner`` acting in ``env`` on ``task``.

    Loop: ``env.reset`` -> (``planner.act_from_obs`` ->
    ``planner_output_to_sim_action`` -> ``env.step``)* until the env reports
    done or ``max_rounds`` steps are taken. Returns a
    :class:`common.schema.Episode` so the evaluation harness works on it
    directly. ``metadata`` carries the ground-truth outcome:
    ``success`` (final env ``check_success`` verdict) and
    ``termination_reason`` (``"stop"`` when the env ended the episode via
    success or a STOP action, ``"max_rounds"`` when the step budget was
    exhausted or the env truncated).

    This helper is reused by the forge runner (Task 3.8).

    Raises:
        ValueError: if ``max_rounds`` is not positive.
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be a positive integer, got {max_rounds}")

    obs = env.reset(task)
    memory = ForgePlannerMemory()
    feedback: Feedback | None = None
    steps: list[EpisodeStep] = []
    done = False
    truncated = False
    success = False
    presence_violations = 0
    violation_stop = False

    while not done and len(steps) < max_rounds:
        step_index = len(steps)
        try:
            rethinker_output = obs_to_rethinker_output(obs)
            output = planner.act_from_obs(
                obs, memory=memory, previous_feedback=feedback
            )
            action = planner_output_to_sim_action(output)
        except ValueError as exc:
            if "not in the DINO label set" not in str(exc):
                raise
            # Planner emitted a label outside the current detections
            # (e.g. picked an occluded/hidden object). Record it as a
            # presence violation and end the episode gracefully so the
            # steps so far stay available for strategy metrics; previously
            # this surfaced as a zero-step crashed episode.
            presence_violations += 1
            violation_stop = True
            logger.warning(
                "rollout_episode: presence violation on task {!r}: {}",
                task.id if hasattr(task, "id") else env.task.id,
                exc,
            )
            break
        obs, reward, done, info = env.step(action)

        env_success = info.get("success")
        truncated = bool(info.get("truncated"))
        success = env_success is True
        feedback = Feedback(
            success=(
                success
                if env_success is not None
                else bool(info.get("primitive_success"))
            ),
            observation=str(info.get("primitive_status", "")),
            reward=float(reward),
        )
        memory.append(
            round=step_index,
            scene_token=str(info.get("task_id", "")),
            query=rethinker_output.reasoning,
            answer=output,
            feedback=feedback,
        )
        steps.append(
            EpisodeStep(
                step_index=step_index,
                task=env.task,
                rethinker_output=rethinker_output,
                planner_output=output,
                feedback=feedback,
            )
        )

    episode_id = f"rollout-{env.task.id}-{uuid.uuid4().hex[:8]}"
    termination = TERMINATION_STOP if (done and not truncated) else TERMINATION_MAX_ROUNDS
    if violation_stop:
        termination = "presence_violation"
    logger.info(
        "rollout_episode: task={!r} steps={} success={} termination={}",
        env.task.id,
        len(steps),
        success,
        termination,
    )
    return Episode(
        id=episode_id,
        task_id=env.task.id,
        steps=steps,
        metadata={
            "success": success,
            "termination_reason": termination,
            "presence_violations": presence_violations,
        },
    )


class PromptValidator:
    """Validation gate: rolls out candidates and accepts only strict gains.

    Args:
        registry: the forge prompt registry used for incumbent lookup and
            outcome recording.
        env: optional :class:`forge.env.SimEnv` for the default rollout.
        planner: optional :class:`ForgePlannerAgent` for the default rollout.
            The caller is responsible for loading the candidate prompt into
            the planner (the forge runner owns prompt swapping).
        max_rounds: step cap for the default rollout.
    """

    def __init__(
        self,
        registry: ForgePromptRegistry,
        *,
        env: Any | None = None,
        planner: ForgePlannerAgent | None = None,
        max_rounds: int = 10,
    ) -> None:
        self._registry = registry
        self._env = env
        self._planner = planner
        self._max_rounds = max_rounds

    def validate(
        self,
        candidate_version: PromptVersion | str,
        val_tasks: list[TaskDefinition],
        rollout_fn: RolloutFn | None = None,
        *,
        use_critic: bool = False,
        critic_fn: CriticFn | None = None,
        timestamp: str | None = None,
    ) -> ValidationResult:
        """Validate ``candidate_version`` on the held-out ``val_tasks``.

        Args:
            candidate_version: the candidate :class:`PromptVersion` (or its
                version id, resolved via the registry). Must still be a
                ``candidate`` in the registry.
            val_tasks: held-out task set; must be non-empty.
            rollout_fn: ``(version_id, task) -> Episode`` rollout callable.
                Injected in tests; defaults to :func:`rollout_episode`
                against the constructor-provided env/planner.
            use_critic: when True, ``critic_fn`` is called per episode for
                stage/video metrics.
            critic_fn: ``(episode, task) -> CriticResult``; required when
                ``use_critic`` is True.
            timestamp: ISO-8601 timestamp forwarded to the registry record.

        Returns:
            The :class:`ValidationResult`; the outcome is also recorded to
            the registry via ``record_validation``.

        Raises:
            ValueError: if ``val_tasks`` is empty, ``use_critic`` lacks a
                ``critic_fn``, no rollout is available, or the version is
                not a candidate.
            KeyError: if a string version id is unknown to the registry.
        """
        if not val_tasks:
            raise ValueError("val_tasks must be a non-empty held-out task set")
        if use_critic and critic_fn is None:
            raise ValueError("use_critic=True requires a critic_fn callable")
        rollout = rollout_fn or self._default_rollout

        version = self._resolve_version(candidate_version)
        if version.status != "candidate":
            raise ValueError(
                f"version {version.version_id!r} is not a candidate "
                f"(status={version.status!r}); refusing re-validation"
            )
        logger.info(
            "Validating candidate {} for {} on {} task(s)",
            version.version_id,
            version.target_agent,
            len(val_tasks),
        )

        active_critic = critic_fn if use_critic else None
        baseline = self._baseline(version, val_tasks, rollout)

        # Group repeat-rollout entries by task id; entry i of a group is
        # rollout round i for that task. Screening = the first
        # _SCREENING_ROUNDS rounds (2 episodes per task with the x4 budget).
        groups: dict[str, list[TaskDefinition]] = {}
        for task in val_tasks:
            groups.setdefault(task.id, []).append(task)
        n_rounds = max(len(entries) for entries in groups.values())
        screen_rounds = min(_SCREENING_ROUNDS, n_rounds)

        def round_tasks(r: int) -> list[TaskDefinition]:
            return [entries[r] for entries in groups.values() if r < len(entries)]

        per_task = [
            self._evaluate_task(version.version_id, task, rollout, active_critic)
            for r in range(screen_rounds)
            for task in round_tasks(r)
        ]
        screen_rate, _ = self._aggregate(per_task)

        early_reject = (
            baseline is not None
            and n_rounds > screen_rounds
            and screen_rate < baseline[0] - _SCREENING_MARGIN
        )
        if early_reject:
            logger.info(
                "Candidate {} rejected at screening: success_rate {:.3f} < "
                "baseline {:.3f} - {:.3f}; skipping {} remaining rollout(s)",
                version.version_id,
                screen_rate,
                baseline[0],
                _SCREENING_MARGIN,
                sum(len(round_tasks(r)) for r in range(screen_rounds, n_rounds)),
            )
        else:
            per_task.extend(
                self._evaluate_task(version.version_id, task, rollout, active_critic)
                for r in range(screen_rounds, n_rounds)
                for task in round_tasks(r)
            )
        success_rate, average_steps = self._aggregate(per_task)
        video_scores = [m.video_score for m in per_task if m.video_score is not None]
        mean_video_score = (
            sum(video_scores) / len(video_scores) if video_scores else None
        )

        strategy = aggregate_strategy_metrics(
            [
                {
                    "action_count": m.action_count,
                    "presence_violations": m.presence_violations,
                    "move_aside_first": m.move_aside_first,
                    "decoy_picks": m.decoy_picks,
                }
                for m in per_task
            ]
        )

        if early_reject:
            accepted = False
            reason = (
                f"screening rejection: success_rate {screen_rate:.3f} below "
                f"baseline {baseline[0]:.3f} - {_SCREENING_MARGIN:.3f}; "
                "full validation skipped"
            )
            accepted_via: str | None = None
        else:
            accepted, reason, accepted_via = self._decide(
                version, success_rate, average_steps, baseline, strategy
            )
        metrics = {
            SUCCESS_RATE_METRIC: success_rate,
            AVERAGE_STEPS_METRIC: average_steps,
            COMPOSITE_METRIC: success_rate,
            NUM_TASKS_METRIC: float(len(per_task)),
            "presence_violation_rate": strategy["presence_violation_rate"],
            "mean_action_count": strategy["mean_action_count"],
            "decoy_pick_rate": strategy["decoy_pick_rate"],
        }
        if strategy["move_aside_first_rate"] is not None:
            metrics["move_aside_first_rate"] = strategy["move_aside_first_rate"]
        if mean_video_score is not None:
            metrics[MEAN_VIDEO_SCORE_METRIC] = mean_video_score
        self._registry.record_validation(
            version.version_id,
            metrics,
            accepted,
            timestamp=timestamp,
            detail=reason,
        )

        return ValidationResult(
            version_id=version.version_id,
            target_agent=version.target_agent,
            per_task=per_task,
            success_rate=success_rate,
            average_steps=average_steps,
            mean_video_score=mean_video_score,
            composite=success_rate,
            baseline_composite=baseline[0] if baseline else None,
            baseline_average_steps=baseline[1] if baseline else None,
            move_aside_first_rate=strategy["move_aside_first_rate"],
            presence_violation_rate=strategy["presence_violation_rate"],
            decoy_pick_rate=strategy["decoy_pick_rate"],
            mean_action_count=strategy["mean_action_count"],
            accepted=accepted,
            reason=reason,
            accepted_via=accepted_via,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resolve_version(self, candidate_version: PromptVersion | str) -> PromptVersion:
        if isinstance(candidate_version, PromptVersion):
            return candidate_version
        return self._registry.get(candidate_version)

    def _default_rollout(self, version_id: str, task: TaskDefinition) -> Episode:
        if self._env is None or self._planner is None:
            raise ValueError(
                "no rollout_fn given and PromptValidator was built without "
                "env/planner; provide one of them"
            )
        logger.debug("Default rollout of {} on task {!r}", version_id, task.id)
        return rollout_episode(self._env, self._planner, task, self._max_rounds)

    def _evaluate_task(
        self,
        version_id: str,
        task: TaskDefinition,
        rollout: RolloutFn,
        critic_fn: CriticFn | None,
    ) -> TaskValidationMetrics:
        episode = rollout(version_id, task)
        evaluation = evaluate_episode(episode, task, checker=RolloutSuccessChecker())
        if task.success_criteria:
            success = evaluation.success
        else:
            success = bool((episode.metadata or {}).get("success", False))

        stage_scores: StageScores | None = None
        video_score: float | None = None
        if critic_fn is not None:
            stage_scores, video_score = _summarize_critic(critic_fn(episode, task))

        strategy = episode_strategy_metrics(episode, task)
        return TaskValidationMetrics(
            task_id=task.id,
            episode_id=episode.id,
            success=success,
            steps=evaluation.steps,
            stage_scores=stage_scores,
            video_score=video_score,
            move_aside_first=strategy["move_aside_first"],
            presence_violations=strategy["presence_violations"],
            action_count=strategy["action_count"],
            decoy_picks=strategy["decoy_picks"],
        )

    def _baseline(
        self,
        version: PromptVersion,
        val_tasks: list[TaskDefinition],
        rollout: RolloutFn,
    ) -> tuple[float, float, bool, dict[str, float] | None] | None:
        """Return ``(success_rate, average_steps, re_evaluated, strategy)``.

        ``strategy`` carries the incumbent's semantic rates
        (presence_violation_rate / decoy_pick_rate) from its recorded
        metrics, or re-aggregated when re-evaluated; None when the record
        predates semantic metrics (semantic channel unavailable).
        """
        try:
            best = self._registry.best(version.target_agent)
        except LookupError:
            return None
        metrics = best.validation.metrics if best.validation else {}
        if SUCCESS_RATE_METRIC in metrics and AVERAGE_STEPS_METRIC in metrics:
            strategy = None
            if "presence_violation_rate" in metrics and "decoy_pick_rate" in metrics:
                strategy = {
                    "presence_violation_rate": metrics["presence_violation_rate"],
                    "decoy_pick_rate": metrics["decoy_pick_rate"],
                }
            return (
                metrics[SUCCESS_RATE_METRIC],
                metrics[AVERAGE_STEPS_METRIC],
                False,
                strategy,
            )
        logger.info(
            "Best {} has no recorded {} metrics; re-evaluating on {} task(s)",
            best.version_id,
            SUCCESS_RATE_METRIC,
            len(val_tasks),
        )
        per_task = [
            self._evaluate_task(best.version_id, task, rollout, None)
            for task in val_tasks
        ]
        success_rate, average_steps = self._aggregate(per_task)
        strategy = aggregate_strategy_metrics(
            [
                {
                    "action_count": m.action_count,
                    "presence_violations": m.presence_violations,
                    "move_aside_first": m.move_aside_first,
                    "decoy_picks": m.decoy_picks,
                }
                for m in per_task
            ]
        )
        return (success_rate, average_steps, True, strategy)

    def _aggregate(
        self, per_task: list[TaskValidationMetrics]
    ) -> tuple[float, float]:
        """Return ``(success_rate, average_steps)`` for one rollout set.

        ``average_steps`` is computed over SUCCESSFUL episodes only: a
        failed episode's step count (including zero-step error rollouts)
        must not drag the tie-break metric down — otherwise a candidate can
        "improve" by failing faster. When no episode succeeded, the metric
        is the worst-case value ``max_rounds`` so a 0-success candidate
        cannot win the step tie-break either.
        """
        success_rate = sum(m.success for m in per_task) / len(per_task)
        successes = [m for m in per_task if m.success]
        if not successes:
            return success_rate, float(self._max_rounds)
        average_steps = sum(m.steps for m in successes) / len(successes)
        return success_rate, average_steps

    @staticmethod
    def _decide(
        version: PromptVersion,
        success_rate: float,
        average_steps: float,
        baseline: tuple[float, float, bool, dict[str, float] | None] | None,
        cand_strategy: dict[str, Any] | None = None,
    ) -> tuple[bool, str, str | None]:
        """Dual-channel accept decision; returns (accepted, reason, via).

        Channel A ("success"): strict success_rate improvement, or a tie
        with strictly fewer average steps.
        Channel B ("semantic"): success_rate within 0.05 of the baseline
        AND at least one semantic rate (presence violations / decoy picks)
        strictly lower with the other not higher. Binary success is
        dominated by physical execution noise; the semantic channel lets
        genuine strategy improvements through when the noise floor ties.
        """
        if baseline is None:
            return True, (
                f"no incumbent best for {version.target_agent!r}; accepting "
                f"{version.version_id} as first champion "
                f"(success_rate={success_rate:.3f})"
            ), "success"
        base_rate, base_steps, re_evaluated, base_strategy = baseline
        source = "re-evaluated" if re_evaluated else "recorded"
        if success_rate > base_rate:
            return True, (
                f"strict improvement over {source} baseline: success_rate "
                f"{success_rate:.3f} > {base_rate:.3f}"
            ), "success"
        if success_rate == base_rate and average_steps < base_steps:
            return True, (
                f"strict improvement over {source} baseline: success_rate "
                f"tied at {success_rate:.3f}, average_steps "
                f"{average_steps:.2f} < {base_steps:.2f}"
            ), "success"
        # Channel B: semantic improvement within the success-rate tolerance.
        if (
            base_strategy is not None
            and cand_strategy is not None
            and success_rate >= base_rate - _SEMANTIC_RATE_TOLERANCE
        ):
            better_pvr = (
                cand_strategy["presence_violation_rate"]
                < base_strategy["presence_violation_rate"]
            )
            not_worse_pvr = (
                cand_strategy["presence_violation_rate"]
                <= base_strategy["presence_violation_rate"]
            )
            better_dpr = (
                cand_strategy["decoy_pick_rate"] < base_strategy["decoy_pick_rate"]
            )
            not_worse_dpr = (
                cand_strategy["decoy_pick_rate"] <= base_strategy["decoy_pick_rate"]
            )
            if (better_pvr and not_worse_dpr) or (better_dpr and not_worse_pvr):
                return True, (
                    f"semantic improvement over {source} baseline: "
                    f"presence_violation_rate "
                    f"{cand_strategy['presence_violation_rate']:.3f} vs "
                    f"{base_strategy['presence_violation_rate']:.3f}, "
                    f"decoy_pick_rate {cand_strategy['decoy_pick_rate']:.3f} vs "
                    f"{base_strategy['decoy_pick_rate']:.3f} "
                    f"(success_rate {success_rate:.3f} vs {base_rate:.3f} "
                    f"within {_SEMANTIC_RATE_TOLERANCE:.2f} tolerance)"
                ), "semantic"
        return False, (
            f"no strict improvement over {source} baseline: success_rate "
            f"{success_rate:.3f} vs {base_rate:.3f}, average_steps "
            f"{average_steps:.2f} vs {base_steps:.2f}"
        ), None


def _summarize_critic(result: CriticResult) -> tuple[StageScores | None, float | None]:
    """Aggregate a :class:`CriticResult` into per-task stage/video scores.

    ``stage_scores`` averages the three dims across all evaluations;
    ``video_score`` is the mean of the three dims of the ``"episode"`` stage
    evaluation (falling back to the overall stage-score mean when no global
    evaluation exists). Both are ``None`` when the critic returned no
    evaluations (e.g. a filtered clean success).
    """
    evaluations = result.evaluations
    if not evaluations:
        return None, None
    n = len(evaluations)
    stage_scores = StageScores(
        correctness=sum(e.scores.correctness for e in evaluations) / n,
        efficiency=sum(e.scores.efficiency for e in evaluations) / n,
        safety=sum(e.scores.safety for e in evaluations) / n,
    )
    episode_eval = next((e for e in evaluations if e.stage == "episode"), None)
    if episode_eval is not None:
        scores = episode_eval.scores
        video_score = (scores.correctness + scores.efficiency + scores.safety) / 3
    else:
        video_score = (
            stage_scores.correctness + stage_scores.efficiency + stage_scores.safety
        ) / 3
    return stage_scores, video_score
