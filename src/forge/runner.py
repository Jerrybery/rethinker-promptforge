"""Forge training loop (Task 3.8): rollout -> critic -> optimizer -> validate.

``ForgeRunner`` ties the Milestone-3 forge components together into the
training loop:

1. Roll out the current best prompt on the FIXED train task set
   (:func:`forge.validator.rollout_episode`), recording video/keyframes per
   episode post-hoc from the env frame buffer (``rollout_episode`` records
   no video; the runner owns the :class:`EpisodeRecorder` lifecycle).
2. The critic (optional) evaluates each recording -> ``StageEvaluation``s.
3. The optimizer proposes bounded edits from the best prompt text, the
   evaluations, and the rejected-edit history (with ``rejected_texts``
   resolved via ``registry.text`` for symmetric dedup); ``apply_edits``
   materializes the candidate text.
4. The candidate is registered in the forge registry and validated on the
   FIXED held-out validation task set; the validator's strict-improvement
   gate records accept/reject.

Both task sets are fixed at construction for the whole run — the registry
does not track which task set produced recorded metrics, so rotating the
held-out set mid-run would silently compare against stale baselines.

Artifacts under ``run_dir`` (conventionally ``results/forge/YYYYMMDD_HHMMSS/``)::

    forge_log.json                  ForgeLog, written at the end of run()
    epochs/epoch_{NNN}.json         EpochLog, flushed after EVERY epoch (resume-safe)
    metrics.jsonl                   one EpochLog JSON line per epoch (metric curves)
    prompts/best_epoch_{NNN}.md     best prompt snapshot after each epoch
    recordings/epoch_{NNN}/<id>/    per-episode video.mp4 + metadata.json
    registry/                       ForgePromptRegistry root (owned by the caller)

Fault tolerance: a failing train episode, critic call, or validation
rollout never kills the run — train/critic failures are logged and skipped;
validation rollout failures are converted into zero-step failed episodes so
the accept/reject math still completes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from common.schema import Episode
from forge.critic import StageEvaluation
from forge.optimizer import PromptEdit, apply_edits
from forge.recorder import EpisodeRecorder, EpisodeRecording
from forge.registry import EditMetadata, ForgePromptRegistry
from forge.validator import PromptValidator, rollout_episode
from tasks.schema import TaskDefinition

DEFAULT_TARGET_AGENT = "planner"
SEED_EDIT_TYPE = "seed"
SEED_SOURCE = "hand"
OPTIMIZER_SOURCE = "optimizer"

_EPOCH_DIR = "epochs"
_METRICS_FILENAME = "metrics.jsonl"
_FORGE_LOG_FILENAME = "forge_log.json"
_PROMPT_SNAPSHOT_DIR = "prompts"
_RECORDINGS_DIR = "recordings"

_MAX_EDIT_REASON_CHARS = 500


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpochLog(BaseModel):
    """Record of one forge epoch (rollout -> optimize -> validate).

    ``accepted`` is ``None`` when no candidate was validated (optimizer
    proposed no edits, or the edits left the prompt unchanged); the reason
    is then in ``validation_reason``.
    """

    model_config = ConfigDict(frozen=True)

    epoch_index: int = Field(..., ge=0)
    incumbent_version_id: str = Field(..., min_length=1)
    train_episodes: int = Field(default=0, ge=0)
    train_failed_episodes: int = Field(default=0, ge=0)
    train_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    train_average_steps: float | None = Field(default=None, ge=0.0)
    critic_evaluations: int = Field(default=0, ge=0)
    edits_proposed: int = Field(default=0, ge=0)
    edit_summary: str = ""
    candidate_version_id: str | None = None
    accepted: bool | None = None
    validation_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_average_steps: float | None = Field(default=None, ge=0.0)
    validation_reason: str = ""
    best_version_id: str = Field(..., min_length=1)
    recordings: list[str] = Field(default_factory=list)
    snapshot_path: str | None = None
    timestamp: str = Field(..., min_length=1)


class ForgeLog(BaseModel):
    """Record of a full forge run: config identity plus per-epoch logs."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., min_length=1)
    target_agent: str = Field(..., min_length=1)
    train_task_ids: list[str]
    val_task_ids: list[str]
    max_rounds: int = Field(..., ge=1)
    use_critic: bool
    seed_version_id: str = Field(..., min_length=1)
    final_best_version_id: str = Field(..., min_length=1)
    epochs: list[EpochLog] = Field(default_factory=list)
    run_dir: str = Field(..., min_length=1)


class ForgeRunner:
    """Forge training loop over fixed train/validation task sets.

    Args:
        registry: forge prompt registry holding the version lineage. Its
            root is caller-owned (the CLI places it inside the run dir).
        env: duck-typed :class:`forge.env.SimEnv` (``reset``/``step``/
            ``task``/``render``), reused across episodes.
        planner: duck-typed :class:`ForgePlannerAgent` with
            ``set_prompt_text``; the runner swaps prompt text per version.
        optimizer: duck-typed :class:`forge.optimizer.OptimizerLLM`.
        critic: optional duck-typed :class:`forge.critic.VideoStageCritic`;
            ``None`` skips the cloud critic entirely (validation gate works
            without it).
        validator: optional :class:`PromptValidator`; defaults to one built
            on the same registry (the runner always injects its rollout).
        train_tasks / val_tasks: FIXED task sets for the whole run.
        target_agent: registry agent namespace (default ``"planner"``).
        max_rounds: step cap per rollout.
        run_dir: artifact root; created with subdirectories on init.
        initial_prompt_text: seed prompt text, REQUIRED when the registry
            has no accepted best for ``target_agent`` yet. The seed is
            registered (``source="hand"``) and validated on ``val_tasks`` so
            it becomes the first champion with recorded baseline metrics.
        recorder_factory: :class:`EpisodeRecorder` factory (test injection).
        fps: recording frame rate.
    """

    def __init__(
        self,
        *,
        registry: ForgePromptRegistry,
        env: Any,
        planner: Any,
        optimizer: Any,
        critic: Any | None = None,
        validator: PromptValidator | None = None,
        train_tasks: Sequence[TaskDefinition],
        val_tasks: Sequence[TaskDefinition],
        target_agent: str = DEFAULT_TARGET_AGENT,
        max_rounds: int = 10,
        run_dir: str | Path,
        initial_prompt_text: str | None = None,
        recorder_factory: Callable[[], EpisodeRecorder] = EpisodeRecorder,
        fps: float = 10.0,
    ) -> None:
        if not train_tasks:
            raise ValueError("train_tasks must be a non-empty fixed task set")
        if not val_tasks:
            raise ValueError("val_tasks must be a non-empty fixed task set")
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
        if not target_agent.strip():
            raise ValueError("target_agent must be non-empty")

        self._registry = registry
        self._env = env
        self._planner = planner
        self._optimizer = optimizer
        self._critic = critic
        self._validator = validator or PromptValidator(
            registry, env=env, planner=planner, max_rounds=max_rounds
        )
        self._train_tasks = list(train_tasks)
        self._val_tasks = list(val_tasks)
        self._target_agent = target_agent
        self._max_rounds = int(max_rounds)
        self._initial_prompt_text = initial_prompt_text
        self._recorder_factory = recorder_factory
        self._fps = float(fps)

        self._run_dir = Path(run_dir)
        self._epochs_dir = self._run_dir / _EPOCH_DIR
        self._prompts_dir = self._run_dir / _PROMPT_SNAPSHOT_DIR
        for directory in (self._epochs_dir, self._prompts_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._run_dir / _METRICS_FILENAME

        logger.info(
            "ForgeRunner initialized: target_agent={}, train_tasks={}, "
            "val_tasks={}, max_rounds={}, critic={}, run_dir={}",
            target_agent,
            [t.id for t in self._train_tasks],
            [t.id for t in self._val_tasks],
            max_rounds,
            critic is not None,
            self._run_dir,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, n_epochs: int) -> ForgeLog:
        """Run ``n_epochs`` forge epochs and return the :class:`ForgeLog`.

        Seeds the registry when no best exists (register initial prompt +
        validate on the fixed val set). Epoch logs are flushed to disk after
        every epoch, so a crash mid-run leaves completed epochs intact.
        """
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        run_id = f"forge-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"
        seed_version_id = self._ensure_seed()

        epochs: list[EpochLog] = []
        for epoch_index in range(n_epochs):
            epoch_log = self.run_epoch(epoch_index)
            self._write_epoch_log(epoch_log)
            self._append_metrics(epoch_log)
            epochs.append(epoch_log)
            logger.info(
                "Epoch {} done: candidate={} accepted={} best={}",
                epoch_index,
                epoch_log.candidate_version_id,
                epoch_log.accepted,
                epoch_log.best_version_id,
            )

        final_best = self._registry.best(self._target_agent)
        forge_log = ForgeLog(
            run_id=run_id,
            target_agent=self._target_agent,
            train_task_ids=[t.id for t in self._train_tasks],
            val_task_ids=[t.id for t in self._val_tasks],
            max_rounds=self._max_rounds,
            use_critic=self._critic is not None,
            seed_version_id=seed_version_id,
            final_best_version_id=final_best.version_id,
            epochs=epochs,
            run_dir=str(self._run_dir),
        )
        log_path = self._run_dir / _FORGE_LOG_FILENAME
        log_path.write_text(forge_log.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "Forge run {} finished: {} epoch(s), final best={}, log={}",
            run_id,
            len(epochs),
            final_best.version_id,
            log_path,
        )
        return forge_log

    def run_epoch(self, epoch_idx: int) -> EpochLog:
        """Run one forge epoch and return its log (not yet flushed)."""
        best = self._registry.best(self._target_agent)
        best_text = self._registry.text(best.version_id)
        self._planner.set_prompt_text(best_text)
        logger.info(
            "Epoch {}: incumbent={} ({} chars)",
            epoch_idx,
            best.version_id,
            len(best_text),
        )

        episodes, recordings, failed = self._train_rollouts(epoch_idx)
        train_success_rate, train_average_steps = _train_metrics(episodes)

        evaluations = self._critic_evaluations(episodes, recordings)

        rejected = self._registry.history(self._target_agent, status="rejected")
        rejected_texts = [self._registry.text(v.version_id) for v in rejected]
        edits: list[PromptEdit] = self._optimizer.propose_edits(
            best_text,
            evaluations,
            rejected_history=rejected,
            rejected_texts=rejected_texts,
        )
        edit_summary = _summarize_edits(edits)

        candidate_id: str | None = None
        accepted: bool | None = None
        val_success_rate: float | None = None
        val_average_steps: float | None = None
        validation_reason: str

        if not edits:
            validation_reason = "optimizer proposed no edits; epoch skipped"
            logger.info("Epoch {}: {}", epoch_idx, validation_reason)
        else:
            candidate_text = apply_edits(best_text, edits)
            if candidate_text == best_text:
                validation_reason = (
                    "edits left the prompt unchanged; validation skipped"
                )
                logger.warning("Epoch {}: {}", epoch_idx, validation_reason)
            else:
                candidate = self._registry.register(
                    candidate_text,
                    self._target_agent,
                    EditMetadata(
                        edit_type=_edit_type_summary(edits),
                        reason=edit_summary[:_MAX_EDIT_REASON_CHARS],
                        source=OPTIMIZER_SOURCE,
                    ),
                    parent_version=best.version_id,
                )
                candidate_id = candidate.version_id
                result = self._validator.validate(
                    candidate,
                    list(self._val_tasks),
                    rollout_fn=self._rollout_for_version,
                )
                accepted = result.accepted
                val_success_rate = result.success_rate
                val_average_steps = result.average_steps
                validation_reason = result.reason

        new_best = self._registry.best(self._target_agent)
        # Restore the planner to the reigning best for the next epoch.
        self._planner.set_prompt_text(self._registry.text(new_best.version_id))
        snapshot = self._snapshot(epoch_idx)

        return EpochLog(
            epoch_index=epoch_idx,
            incumbent_version_id=best.version_id,
            train_episodes=len(episodes),
            train_failed_episodes=failed,
            train_success_rate=train_success_rate,
            train_average_steps=train_average_steps,
            critic_evaluations=len(evaluations),
            edits_proposed=len(edits),
            edit_summary=edit_summary[:_MAX_EDIT_REASON_CHARS],
            candidate_version_id=candidate_id,
            accepted=accepted,
            validation_success_rate=val_success_rate,
            validation_average_steps=val_average_steps,
            validation_reason=validation_reason,
            best_version_id=new_best.version_id,
            recordings=[
                r.video_path or r.metadata_path
                for r in recordings
                if r is not None
            ],
            snapshot_path=str(snapshot) if snapshot else None,
            timestamp=_utc_now(),
        )

    # ------------------------------------------------------------------ #
    # Loop stages
    # ------------------------------------------------------------------ #

    def _ensure_seed(self) -> str:
        """Return the current best version id, seeding the registry if empty."""
        try:
            return self._registry.best(self._target_agent).version_id
        except LookupError:
            pass
        if not self._initial_prompt_text or not self._initial_prompt_text.strip():
            raise ValueError(
                "registry has no best prompt for "
                f"{self._target_agent!r} and no initial_prompt_text was given"
            )
        seed = self._registry.register(
            self._initial_prompt_text,
            self._target_agent,
            EditMetadata(
                edit_type=SEED_EDIT_TYPE,
                reason="initial prompt seed",
                source=SEED_SOURCE,
            ),
        )
        logger.info(
            "Seeding registry with {} as first candidate; validating on {} task(s)",
            seed.version_id,
            len(self._val_tasks),
        )
        result = self._validator.validate(
            seed,
            list(self._val_tasks),
            rollout_fn=self._rollout_for_version,
        )
        logger.info(
            "Seed {} validation: accepted={} success_rate={:.3f} ({})",
            seed.version_id,
            result.accepted,
            result.success_rate,
            result.reason,
        )
        return seed.version_id

    def _train_rollouts(
        self, epoch_idx: int
    ) -> tuple[list[Episode], list[EpisodeRecording | None], int]:
        episodes: list[Episode] = []
        # Index-aligned with `episodes`: a failed recording stays a None
        # slot so the critic loop never pairs an episode with the wrong
        # recording.
        recordings: list[EpisodeRecording | None] = []
        failed = 0
        for task in self._train_tasks:
            try:
                episode = rollout_episode(
                    self._env, self._planner, task, self._max_rounds
                )
            except Exception as exc:
                failed += 1
                logger.exception(
                    "Train rollout failed (task={!r}): {}; continuing", task.id, exc
                )
                continue
            episodes.append(episode)
            try:
                recordings.append(self._record_episode(episode, epoch_idx))
            except Exception as exc:
                recordings.append(None)
                logger.exception(
                    "Recording failed (episode={}): {}; continuing",
                    episode.id,
                    exc,
                )
        return episodes, recordings, failed

    def _critic_evaluations(
        self,
        episodes: list[Episode],
        recordings: list[EpisodeRecording | None],
    ) -> list[StageEvaluation]:
        if self._critic is None:
            return []
        evaluations: list[StageEvaluation] = []
        for episode, recording in zip(episodes, recordings):
            if recording is None:
                logger.debug(
                    "Skipping critic for episode {}: recording failed",
                    episode.id,
                )
                continue
            try:
                result = self._critic.evaluate_episode(
                    recording,
                    final_success=bool(
                        (episode.metadata or {}).get("success", False)
                    ),
                    max_steps=self._max_rounds,
                    stage_logs=_stage_logs(episode),
                )
            except Exception as exc:
                logger.exception(
                    "Critic failed (episode={}): {}; continuing", episode.id, exc
                )
                continue
            evaluations.extend(result.evaluations)
        return evaluations

    def _rollout_for_version(self, version_id: str, task: TaskDefinition) -> Episode:
        """Validation rollout: swap the planner prompt, roll, guard failures.

        A failing rollout becomes a zero-step failed episode so the
        validator's accept/reject math still completes.
        """
        try:
            self._planner.set_prompt_text(self._registry.text(version_id))
            return rollout_episode(self._env, self._planner, task, self._max_rounds)
        except Exception as exc:
            logger.exception(
                "Validation rollout failed (version={}, task={!r}): {}; "
                "marking episode failed",
                version_id,
                task.id,
                exc,
            )
            return Episode(
                id=f"rollout-{task.id}-failed-{uuid.uuid4().hex[:8]}",
                task_id=task.id,
                steps=[],
                metadata={
                    "success": False,
                    "termination_reason": "error",
                    "error": str(exc),
                },
            )

    def _record_episode(self, episode: Episode, epoch_idx: int) -> EpisodeRecording:
        """Record a completed train episode post-hoc from the env frames.

        ``rollout_episode`` records no video; the env frame buffer
        (``render()``) still holds every frame of the last episode, so the
        recorder replays them with the same frame/step alignment an online
        recorder would produce: decision events at the frame the planner
        saw, outcome events at the frame after the step.
        """
        recorder = self._recorder_factory()
        out_dir = self._run_dir / _RECORDINGS_DIR / f"epoch_{epoch_idx:03d}"
        recorder.start_episode(episode.id, out_dir, fps=self._fps)

        frames = self._env.render()
        if frames:
            recorder.add_frame(frames[0])
        for step in episode.steps:
            output = step.planner_output
            detail = (
                f"{output.mission.value} pick={output.pick} place={output.place}"
                if output is not None
                else ""
            )
            recorder.mark_event(step.step_index, "decision", detail)
            frame_index = step.step_index + 1
            if frame_index < len(frames):
                recorder.add_frame(frames[frame_index])
            feedback = step.feedback
            if feedback is not None:
                if feedback.success:
                    recorder.mark_event(
                        step.step_index, "success", str(feedback.observation or "")
                    )
                else:
                    recorder.mark_event(
                        step.step_index, "failure", str(feedback.observation or "")
                    )
        return recorder.finish()

    # ------------------------------------------------------------------ #
    # Artifact writers
    # ------------------------------------------------------------------ #

    def _snapshot(self, epoch_idx: int) -> Path | None:
        try:
            return self._registry.materialize_best(
                self._target_agent,
                self._prompts_dir / f"best_epoch_{epoch_idx:03d}.md",
            )
        except LookupError:
            logger.warning("No best prompt to snapshot after epoch {}", epoch_idx)
            return None

    def _write_epoch_log(self, epoch_log: EpochLog) -> Path:
        path = self._epochs_dir / f"epoch_{epoch_log.epoch_index:03d}.json"
        path.write_text(epoch_log.model_dump_json(indent=2), encoding="utf-8")
        return path

    def _append_metrics(self, epoch_log: EpochLog) -> None:
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(epoch_log.model_dump_json() + "\n")


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _train_metrics(episodes: list[Episode]) -> tuple[float | None, float | None]:
    if not episodes:
        return None, None
    successes = [
        bool((ep.metadata or {}).get("success", False)) for ep in episodes
    ]
    success_rate = sum(successes) / len(episodes)
    average_steps = sum(len(ep.steps) for ep in episodes) / len(episodes)
    return success_rate, average_steps


def _stage_logs(episode: Episode) -> list[str]:
    logs = []
    for step in episode.steps:
        output = step.planner_output
        feedback = step.feedback
        logs.append(
            f"step {step.step_index}: "
            f"mission={output.mission.value if output else '?'} "
            f"success={feedback.success if feedback else '?'} "
            f"status={feedback.observation if feedback else ''}"
        )
    return logs


def _edit_type_summary(edits: Sequence[PromptEdit]) -> str:
    return "+".join(dict.fromkeys(edit.edit_type for edit in edits)) or "rewrite"


def _summarize_edits(edits: Sequence[PromptEdit]) -> str:
    if not edits:
        return ""
    parts = [
        f"{edit.edit_type}@{edit.location}: {edit.reason or '(no reason)'}"
        for edit in edits
    ]
    return " | ".join(parts)
