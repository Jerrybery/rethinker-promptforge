"""Unit tests for the forge training loop runner (Task 3.8).

All heavy components are faked: a duck-typed env, a duck-typed planner, a
scripted critic, and a scripted optimizer. The registry and validator are
real (tmp_path storage) so accept/reject bookkeeping is exercised end to
end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from common.schema import MissionType, PlannerOutput
from forge.critic import (
    CriticModelMetadata,
    CriticResult,
    StageEvaluation,
    StageScores,
)
from forge.optimizer import PromptEdit, apply_edits
from forge.recorder import EpisodeRecorder, EpisodeRecording
from forge.registry import ForgePromptRegistry
from forge.runner import EpochLog, ForgeLog, ForgeRunner
from forge.validator import PromptValidator
from tasks.schema import TaskDefinition

SEED_PROMPT = "# Seed Planner Prompt\n\n## Rules\n\n- Be helpful.\n"

EDIT = PromptEdit(
    target_agent="planner",
    edit_type="add",
    location="Rules",
    new_text="- Always stop on confirmed success.",
    reason="efficiency",
)


def make_task(task_id: str, mission: str = "PICK_AND_PLACE") -> TaskDefinition:
    return TaskDefinition(
        id=task_id,
        instruction="Do the thing.",
        mission_type=mission,
        objects=["obj"],
        initial_scene={"task_name": "fake_task"},
        success_criteria=[],
    )


TRAIN_TASK = make_task("train-0")
VAL_TASK = make_task("val-0")


class FakeEnv:
    """Duck-typed SimEnv: one-step episodes, switchable success."""

    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        self._success = False
        self._fail_on = set(fail_on)
        self._task: TaskDefinition | None = None
        self._frames: list[np.ndarray] = []

    def set_success(self, success: bool) -> None:
        self._success = success

    @property
    def task(self) -> TaskDefinition | None:
        return self._task

    def _obs(self, step_index: int) -> dict[str, Any]:
        assert self._task is not None
        return {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": {
                "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "gripper": 1.0,
                "timestamp": 0.0,
            },
            "detections": [
                {"label": "obj", "bbox": [0, 0, 2, 2], "confidence": 0.9}
            ],
            "task": {
                "id": self._task.id,
                "instruction": self._task.instruction,
                "mission_type": self._task.mission_type.value,
            },
            "step_index": step_index,
        }

    def reset(self, task: TaskDefinition) -> dict[str, Any]:
        self._task = task
        if task.id in self._fail_on:
            raise RuntimeError(f"simulated env failure on {task.id}")
        self._frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        return self._obs(0)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        assert self._task is not None
        self._frames.append(np.zeros((4, 4, 3), dtype=np.uint8))
        success = self._success
        return self._obs(1), 1.0 if success else 0.0, True, {
            "success": success,
            "primitive_success": True,
            "primitive_status": "ok",
            "task_id": self._task.id,
            "step_index": 1,
            "truncated": False,
        }

    def render(self) -> list[np.ndarray]:
        return list(self._frames)


class FakePlanner:
    """Duck-typed ForgePlannerAgent echoing the obs mission type."""

    def __init__(self) -> None:
        self.prompt_text = ""

    def set_prompt_text(self, system_template: str, user_template: str | None = None) -> None:
        self.prompt_text = system_template

    def act_from_obs(self, obs: dict[str, Any], **kwargs: Any) -> PlannerOutput:
        mission = MissionType(obs["task"]["mission_type"])
        if mission is MissionType.STOP:
            return PlannerOutput(plan_id="fake", mission=mission, pick="none")
        if mission is MissionType.PICK_AND_PLACE:
            return PlannerOutput(
                plan_id="fake", mission=mission, pick="obj", place="obj"
            )
        return PlannerOutput(plan_id="fake", mission=mission, pick="obj")


class FakeCritic:
    """Scripted critic returning one episode-stage evaluation per call."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    def evaluate_episode(
        self,
        recording: Any,
        *,
        final_success: bool,
        max_steps: int | None = None,
        stage_logs: Any = "",
    ) -> CriticResult:
        self.calls += 1
        if self._fail:
            raise RuntimeError("critic exploded")
        return CriticResult(
            episode_id=recording.episode_id,
            filtered=False,
            reason="fake",
            prefilter=None,
            evaluations=[
                StageEvaluation(
                    stage="episode",
                    scores=StageScores(
                        correctness=0.5, efficiency=0.5, safety=1.0
                    ),
                    root_cause="fake root cause",
                    evidence="frame 0",
                )
            ],
            model_metadata=CriticModelMetadata(
                cloud_model_id="fake-cloud",
                cloud_temperature=0.0,
                cloud_max_tokens=64,
                prefilter_model_id=None,
                prompt_version="v0",
                recording_schema_version="1.0",
            ),
        )


class FakeOptimizer:
    """Scripted optimizer recording every propose_edits call."""

    def __init__(
        self,
        edits: list[PromptEdit],
        on_call: Callable[[], None] | None = None,
        fail: bool = False,
    ) -> None:
        self._edits = edits
        self._on_call = on_call
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    def propose_edits(
        self,
        best_prompt: str,
        evaluations: Any,
        rejected_history: Any = (),
        budget_chars: int | None = None,
        rejected_texts: Any = None,
    ) -> list[PromptEdit]:
        self.calls.append(
            {
                "best_prompt": best_prompt,
                "evaluations": list(evaluations),
                "rejected_history": list(rejected_history),
                "rejected_texts": (
                    None if rejected_texts is None else list(rejected_texts)
                ),
            }
        )
        if self._fail:
            raise RuntimeError("optimizer exploded")
        if self._on_call is not None:
            self._on_call()
        return list(self._edits)


def make_runner(
    tmp_path: Path,
    *,
    env: FakeEnv | None = None,
    optimizer: FakeOptimizer | None = None,
    critic: Any = None,
    train_tasks: list[TaskDefinition] | None = None,
    val_tasks: list[TaskDefinition] | None = None,
    initial_prompt_text: str | None = SEED_PROMPT,
    recorder_factory: Any = None,
) -> tuple[ForgeRunner, ForgePromptRegistry, FakeEnv, FakeOptimizer]:
    registry = ForgePromptRegistry(tmp_path / "registry")
    env = env or FakeEnv()
    optimizer = optimizer or FakeOptimizer([EDIT])
    runner = ForgeRunner(
        registry=registry,
        env=env,
        planner=FakePlanner(),
        optimizer=optimizer,
        critic=critic,
        train_tasks=train_tasks if train_tasks is not None else [TRAIN_TASK],
        val_tasks=val_tasks if val_tasks is not None else [VAL_TASK],
        run_dir=tmp_path / "run",
        initial_prompt_text=initial_prompt_text,
        recorder_factory=recorder_factory or EpisodeRecorder,
    )
    return runner, registry, env, optimizer


# --------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------- #


def test_empty_train_tasks_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train_tasks"):
        make_runner(tmp_path, train_tasks=[])


def test_empty_val_tasks_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="val_tasks"):
        make_runner(tmp_path, val_tasks=[])


def test_missing_initial_prompt_rejected_when_registry_empty(tmp_path: Path) -> None:
    runner, _, _, _ = make_runner(tmp_path, initial_prompt_text=None)
    with pytest.raises(ValueError, match="initial_prompt_text"):
        runner.run(1)


def test_zero_epochs_rejected(tmp_path: Path) -> None:
    runner, _, _, _ = make_runner(tmp_path)
    with pytest.raises(ValueError, match="n_epochs"):
        runner.run(0)


# --------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------- #


def test_run_seeds_registry_as_first_champion(tmp_path: Path) -> None:
    runner, registry, _, optimizer = make_runner(tmp_path, optimizer=FakeOptimizer([]))
    log = runner.run(1)

    seed = registry.history("planner", status="best")
    assert len(seed) == 1
    assert seed[0].edit.source == "hand"
    assert seed[0].edit.edit_type == "seed"
    assert seed[0].validation is not None
    assert seed[0].validation.accepted is True
    assert seed[0].validation.metrics["success_rate"] == 0.0
    assert log.seed_version_id == seed[0].version_id
    assert registry.text(seed[0].version_id) == SEED_PROMPT


def test_run_skips_seeding_when_best_exists(tmp_path: Path) -> None:
    runner, registry, _, _ = make_runner(tmp_path, optimizer=FakeOptimizer([]))
    first = runner.run(1)
    second = runner.run(1)
    assert second.seed_version_id == first.seed_version_id
    # No extra version registered by the second run's seeding.
    assert len(registry.history("planner")) == 1


# --------------------------------------------------------------------- #
# Accept path
# --------------------------------------------------------------------- #


def test_epoch_accept_path(tmp_path: Path) -> None:
    env = FakeEnv()
    # The env starts failing validation rollouts when the optimizer is
    # called, i.e. only the candidate sees a "fixed" env.
    optimizer = FakeOptimizer([EDIT], on_call=lambda: env.set_success(True))
    runner, registry, _, _ = make_runner(
        tmp_path, env=env, optimizer=optimizer, critic=FakeCritic()
    )
    log = runner.run(1)

    assert isinstance(log, ForgeLog)
    assert len(log.epochs) == 1
    epoch = log.epochs[0]
    assert isinstance(epoch, EpochLog)

    candidate_id = epoch.candidate_version_id
    assert candidate_id is not None
    assert epoch.accepted is True
    assert epoch.validation_success_rate == 1.0
    assert epoch.train_success_rate == 0.0
    assert epoch.critic_evaluations == 1
    assert epoch.edits_proposed == 1

    candidate = registry.get(candidate_id)
    assert candidate.status == "best"
    assert candidate.edit.source == "optimizer"
    assert candidate.parent_version == log.seed_version_id
    assert registry.get(log.seed_version_id).status == "accepted"
    assert log.final_best_version_id == candidate_id

    # Candidate text is the seed with the edit applied.
    assert registry.text(candidate_id) == apply_edits(SEED_PROMPT, [EDIT])

    # Optimizer saw the critic evaluation and an empty reject history.
    call = optimizer.calls[0]
    assert call["best_prompt"] == SEED_PROMPT
    assert len(call["evaluations"]) == 1
    assert call["rejected_history"] == []
    assert call["rejected_texts"] == []


def test_accept_artifacts_written(tmp_path: Path) -> None:
    env = FakeEnv()
    optimizer = FakeOptimizer([EDIT], on_call=lambda: env.set_success(True))
    runner, registry, _, _ = make_runner(tmp_path, env=env, optimizer=optimizer)
    log = runner.run(1)
    run_dir = tmp_path / "run"

    forge_log_path = run_dir / "forge_log.json"
    assert forge_log_path.exists()
    loaded = ForgeLog(**json.loads(forge_log_path.read_text()))
    assert loaded.run_id == log.run_id

    epoch_path = run_dir / "epochs" / "epoch_000.json"
    assert epoch_path.exists()
    epoch = EpochLog(**json.loads(epoch_path.read_text()))
    assert epoch.accepted is True

    metrics_path = run_dir / "metrics.jsonl"
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["epoch_index"] == 0
    assert row["accepted"] is True
    assert row["validation_success_rate"] == 1.0

    snapshot = run_dir / "prompts" / "best_epoch_000.md"
    assert snapshot.exists()
    assert snapshot.read_text() == registry.text(log.final_best_version_id)

    recordings = list((run_dir / "recordings" / "epoch_000").iterdir())
    assert len(recordings) == 1
    assert (recordings[0] / "metadata.json").exists()
    assert (recordings[0] / "video.mp4").exists()


# --------------------------------------------------------------------- #
# Reject path
# --------------------------------------------------------------------- #


def test_epoch_reject_path_keeps_incumbent(tmp_path: Path) -> None:
    runner, registry, _, _ = make_runner(tmp_path)
    log = runner.run(1)

    epoch = log.epochs[0]
    assert epoch.candidate_version_id is not None
    assert epoch.accepted is False
    assert "no strict improvement" in epoch.validation_reason
    assert registry.get(epoch.candidate_version_id).status == "rejected"
    assert log.final_best_version_id == log.seed_version_id


def test_rejected_texts_passed_to_optimizer_next_epoch(tmp_path: Path) -> None:
    runner, registry, _, optimizer = make_runner(tmp_path)
    log = runner.run(2)

    assert len(optimizer.calls) == 2
    second = optimizer.calls[1]
    rejected = second["rejected_history"]
    assert len(rejected) == 1
    first_candidate_id = log.epochs[0].candidate_version_id
    assert rejected[0].version_id == first_candidate_id
    # Symmetric dedup contract: rejected texts resolved via registry.text.
    assert second["rejected_texts"] == [registry.text(first_candidate_id)]


# --------------------------------------------------------------------- #
# No-edit / unchanged-edit paths
# --------------------------------------------------------------------- #


def test_no_edits_proposed_skips_validation(tmp_path: Path) -> None:
    runner, registry, _, _ = make_runner(tmp_path, optimizer=FakeOptimizer([]))
    log = runner.run(1)

    epoch = log.epochs[0]
    assert epoch.edits_proposed == 0
    assert epoch.candidate_version_id is None
    assert epoch.accepted is None
    assert "no edits" in epoch.validation_reason
    # Only the seed exists in the registry.
    assert len(registry.history("planner")) == 1


def test_unchanged_candidate_skips_validation(tmp_path: Path) -> None:
    # Edit targets a missing heading -> apply_edits leaves text unchanged.
    no_op = PromptEdit(
        target_agent="planner",
        edit_type="add",
        location="No Such Section",
        new_text="- nothing happens",
        reason="dud",
    )
    runner, registry, _, _ = make_runner(tmp_path, optimizer=FakeOptimizer([no_op]))
    log = runner.run(1)

    epoch = log.epochs[0]
    assert epoch.edits_proposed == 1
    assert epoch.candidate_version_id is None
    assert epoch.accepted is None
    assert "unchanged" in epoch.validation_reason
    assert len(registry.history("planner")) == 1


# --------------------------------------------------------------------- #
# Fault tolerance
# --------------------------------------------------------------------- #


def test_train_episode_exception_does_not_kill_run(tmp_path: Path) -> None:
    env = FakeEnv(fail_on=("train-bad",))
    runner, registry, _, _ = make_runner(
        tmp_path, env=env, train_tasks=[make_task("train-bad"), TRAIN_TASK]
    )
    log = runner.run(1)

    epoch = log.epochs[0]
    assert epoch.train_failed_episodes == 1
    assert epoch.train_episodes == 1
    assert epoch.candidate_version_id is not None


def test_validation_episode_exception_marked_failed(tmp_path: Path) -> None:
    env = FakeEnv(fail_on=("val-bad",))
    runner, registry, _, _ = make_runner(
        tmp_path, env=env, val_tasks=[make_task("val-bad"), VAL_TASK]
    )
    log = runner.run(1)

    # The run completes; the failed val rollout counts as an unsuccessful
    # zero-step episode, so seed and candidate both score 0.0 -> reject.
    epoch = log.epochs[0]
    assert epoch.accepted is False
    assert epoch.validation_success_rate == 0.0
    assert registry.get(log.seed_version_id).validation is not None


def test_critic_exception_tolerated(tmp_path: Path) -> None:
    runner, _, _, optimizer = make_runner(tmp_path, critic=FakeCritic(fail=True))
    log = runner.run(1)

    assert log.epochs[0].critic_evaluations == 0
    assert optimizer.calls[0]["evaluations"] == []


def test_optimizer_exception_does_not_kill_run(tmp_path: Path) -> None:
    runner, registry, _, optimizer = make_runner(
        tmp_path, optimizer=FakeOptimizer([EDIT], fail=True)
    )
    log = runner.run(2)

    assert len(log.epochs) == 2
    assert len(optimizer.calls) == 2  # every epoch still calls the optimizer
    for epoch in log.epochs:
        assert epoch.edits_proposed == 0
        assert epoch.candidate_version_id is None
        assert epoch.accepted is None
        assert "propose_edits" in epoch.validation_reason
    # Only the seed exists in the registry; best stays the seed.
    assert len(registry.history("planner")) == 1
    assert log.final_best_version_id == log.seed_version_id


class _FlakyValidator:
    """Delegates to a real PromptValidator but raises when ``fail`` is set."""

    def __init__(self, delegate: PromptValidator) -> None:
        self._delegate = delegate
        self.fail = False

    def validate(self, *args: Any, **kwargs: Any) -> Any:
        if self.fail:
            raise RuntimeError("validator exploded")
        return self._delegate.validate(*args, **kwargs)


def test_validator_exception_does_not_kill_run(tmp_path: Path) -> None:
    registry = ForgePromptRegistry(tmp_path / "registry")
    env = FakeEnv()
    planner = FakePlanner()
    delegate = PromptValidator(registry, env=env, planner=planner, max_rounds=10)
    validator = _FlakyValidator(delegate)
    runner = ForgeRunner(
        registry=registry,
        env=env,
        planner=planner,
        optimizer=FakeOptimizer([EDIT]),
        validator=validator,
        train_tasks=[TRAIN_TASK],
        val_tasks=[VAL_TASK],
        run_dir=tmp_path / "run",
        initial_prompt_text=SEED_PROMPT,
    )

    first = runner.run(1)  # seeding + epoch validate normally
    assert first.epochs[0].accepted is False

    validator.fail = True
    second = runner.run(1)  # mid-run validator outage: run must survive

    assert len(second.epochs) == 1
    epoch = second.epochs[0]
    assert epoch.candidate_version_id is not None
    assert epoch.accepted is None
    assert "validate" in epoch.validation_reason
    assert "validator exploded" in epoch.validation_reason
    # Candidate stays registered but unvalidated (no outcome recorded).
    candidate = registry.get(epoch.candidate_version_id)
    assert candidate.validation is None
    assert candidate.status == "candidate"
    assert second.final_best_version_id == second.seed_version_id


def test_no_critic_means_no_evaluations(tmp_path: Path) -> None:
    runner, _, _, optimizer = make_runner(tmp_path, critic=None)
    runner.run(1)
    assert optimizer.calls[0]["evaluations"] == []


class _SuccessOnTasksEnv(FakeEnv):
    """FakeEnv variant reporting success only for the given task ids."""

    def __init__(self, success_tasks: tuple[str, ...]) -> None:
        super().__init__()
        self._success_tasks = set(success_tasks)

    def reset(self, task: TaskDefinition) -> dict[str, Any]:
        self.set_success(task.id in self._success_tasks)
        return super().reset(task)


class _StubRecorder:
    """Minimal EpisodeRecorder stand-in returning a metadata-only recording."""

    def __init__(self) -> None:
        self._episode_id = ""

    def start_episode(self, episode_id: str, out_dir: Any, fps: float = 10.0) -> None:
        self._episode_id = episode_id

    def add_frame(self, frame: Any) -> None:
        pass

    def mark_event(self, step_index: int, kind: str, detail: str = "") -> None:
        pass

    def finish(self) -> EpisodeRecording:
        return EpisodeRecording(
            episode_id=self._episode_id,
            fps=10.0,
            frame_count=0,
            metadata_path="/stub/metadata.json",
        )


class _SpyCritic(FakeCritic):
    """FakeCritic that records every evaluate_episode call's arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict[str, Any]] = []

    def evaluate_episode(
        self,
        recording: Any,
        *,
        final_success: bool,
        max_steps: int | None = None,
        stage_logs: Any = "",
    ) -> CriticResult:
        self.seen.append(
            {"recording": recording, "final_success": final_success}
        )
        return super().evaluate_episode(
            recording,
            final_success=final_success,
            max_steps=max_steps,
            stage_logs=stage_logs,
        )


def test_failed_recording_keeps_episode_critic_alignment(tmp_path: Path) -> None:
    # The recorder factory explodes on the FIRST train episode only; without
    # index-aligned recordings the critic would pair episode 0's outcome
    # with episode 1's recording.
    calls = {"n": 0}

    def factory() -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("recorder exploded")
        return _StubRecorder()

    env = _SuccessOnTasksEnv(("train-1",))
    critic = _SpyCritic()
    runner, _, _, _ = make_runner(
        tmp_path,
        env=env,
        critic=critic,
        train_tasks=[make_task("train-0"), make_task("train-1")],
        recorder_factory=factory,
    )
    log = runner.run(1)

    # Critic saw only the second episode, with ITS recording and outcome.
    assert len(critic.seen) == 1
    seen = critic.seen[0]
    assert seen["recording"].episode_id.startswith("rollout-train-1-")
    assert seen["final_success"] is True
    # The failed recording leaves no path in the epoch log.
    epoch = log.epochs[0]
    assert epoch.train_episodes == 2
    assert epoch.recordings == ["/stub/metadata.json"]


# --------------------------------------------------------------------- #
# Incremental artifact flushing (resume safety)
# --------------------------------------------------------------------- #


def test_epoch_logs_flushed_per_epoch(tmp_path: Path) -> None:
    runner, _, _, _ = make_runner(tmp_path)
    runner.run(3)
    run_dir = tmp_path / "run"

    epoch_files = sorted((run_dir / "epochs").glob("epoch_*.json"))
    assert [p.name for p in epoch_files] == [
        "epoch_000.json",
        "epoch_001.json",
        "epoch_002.json",
    ]
    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
