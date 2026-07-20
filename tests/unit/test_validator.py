"""Unit tests for the forge validation gate (Task 3.7).

Rollouts are either real ``rollout_episode`` loops against a duck-typed
fake RoboTwin env (same approach as ``tests/unit/test_forge_env.py``) with
a mocked VLLM client, or injected ``rollout_fn`` callables returning
scripted episodes. No sim, no LLM, no cloud critic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from common.schema import Episode, EpisodeStep, RethinkerOutput
from forge.critic import (
    CriticModelMetadata,
    CriticResult,
    StageEvaluation,
    StageScores,
)
from forge.registry import EditMetadata, ForgePromptRegistry, PromptVersion
from forge.validator import (
    PromptValidator,
    TaskValidationMetrics,
    ValidationResult,
    rollout_episode,
)
from forge.env import SimEnv
from forge.planner_agent import ForgePlannerAgent
from tasks.schema import TaskDefinition

TS0 = "2026-07-20T00:00:00+00:00"
TS1 = "2026-07-20T00:01:00+00:00"
TS2 = "2026-07-20T00:02:00+00:00"
TS3 = "2026-07-20T00:03:00+00:00"
TS4 = "2026-07-20T00:04:00+00:00"


def _edit(source: str = "optimizer") -> EditMetadata:
    return EditMetadata(edit_type="rewrite", reason="improve", source=source)


# --------------------------------------------------------------------- #
# Fake sim env (duck-typed, same pattern as tests/unit/test_forge_env.py)
# --------------------------------------------------------------------- #


class FakeRobot:
    def __init__(self) -> None:
        self._gripper_val = 1.0

    def get_right_gripper_val(self) -> float:
        return self._gripper_val

    def get_left_gripper_val(self) -> float:
        return self._gripper_val


class FakeRoboTwinEnv:
    """Duck-typed fake matching the RoboTwin base task API used by SimEnv."""

    def __init__(self, success: bool = False) -> None:
        self.robot = FakeRobot()
        self._success = success
        self.stopped = False

    def get_obs(self) -> dict[str, Any]:
        rgb = np.full((8, 8, 3), 17, dtype=np.uint8)
        return {"observation": {"head_camera": {"rgb": rgb}}}

    def get_arm_pose(self, arm_tag: str = "right") -> list[float]:
        return [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]

    def move_to_pose(self, arm_tag: str, pose: list[float]) -> list[str]:
        return ["action"]

    def open_gripper(self, arm_tag: str) -> list[str]:
        self.robot._gripper_val = 1.0
        return ["action"]

    def close_gripper(self, arm_tag: str) -> list[str]:
        self.robot._gripper_val = 0.0
        return ["action"]

    def move(self, actions: list[Any]) -> bool:
        return True

    def check_success(self) -> bool:
        return self._success

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


def make_factory(fake_env: FakeRoboTwinEnv):
    def factory(
        task_name: str,
        task_config_name: str = "demo_clean",
        repo_root: Any = None,
        seed: int = 0,
        render_freq: int = 0,
        overrides: dict[str, Any] | None = None,
    ) -> FakeRoboTwinEnv:
        return fake_env

    return factory


def _mock_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)
    return client


PICK = json.dumps({"plan_id": "p-1", "mission": "PICK_ONLY", "pick": "mock_object"})
STOP = json.dumps({"plan_id": "p-2", "mission": "STOP", "pick": "none"})


def _sim_task(task_id: str = "sim-task", max_rounds: int = 10) -> dict[str, Any]:
    return {
        "id": task_id,
        "instruction": "Pick up the mock object.",
        "mission_type": "PICK_ONLY",
        "objects": ["mock_object"],
        "initial_scene": {
            "task_name": "fake_task",
            "embodiment": ["aloha-agilex"],
            "seed": 0,
            "render_freq": 0,
            "save_data": False,
            "collect_data": False,
        },
        "metadata": {
            "robottwin_task_name": "fake_task",
            "robottwin_task_config": "demo_clean",
            "max_rounds": max_rounds,
        },
    }


# --------------------------------------------------------------------- #
# rollout_episode
# --------------------------------------------------------------------- #


def test_rollout_episode_success_first_step() -> None:
    env = SimEnv(env_factory=make_factory(FakeRoboTwinEnv(success=True)))
    agent = ForgePlannerAgent(vllm_client=_mock_client([PICK]))

    episode = rollout_episode(env, agent, _sim_task(), max_rounds=5)

    assert isinstance(episode, Episode)
    assert episode.task_id == "sim-task"
    assert len(episode.steps) == 1
    step = episode.steps[0]
    assert step.step_index == 0
    assert step.planner_output is not None
    assert step.planner_output.mission.value == "PICK_ONLY"
    assert step.feedback is not None
    assert step.feedback.success is True
    assert episode.metadata is not None
    assert episode.metadata["success"] is True
    assert episode.metadata["termination_reason"] == "stop"


def test_rollout_episode_hits_max_rounds_without_success() -> None:
    env = SimEnv(env_factory=make_factory(FakeRoboTwinEnv(success=False)))
    agent = ForgePlannerAgent(vllm_client=_mock_client([PICK, PICK, PICK]))

    episode = rollout_episode(env, agent, _sim_task(), max_rounds=3)

    assert len(episode.steps) == 3
    assert [step.step_index for step in episode.steps] == [0, 1, 2]
    assert episode.metadata is not None
    assert episode.metadata["success"] is False
    assert episode.metadata["termination_reason"] == "max_rounds"


def test_rollout_episode_stop_action_ends_episode() -> None:
    fake = FakeRoboTwinEnv(success=False)
    env = SimEnv(env_factory=make_factory(fake))
    agent = ForgePlannerAgent(vllm_client=_mock_client([PICK, STOP]))

    episode = rollout_episode(env, agent, _sim_task(), max_rounds=5)

    assert len(episode.steps) == 2
    assert fake.stopped is True
    assert episode.metadata is not None
    assert episode.metadata["success"] is False
    assert episode.metadata["termination_reason"] == "stop"


def test_rollout_episode_rejects_nonpositive_max_rounds() -> None:
    env = SimEnv(env_factory=make_factory(FakeRoboTwinEnv()))
    agent = ForgePlannerAgent(vllm_client=_mock_client([PICK]))
    with pytest.raises(ValueError, match="max_rounds"):
        rollout_episode(env, agent, _sim_task(), max_rounds=0)


# --------------------------------------------------------------------- #
# PromptValidator with injected rollout_fn
# --------------------------------------------------------------------- #


def _task(task_id: str, criteria: list[str] | None = None) -> TaskDefinition:
    return TaskDefinition(
        id=task_id,
        instruction=f"Do {task_id}.",
        mission_type="PICK_ONLY",
        objects=["mock_object"],
        success_criteria=criteria,
    )


def _episode(task_id: str, *, success: bool, steps: int) -> Episode:
    task_unit = _task(task_id)
    return Episode(
        id=f"ep-{task_id}-{'ok' if success else 'fail'}",
        task_id=task_id,
        steps=[
            EpisodeStep(
                step_index=i,
                task=task_unit,
                rethinker_output=RethinkerOutput(
                    mission_type="PICK_ONLY", reasoning="scripted"
                ),
            )
            for i in range(steps)
        ],
        metadata={
            "success": success,
            "termination_reason": "stop" if success else "max_rounds",
        },
    )


class ScriptedRollout:
    """rollout_fn returning scripted episodes per (version_id, task_id)."""

    def __init__(self) -> None:
        self.script: dict[tuple[str, str], Episode] = {}
        self.calls: list[tuple[str, str]] = []

    def set(self, version_id: str, task_id: str, *, success: bool, steps: int) -> None:
        self.script[(version_id, task_id)] = _episode(
            task_id, success=success, steps=steps
        )

    def __call__(self, version_id: str, task: TaskDefinition) -> Episode:
        self.calls.append((version_id, task.id))
        return self.script[(version_id, task.id)]


@pytest.fixture
def registry(tmp_path: Path) -> ForgePromptRegistry:
    return ForgePromptRegistry(tmp_path / "prompts")


def _seed_best(
    registry: ForgePromptRegistry,
    metrics: dict[str, float] | None = None,
) -> PromptVersion:
    best = registry.register("base prompt", "planner", _edit(source="hand"), timestamp=TS0)
    registry.record_validation(
        best.version_id,
        metrics if metrics is not None else {"success_rate": 0.5, "average_steps": 4.0, "composite": 0.5},
        accepted=True,
        timestamp=TS1,
    )
    return registry.get(best.version_id)


def _candidate(registry: ForgePromptRegistry, parent: PromptVersion) -> PromptVersion:
    return registry.register(
        "edited prompt",
        "planner",
        _edit(),
        parent_version=parent.version_id,
        timestamp=TS2,
    )


def test_validate_accepts_on_strict_improvement(
    registry: ForgePromptRegistry,
) -> None:
    best = _seed_best(registry)
    cand = _candidate(registry, best)
    tasks = [_task("t1"), _task("t2")]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=True, steps=2)
    rollout.set(cand.version_id, "t2", success=True, steps=2)

    validator = PromptValidator(registry)
    result = validator.validate(cand, tasks, rollout_fn=rollout, timestamp=TS3)

    assert isinstance(result, ValidationResult)
    assert result.accepted is True
    assert result.version_id == cand.version_id
    assert result.target_agent == "planner"
    assert result.success_rate == 1.0
    assert result.composite == 1.0
    assert result.baseline_composite == 0.5
    assert result.average_steps == 2.0
    assert len(result.per_task) == 2
    assert all(isinstance(m, TaskValidationMetrics) for m in result.per_task)
    assert all(m.success for m in result.per_task)
    assert result.mean_video_score is None
    assert "improvement" in result.reason

    # registry bookkeeping: candidate promoted to best, old best demoted
    updated = registry.get(cand.version_id)
    assert updated.status == "best"
    assert updated.validation is not None
    assert updated.validation.accepted is True
    assert updated.validation.metrics["success_rate"] == 1.0
    assert updated.validation.metrics["average_steps"] == 2.0
    assert updated.validation.metrics["composite"] == 1.0
    assert updated.validation.metrics["num_tasks"] == 2.0
    assert updated.validation.timestamp == TS3
    assert updated.validation.detail == result.reason
    assert registry.get(best.version_id).status == "accepted"
    assert registry.best("planner").version_id == cand.version_id


def test_validate_rejects_on_regression(registry: ForgePromptRegistry) -> None:
    best = _seed_best(registry)
    cand = _candidate(registry, best)
    tasks = [_task("t1"), _task("t2")]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=False, steps=5)
    rollout.set(cand.version_id, "t2", success=True, steps=5)

    result = PromptValidator(registry).validate(
        cand, tasks, rollout_fn=rollout, timestamp=TS3
    )

    assert result.accepted is False
    assert result.success_rate == 0.5
    assert result.baseline_composite == 0.5
    updated = registry.get(cand.version_id)
    assert updated.status == "rejected"
    assert updated.validation is not None
    assert updated.validation.accepted is False
    assert updated.validation.metrics["success_rate"] == 0.5
    assert registry.best("planner").version_id == best.version_id


def test_validate_rejects_on_exact_tie(registry: ForgePromptRegistry) -> None:
    best = _seed_best(registry)  # success_rate 0.5, average_steps 4.0
    cand = _candidate(registry, best)
    tasks = [_task("t1"), _task("t2")]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=True, steps=4)
    rollout.set(cand.version_id, "t2", success=False, steps=4)

    result = PromptValidator(registry).validate(
        cand, tasks, rollout_fn=rollout, timestamp=TS3
    )

    assert result.accepted is False
    assert result.success_rate == 0.5
    assert result.average_steps == 4.0
    assert "no strict improvement" in result.reason
    assert registry.get(cand.version_id).status == "rejected"
    assert registry.best("planner").version_id == best.version_id


def test_validate_accepts_on_tiebreak_fewer_steps(
    registry: ForgePromptRegistry,
) -> None:
    best = _seed_best(registry)  # success_rate 0.5, average_steps 4.0
    cand = _candidate(registry, best)
    tasks = [_task("t1"), _task("t2")]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=True, steps=2)
    rollout.set(cand.version_id, "t2", success=False, steps=2)

    result = PromptValidator(registry).validate(
        cand, tasks, rollout_fn=rollout, timestamp=TS3
    )

    assert result.accepted is True
    assert result.success_rate == 0.5
    assert result.average_steps == 2.0
    assert registry.get(cand.version_id).status == "best"


def test_validate_rejects_on_tiebreak_more_steps(
    registry: ForgePromptRegistry,
) -> None:
    best = _seed_best(registry)  # success_rate 0.5, average_steps 4.0
    cand = _candidate(registry, best)
    tasks = [_task("t1"), _task("t2")]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=True, steps=6)
    rollout.set(cand.version_id, "t2", success=False, steps=6)

    result = PromptValidator(registry).validate(
        cand, tasks, rollout_fn=rollout, timestamp=TS3
    )

    assert result.accepted is False
    assert registry.get(cand.version_id).status == "rejected"


def test_validate_reevaluates_best_when_metrics_missing(
    registry: ForgePromptRegistry,
) -> None:
    best = _seed_best(registry, metrics={})  # accepted without composite keys
    cand = _candidate(registry, best)
    tasks = [_task("t1"), _task("t2")]
    rollout = ScriptedRollout()
    # incumbent best re-evaluated on the same tasks: 1/2 success
    rollout.set(best.version_id, "t1", success=True, steps=3)
    rollout.set(best.version_id, "t2", success=False, steps=3)
    rollout.set(cand.version_id, "t1", success=True, steps=3)
    rollout.set(cand.version_id, "t2", success=True, steps=3)

    result = PromptValidator(registry).validate(
        cand, tasks, rollout_fn=rollout, timestamp=TS3
    )

    assert result.accepted is True
    assert result.baseline_composite == 0.5
    assert result.baseline_average_steps == 3.0
    assert "re-evaluated" in result.reason
    # rollout_fn was invoked for both versions on both tasks
    assert (best.version_id, "t1") in rollout.calls
    assert (best.version_id, "t2") in rollout.calls
    assert (cand.version_id, "t1") in rollout.calls


def test_validate_accepts_first_candidate_when_no_best(
    registry: ForgePromptRegistry,
) -> None:
    cand = registry.register("first prompt", "planner", _edit(), timestamp=TS0)
    tasks = [_task("t1")]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=False, steps=5)

    result = PromptValidator(registry).validate(
        cand, tasks, rollout_fn=rollout, timestamp=TS1
    )

    assert result.accepted is True
    assert result.baseline_composite is None
    assert "no incumbent" in result.reason
    assert registry.best("planner").version_id == cand.version_id


def test_validate_success_uses_harness_with_task_criteria(
    registry: ForgePromptRegistry,
) -> None:
    """Tasks with success_criteria route through the evaluation harness."""
    best = _seed_best(registry)
    cand = _candidate(registry, best)
    tasks = [_task("t1", criteria=["object grasped"]), _task("t2", criteria=["object grasped"])]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=True, steps=1)
    rollout.set(cand.version_id, "t2", success=True, steps=1)

    result = PromptValidator(registry).validate(
        cand, tasks, rollout_fn=rollout, timestamp=TS3
    )

    assert result.accepted is True
    assert result.success_rate == 1.0
    assert all(m.success for m in result.per_task)
    assert [m.steps for m in result.per_task] == [1, 1]


# --------------------------------------------------------------------- #
# Critic integration (optional per call)
# --------------------------------------------------------------------- #


def _critic_result(episode_id: str, evaluations: list[StageEvaluation]) -> CriticResult:
    return CriticResult(
        episode_id=episode_id,
        filtered=not evaluations,
        reason="escalated" if evaluations else "clean_success",
        prefilter=None,
        evaluations=evaluations,
        model_metadata=CriticModelMetadata(
            cloud_model_id="fake-cloud",
            cloud_temperature=0.0,
            cloud_max_tokens=100,
            prefilter_model_id=None,
            prompt_version="v0",
            recording_schema_version="1.0",
        ),
    )


def test_validate_with_critic_populates_stage_and_video_scores(
    registry: ForgePromptRegistry,
) -> None:
    best = _seed_best(registry)
    cand = _candidate(registry, best)
    tasks = [_task("t1")]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=True, steps=2)

    def critic_fn(episode: Episode, task: TaskDefinition) -> CriticResult:
        return _critic_result(
            episode.id,
            [
                StageEvaluation(
                    stage="episode",
                    scores=StageScores(correctness=0.9, efficiency=0.6, safety=0.9),
                    root_cause="none",
                    evidence="frame 0",
                ),
                StageEvaluation(
                    stage="step:0",
                    scores=StageScores(correctness=0.5, efficiency=0.6, safety=1.0),
                    root_cause="slow",
                    evidence="frame 0",
                ),
            ],
        )

    result = PromptValidator(registry).validate(
        cand,
        tasks,
        rollout_fn=rollout,
        use_critic=True,
        critic_fn=critic_fn,
        timestamp=TS3,
    )

    assert result.accepted is True
    metrics = result.per_task[0]
    assert metrics.stage_scores is not None
    # mean across the two evaluations
    assert metrics.stage_scores.correctness == pytest.approx(0.7)
    assert metrics.stage_scores.efficiency == pytest.approx(0.6)
    assert metrics.stage_scores.safety == pytest.approx(0.95)
    # video score = mean of the episode-stage dims
    assert metrics.video_score == pytest.approx((0.9 + 0.6 + 0.9) / 3)
    assert result.mean_video_score == pytest.approx(0.8)
    recorded = registry.get(cand.version_id).validation
    assert recorded is not None
    assert recorded.metrics["mean_video_score"] == pytest.approx(0.8)


def test_validate_critic_filtered_result_gives_absent_scores(
    registry: ForgePromptRegistry,
) -> None:
    best = _seed_best(registry)
    cand = _candidate(registry, best)
    tasks = [_task("t1")]
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=True, steps=1)

    def critic_fn(episode: Episode, task: TaskDefinition) -> CriticResult:
        return _critic_result(episode.id, [])

    result = PromptValidator(registry).validate(
        cand,
        tasks,
        rollout_fn=rollout,
        use_critic=True,
        critic_fn=critic_fn,
        timestamp=TS3,
    )

    assert result.per_task[0].stage_scores is None
    assert result.per_task[0].video_score is None
    assert result.mean_video_score is None
    recorded = registry.get(cand.version_id).validation
    assert recorded is not None
    assert "mean_video_score" not in recorded.metrics


def test_validate_use_critic_without_fn_raises(
    registry: ForgePromptRegistry,
) -> None:
    cand = registry.register("prompt", "planner", _edit(), timestamp=TS0)
    with pytest.raises(ValueError, match="critic_fn"):
        PromptValidator(registry).validate(
            cand, [_task("t1")], rollout_fn=ScriptedRollout(), use_critic=True
        )


# --------------------------------------------------------------------- #
# Input validation / error paths
# --------------------------------------------------------------------- #


def test_validate_rejects_empty_task_list(registry: ForgePromptRegistry) -> None:
    cand = registry.register("prompt", "planner", _edit(), timestamp=TS0)
    with pytest.raises(ValueError, match="val_tasks"):
        PromptValidator(registry).validate(cand, [], rollout_fn=ScriptedRollout())


def test_validate_rejects_non_candidate_version(
    registry: ForgePromptRegistry,
) -> None:
    best = _seed_best(registry)
    with pytest.raises(ValueError, match="candidate"):
        PromptValidator(registry).validate(
            best, [_task("t1")], rollout_fn=ScriptedRollout()
        )


def test_validate_accepts_version_id_string(registry: ForgePromptRegistry) -> None:
    cand = registry.register("prompt", "planner", _edit(), timestamp=TS0)
    rollout = ScriptedRollout()
    rollout.set(cand.version_id, "t1", success=True, steps=1)

    result = PromptValidator(registry).validate(
        cand.version_id, [_task("t1")], rollout_fn=rollout, timestamp=TS1
    )

    assert result.version_id == cand.version_id
    assert result.accepted is True


def test_validate_unknown_version_id_raises(registry: ForgePromptRegistry) -> None:
    with pytest.raises(KeyError, match="v999"):
        PromptValidator(registry).validate(
            "v999", [_task("t1")], rollout_fn=ScriptedRollout()
        )


def test_validate_requires_env_and_planner_without_rollout_fn(
    registry: ForgePromptRegistry,
) -> None:
    cand = registry.register("prompt", "planner", _edit(), timestamp=TS0)
    with pytest.raises(ValueError, match="env"):
        PromptValidator(registry).validate(cand, [_task("t1")])


def test_validate_default_rollout_uses_env_and_planner(
    registry: ForgePromptRegistry,
) -> None:
    cand = registry.register("prompt", "planner", _edit(), timestamp=TS0)
    env = SimEnv(env_factory=make_factory(FakeRoboTwinEnv(success=True)))
    agent = ForgePlannerAgent(vllm_client=_mock_client([PICK]))

    validator = PromptValidator(registry, env=env, planner=agent, max_rounds=5)
    result = validator.validate(
        cand.version_id, [TaskDefinition(**_sim_task("t1"))], timestamp=TS1
    )

    assert result.accepted is True
    assert result.success_rate == 1.0
    assert result.per_task[0].steps == 1
