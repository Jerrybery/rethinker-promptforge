"""Unit tests for the lightweight evaluation harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.schema import (
    Episode,
    EpisodeStep,
    ExecutorOutput,
    Feedback,
    MissionType,
    PlannerOutput,
    RethinkerOutput,
    TaskUnit,
)
from executor.schema import ExecutorAgentOutput
from evaluation.harness import (
    CriterionResult,
    EpisodeEvaluation,
    EvaluationResult,
    KeywordSuccessChecker,
    evaluate_episode,
    evaluate_tasks,
)
from tasks.schema import TaskDefinition


REPO_ROOT = Path(__file__).resolve().parents[2]
HELLO_TASKS_PATH = REPO_ROOT / "data" / "tasks" / "hello_tasks.yaml"


@pytest.fixture
def warning_log(monkeypatch):
    """Capture loguru warning messages emitted by the evaluation harness."""
    messages = []

    def fake_warning(_self, msg, *args, **kwargs):
        messages.append(msg.format(*args, **kwargs) if args else msg)

    monkeypatch.setattr(
        "evaluation.harness.logger", type("L", (), {"warning": fake_warning})()
    )
    return messages


def _task(
    task_id: str = "task-001",
    mission_type: MissionType = MissionType.PICK_AND_PLACE,
    criteria: list[str] | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        id=task_id,
        instruction="mock instruction",
        mission_type=mission_type,
        objects=["object"],
        success_criteria=criteria,
    )


def _step(
    step_index: int = 0,
    task: TaskDefinition | None = None,
    mission_type: MissionType = MissionType.PICK_AND_PLACE,
    feedback_success: bool | None = True,
    gripper_state: float = 0.8,
) -> EpisodeStep:
    resolved_task = task or _task()
    feedback = None if feedback_success is None else Feedback(success=feedback_success)
    return EpisodeStep(
        step_index=step_index,
        task=resolved_task,
        rethinker_output=RethinkerOutput(
            mission_type=mission_type,
            reasoning="mock reasoning",
        ),
        planner_output=PlannerOutput(
            plan_id=f"plan-{step_index}",
            mission=mission_type,
            pick="object",
        ),
        executor_output=ExecutorOutput(
            step_index=step_index,
            joint_angles=[0.0],
            gripper_state=gripper_state,
        ),
        feedback=feedback,
    )


def _episode(
    episode_id: str = "ep-001",
    task_id: str = "task-001",
    steps: list[EpisodeStep] | None = None,
    termination_reason: str | None = "stop",
    metadata: dict | None = None,
) -> Episode:
    resolved_metadata = dict(metadata or {})
    if termination_reason is not None:
        resolved_metadata.setdefault("termination_reason", termination_reason)
    return Episode(
        id=episode_id,
        task_id=task_id,
        steps=steps or [],
        metadata=resolved_metadata or None,
    )


class TestKeywordSuccessChecker:
    def test_stop_criterion_passes_when_terminated_by_stop(self) -> None:
        task = _task(criteria=["episode stopped"])
        episode = _episode(steps=[_step()], termination_reason="stop")
        assert evaluate_episode(episode, task).success is True

    def test_stop_criterion_fails_when_not_stopped(self) -> None:
        task = _task(criteria=["episode stopped"])
        episode = _episode(steps=[_step()], termination_reason="timeout")
        result = evaluate_episode(episode, task)
        assert result.success is False
        assert result.criteria[0].passed is False

    def test_success_criterion_passes_when_all_feedbacks_successful(self) -> None:
        task = _task(criteria=["all steps successful"])
        episode = _episode(
            steps=[
                _step(step_index=0, feedback_success=True),
                _step(step_index=1, feedback_success=True),
            ]
        )
        assert evaluate_episode(episode, task).success is True

    def test_success_criterion_fails_when_any_feedback_unsuccessful(self) -> None:
        task = _task(criteria=["all steps successful"])
        episode = _episode(
            steps=[
                _step(step_index=0, feedback_success=True),
                _step(step_index=1, feedback_success=False),
            ]
        )
        assert evaluate_episode(episode, task).success is False

    def test_grasp_criterion_passes_for_pick_only_stop(self) -> None:
        task = _task(
            task_id="pick-task",
            mission_type=MissionType.PICK_ONLY,
            criteria=["object is grasped"],
        )
        episode = _episode(
            task_id="pick-task",
            steps=[_step(task=task)],
            termination_reason="stop",
        )
        assert evaluate_episode(episode, task).success is True

    def test_grasp_criterion_fails_for_non_pick_mission(self) -> None:
        task = _task(
            task_id="move-task",
            mission_type=MissionType.MOVE_ASIDE,
            criteria=["object is grasped"],
        )
        episode = _episode(
            task_id="move-task",
            steps=[_step(task=task, mission_type=MissionType.MOVE_ASIDE)],
            termination_reason="stop",
        )
        assert evaluate_episode(episode, task).success is False

    def test_place_criterion_passes_for_pick_and_place_stop(self) -> None:
        task = _task(
            task_id="place-task",
            mission_type=MissionType.PICK_AND_PLACE,
            criteria=["object is placed"],
        )
        episode = _episode(
            task_id="place-task",
            steps=[_step(task=task)],
            termination_reason="stop",
        )
        assert evaluate_episode(episode, task).success is True

    def test_place_criterion_fails_for_pick_only(self) -> None:
        task = _task(
            task_id="pick-task",
            mission_type=MissionType.PICK_ONLY,
            criteria=["object is placed"],
        )
        episode = _episode(
            task_id="pick-task",
            steps=[_step(task=task, mission_type=MissionType.PICK_ONLY)],
            termination_reason="stop",
        )
        assert evaluate_episode(episode, task).success is False

    def test_gripper_open_criterion_passes(self) -> None:
        task = _task(criteria=["gripper is open"])
        episode = _episode(steps=[_step(gripper_state=0.8)])
        assert evaluate_episode(episode, task).success is True

    def test_gripper_open_criterion_fails_when_closed(self) -> None:
        task = _task(criteria=["gripper is open"])
        episode = _episode(steps=[_step(gripper_state=0.1)])
        assert evaluate_episode(episode, task).success is False

    def test_unknown_criterion_logs_warning_and_fails(self, warning_log) -> None:
        task = _task(criteria=["does a backflip"])
        episode = _episode(steps=[_step()])
        result = evaluate_episode(episode, task)
        assert result.success is False
        assert result.criteria[0].passed is False
        assert any("Unrecognized success criterion" in msg for msg in warning_log)

    def test_all_criteria_must_pass(self) -> None:
        task = _task(criteria=["episode stopped", "gripper is open"])
        episode = _episode(
            steps=[_step(gripper_state=0.8)],
            termination_reason="stop",
        )
        assert evaluate_episode(episode, task).success is True

        episode_fail = _episode(
            steps=[_step(gripper_state=0.1)],
            termination_reason="stop",
        )
        result = evaluate_episode(episode_fail, task)
        assert result.success is False
        assert any(
            criterion.criterion == "gripper is open" and not criterion.passed
            for criterion in result.criteria
        )


class TestEvaluateTasks:
    def test_episodes_matched_to_correct_tasks(self) -> None:
        pick_task = _task(
            task_id="pick",
            mission_type=MissionType.PICK_ONLY,
            criteria=["object is grasped"],
        )
        place_task = _task(
            task_id="place",
            mission_type=MissionType.PICK_AND_PLACE,
            criteria=["object is placed"],
        )
        pick_episode = _episode(
            episode_id="ep-pick",
            task_id="pick",
            steps=[_step(task=pick_task, mission_type=MissionType.PICK_ONLY)],
            termination_reason="stop",
        )
        place_episode = _episode(
            episode_id="ep-place",
            task_id="place",
            steps=[_step(task=place_task)],
            termination_reason="stop",
        )

        result = evaluate_tasks([pick_task, place_task], [pick_episode, place_episode])
        assert result.total == 2
        assert result.success_count == 2
        by_id = {ev.episode_id: ev for ev in result.per_episode}
        assert by_id["ep-pick"].task_id == "pick"
        assert by_id["ep-place"].task_id == "place"

    def test_aggregate_metrics(self) -> None:
        task = _task(criteria=["episode stopped"])
        episodes = [
            _episode(
                episode_id="ep-1",
                steps=[_step(step_index=0), _step(step_index=1)],
                termination_reason="stop",
            ),
            _episode(
                episode_id="ep-2",
                steps=[_step(step_index=0)],
                termination_reason="timeout",
            ),
        ]
        result = evaluate_tasks([task], episodes)
        assert result.total == 2
        assert result.success_count == 1
        assert result.success_rate == 0.5
        assert result.average_steps == 1.5
        assert result.min_steps == 1
        assert result.max_steps == 2
        assert result.termination_reason_counts == {"stop": 1, "timeout": 1}

    def test_empty_episode_list(self) -> None:
        task = _task(criteria=["episode stopped"])
        result = evaluate_tasks([task], [])
        assert result == EvaluationResult(
            total=0,
            success_count=0,
            success_rate=0.0,
            average_steps=0.0,
            min_steps=0,
            max_steps=0,
            termination_reason_counts={},
            per_episode=[],
        )

    def test_missing_termination_reason_defaults_to_unknown(self) -> None:
        task = _task(criteria=["episode stopped"])
        episode = _episode(
            steps=[_step()],
            termination_reason=None,
            metadata={},
        )
        result = evaluate_tasks([task], [episode])
        assert result.per_episode[0].termination_reason == "unknown"
        assert result.termination_reason_counts == {"unknown": 1}
        assert result.success_count == 0

    def test_task_not_found_warns_and_marks_failure(self, warning_log) -> None:
        task = _task(task_id="known", criteria=["episode stopped"])
        episode = _episode(
            episode_id="orphan",
            task_id="unknown",
            steps=[_step()],
            termination_reason="stop",
        )
        result = evaluate_tasks([task], [episode])
        assert result.total == 1
        assert result.success_count == 0
        assert result.per_episode[0].success is False
        assert result.per_episode[0].criteria == []
        assert any("unknown task_id" in msg for msg in warning_log)

    def test_custom_checker_overrides_default(self) -> None:
        class AlwaysTrueChecker:
            def check(self, criterion: str, episode: Episode, task: TaskDefinition) -> bool:
                return True

        task = _task(criteria=["anything"])
        episode = _episode(steps=[], termination_reason="timeout")
        result = evaluate_episode(episode, task, checker=AlwaysTrueChecker())
        assert result.success is True
        assert result.criteria == [
            CriterionResult(criterion="anything", passed=True, reason="passed")
        ]


class TestPublicAPI:
    def test_imports_exposed(self) -> None:
        from evaluation import (
            CriterionResult,
            EpisodeEvaluation,
            EvaluationResult,
            KeywordSuccessChecker,
            SuccessChecker,
            evaluate_episode,
            evaluate_tasks,
        )

        assert evaluate_tasks is not None
        assert SuccessChecker is not None
        assert KeywordSuccessChecker is not None


def _agent_step(
    step_index: int,
    *,
    risk: bool = False,
    timestamp: float | None = None,
    feedback_success: bool | None = True,
) -> EpisodeStep:
    """Step carrying an ExecutorAgentOutput so risk/timestamp fields exist."""
    feedback = None if feedback_success is None else Feedback(success=feedback_success)
    return EpisodeStep(
        step_index=step_index,
        task=_task(),
        rethinker_output=RethinkerOutput(
            mission_type=MissionType.PICK_AND_PLACE,
            reasoning="mock reasoning",
        ),
        executor_output=ExecutorAgentOutput(
            step_index=step_index,
            joint_angles=[0.0],
            gripper_state=0.5,
            risk=risk,
            timestamp=timestamp,
            success=True,
        ),
        feedback=feedback,
    )


class TestRiskyActionMetric:
    def test_counts_steps_whose_executor_output_flags_risk(self) -> None:
        episode = _episode(
            steps=[
                _agent_step(0, risk=True),
                _agent_step(1, risk=False),
                _agent_step(2, risk=True),
            ]
        )
        result = evaluate_episode(episode, _task())
        assert result.risky_actions == 2

    def test_base_executor_output_without_risk_field_counts_zero(self) -> None:
        episode = _episode(steps=[_step(step_index=0), _step(step_index=1)])
        assert evaluate_episode(episode, _task()).risky_actions == 0

    def test_missing_executor_output_counts_zero(self) -> None:
        step = EpisodeStep(
            step_index=0,
            task=_task(),
            rethinker_output=RethinkerOutput(
                mission_type=MissionType.STOP,
                reasoning="stop immediately",
            ),
            executor_output=None,
        )
        episode = _episode(steps=[step])
        assert evaluate_episode(episode, _task()).risky_actions == 0


class TestReflectionMetric:
    def test_counts_rounds_following_feedback(self) -> None:
        steps = [
            _agent_step(0, feedback_success=True),
            _agent_step(1, feedback_success=None),
            _agent_step(2, feedback_success=True),
        ]
        episode = _episode(steps=steps)
        result = evaluate_episode(episode, _task())
        # step 1 follows feedback -> reflection; step 2 follows no feedback -> not.
        assert result.reflections == 1

    def test_single_step_episode_has_no_reflections(self) -> None:
        episode = _episode(steps=[_agent_step(0)])
        assert evaluate_episode(episode, _task()).reflections == 0

    def test_all_rounds_after_first_count_when_feedback_present(self) -> None:
        episode = _episode(steps=[_agent_step(i) for i in range(3)])
        assert evaluate_episode(episode, _task()).reflections == 2


class TestRuntimeMetric:
    def test_runtime_from_metadata(self) -> None:
        episode = _episode(steps=[_agent_step(0)], metadata={"runtime_seconds": 12.5})
        assert evaluate_episode(episode, _task()).runtime_seconds == 12.5

    def test_runtime_from_executor_timestamps(self) -> None:
        episode = _episode(
            steps=[
                _agent_step(0, timestamp=100.0),
                _agent_step(1, timestamp=104.5),
            ]
        )
        assert evaluate_episode(episode, _task()).runtime_seconds == 4.5

    def test_runtime_none_when_not_derivable(self) -> None:
        episode = _episode(steps=[_agent_step(0)])
        assert evaluate_episode(episode, _task()).runtime_seconds is None

    def test_metadata_runtime_takes_precedence_over_timestamps(self) -> None:
        episode = _episode(
            steps=[
                _agent_step(0, timestamp=1.0),
                _agent_step(1, timestamp=2.0),
            ],
            metadata={"runtime_seconds": 9.0},
        )
        assert evaluate_episode(episode, _task()).runtime_seconds == 9.0


class TestAggregateMetricExtensions:
    def test_failure_count_and_action_aggregates(self) -> None:
        task = _task(criteria=["episode stopped"])
        ep_ok = _episode(
            episode_id="ok",
            steps=[_agent_step(0, risk=True), _agent_step(1)],
            termination_reason="stop",
        )
        ep_bad = _episode(
            episode_id="bad",
            steps=[_agent_step(0)],
            termination_reason="failure",
        )
        result = evaluate_tasks([task], [ep_ok, ep_bad])
        assert result.failure_count == 1
        assert result.risky_action_count == 1
        assert result.reflection_count == 1

    def test_runtime_aggregates_over_available_episodes(self) -> None:
        task = _task()
        ep1 = _episode(
            episode_id="e1", steps=[_agent_step(0)], metadata={"runtime_seconds": 10.0}
        )
        ep2 = _episode(
            episode_id="e2", steps=[_agent_step(0)], metadata={"runtime_seconds": 20.0}
        )
        ep3 = _episode(episode_id="e3", steps=[_agent_step(0)])
        result = evaluate_tasks([task], [ep1, ep2, ep3])
        assert result.average_runtime_seconds == 15.0
        assert result.min_runtime_seconds == 10.0
        assert result.max_runtime_seconds == 20.0

    def test_runtime_aggregates_none_without_data(self) -> None:
        task = _task()
        result = evaluate_tasks([task], [_episode(steps=[_step()])])
        assert result.average_runtime_seconds is None
        assert result.min_runtime_seconds is None
        assert result.max_runtime_seconds is None
