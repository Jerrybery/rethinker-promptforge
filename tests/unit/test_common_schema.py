"""Unit tests for shared Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


@pytest.fixture
def task() -> TaskUnit:
    return TaskUnit(
        id="task-001",
        instruction="pick the red mug and place it on the saucer",
        mission_type=MissionType.PICK_AND_PLACE,
        objects=["red_mug", "saucer"],
    )


@pytest.fixture
def rethinker_output() -> RethinkerOutput:
    return RethinkerOutput(
        mission_type=MissionType.PICK_AND_PLACE,
        reasoning="Target object is visible and reachable.",
        target_object="red_mug",
        target_container="saucer",
        arm_hint="right",
    )


@pytest.fixture
def planner_output() -> PlannerOutput:
    return PlannerOutput(
        plan_id="plan-001",
        mission=MissionType.PICK_AND_PLACE,
        pick="red_mug",
        place="saucer",
        trajectory_name="top_down_approach",
        waypoints=["approach", "grasp", "lift", "place"],
        gripper_action="open",
    )


@pytest.fixture
def executor_output() -> ExecutorOutput:
    return ExecutorOutput(
        step_index=0,
        joint_angles=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        gripper_state=0.8,
    )


@pytest.fixture
def feedback() -> Feedback:
    return Feedback(success=True, observation="mug grasped firmly", reward=1.0)


class TestMissionType:
    def test_all_values_present(self) -> None:
        assert MissionType.PICK_AND_PLACE.value == "PICK_AND_PLACE"
        assert MissionType.PICK_ONLY.value == "PICK_ONLY"
        assert MissionType.MOVE_ASIDE.value == "MOVE_ASIDE"
        assert MissionType.REOBSERVE.value == "REOBSERVE"
        assert MissionType.STOP.value == "STOP"


class TestRethinkerOutput:
    def test_valid(self) -> None:
        out = RethinkerOutput(
            mission_type=MissionType.MOVE_ASIDE,
            reasoning="Obstacle blocks the target.",
        )
        assert out.mission_type == MissionType.MOVE_ASIDE
        assert out.reasoning == "Obstacle blocks the target."

    def test_missing_reasoning_raises(self) -> None:
        with pytest.raises(ValidationError):
            RethinkerOutput(mission_type=MissionType.STOP)

    def test_joint_angles_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RethinkerOutput(
                mission_type=MissionType.PICK_ONLY,
                reasoning="Should not carry joint angles.",
                joint_angles=[0.0, 0.0, 0.0],
            )

    def test_left_arm_joint_angles_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RethinkerOutput(
                mission_type=MissionType.PICK_ONLY,
                reasoning="Should not carry joint angles.",
                left_arm_joint_angles=[0.0, 0.0, 0.0],
            )


class TestPlannerOutput:
    def test_valid(self) -> None:
        out = PlannerOutput(
            plan_id="plan-1",
            mission=MissionType.PICK_AND_PLACE,
            pick="red_mug",
            place="saucer",
            waypoints=["pre_grasp", "grasp"],
        )
        assert out.plan_id == "plan-1"
        assert out.mission == MissionType.PICK_AND_PLACE
        assert out.pick == "red_mug"
        assert out.place == "saucer"
        assert out.waypoints == ["pre_grasp", "grasp"]

    def test_missing_plan_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlannerOutput()

    @pytest.mark.parametrize(
        "bad_field,bad_value",
        [
            ("grasp_coordinates", [0.1, 0.2, 0.3]),
            ("place_coordinates", [0.4, 0.5, 0.6]),
            ("grasp_position", {"x": 0.1, "y": 0.2, "z": 0.3}),
            ("place_pose", [0.0] * 7),
        ],
    )
    def test_grasp_place_coordinates_rejected(
        self, bad_field: str, bad_value: object
    ) -> None:
        data = {"plan_id": "plan-bad", bad_field: bad_value}
        with pytest.raises(ValidationError):
            PlannerOutput(**data)


class TestExecutorOutput:
    def test_valid(self, executor_output: ExecutorOutput) -> None:
        assert executor_output.step_index == 0
        assert len(executor_output.joint_angles) == 6

    def test_negative_step_index_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExecutorOutput(
                step_index=-1,
                joint_angles=[0.0],
                gripper_state=0.5,
            )

    def test_gripper_state_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExecutorOutput(
                step_index=0,
                joint_angles=[0.0],
                gripper_state=1.5,
            )


class TestTaskUnit:
    def test_valid(self, task: TaskUnit) -> None:
        assert task.id == "task-001"
        assert task.mission_type == MissionType.PICK_AND_PLACE

    def test_empty_instruction_raises(self) -> None:
        with pytest.raises(ValidationError):
            TaskUnit(
                id="task-bad",
                instruction="",
                mission_type=MissionType.STOP,
            )


class TestFeedback:
    def test_valid(self, feedback: Feedback) -> None:
        assert feedback.success is True
        assert feedback.reward == 1.0


class TestEpisodeStep:
    def test_valid(
        self,
        task: TaskUnit,
        rethinker_output: RethinkerOutput,
        planner_output: PlannerOutput,
        executor_output: ExecutorOutput,
        feedback: Feedback,
    ) -> None:
        step = EpisodeStep(
            step_index=0,
            task=task,
            rethinker_output=rethinker_output,
            planner_output=planner_output,
            executor_output=executor_output,
            feedback=feedback,
        )
        assert step.task.id == "task-001"
        assert step.rethinker_output.mission_type == MissionType.PICK_AND_PLACE


class TestEpisode:
    def test_valid_empty(self) -> None:
        ep = Episode(id="ep-001", task_id="task-001")
        assert ep.steps == []

    def test_valid_with_step(
        self,
        task: TaskUnit,
        rethinker_output: RethinkerOutput,
    ) -> None:
        step = EpisodeStep(
            step_index=0,
            task=task,
            rethinker_output=rethinker_output,
        )
        ep = Episode(id="ep-002", task_id="task-001", steps=[step])
        assert len(ep.steps) == 1

    def test_episode_is_frozen(self) -> None:
        ep = Episode(id="ep-003", task_id="task-001")
        with pytest.raises(ValidationError):
            ep.task_id = "mutated"


class TestRethinkerOutputHypothesis:
    def test_hidden_hypothesis_and_risk_note_valid(self) -> None:
        out = RethinkerOutput(
            mission_type=MissionType.PICK_AND_PLACE,
            reasoning="Mug visible; handle may be hidden.",
            target_object="mug",
            target_container="saucer",
            hidden_hypothesis="The mug handle may be occluded by the box.",
            risk_note="Grasp may slip on the smooth rim.",
        )
        assert out.hidden_hypothesis == "The mug handle may be occluded by the box."
        assert out.risk_note == "Grasp may slip on the smooth rim."

    def test_hypothesis_fields_default_to_none(self) -> None:
        out = RethinkerOutput(
            mission_type=MissionType.STOP,
            reasoning="Task complete.",
        )
        assert out.hidden_hypothesis is None
        assert out.risk_note is None
