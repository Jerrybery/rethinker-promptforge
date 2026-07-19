"""Unit tests for the ablation baselines (Task 2.2).

Covers the three baselines: Monolithic Planner, No-Hidden-Hypothesis, and
No-Reflection. All LLM calls are mocked via ``VLLMClient`` doubles.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from baselines.monolithic_planner import (
    MonolithicPlannerAgent,
    PassThroughPlanner,
    build_monolithic_agents,
)
from baselines.no_hidden_hypothesis import NoHiddenHypothesisRethinker
from baselines.no_reflection import (
    NoReflectionPlanner,
    NoReflectionRethinker,
    build_no_reflection_agents,
)
from baselines.schema import MonolithicDecision
from common.schema import (
    DetectedObject,
    Feedback,
    MissionType,
    PlannerOutput,
    RethinkerOutput,
)
from planner.agent import PlannerAgent
from planner.memory import PlannerMemory
from rethinker.agent import RethinkerAgent
from rethinker.memory import RethinkerMemory
from rethinker.prompts.registry import PromptRegistry


@pytest.fixture
def rgb_image() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def detections() -> list[DetectedObject]:
    return [
        DetectedObject(label="mug", bbox=[10.0, 20.0, 40.0, 50.0], confidence=0.92),
        DetectedObject(label="saucer", bbox=[5.0, 5.0, 30.0, 30.0], confidence=0.88),
    ]


@pytest.fixture
def decision_payload() -> dict:
    return {
        "mission_type": "PICK_AND_PLACE",
        "reasoning": "Mug is visible on the table; place it on the saucer.",
        "target_object": "mug",
        "target_container": "saucer",
        "arm_hint": "right",
        "hidden_hypothesis": "The mug handle may be occluded by the saucer.",
        "risk_note": "Saucer edge is close to the mug; approach carefully.",
        "pick": "mug",
        "place": "saucer",
    }


def _make_client(*responses: str) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock(side_effect=list(responses))
    return client


class TestMonolithicDecision:
    def test_valid(self, decision_payload: dict) -> None:
        decision = MonolithicDecision.model_validate(decision_payload)
        assert decision.mission_type is MissionType.PICK_AND_PLACE
        assert decision.pick == "mug"
        assert decision.place == "saucer"

    def test_missing_pick_raises(self, decision_payload: dict) -> None:
        payload = {k: v for k, v in decision_payload.items() if k != "pick"}
        with pytest.raises(ValidationError):
            MonolithicDecision.model_validate(payload)

    def test_extra_keys_rejected(self, decision_payload: dict) -> None:
        payload = {**decision_payload, "grasp_point": [0.1, 0.2, 0.3]}
        with pytest.raises(ValidationError):
            MonolithicDecision.model_validate(payload)

    def test_to_rethinker_output(self, decision_payload: dict) -> None:
        decision = MonolithicDecision.model_validate(decision_payload)
        out = decision.to_rethinker_output()
        assert isinstance(out, RethinkerOutput)
        assert out.mission_type is MissionType.PICK_AND_PLACE
        assert out.hidden_hypothesis == decision.hidden_hypothesis
        assert out.risk_note == decision.risk_note

    def test_to_planner_output(self, decision_payload: dict) -> None:
        decision = MonolithicDecision.model_validate(decision_payload)
        plan = decision.to_planner_output(plan_id="mono-test")
        assert isinstance(plan, PlannerOutput)
        assert plan.plan_id == "mono-test"
        assert plan.mission is MissionType.PICK_AND_PLACE
        assert plan.pick == "mug"
        assert plan.place == "saucer"


class TestMonolithicPlannerAgent:
    def test_act_returns_rethinker_output_and_stores_decision(
        self,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
        decision_payload: dict,
    ) -> None:
        agent = MonolithicPlannerAgent(
            vllm_client=_make_client(json.dumps(decision_payload))
        )
        out = agent.act(
            task_goal="Put the mug on the saucer.",
            rgb_image=rgb_image,
            detections=detections,
        )
        assert isinstance(out, RethinkerOutput)
        assert out.mission_type is MissionType.PICK_AND_PLACE
        assert out.hidden_hypothesis == decision_payload["hidden_hypothesis"]
        # One single LLM call that includes the raw image.
        assert agent.vllm_client.chat.call_count == 1
        assert agent.vllm_client.chat.call_args.kwargs["images"] == [rgb_image]
        user_text = agent.vllm_client.chat.call_args.args[0][1]["content"]
        assert "Put the mug on the saucer." in user_text
        assert agent.last_decision is not None
        assert agent.last_decision.pick == "mug"

    def test_act_invalid_json_raises(
        self,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
    ) -> None:
        agent = MonolithicPlannerAgent(vllm_client=_make_client("not json"))
        with pytest.raises(ValueError, match="could not parse"):
            agent.act(
                task_goal="Do something.",
                rgb_image=rgb_image,
                detections=detections,
            )


class TestPassThroughPlanner:
    def _run_monolithic(
        self,
        decision_payload: dict,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
    ) -> tuple[MonolithicPlannerAgent, RethinkerOutput]:
        agent = MonolithicPlannerAgent(
            vllm_client=_make_client(json.dumps(decision_payload))
        )
        out = agent.act(
            task_goal="Put the mug on the saucer.",
            rgb_image=rgb_image,
            detections=detections,
        )
        return agent, out

    def test_translates_stored_decision(
        self,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
        decision_payload: dict,
    ) -> None:
        agent, rethinker_out = self._run_monolithic(
            decision_payload, rgb_image, detections
        )
        planner = PassThroughPlanner(agent)
        plan = planner.act(
            rethinker_output=rethinker_out,
            dino_labels=["mug", "saucer"],
            action_library=["pick", "place"],
        )
        assert isinstance(plan, PlannerOutput)
        assert plan.mission is MissionType.PICK_AND_PLACE
        assert plan.pick == "mug"
        assert plan.place == "saucer"
        assert plan.plan_id

    def test_requires_prior_decision(self) -> None:
        planner = PassThroughPlanner(
            MonolithicPlannerAgent(vllm_client=_make_client("{}"))
        )
        with pytest.raises(ValueError, match="decision"):
            planner.act(
                rethinker_output=RethinkerOutput(
                    mission_type=MissionType.STOP, reasoning="done"
                ),
                dino_labels=["mug"],
            )

    def test_rejects_mismatched_rethinker_output(
        self,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
        decision_payload: dict,
    ) -> None:
        agent, _ = self._run_monolithic(decision_payload, rgb_image, detections)
        planner = PassThroughPlanner(agent)
        foreign = RethinkerOutput(
            mission_type=MissionType.STOP, reasoning="different output"
        )
        with pytest.raises(ValueError, match="does not match"):
            planner.act(rethinker_output=foreign, dino_labels=["mug", "saucer"])

    def test_rejects_unknown_pick_label(
        self,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
        decision_payload: dict,
    ) -> None:
        payload = {**decision_payload, "pick": "cup"}
        agent, rethinker_out = self._run_monolithic(payload, rgb_image, detections)
        planner = PassThroughPlanner(agent)
        with pytest.raises(ValueError, match="Pick label"):
            planner.act(
                rethinker_output=rethinker_out,
                dino_labels=["mug", "saucer"],
            )

    def test_build_monolithic_agents_factory(self) -> None:
        client = _make_client("{}")
        agents = build_monolithic_agents(vllm_client=client)
        assert set(agents) == {"rethinker", "planner"}
        assert isinstance(agents["rethinker"], MonolithicPlannerAgent)
        assert isinstance(agents["planner"], PassThroughPlanner)
        assert agents["planner"].monolithic is agents["rethinker"]


class TestNoHiddenHypothesis:
    def test_prompt_variant_excludes_hypothesis_only(self) -> None:
        system_full, user_full = PromptRegistry.load("v1")
        system_nohh, user_nohh = PromptRegistry.load("v1_nohh")
        assert "hidden_hypothesis" in system_full
        assert "hidden_hypothesis" not in system_nohh
        assert "hidden_hypothesis" not in user_nohh
        # The ablation isolates only the hypothesis field; risk_note stays.
        assert "risk_note" in system_nohh
        assert "risk_note" in system_full
        assert "v1_nohh" in PromptRegistry.versions()

    def test_output_validates_without_hypothesis(
        self,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
    ) -> None:
        payload = {
            "mission_type": "PICK_AND_PLACE",
            "reasoning": "Mug visible; place it on the saucer.",
            "target_object": "mug",
            "target_container": "saucer",
            "risk_note": "Saucer edge close to the mug.",
        }
        agent = NoHiddenHypothesisRethinker(
            vllm_client=_make_client(json.dumps(payload))
        )
        assert agent.prompt_version == "v1_nohh"
        assert "hidden_hypothesis" not in agent._system_template
        out = agent.act(
            task_goal="Put the mug on the saucer.",
            rgb_image=rgb_image,
            detections=detections,
        )
        assert isinstance(out, RethinkerOutput)
        assert out.hidden_hypothesis is None
        assert out.risk_note == "Saucer edge close to the mug."


class TestNoReflection:
    def test_rethinker_wrapper_drops_feedback_and_memory(
        self,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
    ) -> None:
        inner = MagicMock()
        inner.act = MagicMock(return_value="sentinel")
        wrapper = NoReflectionRethinker(inner)
        out = wrapper.act(
            task_goal="goal",
            rgb_image=rgb_image,
            detections=detections,
            memory=object(),
            previous_feedback=Feedback(success=False, observation="missed"),
        )
        assert out == "sentinel"
        kwargs = inner.act.call_args.kwargs
        assert kwargs["task_goal"] == "goal"
        assert kwargs["rgb_image"] is rgb_image
        assert kwargs["memory"] is None
        assert kwargs["previous_feedback"] is None

    def test_planner_wrapper_drops_feedback_and_memory(self) -> None:
        inner = MagicMock()
        inner.act = MagicMock(return_value="sentinel")
        wrapper = NoReflectionPlanner(inner)
        rethinker_out = RethinkerOutput(
            mission_type=MissionType.PICK_ONLY,
            reasoning="Pick the mug.",
            target_object="mug",
        )
        out = wrapper.act(
            rethinker_output=rethinker_out,
            dino_labels=["mug"],
            action_library=["pick"],
            memory=object(),
            previous_feedback=Feedback(success=False, observation="missed"),
        )
        assert out == "sentinel"
        kwargs = inner.act.call_args.kwargs
        assert kwargs["rethinker_output"] is rethinker_out
        assert kwargs["dino_labels"] == ["mug"]
        assert kwargs["memory"] is None
        assert kwargs["previous_feedback"] is None

    def test_wrapped_rethinker_prompt_shows_no_memory_across_rounds(
        self,
        rgb_image: np.ndarray,
        detections: list[DetectedObject],
    ) -> None:
        payload = {"mission_type": "REOBSERVE", "reasoning": "Scene unclear."}
        client = _make_client(json.dumps(payload))
        wrapper = NoReflectionRethinker(RethinkerAgent(vllm_client=client))
        memory = RethinkerMemory(capacity=2)
        memory.append(
            round=0,
            scene_token="scene-000",
            query="q0",
            answer=RethinkerOutput(
                mission_type=MissionType.STOP, reasoning="done"
            ),
            feedback=Feedback(success=False, observation="missed grasp"),
        )
        out = wrapper.act(
            task_goal="Put the mug on the saucer.",
            rgb_image=rgb_image,
            detections=detections,
            memory=memory,
            previous_feedback=Feedback(success=False, observation="missed"),
        )
        assert out.mission_type is MissionType.REOBSERVE
        user_text = client.chat.call_args.args[0][1]["content"]
        assert "No prior rounds." in user_text
        assert "missed" not in user_text

    def test_wrapped_planner_prompt_shows_no_memory_across_rounds(self) -> None:
        payload = {
            "plan_id": "plan-nr",
            "mission": "PICK_AND_PLACE",
            "pick": "mug",
            "place": "saucer",
        }
        client = _make_client(json.dumps(payload))
        wrapper = NoReflectionPlanner(PlannerAgent(vllm_client=client))
        memory = PlannerMemory(capacity=2)
        memory.append(
            round=0,
            scene_token="scene-000",
            query="q0",
            answer=PlannerOutput(
                plan_id="plan-old",
                mission=MissionType.PICK_AND_PLACE,
                pick="mug",
                place="saucer",
            ),
            feedback=Feedback(success=False, observation="missed placement"),
        )
        out = wrapper.act(
            rethinker_output=RethinkerOutput(
                mission_type=MissionType.PICK_AND_PLACE,
                reasoning="Pick the mug and place on saucer.",
                target_object="mug",
                target_container="saucer",
            ),
            dino_labels=["mug", "saucer"],
            memory=memory,
            previous_feedback=Feedback(success=False, observation="missed"),
        )
        assert out.plan_id == "plan-nr"
        user_text = client.chat.call_args.args[0][1]["content"]
        assert "No prior rounds." in user_text
        assert "missed" not in user_text

    def test_build_no_reflection_agents_factory(self) -> None:
        rethinker = MagicMock()
        planner = MagicMock()
        agents = build_no_reflection_agents(rethinker=rethinker, planner=planner)
        assert set(agents) == {"rethinker", "planner"}
        assert isinstance(agents["rethinker"], NoReflectionRethinker)
        assert isinstance(agents["planner"], NoReflectionPlanner)
