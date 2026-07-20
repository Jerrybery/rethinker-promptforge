"""Integration test: full forge training epoch with real forge components.

Wires SimEnv (fake duck-typed RoboTwin env), the real ForgePlannerAgent
(mocked VLLM client), the real OptimizerLLM (scripted mock client), the real
ForgePromptRegistry, and the real PromptValidator through ForgeRunner. One
epoch runs rollout -> (no critic) -> optimizer -> candidate -> validate ->
reject, and all run artifacts are checked on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from forge.env import SimEnv
from forge.optimizer import OptimizerLLM
from forge.planner_agent import ForgePlannerAgent
from forge.registry import ForgePromptRegistry
from forge.runner import ForgeRunner
from tasks.schema import TaskDefinition

SEED_PROMPT = "# Integration Seed Prompt\n\n## Rules\n\n- Follow the mission.\n"

STOP_JSON = json.dumps(
    {"plan_id": "p-stop", "mission": "STOP", "pick": "none"}
)

EDIT_JSON = json.dumps(
    [
        {
            "target_agent": "planner",
            "edit_type": "add",
            "location": "Rules",
            "new_text": "- Stop promptly once the task is confirmed complete.",
            "reason": "efficiency (scripted)",
        }
    ]
)


class FakeRobot:
    def get_right_gripper_val(self) -> float:
        return 1.0

    def get_left_gripper_val(self) -> float:
        return 1.0


class FakeRoboTwinEnv:
    """Duck-typed fake matching the RoboTwin base task API used by SimEnv."""

    def __init__(self) -> None:
        self.robot = FakeRobot()
        self.stopped = False

    def get_obs(self) -> dict[str, Any]:
        rgb = np.full((8, 8, 3), 17, dtype=np.uint8)
        return {"observation": {"head_camera": {"rgb": rgb}}}

    def get_arm_pose(self, arm_tag: str = "right") -> list[float]:
        return [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]

    def move_to_pose(self, arm_tag: str, pose: list[float]) -> list[str]:
        return ["action"]

    def open_gripper(self, arm_tag: str) -> list[str]:
        return ["action"]

    def close_gripper(self, arm_tag: str) -> list[str]:
        return ["action"]

    def move(self, actions: list[Any]) -> bool:
        return True

    def check_success(self) -> bool:
        return False

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


def make_task(task_id: str) -> TaskDefinition:
    return TaskDefinition(
        id=task_id,
        instruction="Pick up the mock object.",
        mission_type="PICK_ONLY",
        objects=["mock_object"],
        initial_scene={"task_name": "fake_task"},
        success_criteria=[],
        metadata={
            "robottwin_task_name": "fake_task",
            "robottwin_task_config": "demo_clean",
        },
    )


def test_full_forge_epoch_end_to_end(tmp_path: Path) -> None:
    env = SimEnv(env_factory=make_factory(FakeRoboTwinEnv()))

    planner_client = MagicMock()
    planner_client.chat = MagicMock(return_value=STOP_JSON)
    planner = ForgePlannerAgent(vllm_client=planner_client)

    optimizer_client = MagicMock()
    optimizer_client.chat = MagicMock(return_value=EDIT_JSON)
    optimizer = OptimizerLLM(
        client=optimizer_client, target_agent="planner", budget_chars=400
    )

    registry = ForgePromptRegistry(tmp_path / "registry")
    runner = ForgeRunner(
        registry=registry,
        env=env,
        planner=planner,
        optimizer=optimizer,
        critic=None,
        train_tasks=[make_task("forge-train-0")],
        val_tasks=[make_task("forge-val-0")],
        run_dir=tmp_path / "run",
        initial_prompt_text=SEED_PROMPT,
        max_rounds=3,
    )
    log = runner.run(1)

    # Registry lineage: seed champion + rejected candidate with parentage.
    versions = registry.history("planner")
    assert len(versions) == 2
    seed, candidate = versions
    assert seed.status == "best"
    assert seed.edit.source == "hand"
    assert candidate.status == "rejected"
    assert candidate.parent_version == seed.version_id
    assert candidate.edit.source == "optimizer"
    assert candidate.validation is not None
    assert candidate.validation.metrics["success_rate"] == 0.0

    epoch = log.epochs[0]
    assert epoch.accepted is False
    assert epoch.candidate_version_id == candidate.version_id
    assert log.final_best_version_id == seed.version_id

    # The real optimizer was consulted with the seed text and no rejects.
    optimizer_client.chat.assert_called_once()
    prompt_sent = optimizer_client.chat.call_args[0][0][0]["content"]
    assert SEED_PROMPT in prompt_sent

    # Artifacts.
    run_dir = tmp_path / "run"
    assert (run_dir / "forge_log.json").exists()
    assert (run_dir / "epochs" / "epoch_000.json").exists()
    metrics_lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics_lines) == 1
    assert json.loads(metrics_lines[0])["accepted"] is False
    snapshot = run_dir / "prompts" / "best_epoch_000.md"
    assert snapshot.read_text() == SEED_PROMPT

    episode_dirs = list((run_dir / "recordings" / "epoch_000").iterdir())
    assert len(episode_dirs) == 1
    recording_meta = json.loads(
        (episode_dirs[0] / "metadata.json").read_text()
    )
    # STOP on the first step -> reset frame + one step frame.
    assert recording_meta["frame_count"] == 2
    assert (episode_dirs[0] / "video.mp4").exists()

    # Registry persisted on disk inside the run dir.
    assert (tmp_path / "registry" / "registry.json").exists()
