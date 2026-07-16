#!/usr/bin/env python3
"""Hello-loop runner: closed-loop runner against a real RoboTwin task.

This script wires stub perception/reasoning agents so that no VLLM endpoint or
real DINO model is required. It loads a task definition from the catalogue,
builds the corresponding RoboTwin environment, wraps it in ``RobotInterface``,
and runs ``ClosedLoopRunner``.

For simulator-only runs the ``RoboTwinBackend`` is constructed with
``strict_stop=False`` because the wrapped RoboTwin environment does not expose
a ``stop()`` or ``halt()`` method.

Usage::

    PYTHONPATH=src python scripts/run_hello_loop.py \\
        --task-id hello-place-a2b-right
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from common.schema import MissionType, PlannerOutput, RethinkerOutput
from executor.agent import ExecutorAgent
from executor.primitives import PrimitiveLibrary
from perception.dino_client import DINOClient
from rethinker.runner import ClosedLoopRunner
from robot.interface import RoboTwinBackend, RobotInterface
from robot.robottwin_env import make_robottwin_env
from tasks.loader import get_task_by_id, load_task_definitions


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_PATH = REPO_ROOT / "data" / "tasks" / "hello_tasks.yaml"
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robot.yaml"
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"


class _SequenceAgent:
    """Generic agent that returns a fixed sequence of outputs."""

    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = outputs
        self._index = 0

    def act(self, **kwargs: Any) -> Any:
        output = self._outputs[self._index]
        self._index = min(self._index + 1, len(self._outputs) - 1)
        return output


def _build_stub_agents() -> dict[str, Any]:
    """Return deterministic stub agents for the hello-loop."""
    rethinker_outputs = [
        RethinkerOutput(
            mission_type=MissionType.PICK_AND_PLACE,
            reasoning="Mock object is visible; pick it and place at target.",
            target_object="mock_object",
            target_container="target_object",
            arm_hint="right",
        ),
        RethinkerOutput(
            mission_type=MissionType.STOP,
            reasoning="Finished the hello-loop iteration.",
        ),
    ]
    planner_outputs = [
        PlannerOutput(
            plan_id="hello-plan-0",
            mission=MissionType.PICK_AND_PLACE,
            pick="mock_object",
            place="target_object",
        ),
        PlannerOutput(
            plan_id="hello-plan-1",
            mission=MissionType.STOP,
            pick="none",
        ),
    ]
    return {
        "rethinker": _SequenceAgent(rethinker_outputs),
        "planner": _SequenceAgent(planner_outputs),
    }


def _build_executor(robot: RobotInterface, dino: DINOClient) -> ExecutorAgent:
    """Build a real executor that exercises the robot primitives."""
    primitives = PrimitiveLibrary(robot=robot, dino=dino)
    return ExecutorAgent(primitives=primitives, dino=dino, robot=robot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a RoboTwin hello-loop episode.")
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="Task id from the task catalogue.",
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help="Path to the task catalogue YAML.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_ROBOT_CONFIG,
        help="Path to the robot/perception YAML config.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum closed-loop rounds.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the mock robot backend instead of RoboTwin.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to RoboTwin setup_demo.",
    )
    args = parser.parse_args(argv)

    if not args.tasks_path.exists():
        logger.error("Task catalogue not found: {}", args.tasks_path)
        return 1

    tasks = load_task_definitions(args.tasks_path)
    task = get_task_by_id(tasks, args.task_id)

    if args.mock:
        robot = RobotInterface(config_path=args.config_path, mock=True)
    else:
        initial_scene = task.initial_scene or {}
        robottwin_task_name = initial_scene.get(
            "task_name"
        ) or task.metadata.get("robottwin_task_name") if task.metadata else None
        if robottwin_task_name is None:
            logger.error(
                "Task {} does not specify a RoboTwin task name "
                "(expected initial_scene.task_name or metadata.robottwin_task_name).",
                task.id,
            )
            return 1

        robottwin_task_config = (
            task.metadata.get("robottwin_task_config", "demo_clean")
            if task.metadata
            else "demo_clean"
        )

        logger.info(
            "Building RoboTwin env: task_name={}, task_config={}",
            robottwin_task_name,
            robottwin_task_config,
        )
        env = make_robottwin_env(
            task_name=robottwin_task_name,
            task_config_name=robottwin_task_config,
            repo_root=REPO_ROOT,
            seed=args.seed,
            render_freq=0,
            overrides=initial_scene,
        )
        backend = RoboTwinBackend(config=task.initial_scene, env=env, strict_stop=False)
        robot = RobotInterface(config_path=args.config_path, backend=backend)

    logger.info("Reading initial observation...")
    initial_state = robot.read_state()
    logger.info("Initial state: {}", initial_state)

    dino = DINOClient(config_path=DEFAULT_MODELS_CONFIG, mode="mock")

    agents = _build_stub_agents()
    agents["executor"] = _build_executor(robot=robot, dino=dino)

    runner = ClosedLoopRunner(
        task=task,
        config_path=args.config_path,
        agents=agents,
        robot=robot,
        dino=dino,
        max_rounds=args.max_rounds,
    )
    episode = runner.run()

    logger.info(
        "Episode {} finished with {} step(s). Log: {}",
        episode.id,
        len(episode.steps),
        runner.log_path,
    )
    print(runner.log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
