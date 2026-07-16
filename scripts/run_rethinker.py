#!/usr/bin/env python
"""CLI entry point for running a closed-loop Rethinker episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.schema import MissionType, TaskUnit
from executor.agent import ExecutorAgent
from executor.primitives import PrimitiveLibrary
from perception.dino_client import DINOClient
from planner.agent import PlannerAgent
from rethinker.agent import RethinkerAgent
from rethinker.runner import ClosedLoopRunner
from robot.interface import RobotInterface


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a closed-loop Rethinker episode."
    )
    parser.add_argument(
        "--task-goal",
        required=True,
        help="Natural-language task goal, e.g. 'pick the mug and place it on the saucer'.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the project YAML config (e.g. configs/models.yaml).",
    )
    parser.add_argument(
        "--task-id",
        default="task-cli-001",
        help="Task identifier for the episode.",
    )
    parser.add_argument(
        "--mission-type",
        default=MissionType.PICK_AND_PLACE.value,
        choices=[m.value for m in MissionType],
        help="High-level mission type for the task unit.",
    )
    parser.add_argument(
        "--objects",
        default="",
        help="Comma-separated list of relevant object labels.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the mock robot backend instead of real hardware/simulator.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum closed-loop iterations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    task = TaskUnit(
        id=args.task_id,
        instruction=args.task_goal,
        mission_type=MissionType(args.mission_type),
        objects=objects,
    )

    config_path = Path(args.config_path)
    robot = RobotInterface(config_path=config_path, mock=args.mock)
    dino = DINOClient(config_path=config_path)

    agents = {
        "rethinker": RethinkerAgent(config_path=config_path),
        "planner": PlannerAgent(config_path=config_path),
        "executor": ExecutorAgent(
            primitives=PrimitiveLibrary(robot=robot, dino=dino),
            dino=dino,
            robot=robot,
        ),
    }

    runner = ClosedLoopRunner(
        task=task,
        config_path=config_path,
        agents=agents,
        robot=robot,
        dino=dino,
        max_rounds=args.max_rounds,
    )

    episode = runner.run()
    print(f"Episode {episode.id} finished with {len(episode.steps)} step(s).")
    print(f"Log: {runner.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
