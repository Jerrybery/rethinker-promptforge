#!/usr/bin/env python3
"""Batch evaluation over (method, task, trials) with CSV/JSON summaries.

Runs N trials per (method, task) pair, evaluates each episode with the
evaluation harness, and persists per-trial JSON plus aggregate
``summary.json`` / ``summary.csv`` under
``<out>/YYYYMMDD_HHMMSS/`` (default ``results/real/...``).

Methods are named agent-stack factories: ``full`` (Rethinker + Planner),
``monolithic``, ``no_hidden_hypothesis``, and ``no_reflection``. Real runs
wire :class:`rethinker.runner.ClosedLoopRunner` behind a thin producer
adapter and require the robot/sim stack plus a VLLM endpoint. ``--mock``
uses deterministic stub agents, the mock robot backend, and the mock DINO
client so the harness can be smoke-tested without hardware.

Usage::

    PYTHONPATH=src python scripts/evaluate_real.py --mock \
        --methods full,monolithic --trials 2 --out results/real

    PYTHONPATH=src python scripts/evaluate_real.py \
        --methods full,no_reflection --trials 5 \
        --tasks-path data/tasks/real_tasks.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger

from common.schema import (
    Feedback,
    MissionType,
    PlannerOutput,
    RethinkerOutput,
)
from evaluation.batch import (
    BatchResult,
    EpisodeProducer,
    make_run_dir,
    run_batch,
    write_batch_results,
)
from executor.schema import ExecutorAgentOutput
from perception.dino_client import DINOClient
from rethinker.runner import ClosedLoopRunner
from robot.interface import RobotInterface
from tasks.loader import get_task_by_id, load_task_definitions
from tasks.schema import TaskDefinition


DEFAULT_TASKS_PATH = REPO_ROOT / "data" / "tasks" / "real_tasks.yaml"
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robot.yaml"
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"

METHOD_NAMES = ("full", "monolithic", "no_hidden_hypothesis", "no_reflection")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation over (method, task, trials)."
    )
    parser.add_argument(
        "--methods",
        default="full",
        help=(
            "Comma-separated method names subset of: "
            + ", ".join(METHOD_NAMES)
            + " (default: full)."
        ),
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help="Path to the task catalogue YAML (default: data/tasks/real_tasks.yaml).",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids to evaluate (default: all in catalogue).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Trials per (method, task) pair (default: 1).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/real"),
        help="Base output directory; a YYYYMMDD_HHMMSS run dir is created inside.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_ROBOT_CONFIG,
        help="Robot/runner YAML config path.",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=DEFAULT_MODELS_CONFIG,
        help="Models (VLLM/DINO) YAML config path.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum closed-loop rounds per episode.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use stub agents, mock robot, and mock DINO (no hardware/LLM).",
    )
    return parser.parse_args(argv)


class _SequenceAgent:
    """Generic agent that returns a fixed sequence of outputs."""

    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = outputs
        self._index = 0

    def act(self, **kwargs: Any) -> Any:
        output = self._outputs[self._index]
        self._index = min(self._index + 1, len(self._outputs) - 1)
        return output


def _mock_agent_stack() -> dict[str, Any]:
    """Deterministic stub agent stack for --mock smoke runs."""
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
            reasoning="Mock run finished.",
        ),
    ]
    planner_outputs = [
        PlannerOutput(
            plan_id="mock-plan-0",
            mission=MissionType.PICK_AND_PLACE,
            pick="mock_object",
            place="target_object",
        ),
        PlannerOutput(
            plan_id="mock-plan-1",
            mission=MissionType.STOP,
            pick="none",
        ),
    ]
    executor_outputs = [
        ExecutorAgentOutput(
            step_index=0,
            success=True,
            feedback=Feedback(success=True, observation="mock execution ok"),
        )
    ]
    return {
        "rethinker": _SequenceAgent(rethinker_outputs),
        "planner": _SequenceAgent(planner_outputs),
        "executor": _SequenceAgent(executor_outputs),
    }


def _build_mock_producers(
    method_names: list[str], args: argparse.Namespace
) -> dict[str, EpisodeProducer]:
    """Build mock producers for each method (same stub stack, tagged by name)."""
    robot = RobotInterface(config_path=args.config_path, mock=True)
    dino = DINOClient(config_path=args.models_config, mode="mock")

    def make_producer() -> EpisodeProducer:
        def produce(task: TaskDefinition, trial_index: int):
            runner = ClosedLoopRunner(
                task=task,
                config_path=args.config_path,
                agents=_mock_agent_stack(),
                robot=robot,
                dino=dino,
                max_rounds=args.max_rounds,
            )
            return runner.run()

        return produce

    return {name: make_producer() for name in method_names}


def _real_agent_builders(args: argparse.Namespace) -> dict[str, Any]:
    """Return per-method factories for the runner's rethinker/planner slots."""
    from baselines.monolithic_planner import build_monolithic_agents
    from baselines.no_hidden_hypothesis import NoHiddenHypothesisRethinker
    from baselines.no_reflection import NoReflectionPlanner, NoReflectionRethinker
    from planner.agent import PlannerAgent
    from rethinker.agent import RethinkerAgent

    def full() -> dict[str, Any]:
        return {
            "rethinker": RethinkerAgent(config_path=args.models_config),
            "planner": PlannerAgent(config_path=args.models_config),
        }

    def monolithic() -> dict[str, Any]:
        return build_monolithic_agents(config_path=args.models_config)

    def no_hidden_hypothesis() -> dict[str, Any]:
        return {
            "rethinker": NoHiddenHypothesisRethinker(config_path=args.models_config),
            "planner": PlannerAgent(config_path=args.models_config),
        }

    def no_reflection() -> dict[str, Any]:
        return {
            "rethinker": NoReflectionRethinker(
                RethinkerAgent(config_path=args.models_config)
            ),
            "planner": NoReflectionPlanner(
                PlannerAgent(config_path=args.models_config)
            ),
        }

    return {
        "full": full,
        "monolithic": monolithic,
        "no_hidden_hypothesis": no_hidden_hypothesis,
        "no_reflection": no_reflection,
    }


def _build_robot_for_task(
    task: TaskDefinition, args: argparse.Namespace
) -> RobotInterface:
    """Build a RoboTwin-backed robot for the task's initial scene."""
    from robot.interface import RoboTwinBackend
    from robot.robottwin_env import make_robottwin_env

    initial_scene = task.initial_scene or {}
    metadata = task.metadata or {}
    task_name = initial_scene.get("task_name") or metadata.get("robottwin_task_name")
    if task_name is None:
        raise ValueError(
            f"Task {task.id} does not specify a RoboTwin task name "
            "(expected initial_scene.task_name or metadata.robottwin_task_name)."
        )
    # Same kwarg promotion as forge.env.SimEnv: task_name/seed/render_freq
    # become explicit kwargs; only the remaining scene keys are overrides.
    # Single source of truth: forge.env._SCENE_KWARG_KEYS (keep in sync).
    from forge.env import _SCENE_KWARG_KEYS

    overrides = {
        k: v for k, v in initial_scene.items() if k not in _SCENE_KWARG_KEYS
    }
    env = make_robottwin_env(
        task_name=task_name,
        task_config_name=metadata.get("robottwin_task_config", "demo_clean"),
        repo_root=REPO_ROOT,
        seed=int(initial_scene.get("seed", 0)),
        render_freq=0,
        overrides=overrides,
    )
    backend = RoboTwinBackend(config=task.initial_scene, env=env, strict_stop=False)
    return RobotInterface(config_path=args.config_path, backend=backend)


def _build_real_producers(
    method_names: list[str], args: argparse.Namespace
) -> dict[str, EpisodeProducer]:
    """Thin adapter wiring ClosedLoopRunner per trial for each method."""
    from executor.agent import ExecutorAgent
    from executor.primitives import PrimitiveLibrary

    builders = _real_agent_builders(args)

    def make_producer(method_name: str) -> EpisodeProducer:
        agent_builder = builders[method_name]

        def produce(task: TaskDefinition, trial_index: int):
            robot = _build_robot_for_task(task, args)
            dino = DINOClient(config_path=args.models_config)
            agents = agent_builder()
            agents["executor"] = ExecutorAgent(
                primitives=PrimitiveLibrary(robot=robot, dino=dino),
                dino=dino,
                robot=robot,
            )
            runner = ClosedLoopRunner(
                task=task,
                config_path=args.config_path,
                agents=agents,
                robot=robot,
                dino=dino,
                max_rounds=args.max_rounds,
            )
            return runner.run()

        return produce

    return {name: make_producer(name) for name in method_names}


def _print_summaries(result: BatchResult) -> None:
    for summary in result.summaries:
        runtime = (
            f"{summary.average_runtime_seconds:.2f}s"
            if summary.average_runtime_seconds is not None
            else "n/a"
        )
        print(
            f"{summary.method} / {summary.task_id}: "
            f"trials={summary.trials} success_rate={summary.success_rate:.2%} "
            f"avg_steps={summary.average_steps:.2f} failures={summary.failure_count} "
            f"risky_actions={summary.risky_actions} "
            f"reflections={summary.reflections} avg_runtime={runtime}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(method_names) - set(METHOD_NAMES))
    if unknown:
        logger.error(
            "Unknown methods: {}. Available: {}", unknown, list(METHOD_NAMES)
        )
        return 2

    if not args.tasks_path.exists():
        logger.error("Task catalogue not found: {}", args.tasks_path)
        return 1

    tasks = load_task_definitions(args.tasks_path)
    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        try:
            tasks = [get_task_by_id(tasks, task_id) for task_id in wanted]
        except ValueError as exc:
            logger.error("{}", exc)
            return 2
    if not tasks:
        logger.error("No tasks selected from {}", args.tasks_path)
        return 2

    if args.mock:
        producers = _build_mock_producers(method_names, args)
    else:
        producers = _build_real_producers(method_names, args)

    logger.info(
        "Starting batch evaluation: methods={} tasks={} trials={}",
        method_names,
        [task.id for task in tasks],
        args.trials,
    )
    result = run_batch(producers, tasks, trials=args.trials)

    run_dir = make_run_dir(args.out)
    paths = write_batch_results(result, run_dir)
    _print_summaries(result)
    print(f"Wrote summary JSON: {paths['summary_json']}")
    print(f"Wrote summary CSV:  {paths['summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
