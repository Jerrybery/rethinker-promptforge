"""Closed-loop runner wiring Rethinker, Planner, Executor, DINO, robot, and memory."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from loguru import logger

from common.logger import EpisodeLogger
from common.schema import Episode, EpisodeStep, Feedback, MissionType, TaskUnit
from executor.memory import ExecutorMemory
from perception.dino_client import DINOClient
from planner.memory import PlannerMemory
from rethinker.memory import RethinkerMemory
from rethinker_promptforge.config import load_config
from robot.interface import RobotInterface


class ClosedLoopRunner:
    """Run the full observe → DINO → Rethinker → Planner → Executor loop.

    The runner is deterministic and logs all parameters, every step, and all
    feedback to a JSON-lines episode log.  It enforces a maximum number of
    rounds and terminates early on STOP missions or execution failures.

    Args:
        task: The manipulation task to perform.
        config_path: Path to the project YAML config used by agents and the
            robot/perception stack.  A ``runner`` section may configure the
            action library, memory capacity, and log directory.
        agents: Mapping with keys ``"rethinker"``, ``"planner"``, and
            ``"executor"`` to the corresponding agent instances.
        robot: Initialized robot interface for reading state.
        dino: Initialized DINO client for object detection.
        max_rounds: Hard upper bound on the number of loop iterations.
    """

    DEFAULT_ACTION_LIBRARY = ["pick", "place", "move_aside", "reobserve", "stop"]

    def __init__(
        self,
        task: TaskUnit,
        config_path: str | Path,
        agents: dict[str, Any],
        robot: RobotInterface,
        dino: DINOClient,
        max_rounds: int = 10,
    ) -> None:
        self.task = task
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.robot = robot
        self.dino = dino
        self.max_rounds = max_rounds

        required_agents = {"rethinker", "planner", "executor"}
        missing = required_agents - set(agents.keys())
        if missing:
            raise ValueError(f"Missing required agents: {sorted(missing)}")
        self.agents = agents

        runner_cfg = self.config.get("runner", {}) or {}
        memory_cfg = runner_cfg.get("memory", {}) or {}
        capacity = int(memory_cfg.get("capacity", 100))
        self.rethinker_memory = RethinkerMemory(capacity=capacity)
        self.planner_memory = PlannerMemory(capacity=capacity)
        self.executor_memory = ExecutorMemory(capacity=capacity)

        log_dir = Path(runner_cfg.get("log_dir", "logs"))
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.episode_id = f"ep-{task.id}-{timestamp}"
        self.log_path = log_dir / f"{self.episode_id}.jsonl"
        self.logger = EpisodeLogger(self.log_path)
        self.logger.log_metadata(
            {
                "episode_id": self.episode_id,
                "task": self.task.model_dump(mode="json"),
                "config_path": str(self.config_path),
                "max_rounds": self.max_rounds,
                "agents": {k: type(v).__name__ for k, v in self.agents.items()},
                "robot": type(self.robot).__name__,
                "dino_mode": getattr(self.dino, "mode", "unknown"),
            }
        )

        logger.info(
            "ClosedLoopRunner initialized: episode_id={}, max_rounds={}",
            self.episode_id,
            self.max_rounds,
        )

    def run(self) -> Episode:
        """Execute the closed loop and return the populated :class:`Episode`."""
        episode = Episode(
            id=self.episode_id,
            task_id=self.task.id,
            metadata={
                "config_path": str(self.config_path),
                "max_rounds": self.max_rounds,
            },
        )
        previous_feedback: Feedback | None = None
        action_library = list(
            self.config.get("runner", {}).get("action_library")
            or self.DEFAULT_ACTION_LIBRARY
        )

        for round_idx in range(self.max_rounds):
            logger.info(
                "ClosedLoopRunner round {}/{} for episode {}",
                round_idx + 1,
                self.max_rounds,
                self.episode_id,
            )

            state = self.robot.read_state()
            rgb = state.camera_image
            detections = self.dino.detect(rgb)
            dino_labels = sorted({det.label for det in detections})

            rethinker_out = self.agents["rethinker"].act(
                task_goal=self.task.instruction,
                rgb_image=rgb,
                detections=detections,
                memory=self.rethinker_memory,
                previous_feedback=previous_feedback,
            )

            planner_out = self.agents["planner"].act(
                rethinker_output=rethinker_out,
                dino_labels=dino_labels,
                action_library=action_library,
                memory=self.planner_memory,
                previous_feedback=previous_feedback,
            )

            executor_out = self.agents["executor"].act(
                planner_output=planner_out,
                rgb=rgb,
                depth=None,
            )

            feedback = executor_out.feedback
            if feedback is None:
                feedback = Feedback(
                    success=bool(executor_out.success),
                    observation=executor_out.status,
                    error_message=None if executor_out.success else executor_out.status,
                )

            step = EpisodeStep(
                step_index=round_idx,
                task=self.task,
                rethinker_output=rethinker_out,
                planner_output=planner_out,
                executor_output=executor_out,
                feedback=feedback,
            )
            episode = episode.model_copy(
                update={"steps": list(episode.steps) + [step]}
            )
            self.logger.log_step(step)

            scene_token = f"scene-{round_idx:03d}"
            self.rethinker_memory.append(
                round=round_idx,
                scene_token=scene_token,
                query=self.task.instruction,
                answer=rethinker_out,
                feedback=feedback,
            )
            self.planner_memory.append(
                round=round_idx,
                scene_token=scene_token,
                query=rethinker_out.model_dump_json(),
                answer=planner_out,
                feedback=feedback,
            )
            self.executor_memory.append(
                round=round_idx,
                scene_token=scene_token,
                query=planner_out.plan_id,
                answer=executor_out,
                feedback=feedback,
            )

            if rethinker_out.mission_type is MissionType.STOP:
                logger.info("ClosedLoopRunner: STOP mission, terminating loop.")
                break
            if not feedback.success:
                logger.warning(
                    "ClosedLoopRunner: step {} failed, terminating loop.",
                    round_idx,
                )
                break

            previous_feedback = feedback

        self.logger.close()
        logger.info(
            "ClosedLoopRunner finished episode {} with {} step(s)",
            self.episode_id,
            len(episode.steps),
        )
        return episode
