"""Gym-like simulation environment wrapper for the EmbodiedPromptForge loop.

``SimEnv`` composes the existing RoboTwin infrastructure:

* :func:`robot.robottwin_env.make_robottwin_env` to construct a headless
  RoboTwin task env,
* :class:`robot.interface.RoboTwinBackend` + ``RobotInterface`` for motion,
* :class:`executor.primitives.PrimitiveLibrary` for symbolic actions.

Observation contract (consumed by the forge planner, Task 3.2)::

    obs = {
        "image": np.ndarray,            # (H, W, 3) uint8 head-camera RGB
        "state": {                      # text-serializable robot state
            "pose": [x, y, z, qx, qy, qz, qw],
            "gripper": float,           # normalized opening in [0, 1]
            "timestamp": float,
        },
        "detections": [                 # DINO detections on the current image
            {"label": str, "bbox": [x1, y1, x2, y2], "confidence": float},
            ...
        ],
        "task": {
            "id": str,
            "instruction": str,
            "mission_type": str,        # MissionType value
        },
        "step_index": int,              # 0 at reset, +1 per step
    }

Action contract (matches ``PlannerOutput`` semantics — prompt-level symbolic
decisions, no low-level control)::

    SimAction(
        mission=MissionType.PICK_AND_PLACE | PICK_ONLY | MOVE_ASIDE
                 | REOBSERVE | STOP,
        target="mug",           # object label for pick / move_aside
        place_target="plate",   # optional place target label
        arm="right",            # arm hint
    )

``step`` returns ``(obs, reward, done, info)``. Reward is a sparse
task-success signal: ``1.0`` when the wrapped RoboTwin env exposes
``check_success()`` and it returns ``True``, else ``0.0``. ``info`` carries::

    info = {
        "success": bool | None,   # None when the env has no check_success
        "primitive_success": bool,
        "primitive_status": str,
        "task_id": str,
        "step_index": int,
        "truncated": bool,        # True when metadata.max_rounds is exceeded
    }

``done`` is True on task success, on a STOP action, or on truncation when
``metadata.max_rounds`` steps are exceeded without success.

``render()`` returns the list of frames captured since the last reset: one
frame at reset plus one frame after every step (captured synchronously during
``reset``/``step``; never re-queried).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

from common.schema import MissionType
from executor.primitives import PrimitiveLibrary, PrimitiveResult
from perception.dino_client import DINOClient
from robot.interface import RoboTwinBackend, RobotInterface
from robot.robottwin_env import make_robottwin_env
from robot.state import RobotState
from tasks.loader import _normalize_task
from tasks.schema import TaskDefinition

# Keys promoted from ``initial_scene`` to explicit ``make_robottwin_env``
# kwargs; everything else in ``initial_scene`` is forwarded as overrides.
_SCENE_KWARG_KEYS = ("task_name", "seed", "render_freq")

_MISSIONS_REQUIRING_TARGET = {
    MissionType.PICK_AND_PLACE,
    MissionType.PICK_ONLY,
    MissionType.MOVE_ASIDE,
}


class SimAction(BaseModel):
    """Structured symbolic action consumed by :meth:`SimEnv.step`.

    Mirrors :class:`common.schema.PlannerOutput` semantics: a mission type
    plus object labels. No grasp/place coordinates are allowed here; pose
    resolution stays inside the executor primitives.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mission: MissionType
    target: str | None = None
    place_target: str | None = None
    arm: Literal["left", "right"] = "right"

    @model_validator(mode="after")
    def _target_required_for_manipulation(self) -> "SimAction":
        if self.mission in _MISSIONS_REQUIRING_TARGET and not self.target:
            raise ValueError(
                f"mission {self.mission.value} requires a 'target' object label"
            )
        return self


class SimEnv:
    """Gym-like wrapper around a RoboTwin task environment.

    Args:
        env_factory: Callable with the same signature as
            :func:`robot.robottwin_env.make_robottwin_env`. Injected for
            testing; defaults to the real factory.
        dino: Object exposing ``detect(image) -> list[DetectedObject]``.
            Defaults to a mock-mode :class:`DINOClient` (deterministic fake
            detections) since grasp/place pose resolution is stubbed anyway.
        repo_root: Optional repository root forwarded to the env factory.
    """

    def __init__(
        self,
        env_factory: Callable[..., Any] = make_robottwin_env,
        dino: Any | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self._env_factory = env_factory
        self._dino = dino if dino is not None else DINOClient(mode="mock")
        self._repo_root = repo_root

        self._env: Any | None = None
        self._robot: RobotInterface | None = None
        self._primitives: PrimitiveLibrary | None = None
        self._task: TaskDefinition | None = None
        self._frames: list[np.ndarray] = []
        self._step_index = 0
        self._done = False
        self._max_steps: int | None = None

    # ------------------------------------------------------------------ #
    # Properties for downstream consumers (recorder/critic)
    # ------------------------------------------------------------------ #

    @property
    def task(self) -> TaskDefinition | None:
        """The task definition of the current episode (None before reset)."""
        return self._task

    @property
    def step_index(self) -> int:
        """Number of steps taken since the last reset."""
        return self._step_index

    @property
    def primitives(self) -> PrimitiveLibrary:
        """The primitive library of the current episode."""
        if self._primitives is None:
            raise RuntimeError("SimEnv.primitives is unavailable before reset()")
        return self._primitives

    # ------------------------------------------------------------------ #
    # Gym-like API
    # ------------------------------------------------------------------ #

    def reset(self, task_config: TaskDefinition | dict[str, Any]) -> dict[str, Any]:
        """Start a new episode for ``task_config`` and return the first obs.

        Args:
            task_config: A :class:`TaskDefinition` or a raw dict with the same
                shape as entries in ``data/tasks/*.yaml``. Must contain
                ``initial_scene`` with at least ``task_name``.

        Raises:
            ValueError: If the task config is invalid or lacks
                ``initial_scene``.
        """
        task = self._coerce_task(task_config)
        scene = dict(task.initial_scene or {})
        if "task_name" not in scene:
            raise ValueError(
                f"Task {task.id!r} initial_scene must contain 'task_name'"
            )

        metadata = task.metadata or {}
        task_name = metadata.get("robottwin_task_name") or scene["task_name"]
        task_config_name = metadata.get("robottwin_task_config", "demo_clean")
        overrides = {k: v for k, v in scene.items() if k not in _SCENE_KWARG_KEYS}

        logger.info(
            "SimEnv.reset: task={!r} robottwin_task={!r} config={!r}",
            task.id,
            task_name,
            task_config_name,
        )
        env = self._env_factory(
            task_name,
            task_config_name,
            repo_root=self._repo_root,
            seed=int(scene.get("seed", 0)),
            render_freq=int(scene.get("render_freq", 0)),
            overrides=overrides,
        )

        backend = RoboTwinBackend(env=env, strict_stop=False)
        robot = RobotInterface(backend=backend)

        self._env = env
        self._robot = robot
        self._primitives = PrimitiveLibrary(robot=robot, dino=self._dino)
        self._task = task
        self._frames = []
        self._step_index = 0
        self._done = False
        max_rounds = metadata.get("max_rounds")
        self._max_steps = int(max_rounds) if max_rounds is not None else None

        return self._observe()

    def step(
        self, action: SimAction | dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute one symbolic action and return ``(obs, reward, done, info)``.

        Raises:
            RuntimeError: If called before :meth:`reset` or after the episode
                ended (call :meth:`reset` again).
            pydantic.ValidationError: If ``action`` is not a valid
                :class:`SimAction`.
        """
        if self._task is None or self._primitives is None:
            raise RuntimeError("SimEnv.step() called before reset()")
        if self._done:
            raise RuntimeError(
                "SimEnv.step() called after episode done; call reset() first"
            )

        sim_action = (
            action if isinstance(action, SimAction) else SimAction(**action)
        )
        result = self._dispatch(sim_action)
        self._step_index += 1

        success = self._check_success()
        reward = 1.0 if success else 0.0
        truncated = (
            success is not True
            and self._max_steps is not None
            and self._step_index >= self._max_steps
        )
        self._done = bool(success) or sim_action.mission is MissionType.STOP or truncated

        obs = self._observe()
        info: dict[str, Any] = {
            "success": success,
            "primitive_success": result.success,
            "primitive_status": result.status,
            "task_id": self._task.id,
            "step_index": self._step_index,
            "truncated": truncated,
        }
        logger.debug(
            "SimEnv.step: mission={} reward={} done={} info={}",
            sim_action.mission.value,
            reward,
            self._done,
            info,
        )
        return obs, reward, self._done, info

    def render(self) -> list[np.ndarray]:
        """Return frames captured since the last reset (reset + each step)."""
        return list(self._frames)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_task(task_config: TaskDefinition | dict[str, Any]) -> TaskDefinition:
        if isinstance(task_config, TaskDefinition):
            task = task_config
        elif isinstance(task_config, dict):
            task = TaskDefinition(**_normalize_task(task_config))
        else:
            raise ValueError(
                f"task_config must be a TaskDefinition or dict, got {type(task_config)}"
            )
        if not task.initial_scene:
            raise ValueError(
                f"Task {task.id!r} must define 'initial_scene' for simulation"
            )
        return task

    def _dispatch(self, action: SimAction) -> PrimitiveResult:
        """Map a symbolic action onto executor primitives."""
        primitives = self.primitives
        mission = action.mission
        if mission is MissionType.PICK_AND_PLACE:
            pick_result = primitives.pick(action.target, arm_tag=action.arm)
            if not pick_result.success:
                return pick_result
            return primitives.place(action.place_target, arm_tag=action.arm)
        if mission is MissionType.PICK_ONLY:
            return primitives.pick(action.target, arm_tag=action.arm)
        if mission is MissionType.MOVE_ASIDE:
            return primitives.move_aside(action.target, arm_tag=action.arm)
        if mission is MissionType.REOBSERVE:
            return primitives.reobserve()
        if mission is MissionType.STOP:
            return primitives.stop()
        # Defensive: MissionType is an enum, so this should be unreachable.
        raise ValueError(f"Unsupported mission: {mission}")

    def _check_success(self) -> bool | None:
        """Query the wrapped env's success check, if it exposes one."""
        env = self._env
        checker = getattr(env, "check_success", None)
        if not callable(checker):
            return None
        try:
            return bool(checker())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("check_success() raised, treating as unknown: {}", exc)
            return None

    def _observe(self) -> dict[str, Any]:
        """Build the observation dict and capture a frame synchronously."""
        assert self._robot is not None and self._task is not None
        state: RobotState = self._robot.read_state()
        image = np.asarray(state.camera_image)
        self._frames.append(image)

        detections = [
            {
                "label": det.label,
                "bbox": list(det.bbox),
                "confidence": det.confidence,
            }
            for det in self._dino.detect(image)
        ]

        return {
            "image": image,
            "state": {
                "pose": state.pose.to_list(),
                "gripper": state.gripper,
                "timestamp": state.timestamp,
            },
            "detections": detections,
            "task": {
                "id": self._task.id,
                "instruction": self._task.instruction,
                "mission_type": self._task.mission_type.value,
            },
            "step_index": self._step_index,
        }
