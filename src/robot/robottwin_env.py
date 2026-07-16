"""Helper to safely import and initialize a RoboTwin task environment.

RoboTwin relies on relative paths and on ``sys.path`` containing the RoboTwin
root directory. This module isolates those side effects so that the rest of the
project does not need to run with a special working directory.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


class RoboTwinEnvError(RuntimeError):
    """Raised when a RoboTwin environment cannot be constructed."""

    pass


def _robottwin_root(repo_root: str | Path | None = None) -> Path:
    """Return the absolute path to the ``third_party/RoboTwin`` directory."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    return Path(repo_root) / "third_party" / "RoboTwin"


@contextmanager
def _robottwin_context(repo_root: str | Path | None = None) -> Iterator[Path]:
    """Temporarily switch cwd and ``sys.path`` to the RoboTwin root.

    Yields the RoboTwin root path. The original cwd and ``sys.path`` are
    restored on exit, even if an exception is raised.
    """
    rt_root = _robottwin_root(repo_root)
    if not rt_root.is_dir():
        raise RoboTwinEnvError(f"RoboTwin directory not found: {rt_root}")

    original_cwd = os.getcwd()
    original_sys_path = list(sys.path)

    # Avoid duplicate entries and ensure the RoboTwin root is first.
    sys.path = [str(rt_root)] + [p for p in sys.path if p != str(rt_root)]
    os.chdir(rt_root)

    try:
        yield rt_root
    finally:
        os.chdir(original_cwd)
        sys.path = original_sys_path


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``.

    Raises:
        RoboTwinEnvError: If the file cannot be read or is not a mapping.
    """
    if not path.is_file():
        raise RoboTwinEnvError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RoboTwinEnvError(f"YAML file {path} must contain a mapping")
    return data


def _resolve_embodiment_args(
    task_config: dict[str, Any], rt_root: Path
) -> dict[str, Any]:
    """Resolve embodiment names from ``task_config`` into RoboTwin kwargs."""
    embodiment_names = task_config.get("embodiment")
    if embodiment_names is None:
        embodiment_names = ["aloha-agilex"]
    if isinstance(embodiment_names, str):
        embodiment_names = [embodiment_names]

    embodiment_config_path = rt_root / "task_config" / "_embodiment_config.yml"
    embodiment_types = _load_yaml(embodiment_config_path)

    def get_embodiment_file(name: str) -> str:
        entry = embodiment_types.get(name)
        if entry is None or not isinstance(entry, dict):
            raise RoboTwinEnvError(f"Unknown embodiment {name!r}")
        file_path = entry.get("file_path")
        if file_path is None:
            raise RoboTwinEnvError(f"Missing embodiment file_path for {name!r}")
        return str(rt_root / file_path)

    def get_embodiment_config(robot_file: str) -> dict[str, Any]:
        robot_config_file = Path(robot_file) / "config.yml"
        return _load_yaml(robot_config_file)

    args: dict[str, Any] = {}
    if len(embodiment_names) == 1:
        robot_file = get_embodiment_file(embodiment_names[0])
        left = robot_file
        right = robot_file
        args["dual_arm_embodied"] = True
    elif len(embodiment_names) == 3:
        left = get_embodiment_file(embodiment_names[0])
        right = get_embodiment_file(embodiment_names[1])
        args["embodiment_dis"] = embodiment_names[2]
        args["dual_arm_embodied"] = False
    else:
        raise RoboTwinEnvError(
            "embodiment list must have length 1 or 3, "
            f"got {len(embodiment_names)}"
        )

    args["left_robot_file"] = left
    args["right_robot_file"] = right
    args["left_embodiment_config"] = get_embodiment_config(left)
    args["right_embodiment_config"] = get_embodiment_config(right)
    args["embodiment_name"] = (
        embodiment_names[0]
        if len(embodiment_names) == 1
        else f"{embodiment_names[0]}+{embodiment_names[1]}"
    )
    return args


def _build_setup_kwargs(
    task_name: str,
    task_config_name: str,
    rt_root: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the keyword arguments passed to ``env.setup_demo``.

    This mirrors the setup logic in ``script/collect_data.py`` but without
    hard-coded episode collection behavior.
    """
    config_path = rt_root / "task_config" / f"{task_config_name}.yml"
    args = _load_yaml(config_path)
    args["task_name"] = task_name
    args["task_config"] = task_config_name

    embodiment_args = _resolve_embodiment_args(args, rt_root)
    args.update(embodiment_args)

    # Default to headless, single-episode behavior unless overridden.
    args.setdefault("render_freq", 0)
    args.setdefault("save_data", False)
    args.setdefault("collect_data", False)
    args.setdefault("episode_num", 1)
    args.setdefault("use_seed", False)
    args.setdefault("now_ep_num", 0)
    args.setdefault("save_path", "./data")

    if overrides:
        args.update(overrides)

    return args


def make_robottwin_env(
    task_name: str,
    task_config_name: str = "demo_clean",
    repo_root: str | Path | None = None,
    seed: int = 0,
    render_freq: int = 0,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Import and initialize a RoboTwin task environment.

    This helper temporarily changes the working directory and ``sys.path`` so
    that RoboTwin's relative-path assumptions are satisfied, then imports the
    requested task class, builds the configuration, and calls
    ``env.setup_demo(...)``.

    Args:
        task_name: RoboTwin task class name (e.g. ``"place_a2b_right"``).
        task_config_name: Task config stem under ``task_config/`` (default
            ``demo_clean``).
        repo_root: Optional repository root. Defaults to the repository
            containing this file.
        seed: Random seed passed to ``setup_demo``.
        render_freq: Viewer render frequency; keep at ``0`` for headless runs.
        overrides: Optional dictionary of extra kwargs merged into the
            ``setup_demo`` call.

    Returns:
        The initialized RoboTwin environment object, compatible with
        :class:`robot.interface.RoboTwinBackend`.

    Raises:
        RoboTwinEnvError: If the task class cannot be imported, the config is
            missing, or setup fails.
    """
    setup_overrides = dict(overrides or {})
    setup_overrides.setdefault("seed", seed)
    setup_overrides.setdefault("render_freq", render_freq)

    with _robottwin_context(repo_root) as rt_root:
        try:
            envs_module = importlib.import_module(f"envs.{task_name}")
            env_class = getattr(envs_module, task_name)
        except Exception as exc:
            raise RoboTwinEnvError(
                f"Failed to import RoboTwin task {task_name!r}: {exc}"
            ) from exc

        try:
            env = env_class()
        except Exception as exc:
            raise RoboTwinEnvError(
                f"Failed to instantiate RoboTwin task {task_name!r}: {exc}"
            ) from exc

        setup_kwargs = _build_setup_kwargs(
            task_name=task_name,
            task_config_name=task_config_name,
            rt_root=rt_root,
            overrides=setup_overrides,
        )

        try:
            env.setup_demo(**setup_kwargs)
        except Exception as exc:
            raise RoboTwinEnvError(
                f"RoboTwin task {task_name!r} setup_demo failed: {exc}"
            ) from exc

    return env
