"""Unit tests for RoboTwinBackend actor-level skills and primitive wiring.

All tests run against fake env objects that record the actor-skill calls, so
no simulator is required. ``RoboTwinBackend._arm_tag`` is patched to avoid
importing the RoboTwin ``envs`` package.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from executor.primitives import PrimitiveLibrary
from perception.oracle_detector import OracleDetector
from robot.interface import RoboTwinBackend, RobotInterface


@pytest.fixture(autouse=True)
def _patch_arm_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid importing envs.utils.ArmTag (drags in the simulator)."""
    monkeypatch.setattr(
        RoboTwinBackend, "_arm_tag", staticmethod(lambda arm: f"ArmTag({arm})")
    )


class _FakeActor:
    def __init__(
        self,
        pose: tuple[float, float, float] = (0.2, -0.1, 0.75),
        functional_point: Any = None,
        functional_point_raises: bool = False,
    ) -> None:
        self._pose = np.array(pose, dtype=float)
        self._fp = functional_point
        self._fp_raises = functional_point_raises

    def get_pose(self) -> Any:
        return SimpleNamespace(p=self._pose)

    def get_functional_point(self, idx: int, fmt: Any = None) -> Any:
        if self._fp_raises:
            raise RuntimeError("no functional point")
        return self._fp


class _FakeSkillEnv:
    """Records grasp_actor/place_actor/move_by_displacement calls."""

    def __init__(
        self,
        actors: dict[str, _FakeActor],
        move_result: bool = True,
        grasp_raises_without_contact_id: bool = False,
        fail_moves: int = 0,
        no_carry: bool = False,
    ) -> None:
        self._actors = actors
        self.calls: list[tuple] = []
        self._move_result = move_result
        self._grasp_raises = grasp_raises_without_contact_id
        self._fail_moves = fail_moves
        self._no_carry = no_carry
        self._grasped: Any = None
        self._pending_lift = 0.0
        self.plan_success = True

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name in self._actors:
            return self._actors[name]
        raise AttributeError(name)

    def grasp_actor(
        self,
        actor: Any,
        arm_tag: Any,
        pre_grasp_dis: float = 0.1,
        contact_point_id: Any = None,
    ) -> tuple:
        if contact_point_id is None and self._grasp_raises:
            raise TypeError("contact_point_id required")
        self.calls.append(("grasp_actor", arm_tag, pre_grasp_dis, contact_point_id))
        self._grasped = actor
        return ("grasp_actions",)

    def place_actor(
        self,
        actor: Any,
        arm_tag: Any,
        target_pose: Any = None,
        functional_point_id: Any = None,
        pre_dis: float = 0.1,
        **kwargs: Any,
    ) -> tuple:
        pose = [float(v) for v in target_pose] if target_pose is not None else None
        self.calls.append(
            ("place_actor", arm_tag, pose, functional_point_id, pre_dis)
        )
        return ("place_actions",)

    def move_by_displacement(
        self, arm_tag: Any, x: float = 0.0, y: float = 0.0, z: float = 0.0,
        move_axis: str = "world",
    ) -> tuple:
        self.calls.append(("move_by_displacement", arm_tag, z, move_axis))
        self._pending_lift = z
        return ("lift_actions",)

    def move(self, actions: Any) -> bool:
        self.calls.append(("move", actions))
        if self._pending_lift and self._grasped is not None and not self._no_carry:
            self._grasped._pose[2] += self._pending_lift
        self._pending_lift = 0.0
        if self._fail_moves > 0:
            self._fail_moves -= 1
            self.plan_success = False
            return False
        return self._move_result


def _backend(env: _FakeSkillEnv) -> RoboTwinBackend:
    return RoboTwinBackend(env=env, strict_stop=False)


# ---------------------------------------------------------------------- #
# resolve_actor
# ---------------------------------------------------------------------- #


def test_resolve_actor_returns_env_attribute() -> None:
    cup = _FakeActor()
    backend = _backend(_FakeSkillEnv({"cup": cup}))
    assert backend.resolve_actor("cup") is cup


def test_resolve_actor_missing_raises_with_attr_name() -> None:
    backend = _backend(_FakeSkillEnv({}))
    with pytest.raises(RuntimeError, match="'mug'"):
        backend.resolve_actor("mug")


# ---------------------------------------------------------------------- #
# grasp_object
# ---------------------------------------------------------------------- #


def test_grasp_object_auto_arm_right_no_contact_point_by_default() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor(pose=(0.2, -0.1, 0.75))})
    backend = _backend(env)

    result = backend.grasp_object("cup")

    assert result["success"] is True
    assert result["arm_tag"] == "right"
    grasp_calls = [c for c in env.calls if c[0] == "grasp_actor"]
    assert grasp_calls == [("grasp_actor", "ArmTag(right)", 0.1, None)]


def test_grasp_object_auto_arm_left_no_contact_point_by_default() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor(pose=(-0.2, -0.1, 0.75))})
    backend = _backend(env)

    result = backend.grasp_object("cup")

    assert result["arm_tag"] == "left"
    grasp_calls = [c for c in env.calls if c[0] == "grasp_actor"]
    assert grasp_calls == [("grasp_actor", "ArmTag(left)", 0.1, None)]


def test_grasp_object_contact_point_id_flag_tries_cpid_first() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor(pose=(0.2, -0.1, 0.75))})
    backend = _backend(env)

    result = backend.grasp_object("cup", contact_point_id=True)

    assert result["success"] is True
    grasp_calls = [c for c in env.calls if c[0] == "grasp_actor"]
    assert grasp_calls == [("grasp_actor", "ArmTag(right)", 0.1, 0)]


def test_grasp_object_contact_point_id_flag_falls_back_to_plain() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor()}, fail_moves=1)
    backend = _backend(env)

    result = backend.grasp_object("cup", contact_point_id=True)

    assert result["success"] is True
    grasp_calls = [c for c in env.calls if c[0] == "grasp_actor"]
    assert grasp_calls == [
        ("grasp_actor", "ArmTag(right)", 0.1, 0),
        ("grasp_actor", "ArmTag(right)", 0.1, None),
    ]


def test_grasp_object_retries_with_contact_point_id_on_move_failure() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor()}, fail_moves=1)
    backend = _backend(env)

    result = backend.grasp_object("cup")

    assert result["success"] is True
    assert env.plan_success is True  # reset before the retry
    grasp_calls = [c for c in env.calls if c[0] == "grasp_actor"]
    assert grasp_calls == [
        ("grasp_actor", "ArmTag(right)", 0.1, None),
        ("grasp_actor", "ArmTag(right)", 0.1, 0),
    ]


def test_grasp_object_retries_with_contact_point_id_on_exception() -> None:
    env = _FakeSkillEnv(
        {"cup": _FakeActor(pose=(-0.2, -0.1, 0.75))},
        grasp_raises_without_contact_id=True,
    )
    backend = _backend(env)

    result = backend.grasp_object("cup")

    assert result["success"] is True
    grasp_calls = [c for c in env.calls if c[0] == "grasp_actor"]
    assert grasp_calls == [("grasp_actor", "ArmTag(left)", 0.1, 2)]


def test_grasp_object_records_grasp_for_later_place() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor(pose=(0.2, -0.1, 0.75))})
    backend = _backend(env)
    backend.grasp_object("cup")
    assert backend._grasped_attr == "cup"
    assert backend._grasped_arm == "right"
    assert backend._grasp_pose == [0.2, -0.1, 0.75]


def test_grasp_object_move_failure_returns_unsuccessful() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor()}, move_result=False)
    backend = _backend(env)
    result = backend.grasp_object("cup")
    assert result["success"] is False
    assert backend._grasped_attr is None


# ---------------------------------------------------------------------- #
# place_object_at
# ---------------------------------------------------------------------- #


def test_place_object_at_uses_functional_point_when_available() -> None:
    fp = [0.1, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0]
    env = _FakeSkillEnv(
        {
            "cup": _FakeActor(pose=(0.2, -0.1, 0.75), functional_point=[0] * 7),
            "coaster": _FakeActor(functional_point=fp),
        }
    )
    backend = _backend(env)
    backend.grasp_object("cup")

    result = backend.place_object_at(target_attr_name="coaster")

    assert result["success"] is True
    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls == [
        ("place_actor", "ArmTag(right)", fp, 0, 0.05)
    ]


def test_place_object_at_falls_back_to_target_pose_plus_offset() -> None:
    env = _FakeSkillEnv(
        {
            "object": _FakeActor(pose=(0.1, -0.1, 0.75)),
            "target_object": _FakeActor(
                pose=(0.3, 0.0, 0.74), functional_point_raises=True
            ),
        }
    )
    backend = _backend(env)
    backend.grasp_object("object")

    result = backend.place_object_at(
        target_attr_name="target_object", offset=[0.13, 0.0, 0.0]
    )

    assert result["success"] is True
    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls[0][2] == pytest.approx([0.43, 0.0, 0.74])
    assert place_calls[0][3] is None  # no functional point


def test_place_object_at_falls_back_when_functional_point_is_none() -> None:
    env = _FakeSkillEnv(
        {
            "object": _FakeActor(pose=(0.1, -0.1, 0.75)),
            # get_functional_point exists but returns None (no functional points)
            "target_object": _FakeActor(pose=(0.3, 0.0, 0.74), functional_point=None),
        }
    )
    backend = _backend(env)
    backend.grasp_object("object")

    result = backend.place_object_at(
        target_attr_name="target_object", offset=[0.13, 0.0, 0.0]
    )

    assert result["success"] is True
    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls[0][2] == pytest.approx([0.43, 0.0, 0.74])
    assert place_calls[0][3] is None  # functional_point_id not forwarded


def test_place_object_at_drops_fpid_when_held_actor_has_no_fp() -> None:
    """place_actor's functional_point_id refers to the GRASPED actor; with
    a target fp but no held-actor fp (the can-into-basket case) it must not
    be forwarded."""
    fp = [0.1, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0]
    env = _FakeSkillEnv(
        {
            "can": _FakeActor(pose=(0.2, 0.1, 0.75)),  # no functional point
            "basket": _FakeActor(functional_point=fp),
        }
    )
    backend = _backend(env)
    backend.grasp_object("can")

    result = backend.place_object_at(target_attr_name="basket")

    assert result["success"] is True
    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls[0][2] == fp
    assert place_calls[0][3] is None


def test_place_object_at_mirror_offset_with_left_arm() -> None:
    env = _FakeSkillEnv(
        {
            "can": _FakeActor(pose=(-0.25, 0.1, 0.75)),
            "pot": _FakeActor(pose=(0.0, 0.0, 0.74), functional_point_raises=True),
        }
    )
    backend = _backend(env)
    backend.grasp_object("can")  # auto arm -> left (x < 0)

    backend.place_object_at(
        target_attr_name="pot",
        offset=[0.18, 0.0, 0.0],
        mirror_offset_with_arm=True,
    )

    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls[0][2] == pytest.approx([-0.18, 0.0, 0.74])


def test_place_object_at_no_target_uses_grasp_pose_plus_offset() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor(pose=(0.2, -0.1, 0.75))})
    backend = _backend(env)
    backend.grasp_object("cup")

    backend.place_object_at(offset=[0.0, -0.15, 0.0])

    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls[0][2] == pytest.approx([0.2, -0.25, 0.75])
    assert backend._grasped_attr is None  # released after successful place


def test_place_object_at_before_grasp_raises() -> None:
    backend = _backend(_FakeSkillEnv({"cup": _FakeActor()}))
    with pytest.raises(RuntimeError, match="before any successful grasp"):
        backend.place_object_at()


# ---------------------------------------------------------------------- #
# lift
# ---------------------------------------------------------------------- #


def test_lift_uses_arm_displacement() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor()})
    backend = _backend(env)
    backend.grasp_object("cup")

    result = backend.lift(dz=0.08)

    assert result == {"success": True, "arm_tag": "right", "dz": 0.08, "carried": True}
    lift_calls = [c for c in env.calls if c[0] == "move_by_displacement"]
    assert lift_calls == [("move_by_displacement", "ArmTag(right)", 0.08, "arm")]


def test_lift_reports_not_carried_when_object_left_behind() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor()}, no_carry=True)
    backend = _backend(env)
    backend.grasp_object("cup")

    result = backend.lift(dz=0.08)

    assert result["success"] is False
    assert result["carried"] is False


def test_pick_fails_when_grasp_cannot_hold() -> None:
    """Kinematic success without physical carry -> honest failure."""
    env = _FakeSkillEnv({"cup": _FakeActor()}, no_carry=True)
    backend = RoboTwinBackend(env=env, strict_stop=False)
    robot = RobotInterface(backend=backend)
    dino = OracleDetector(labels_provider=lambda: ["cup"])
    library = PrimitiveLibrary(robot=robot, dino=dino, object_actors={"cup": "cup"})

    result = library.pick("cup")

    assert result.success is False
    assert "failed to hold" in result.status
    # plain grasp + alternate-contact retry were both attempted
    grasp_calls = [c for c in env.calls if c[0] == "grasp_actor"]
    assert len(grasp_calls) == 2


# ---------------------------------------------------------------------- #
# PrimitiveLibrary real-path wiring
# ---------------------------------------------------------------------- #


def _real_library(env: _FakeSkillEnv, **kwargs: Any) -> PrimitiveLibrary:
    backend = RoboTwinBackend(env=env, strict_stop=False)
    robot = RobotInterface(backend=backend)
    dino = OracleDetector(labels_provider=lambda: list(kwargs["object_actors"]))
    return PrimitiveLibrary(robot=robot, dino=dino, **kwargs)


def test_pick_real_path_grasps_and_lifts() -> None:
    fp = [0.1, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0]
    env = _FakeSkillEnv(
        {
            "cup": _FakeActor(pose=(0.2, -0.1, 0.75)),
            "coaster": _FakeActor(functional_point=fp),
        }
    )
    library = _real_library(env, object_actors={"cup": "cup", "coaster": "coaster"})

    result = library.pick("cup")

    assert result.success is True
    assert result.status == "picked 'cup'"
    kinds = [c[0] for c in env.calls]
    assert kinds == ["grasp_actor", "move", "move_by_displacement", "move"]


def test_pick_real_path_grasp_failure_returns_failure() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor()}, move_result=False)
    library = _real_library(env, object_actors={"cup": "cup"})

    result = library.pick("cup")

    assert result.success is False
    assert "cup" in result.status


def test_place_real_path_onto_functional_point_target() -> None:
    fp = [0.1, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0]
    env = _FakeSkillEnv(
        {
            "cup": _FakeActor(pose=(0.2, -0.1, 0.75), functional_point=[0] * 7),
            "coaster": _FakeActor(functional_point=fp),
        }
    )
    library = _real_library(env, object_actors={"cup": "cup", "coaster": "coaster"})
    library.pick("cup")

    result = library.place("coaster")

    assert result.success is True
    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert len(place_calls) == 1
    assert place_calls[0][2] == fp


def test_place_real_path_applies_place_offsets() -> None:
    env = _FakeSkillEnv(
        {
            "object": _FakeActor(pose=(0.1, -0.1, 0.75)),
            "target_object": _FakeActor(
                pose=(0.3, 0.0, 0.74), functional_point_raises=True
            ),
        }
    )
    library = _real_library(
        env,
        object_actors={"source_object": "object", "target_object": "target_object"},
        place_offsets={"target_object": [0.13, 0.0, 0.0]},
    )
    library.pick("source_object")

    result = library.place("target_object")

    assert result.success is True
    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls[0][2] == pytest.approx([0.43, 0.0, 0.74])


def test_place_real_path_unmapped_label_places_without_target() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor(pose=(0.2, -0.1, 0.75))})
    library = _real_library(env, object_actors={"cup": "cup"})
    library.pick("cup")

    result = library.place("table")

    assert result.success is True
    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls[0][2] == pytest.approx([0.2, -0.1, 0.75])


def test_move_aside_real_path_shifts_laterally() -> None:
    env = _FakeSkillEnv({"cup": _FakeActor(pose=(0.2, -0.1, 0.75))})
    library = _real_library(env, object_actors={"cup": "cup"})

    result = library.move_aside("cup")

    assert result.success is True
    place_calls = [c for c in env.calls if c[0] == "place_actor"]
    assert place_calls[0][2] == pytest.approx([0.25, 0.2, 0.75])


def test_pick_without_mapping_keeps_stub_path() -> None:
    """object_actors present but mock backend -> stub sequence still runs."""
    from perception.dino_client import DINOClient

    robot = RobotInterface(mock=True)
    library = PrimitiveLibrary(
        robot=robot,
        dino=DINOClient(mode="mock"),
        object_actors={"mock_object": "mock_object"},
    )
    result = library.pick("mock_object")
    assert result.success is True
    assert "grasp_pose" in result.data  # stub payload
