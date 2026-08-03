"""Unit tests for raycast visibility and the occlusion-aware OracleDetector."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from perception.oracle_detector import OracleDetector
from perception.visibility import RaycastVisibilityProvider


class _FakePhysxSystem:
    """Scripted raycast: each entry in ``hits`` is returned in call order."""

    def __init__(self) -> None:
        self.hits: list[Any] = []
        self.calls = 0

    def raycast(self, position: Any, direction: Any, distance: float) -> Any:
        self.calls += 1
        if not self.hits:
            return None
        return self.hits.pop(0)


class _FakeRenderComponent:
    def compute_global_aabb_tight(self) -> tuple:
        return (
            np.array([0.09, -0.01, 0.73]),
            np.array([0.11, 0.01, 0.77]),
        )


class _FakeActorWrapper:
    def __init__(self, physx_component: object) -> None:
        self.actor = SimpleNamespace(
            get_components=lambda: [physx_component, _FakeRenderComponent()]
        )

    def get_pose(self) -> Any:
        return SimpleNamespace(p=np.array([0.1, 0.0, 0.75]))


class _FakeCamera:
    def get_pose(self) -> Any:
        return SimpleNamespace(p=np.array([0.0, -0.45, 1.35]))


def _env(physx: _FakePhysxSystem, actors: dict[str, _FakeActorWrapper]) -> Any:
    scene = SimpleNamespace(physx_system=physx)
    cameras = SimpleNamespace(
        static_camera_name=["head_camera"], static_camera_list=[_FakeCamera()]
    )
    return SimpleNamespace(scene=scene, cameras=cameras, **actors)


def test_visible_when_nothing_blocks() -> None:
    physx = _FakePhysxSystem()  # all rays miss
    cup_comp = object()
    env = _env(physx, {"cup": _FakeActorWrapper(cup_comp)})
    provider = RaycastVisibilityProvider(env, {"cup": "cup"})

    assert provider() == ["cup"]
    assert physx.calls == 9  # center + 8 AABB corners


def test_occluded_when_all_rays_hit_other_actor() -> None:
    physx = _FakePhysxSystem()
    cup_comp, box_comp = object(), object()
    physx.hits = [SimpleNamespace(component=box_comp)] * 9
    env = _env(physx, {"cup": _FakeActorWrapper(cup_comp)})
    provider = RaycastVisibilityProvider(env, {"cup": "cup"})

    assert provider() == []


def test_self_hit_counts_as_visible() -> None:
    physx = _FakePhysxSystem()
    cup_comp = object()
    physx.hits = [SimpleNamespace(component=cup_comp)] * 9
    env = _env(physx, {"cup": _FakeActorWrapper(cup_comp)})
    provider = RaycastVisibilityProvider(env, {"cup": "cup"})

    assert provider() == ["cup"]


def test_partial_occlusion_respects_min_fraction() -> None:
    physx = _FakePhysxSystem()
    cup_comp, box_comp = object(), object()
    # 7 of 9 rays blocked -> 2/9 ≈ 0.22 >= 0.2 -> visible
    physx.hits = [SimpleNamespace(component=box_comp)] * 7 + [None, None]
    env = _env(physx, {"cup": _FakeActorWrapper(cup_comp)})
    provider = RaycastVisibilityProvider(env, {"cup": "cup"}, min_fraction=0.2)
    assert provider() == ["cup"]

    # 8 of 9 blocked -> 1/9 ≈ 0.11 < 0.2 -> hidden
    physx2 = _FakePhysxSystem()
    physx2.hits = [SimpleNamespace(component=box_comp)] * 8 + [None]
    env2 = _env(physx2, {"cup": _FakeActorWrapper(cup_comp)})
    provider2 = RaycastVisibilityProvider(env2, {"cup": "cup"}, min_fraction=0.2)
    assert provider2() == []


def test_articulation_hits_do_not_occlude() -> None:
    """The robot's own arm links are ignored as occluders."""

    class PhysxArticulationLinkComponent:
        pass

    physx = _FakePhysxSystem()
    cup_comp = object()
    physx.hits = [
        SimpleNamespace(component=PhysxArticulationLinkComponent())
    ] * 9
    env = _env(physx, {"cup": _FakeActorWrapper(cup_comp)})
    provider = RaycastVisibilityProvider(env, {"cup": "cup"})

    assert provider() == ["cup"]


def test_hidden_by_cover_rule() -> None:
    """Declared cover: label hidden while occluder stays within radius."""
    physx = _FakePhysxSystem()  # rays all miss -> raycast-visible
    apple = _FakeActorWrapper(object())
    bottle = _FakeActorWrapper(object())  # same pose as apple: covered
    env = _env(physx, {"apple": apple, "bottle": bottle})
    provider = RaycastVisibilityProvider(
        env,
        {"apple": "apple", "bottle": "bottle"},
        hidden_by={"apple": ["bottle"]},
    )
    assert provider() == ["bottle"]

    # bottle moved aside -> cover lifted
    bottle_far = _FakeActorWrapper(object())
    bottle_far.get_pose = lambda: SimpleNamespace(p=np.array([0.6, 0.0, 0.75]))
    env2 = _env(physx, {"apple": apple, "bottle": bottle_far})
    provider2 = RaycastVisibilityProvider(
        env2,
        {"apple": "apple", "bottle": "bottle"},
        hidden_by={"apple": ["bottle"]},
    )
    assert provider2() == ["apple", "bottle"]


def test_raycast_failure_fails_open() -> None:
    class _BrokenPhysx:
        def raycast(self, *args: Any) -> Any:
            raise RuntimeError("physx exploded")

    env = _env(_BrokenPhysx(), {"cup": _FakeActorWrapper(object())})
    provider = RaycastVisibilityProvider(env, {"cup": "cup"})
    assert provider() == ["cup"]


def test_missing_actor_attribute_skipped() -> None:
    physx = _FakePhysxSystem()
    env = _env(physx, {})
    provider = RaycastVisibilityProvider(env, {"cup": "cup"})
    assert provider() == []


# ---------------------------------------------------------------------- #
# OracleDetector hysteresis
# ---------------------------------------------------------------------- #


def test_oracle_hysteresis_hides_after_consecutive_misses() -> None:
    visible = {"cup", "coaster"}
    detector = OracleDetector(
        labels_provider=lambda: ["cup", "coaster"],
        visibility_provider=lambda: sorted(visible),
        hide_after=2,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # establish both labels as seen-visible first
    assert [d.label for d in detector.detect(image)] == ["cup", "coaster"]
    visible.discard("coaster")
    # first invisible observation: streak 1 < 2 -> still reported
    assert [d.label for d in detector.detect(image)] == ["cup", "coaster"]
    # second consecutive miss: streak 2 -> hidden
    assert [d.label for d in detector.detect(image)] == ["cup"]
    assert [d.label for d in detector.detect(image)] == ["cup"]
    # visible again: reappears immediately
    visible.add("coaster")
    assert [d.label for d in detector.detect(image)] == ["cup", "coaster"]


def test_oracle_initial_state_reflects_ground_truth_immediately() -> None:
    """A label occluded at episode start is hidden from the first detect."""
    detector = OracleDetector(
        labels_provider=lambda: ["cup", "apple"],
        visibility_provider=lambda: ["cup"],  # apple initially occluded
        hide_after=2,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert [d.label for d in detector.detect(image)] == ["cup"]


def test_oracle_without_provider_stays_all_knowing() -> None:
    detector = OracleDetector(labels_provider=lambda: ["cup", "coaster"])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert [d.label for d in detector.detect(image)] == ["cup", "coaster"]


def test_oracle_hide_after_validation() -> None:
    with pytest.raises(ValueError, match="hide_after"):
        OracleDetector(labels_provider=lambda: ["x"], hide_after=0)
