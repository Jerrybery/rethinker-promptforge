"""Ground-truth line-of-sight visibility via PhysX raycasts.

Used by the occlusion-aware OracleDetector: a mapped actor counts as
"visible" when at least a fraction of sample points on/near it can be
reached by an unobstructed ray from the head camera. This replaces the
SAPIEN segmentation approach — in sapien 3.0.0b1 the two segmentation
channels use different id namespaces (shape-level vs actor-level) with
numeric collisions, making per-actor pixel counting unreliable. PhysX
raycasts (`scene.physx_system.raycast`) are cheap (µs per ray) and
unambiguous: whatever the first hit is, that's the occluder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger


class RaycastVisibilityProvider:
    """Return the subset of mapped labels currently visible from the head camera.

    Args:
        env: A live RoboTwin task env (needs ``cameras``, ``scene`` and the
            actor attributes named in ``object_actors``).
        object_actors: Semantic label -> env actor attribute name.
        min_fraction: Minimum fraction of unblocked sample points for an
            actor to count as visible (default 0.2, i.e. ~2 of 9 samples).
        occlusion_eps: Ray length is shortened by this many meters so a hit
            on the actor's own surface is not counted as occlusion.
    """

    def __init__(
        self,
        env: Any,
        object_actors: dict[str, str],
        min_fraction: float = 0.2,
        occlusion_eps: float = 0.03,
        hidden_by: dict[str, list[str]] | None = None,
        cover_radius: float = 0.15,
    ) -> None:
        self._env = env
        self._mapping = dict(object_actors)
        self._min_fraction = float(min_fraction)
        self._eps = float(occlusion_eps)
        # Declared cover relations: label stays hidden while any of its
        # occluders' actors remains within ``cover_radius`` (xy) of it.
        # This encodes the scene's intended cover structure (e.g. a bottle
        # standing in front of a can) robustly, complementing the raycast
        # test which can be marginal for short occluders.
        self._hidden_by = {k: list(v) for k, v in (hidden_by or {}).items()}
        self._cover_radius = float(cover_radius)
        self._origin = self._head_camera_origin(env)

    @staticmethod
    def _head_camera_origin(env: Any) -> np.ndarray:
        cameras = env.cameras
        names = list(cameras.static_camera_name)
        cam = cameras.static_camera_list[names.index("head_camera")]
        for attr in ("get_pose", "get_global_pose", "local_pose"):
            getter = getattr(cam, attr, None)
            if callable(getter):
                pose = getter()
                p = getattr(pose, "p", pose)
                return np.asarray(p, dtype=np.float64)
        raise RuntimeError(
            "RaycastVisibilityProvider: cannot determine head camera pose"
        )

    @staticmethod
    def _sample_points(actor_wrapper: Any) -> list[np.ndarray]:
        """Sample points on the actor: center plus AABB corners when available."""
        center = np.asarray(actor_wrapper.get_pose().p, dtype=np.float64)
        points = [center]
        comp = None
        try:
            # URDF/articulation wrappers (e.g. the kitchen pot) may not
            # expose get_components; fall back to center+fixed offsets.
            components = actor_wrapper.actor.get_components()
        except Exception:
            components = []
        comp = next(
            (
                c
                for c in components
                if "Render" in type(c).__name__
            ),
            None,
        )
        corners = None
        if comp is not None:
            try:
                aabb = comp.compute_global_aabb_tight()
                lo, hi = np.asarray(aabb[0]), np.asarray(aabb[1])
                corners = (lo, hi)
            except Exception:
                corners = None
        if corners is None:
            lo, hi = center - 0.02, center + 0.02
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    points.append(np.array([x, y, z], dtype=np.float64))
        return points

    def _is_visible(self, actor_wrapper: Any) -> bool:
        physx_system = self._env.scene.physx_system
        try:
            own_components = set(actor_wrapper.actor.get_components())
        except Exception:
            # Articulation actors (URDF pot): their link components contain
            # "Articulation" in the type name and are skipped by the
            # articulation rule below anyway.
            own_components = set()
        unblocked = 0
        points = self._sample_points(actor_wrapper)
        for point in points:
            direction = point - self._origin
            distance = float(np.linalg.norm(direction))
            if distance < 1e-6:
                unblocked += 1
                continue
            unit = direction / distance
            # The head camera sits inside the robot's own link geometry
            # (panda_link4 at the mount point); start the ray past it.
            start_offset = min(0.2, distance * 0.3)
            start = self._origin + unit * start_offset
            hit = physx_system.raycast(
                start.astype(np.float32),
                unit.astype(np.float32),
                max(distance - start_offset - self._eps, 0.0),
            )
            if (
                hit is None
                or hit.component in own_components
                # The robot's own arm links never count as occluders: their
                # occlusion is transient by nature (the arm moves).
                or "Articulation" in type(hit.component).__name__
            ):
                unblocked += 1
        return (unblocked / len(points)) >= self._min_fraction

    def _covered(self, label: str) -> bool:
        """True when a declared occluder actor is still next to ``label``."""
        occluders = self._hidden_by.get(label)
        if not occluders:
            return False
        target = getattr(self._env, self._mapping[label], None)
        if target is None:
            return False
        tp = np.asarray(target.get_pose().p, dtype=np.float64)
        for attr_name in occluders:
            occluder = getattr(self._env, attr_name, None)
            if occluder is None:
                continue
            op = np.asarray(occluder.get_pose().p, dtype=np.float64)
            if float(np.linalg.norm(tp[:2] - op[:2])) <= self._cover_radius:
                return True
        return False

    def __call__(self) -> list[str]:
        """Labels whose actors are currently visible (line-of-sight)."""
        visible = []
        for label, attr_name in self._mapping.items():
            actor_wrapper = getattr(self._env, attr_name, None)
            if actor_wrapper is None:
                logger.warning(
                    "visibility: env has no actor attribute {!r} for label {!r}",
                    attr_name,
                    label,
                )
                continue
            try:
                if self._covered(label):
                    continue
                if self._is_visible(actor_wrapper):
                    visible.append(label)
            except Exception as exc:
                # Raycast failure must not kill the episode; treat as visible
                # (fail-open keeps the all-knowing behavior as fallback).
                logger.warning(
                    "visibility: raycast failed for label {!r}: {}; treating as visible",
                    label,
                    exc,
                )
                visible.append(label)
        return visible
