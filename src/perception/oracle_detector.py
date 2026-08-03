"""Oracle object detector for simulation episodes with known actor labels.

In simulation the set of manipulable object labels is known from the task
metadata (``object_actors``). ``OracleDetector`` turns that ground-truth
label set into ``DetectedObject`` observations so the planner works with
real task labels instead of the mock DINO client's fake ``mock_object``.
Bounding boxes are fixed centered placeholders: grasp/place pose resolution
goes through the RoboTwin actor-level skills, not through the boxes.

When a ``visibility_provider`` is given (e.g.
:class:`perception.visibility.RaycastVisibilityProvider`), labels whose
actors are not currently visible are withheld from the detections — the
planner must reason about hidden state (move occluders aside, reobserve)
instead of seeing an all-knowing label set. A hysteresis counter smooths
out flicker (e.g. the robot arm briefly passing in front of an object):
a label disappears only after ``hide_after`` consecutive invisible
observations and reappears on the first visible one.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from common.schema import DetectedObject


class OracleDetector:
    """Detection stub that echoes the task's known object labels.

    Args:
        labels_provider: Callable returning the current list of object
            labels (e.g. the keys of the task's ``object_actors`` mapping).
        visibility_provider: Optional callable returning the subset of
            labels currently visible from the head camera. ``None`` keeps
            the all-knowing behavior.
        hide_after: Consecutive invisible observations before a label is
            withheld (hysteresis against transient occlusion).
    """

    def __init__(
        self,
        labels_provider: Callable[[], list[str]],
        visibility_provider: Callable[[], list[str]] | None = None,
        hide_after: int = 2,
    ) -> None:
        if hide_after < 1:
            raise ValueError(f"hide_after must be >= 1, got {hide_after}")
        self._labels_provider = labels_provider
        self._visibility_provider = visibility_provider
        self._hide_after = int(hide_after)
        self._invisible_streak: dict[str, int] = {}

    def _visible_labels(self) -> list[str]:
        labels = list(self._labels_provider())
        if self._visibility_provider is None:
            return labels
        visible = set(self._visibility_provider())
        kept = []
        for label in labels:
            if label in visible:
                self._invisible_streak[label] = 0
                kept.append(label)
            else:
                if label in self._invisible_streak:
                    streak = self._invisible_streak[label] + 1
                else:
                    # First observation of this label: report ground truth
                    # immediately (no grace period at episode start).
                    streak = self._hide_after
                self._invisible_streak[label] = streak
                if streak < self._hide_after:
                    kept.append(label)
        return kept

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Return one centered detection per visible label, confidence 1.0."""
        height, width = image.shape[:2]
        return [
            DetectedObject(
                label=label,
                bbox=[width * 0.4, height * 0.4, width * 0.6, height * 0.6],
                confidence=1.0,
            )
            for label in self._visible_labels()
        ]
