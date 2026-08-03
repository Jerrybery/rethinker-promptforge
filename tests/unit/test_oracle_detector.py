"""Unit tests for the oracle detector used in object_actors episodes."""

from __future__ import annotations

import numpy as np

from common.schema import DetectedObject
from perception.oracle_detector import OracleDetector


def _labels() -> list[str]:
    return ["cup", "coaster"]


def test_detect_returns_one_detection_per_label() -> None:
    detector = OracleDetector(labels_provider=_labels)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert [det.label for det in detections] == ["cup", "coaster"]
    assert all(isinstance(det, DetectedObject) for det in detections)
    assert all(det.confidence == 1.0 for det in detections)


def test_detect_bboxes_are_valid_and_centered() -> None:
    detector = OracleDetector(labels_provider=_labels)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    for det in detector.detect(image):
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 480
        assert (x1 + x2) / 2 == 640 * 0.5
        assert (y1 + y2) / 2 == 480 * 0.5


def test_detect_reflects_current_provider_labels() -> None:
    labels = ["can"]
    detector = OracleDetector(labels_provider=lambda: labels)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    assert [det.label for det in detector.detect(image)] == ["can"]
    labels.append("pot")
    assert [det.label for det in detector.detect(image)] == ["can", "pot"]
