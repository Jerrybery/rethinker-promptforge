"""Integration test for the DINO client on a synthetic RGB image."""

from __future__ import annotations

import numpy as np

from common.schema import DetectedObject
from perception.dino_client import DINOClient


def test_mock_detect_returns_detected_objects_on_synthetic_rgb() -> None:
    """End-to-end detection on a synthetic RGB image using mock mode."""
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    client = DINOClient(mode="mock")

    results = client.detect(image)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(isinstance(det, DetectedObject) for det in results)

    det = results[0]
    assert det.label == "mock_object"
    assert 0.0 <= det.confidence <= 1.0
    x1, y1, x2, y2 = det.bbox
    assert 0 <= x1 < x2 <= image.shape[1]
    assert 0 <= y1 < y2 <= image.shape[0]
