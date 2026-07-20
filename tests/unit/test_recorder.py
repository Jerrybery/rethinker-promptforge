"""Unit tests for the forge episode recorder.

All tests use synthetic numpy frames and ``tmp_path`` output — no sim, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from forge.recorder import EpisodeRecorder, EpisodeRecording


def _frame(h: int = 8, w: int = 12, value: int = 42) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


@pytest.fixture
def recorder(tmp_path: Path) -> EpisodeRecorder:
    rec = EpisodeRecorder()
    rec.start_episode("ep-001", tmp_path, fps=5.0)
    return rec


# --------------------------------------------------------------------- #
# happy path: video + metadata
# --------------------------------------------------------------------- #


def test_finish_writes_readable_video(recorder: EpisodeRecorder, tmp_path: Path) -> None:
    for _ in range(4):
        recorder.add_frame(_frame())
    recording = recorder.finish()

    video_path = tmp_path / "ep-001" / "video.mp4"
    assert recording.video_path == str(video_path)
    assert video_path.exists()

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == 4


def test_metadata_round_trips(recorder: EpisodeRecorder, tmp_path: Path) -> None:
    recorder.add_frame(_frame())
    recorder.mark_event(0, "decision", "PICK_ONLY mock_object")
    recorder.add_frame(_frame())
    recorder.mark_event(1, "failure", "pick failed")
    recording = recorder.finish()

    metadata_path = tmp_path / "ep-001" / "metadata.json"
    assert recording.metadata_path == str(metadata_path)
    assert metadata_path.exists()

    loaded = EpisodeRecording(**json.loads(metadata_path.read_text()))
    assert loaded == recording
    assert loaded.schema_version
    assert loaded.episode_id == "ep-001"
    assert loaded.fps == 5.0
    assert loaded.frame_count == 2
    assert loaded.frame_width == 12
    assert loaded.frame_height == 8


def test_keyframes_map_events_to_current_frame(recorder: EpisodeRecorder) -> None:
    recorder.add_frame(_frame())
    recorder.add_frame(_frame())
    recorder.mark_event(1, "decision", "reobserve")
    recorder.add_frame(_frame())
    recorder.mark_event(2, "risk", "target possibly occluded")
    recording = recorder.finish()

    events = {kf.kind: kf for kf in recording.keyframes if kf.kind in ("decision", "risk")}
    assert events["decision"].frame_index == 1
    assert events["decision"].step_index == 1
    assert events["risk"].frame_index == 2
    assert events["risk"].step_index == 2
    # Timestamps derive from fps: frame_index / fps.
    assert events["decision"].timestamp_sec == pytest.approx(1 / 5.0)
    assert events["risk"].timestamp_sec == pytest.approx(2 / 5.0)


def test_first_and_last_frames_always_marked(recorder: EpisodeRecorder) -> None:
    for _ in range(3):
        recorder.add_frame(_frame())
    recording = recorder.finish()

    kinds = [kf.kind for kf in recording.keyframes]
    assert kinds[0] == "start"
    assert kinds[-1] == "end"
    start = recording.keyframes[0]
    end = recording.keyframes[-1]
    assert start.frame_index == 0
    assert end.frame_index == 2
    assert end.timestamp_sec == pytest.approx(2 / 5.0)


def test_all_event_kinds_accepted(recorder: EpisodeRecorder) -> None:
    recorder.add_frame(_frame())
    for kind in ("decision", "failure", "risk", "success"):
        recorder.mark_event(0, kind, f"detail-{kind}")
    recording = recorder.finish()
    marked = [kf.kind for kf in recording.keyframes]
    for kind in ("decision", "failure", "risk", "success"):
        assert kind in marked


def test_invalid_event_kind_rejected(recorder: EpisodeRecorder) -> None:
    recorder.add_frame(_frame())
    with pytest.raises(ValueError, match="kind"):
        recorder.mark_event(0, "mystery", "nope")


# --------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------- #


def test_zero_frame_episode_writes_metadata_only(tmp_path: Path) -> None:
    rec = EpisodeRecorder()
    rec.start_episode("ep-empty", tmp_path, fps=5.0)
    recording = rec.finish()

    assert recording.frame_count == 0
    assert recording.video_path is None
    assert recording.frame_width is None
    assert not (tmp_path / "ep-empty" / "video.mp4").exists()
    metadata_path = tmp_path / "ep-empty" / "metadata.json"
    assert metadata_path.exists()
    loaded = EpisodeRecording(**json.loads(metadata_path.read_text()))
    assert loaded == recording


def test_frame_size_change_raises(recorder: EpisodeRecorder) -> None:
    recorder.add_frame(_frame(h=8, w=12))
    with pytest.raises(ValueError, match="size"):
        recorder.add_frame(_frame(h=10, w=12))


def test_finish_is_idempotent(recorder: EpisodeRecorder, tmp_path: Path) -> None:
    recorder.add_frame(_frame())
    first = recorder.finish()
    second = recorder.finish()
    assert first is second
    # No duplicate writes / state changes.
    assert len(list((tmp_path / "ep-001").iterdir())) == 2


def test_calls_before_start_raise() -> None:
    rec = EpisodeRecorder()
    with pytest.raises(RuntimeError, match="start_episode"):
        rec.add_frame(_frame())
    with pytest.raises(RuntimeError, match="start_episode"):
        rec.mark_event(0, "decision", "x")
    with pytest.raises(RuntimeError, match="start_episode"):
        rec.finish()


def test_start_while_active_raises(tmp_path: Path) -> None:
    rec = EpisodeRecorder()
    rec.start_episode("ep-a", tmp_path)
    with pytest.raises(RuntimeError, match="finish"):
        rec.start_episode("ep-b", tmp_path)


def test_recorder_reusable_across_episodes(tmp_path: Path) -> None:
    rec = EpisodeRecorder()
    rec.start_episode("ep-1", tmp_path)
    rec.add_frame(_frame())
    first = rec.finish()
    rec.start_episode("ep-2", tmp_path)
    rec.add_frame(_frame())
    rec.add_frame(_frame())
    second = rec.finish()
    assert first.episode_id == "ep-1"
    assert first.frame_count == 1
    assert second.episode_id == "ep-2"
    assert second.frame_count == 2
    assert (tmp_path / "ep-1" / "video.mp4").exists()
    assert (tmp_path / "ep-2" / "video.mp4").exists()
