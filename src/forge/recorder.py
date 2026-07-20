"""Episode video recorder with event-driven keyframe extraction.

The recorder captures every frame of a forge rollout (one per reset/step, as
produced by ``SimEnv`` observations) into an mp4 video, and lets the caller
annotate per-step events (decision boundaries, failures, risks, successes).
Each marked event maps to the frame index current at mark time; the first and
last frame are always marked. ``finish()`` returns an
:class:`EpisodeRecording` (pydantic) and writes ``metadata.json`` for the
forge critic (Task 3.4).

Output layout::

    <out_dir>/<episode_id>/video.mp4       (omitted for zero-frame episodes)
    <out_dir>/<episode_id>/metadata.json   (always written by finish())
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

# Versioned so the critic can gate on schema evolution.
SCHEMA_VERSION = "1.0"

VIDEO_FILENAME = "video.mp4"
METADATA_FILENAME = "metadata.json"

EventKind = Literal["decision", "failure", "risk", "success"]
KeyframeKind = Literal["decision", "failure", "risk", "success", "start", "end"]


class KeyframeEvent(BaseModel):
    """One keyframe: an event bound to a frame index and step index."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(..., ge=0)
    step_index: int = Field(..., ge=0)
    kind: KeyframeKind
    detail: str = ""
    timestamp_sec: float = Field(..., ge=0.0)


class EpisodeRecording(BaseModel):
    """Result of a finished episode: video path plus keyframe metadata.

    This model is serialized verbatim to ``metadata.json``; the forge critic
    consumes both the video file and this schema.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    fps: float = Field(..., gt=0)
    frame_count: int = Field(..., ge=0)
    frame_width: int | None = None
    frame_height: int | None = None
    video_path: str | None = None
    metadata_path: str
    keyframes: list[KeyframeEvent] = Field(default_factory=list)


class EpisodeRecorder:
    """Records episode frames to video and collects event keyframes.

    Lifecycle: :meth:`start_episode` -> (:meth:`add_frame` /
    :meth:`mark_event`)* -> :meth:`finish`. The recorder is reusable across
    episodes; ``finish`` is idempotent and returns the same
    :class:`EpisodeRecording` on repeat calls.
    """

    def __init__(self) -> None:
        self._active = False
        self._finished: EpisodeRecording | None = None
        self._episode_id: str | None = None
        self._episode_dir: Path | None = None
        self._fps = 10.0
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._frame_size: tuple[int, int] | None = None  # (height, width)
        self._events: list[KeyframeEvent] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start_episode(
        self, episode_id: str, out_dir: str | Path, fps: float = 10.0
    ) -> None:
        """Begin a new episode, creating ``<out_dir>/<episode_id>``.

        Raises:
            ValueError: if ``episode_id`` is empty or ``fps`` is not positive.
            RuntimeError: if an episode is already in progress.
        """
        if self._active:
            raise RuntimeError(
                "EpisodeRecorder: episode already in progress; call finish() first"
            )
        if not episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        self._episode_id = episode_id
        self._episode_dir = Path(out_dir) / episode_id
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._writer = None
        self._frame_count = 0
        self._frame_size = None
        self._events = []
        self._finished = None
        self._active = True
        logger.info(
            "EpisodeRecorder: started episode {!r} in {}", episode_id, self._episode_dir
        )

    def add_frame(self, frame: np.ndarray) -> None:
        """Append one RGB frame (H, W, 3) to the episode video.

        Raises:
            RuntimeError: if no episode is in progress.
            ValueError: if the frame is not (H, W, 3) or its size differs
                from earlier frames in this episode.
        """
        self._require_active("add_frame")
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must have shape (H, W, 3) RGB, got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        size = (int(frame.shape[0]), int(frame.shape[1]))
        if self._frame_size is None:
            self._frame_size = size
        elif self._frame_size != size:
            raise ValueError(
                f"frame size changed mid-episode from {self._frame_size} to {size}"
            )

        if self._writer is None:
            self._writer = self._open_writer(size)
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self._frame_count += 1

    def mark_event(self, step_index: int, kind: EventKind, detail: str = "") -> None:
        """Annotate an event at the frame index current at mark time.

        Args:
            step_index: the episode step the event belongs to.
            kind: one of ``decision``, ``failure``, ``risk``, ``success``.
            detail: free-text context (e.g. the planner decision or the
                primitive failure status).

        Raises:
            RuntimeError: if no episode is in progress.
            ValueError: if ``kind`` is not a valid event kind.
        """
        self._require_active("mark_event")
        if kind not in get_args(EventKind):
            raise ValueError(
                f"invalid event kind {kind!r}; expected one of {get_args(EventKind)}"
            )
        frame_index = max(self._frame_count - 1, 0)
        self._events.append(
            KeyframeEvent(
                frame_index=frame_index,
                step_index=int(step_index),
                kind=kind,
                detail=detail,
                timestamp_sec=frame_index / self._fps,
            )
        )

    def finish(self) -> EpisodeRecording:
        """Close the episode, write ``metadata.json``, return the recording.

        Idempotent: repeat calls return the same object without rewriting
        anything. Zero-frame episodes produce metadata only (no video file);
        first/last keyframes are added when at least one frame exists.

        Raises:
            RuntimeError: if no episode is in progress.
        """
        if self._finished is not None:
            return self._finished
        self._require_active("finish")
        assert self._episode_dir is not None and self._episode_id is not None

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        video_path: str | None = None
        if self._frame_count > 0:
            video_path = str(self._episode_dir / VIDEO_FILENAME)

        keyframes = self._build_keyframes()
        metadata_path = str(self._episode_dir / METADATA_FILENAME)
        recording = EpisodeRecording(
            episode_id=self._episode_id,
            fps=self._fps,
            frame_count=self._frame_count,
            frame_width=self._frame_size[1] if self._frame_size else None,
            frame_height=self._frame_size[0] if self._frame_size else None,
            video_path=video_path,
            metadata_path=metadata_path,
            keyframes=keyframes,
        )
        Path(metadata_path).write_text(
            recording.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(
            "EpisodeRecorder: finished episode {!r} frames={} keyframes={}",
            self._episode_id,
            self._frame_count,
            len(keyframes),
        )
        self._active = False
        self._finished = recording
        return recording

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _require_active(self, method: str) -> None:
        if not self._active:
            raise RuntimeError(
                f"EpisodeRecorder.{method} called outside an episode; "
                "call start_episode() first"
            )

    def _open_writer(self, size: tuple[int, int]) -> cv2.VideoWriter:
        assert self._episode_dir is not None
        height, width = size
        video_path = self._episode_dir / VIDEO_FILENAME
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {video_path}")
        return writer

    def _build_keyframes(self) -> list[KeyframeEvent]:
        if self._frame_count == 0:
            return list(self._events)
        last_index = self._frame_count - 1
        start = KeyframeEvent(
            frame_index=0,
            step_index=0,
            kind="start",
            detail="episode start",
            timestamp_sec=0.0,
        )
        end = KeyframeEvent(
            frame_index=last_index,
            step_index=last_index,
            kind="end",
            detail="episode end",
            timestamp_sec=last_index / self._fps,
        )
        return [start, *self._events, end]
