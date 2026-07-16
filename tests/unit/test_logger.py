"""Unit tests for the episode logger."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.logger import EpisodeLogger


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Calling close() more than once must not raise."""
    log_path = tmp_path / "episode.jsonl"
    logger = EpisodeLogger(log_path)
    logger.close()
    logger.close()
    assert logger._closed is True


def test_context_manager_closes_on_exit(tmp_path: Path) -> None:
    """EpisodeLogger should act as a context manager and close on exit."""
    log_path = tmp_path / "episode.jsonl"
    with EpisodeLogger(log_path) as logger:
        assert logger._closed is False
        logger.log_event("test", {"value": 1})
    assert logger._closed is True
    logger.close()  # must remain idempotent after context-manager exit


def test_context_manager_closes_even_if_log_event_raises(tmp_path: Path) -> None:
    """The context manager must close the handle even if logging raises."""
    log_path = tmp_path / "episode.jsonl"

    class BrokenLogger(EpisodeLogger):
        """Logger whose writes always fail."""

        def _write(self, record) -> None:  # type: ignore[override]
            raise RuntimeError("write failed")

    logger = BrokenLogger(log_path)
    with pytest.raises(RuntimeError, match="write failed"):
        with logger:
            logger.log_event("test", {"value": 1})
    assert logger._closed is True


def test_log_event_persisted(tmp_path: Path) -> None:
    """Logged events should be flushed to disk as JSON lines."""
    log_path = tmp_path / "episode.jsonl"
    with EpisodeLogger(log_path) as logger:
        logger.log_event("episode_finished", {"termination_reason": "stop", "steps": 2})

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    event = __import__("json").loads(lines[0])
    assert event["event"] == "episode_finished"
    assert event["payload"]["termination_reason"] == "stop"
    assert event["payload"]["steps"] == 2
