"""Structured episode logger writing JSON lines per step."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.schema import EpisodeStep


class EpisodeLogger:
    """Append-only JSON-lines logger for episodes.

    Each call writes a single JSON object terminated by a newline and flushed
    to disk so logs are durable even if the process crashes mid-episode.
    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("a", encoding="utf-8")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write(self, record: dict[str, Any]) -> None:
        json.dump(record, self._file, ensure_ascii=False)
        self._file.write("\n")
        self._file.flush()

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        """Log an episode-level metadata record."""
        self._write(
            {
                "event": "metadata",
                "timestamp": self._now(),
                "payload": metadata,
            }
        )

    def log_step(self, step: EpisodeStep) -> None:
        """Log a single :class:`common.schema.EpisodeStep`."""
        self._write(
            {
                "event": "step",
                "timestamp": self._now(),
                "step_index": step.step_index,
                "step": step.model_dump(mode="json"),
            }
        )

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Log an arbitrary event with the given type and payload."""
        self._write(
            {
                "event": event_type,
                "timestamp": self._now(),
                "payload": payload,
            }
        )

    def close(self) -> None:
        """Close the underlying log file handle."""
        self._file.close()
