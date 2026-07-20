"""Forge prompt registry: versioned prompt lineage with accept/reject bookkeeping.

Manages an evolving lineage of candidate prompts per target agent (e.g.
``"planner"``, ``"rethinker"``). Each registered prompt becomes a
:class:`PromptVersion` with edit provenance; the forge validator (Task 3.7)
then calls :meth:`ForgePromptRegistry.record_validation` with the validation
metrics and an accept/reject decision. The registry enforces the bookkeeping
invariants — the accept/reject *rule* itself lives in the caller:

- new registrations start as ``candidate``;
- a rejected candidate becomes ``rejected`` and stays queryable with its
  metrics (the optimizer, Task 3.6, reads this rejected-edit history);
- an accepted candidate becomes the single ``best`` for its target agent,
  demoting any previous best to ``accepted``;
- only ``candidate`` versions can be validated (no silent re-validation).

Storage layout (human-inspectable; Global Constraint: prompts versioned and
tracked)::

    <root>/v000.md          one prompt-text file per version
    <root>/registry.json    index of all PromptVersion records

All writes are atomic-ish (write ``*.tmp`` then ``os.replace``). Timestamps
are injected by callers so unit tests need no wall clock.
"""

from __future__ import annotations

import difflib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

# Versioned so downstream tasks (3.6-3.8) can gate on index schema evolution.
SCHEMA_VERSION = "1.0"

INDEX_FILENAME = "registry.json"

VersionStatus = Literal["candidate", "accepted", "rejected", "best"]

_VERSION_ID_PATTERN = re.compile(r"^v(\d+)$")


class EditMetadata(BaseModel):
    """Provenance of one prompt edit.

    ``source`` records who produced the edit (e.g. ``"hand"`` for seeded
    prompts, ``"optimizer"`` for Task 3.6 proposals); ``edit_type`` describes
    the edit kind (e.g. ``"rewrite"``, ``"append_rule"``); ``reason`` is the
    human/optimizer rationale.
    """

    model_config = ConfigDict(frozen=True)

    edit_type: str = Field(..., min_length=1)
    reason: str = ""
    source: str = Field(..., min_length=1)


class ValidationRecord(BaseModel):
    """Outcome of validating one candidate prompt.

    ``metrics`` is an open dict of floats (e.g. success rate, mean critic
    score) written by the validator; ``accepted`` is the caller's
    accept/reject decision; ``timestamp`` is an ISO-8601 string.
    """

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float] = Field(default_factory=dict)
    accepted: bool
    timestamp: str = Field(..., min_length=1)
    detail: str = ""


class PromptVersion(BaseModel):
    """One versioned prompt in a target agent's lineage.

    ``prompt_path`` is the prompt-text filename relative to the registry
    root, so records stay portable across checkouts.
    """

    model_config = ConfigDict(frozen=True)

    version_id: str = Field(..., min_length=1)
    target_agent: str = Field(..., min_length=1)
    parent_version: str | None
    edit: EditMetadata
    validation: ValidationRecord | None
    status: VersionStatus
    prompt_path: str = Field(..., min_length=1)
    registered_at: str = Field(..., min_length=1)


class _RegistryIndex(BaseModel):
    """On-disk ``registry.json`` schema."""

    schema_version: str = SCHEMA_VERSION
    versions: list[PromptVersion] = Field(default_factory=list)


class ForgePromptRegistry:
    """Versioned prompt lineage store with accept/reject bookkeeping.

    Args:
        root: directory holding the prompt files and ``registry.json``;
            created (with parents) if missing. An existing directory is
            reloaded, so a new instance on the same root resumes the lineage.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, PromptVersion] = {}
        index_path = self._root / INDEX_FILENAME
        if index_path.exists():
            index = _RegistryIndex.model_validate_json(
                index_path.read_text(encoding="utf-8")
            )
            if index.schema_version != SCHEMA_VERSION:
                logger.warning(
                    "Registry index schema_version {} != expected {}",
                    index.schema_version,
                    SCHEMA_VERSION,
                )
            for version in index.versions:
                self._versions[version.version_id] = version
        logger.info(
            "ForgePromptRegistry: root={}, {} versions loaded",
            self._root,
            len(self._versions),
        )

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(
        self,
        text: str,
        target_agent: str,
        metadata: EditMetadata,
        *,
        parent_version: str | None = None,
        timestamp: str | None = None,
    ) -> PromptVersion:
        """Register a new candidate prompt, returning its :class:`PromptVersion`.

        Args:
            text: full prompt text; stored as ``<version_id>.md``.
            target_agent: agent this prompt targets (e.g. ``"planner"``).
            metadata: edit provenance (edit type, reason, source).
            parent_version: optional lineage parent; must exist and belong
                to the same ``target_agent``.
            timestamp: ISO-8601 registration time; defaults to now (UTC).

        Raises:
            ValueError: if ``text`` is blank, ``target_agent`` is empty, or
                the parent belongs to a different target agent.
            KeyError: if ``parent_version`` is unknown.
        """
        if not text.strip():
            raise ValueError("text must be a non-blank prompt string")
        if not target_agent:
            raise ValueError("target_agent must be a non-empty string")
        if parent_version is not None:
            parent = self._get(parent_version)
            if parent.target_agent != target_agent:
                raise ValueError(
                    f"parent {parent_version!r} targets {parent.target_agent!r}, "
                    f"not {target_agent!r}"
                )

        version_id = self._next_version_id()
        prompt_path = f"{version_id}.md"
        _write_atomic(self._root / prompt_path, text)
        version = PromptVersion(
            version_id=version_id,
            target_agent=target_agent,
            parent_version=parent_version,
            edit=metadata,
            validation=None,
            status="candidate",
            prompt_path=prompt_path,
            registered_at=timestamp or _utc_now(),
        )
        self._versions[version_id] = version
        self._save_index()
        logger.info(
            "Registered {} for {} (parent={}, source={})",
            version_id,
            target_agent,
            parent_version,
            metadata.source,
        )
        return version

    # ------------------------------------------------------------------ #
    # Validation bookkeeping
    # ------------------------------------------------------------------ #

    def record_validation(
        self,
        version_id: str,
        metrics: dict[str, float],
        accepted: bool,
        *,
        timestamp: str | None = None,
        detail: str = "",
    ) -> PromptVersion:
        """Record a validation outcome for a candidate version.

        On ``accepted=True`` the version becomes the single ``best`` for its
        target agent, demoting any previous best to ``accepted``. On
        ``accepted=False`` it becomes ``rejected`` but stays queryable with
        its metrics. Returns the updated :class:`PromptVersion`.

        Raises:
            KeyError: if ``version_id`` is unknown.
            ValueError: if the version is not a ``candidate``.
        """
        version = self._get(version_id)
        if version.status != "candidate":
            raise ValueError(
                f"version {version_id!r} is not a candidate "
                f"(status={version.status!r}); refusing re-validation"
            )
        validation = ValidationRecord(
            metrics=dict(metrics),
            accepted=accepted,
            timestamp=timestamp or _utc_now(),
            detail=detail,
        )
        if accepted:
            current_best = self._best_or_none(version.target_agent)
            if current_best is not None:
                self._versions[current_best.version_id] = current_best.model_copy(
                    update={"status": "accepted"}
                )
                logger.info(
                    "Demoted previous best {} for {}",
                    current_best.version_id,
                    version.target_agent,
                )
            new_status: VersionStatus = "best"
        else:
            new_status = "rejected"
        updated = version.model_copy(
            update={"validation": validation, "status": new_status}
        )
        self._versions[version_id] = updated
        self._save_index()
        logger.info(
            "Validated {} for {}: accepted={}, status={}",
            version_id,
            version.target_agent,
            accepted,
            new_status,
        )
        return updated

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def best(self, target_agent: str) -> PromptVersion:
        """Return the current best version for ``target_agent``.

        Raises:
            LookupError: if no version has been accepted for the agent.
        """
        version = self._best_or_none(target_agent)
        if version is None:
            raise LookupError(
                f"no best prompt for target_agent {target_agent!r} "
                "(nothing accepted yet)"
            )
        return version

    def get(self, version_id: str) -> PromptVersion:
        """Return the version record for ``version_id``.

        Raises:
            KeyError: if ``version_id`` is unknown.
        """
        return self._get(version_id)

    def history(
        self, target_agent: str, status: VersionStatus | None = None
    ) -> list[PromptVersion]:
        """Return versions for ``target_agent`` in registration order.

        Args:
            status: optional filter (e.g. ``"rejected"`` for the optimizer's
                rejected-edit history).
        """
        versions = [
            v for v in self._sorted_versions() if v.target_agent == target_agent
        ]
        if status is not None:
            versions = [v for v in versions if v.status == status]
        return versions

    def text(self, version_id: str) -> str:
        """Return the prompt text for a version.

        Raises:
            KeyError: if ``version_id`` is unknown.
        """
        version = self._get(version_id)
        return (self._root / version.prompt_path).read_text(encoding="utf-8")

    def diff(self, version_a: str, version_b: str) -> str:
        """Return a unified diff of the prompt texts of two versions."""
        a_lines = self.text(version_a).splitlines()
        b_lines = self.text(version_b).splitlines()
        return "\n".join(
            difflib.unified_diff(
                a_lines, b_lines, fromfile=version_a, tofile=version_b, lineterm=""
            )
        )

    def materialize_best(self, target_agent: str, out_path: str | Path) -> Path:
        """Write the current best prompt text for ``target_agent`` to ``out_path``.

        Used by the forge runner (Task 3.8) to snapshot
        ``best_planner_prompt.md``. Parent directories are created.

        Raises:
            LookupError: if no best version exists for the agent.
        """
        version = self.best(target_agent)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(out, self.text(version.version_id))
        logger.info(
            "Materialized best {} prompt ({}) to {}",
            target_agent,
            version.version_id,
            out,
        )
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _get(self, version_id: str) -> PromptVersion:
        try:
            return self._versions[version_id]
        except KeyError:
            raise KeyError(f"unknown prompt version {version_id!r}") from None

    def _best_or_none(self, target_agent: str) -> PromptVersion | None:
        for version in self._versions.values():
            if version.target_agent == target_agent and version.status == "best":
                return version
        return None

    def _next_version_id(self) -> str:
        highest = -1
        for version_id in self._versions:
            match = _VERSION_ID_PATTERN.match(version_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"v{highest + 1:03d}"

    def _sorted_versions(self) -> list[PromptVersion]:
        return sorted(self._versions.values(), key=_version_sort_key)

    def _save_index(self) -> None:
        index = _RegistryIndex(versions=self._sorted_versions())
        _write_atomic(
            self._root / INDEX_FILENAME, index.model_dump_json(indent=2) + "\n"
        )


def _version_sort_key(version: PromptVersion) -> tuple[int, str]:
    match = _VERSION_ID_PATTERN.match(version.version_id)
    return (int(match.group(1)) if match else -1, version.version_id)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a sibling tmp file + rename."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)
