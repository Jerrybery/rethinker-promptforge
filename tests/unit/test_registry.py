"""Unit tests for the forge prompt registry (Task 3.5).

All tests use ``tmp_path`` roots and injected timestamps — no wall clock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.registry import (
    EditMetadata,
    ForgePromptRegistry,
    PromptVersion,
    ValidationRecord,
)

TS0 = "2026-07-20T00:00:00+00:00"
TS1 = "2026-07-20T00:01:00+00:00"
TS2 = "2026-07-20T00:02:00+00:00"
TS3 = "2026-07-20T00:03:00+00:00"


def _edit(source: str = "optimizer", reason: str = "improve scores") -> EditMetadata:
    return EditMetadata(edit_type="rewrite", reason=reason, source=source)


@pytest.fixture
def registry(tmp_path: Path) -> ForgePromptRegistry:
    return ForgePromptRegistry(tmp_path / "prompts")


# --------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------- #


def test_register_creates_candidate_with_prompt_file(
    registry: ForgePromptRegistry, tmp_path: Path
) -> None:
    version = registry.register(
        "You are a planner.", "planner", _edit(source="hand"), timestamp=TS0
    )

    assert isinstance(version, PromptVersion)
    assert version.version_id == "v000"
    assert version.target_agent == "planner"
    assert version.parent_version is None
    assert version.status == "candidate"
    assert version.validation is None
    assert version.registered_at == TS0
    assert version.edit.source == "hand"
    prompt_file = tmp_path / "prompts" / "v000.md"
    assert version.prompt_path == "v000.md"
    assert prompt_file.read_text(encoding="utf-8") == "You are a planner."


def test_register_assigns_sequential_ids_with_parent(
    registry: ForgePromptRegistry,
) -> None:
    v0 = registry.register("base prompt", "planner", _edit(source="hand"), timestamp=TS0)
    v1 = registry.register(
        "edited prompt",
        "planner",
        _edit(),
        parent_version=v0.version_id,
        timestamp=TS1,
    )

    assert v1.version_id == "v001"
    assert v1.parent_version == "v000"


def test_register_rejects_empty_text(registry: ForgePromptRegistry) -> None:
    with pytest.raises(ValueError, match="text"):
        registry.register("   ", "planner", _edit(), timestamp=TS0)


def test_register_rejects_empty_target_agent(registry: ForgePromptRegistry) -> None:
    with pytest.raises(ValueError, match="target_agent"):
        registry.register("prompt", "", _edit(), timestamp=TS0)


def test_register_rejects_unknown_parent(registry: ForgePromptRegistry) -> None:
    with pytest.raises(KeyError, match="v999"):
        registry.register(
            "prompt", "planner", _edit(), parent_version="v999", timestamp=TS0
        )


def test_register_rejects_parent_from_other_target(
    registry: ForgePromptRegistry,
) -> None:
    v0 = registry.register("base", "planner", _edit(source="hand"), timestamp=TS0)
    with pytest.raises(ValueError, match="targets"):
        registry.register(
            "other", "rethinker", _edit(), parent_version=v0.version_id, timestamp=TS1
        )


# --------------------------------------------------------------------- #
# record_validation: accept/reject bookkeeping
# --------------------------------------------------------------------- #


def test_reject_keeps_version_queryable_with_metrics(
    registry: ForgePromptRegistry,
) -> None:
    v0 = registry.register("bad prompt", "planner", _edit(), timestamp=TS0)
    updated = registry.record_validation(
        v0.version_id, {"score": 0.2, "success_rate": 0.1}, accepted=False, timestamp=TS1
    )

    assert updated.status == "rejected"
    assert updated.validation is not None
    assert updated.validation.accepted is False
    assert updated.validation.metrics == {"score": 0.2, "success_rate": 0.1}
    assert updated.validation.timestamp == TS1
    rejected = registry.history("planner", status="rejected")
    assert [v.version_id for v in rejected] == ["v000"]


def test_accept_promotes_candidate_to_best(registry: ForgePromptRegistry) -> None:
    v0 = registry.register("good prompt", "planner", _edit(), timestamp=TS0)
    updated = registry.record_validation(
        v0.version_id, {"score": 0.9}, accepted=True, timestamp=TS1
    )

    assert updated.status == "best"
    assert registry.best("planner").version_id == "v000"


def test_accept_demotes_previous_best_single_best_invariant(
    registry: ForgePromptRegistry,
) -> None:
    v0 = registry.register("v0", "planner", _edit(source="hand"), timestamp=TS0)
    registry.record_validation(v0.version_id, {"score": 0.8}, accepted=True, timestamp=TS1)
    v1 = registry.register(
        "v1", "planner", _edit(), parent_version="v000", timestamp=TS2
    )
    registry.record_validation(v1.version_id, {"score": 0.9}, accepted=True, timestamp=TS3)

    best = registry.best("planner")
    assert best.version_id == "v001"
    former = registry.history("planner", status="accepted")
    assert [v.version_id for v in former] == ["v000"]
    # exactly one best across the whole history
    statuses = [v.status for v in registry.history("planner")]
    assert statuses.count("best") == 1


def test_record_validation_rejects_non_candidate(
    registry: ForgePromptRegistry,
) -> None:
    v0 = registry.register("prompt", "planner", _edit(), timestamp=TS0)
    registry.record_validation(v0.version_id, {"score": 0.5}, accepted=False, timestamp=TS1)
    with pytest.raises(ValueError, match="candidate"):
        registry.record_validation(
            v0.version_id, {"score": 0.6}, accepted=True, timestamp=TS2
        )


def test_record_validation_unknown_version(registry: ForgePromptRegistry) -> None:
    with pytest.raises(KeyError, match="v000"):
        registry.record_validation("v000", {"score": 0.5}, accepted=True, timestamp=TS0)


def test_best_raises_when_none(registry: ForgePromptRegistry) -> None:
    registry.register("unvalidated", "planner", _edit(), timestamp=TS0)
    with pytest.raises(LookupError, match="planner"):
        registry.best("planner")


# --------------------------------------------------------------------- #
# history / diff
# --------------------------------------------------------------------- #


def test_history_filters_by_status(registry: ForgePromptRegistry) -> None:
    v0 = registry.register("a", "planner", _edit(source="hand"), timestamp=TS0)
    v1 = registry.register("b", "planner", _edit(), timestamp=TS1)
    registry.record_validation(v0.version_id, {"score": 0.1}, accepted=False, timestamp=TS2)

    candidates = registry.history("planner", status="candidate")
    assert [v.version_id for v in candidates] == [v1.version_id]
    full = registry.history("planner")
    assert [v.version_id for v in full] == ["v000", "v001"]


def test_history_isolates_target_agents(registry: ForgePromptRegistry) -> None:
    registry.register("planner prompt", "planner", _edit(source="hand"), timestamp=TS0)
    registry.register("rethinker prompt", "rethinker", _edit(source="hand"), timestamp=TS1)

    planner = registry.history("planner")
    rethinker = registry.history("rethinker")
    assert [v.target_agent for v in planner] == ["planner"]
    assert [v.target_agent for v in rethinker] == ["rethinker"]
    assert registry.history("critic") == []


def test_diff_returns_unified_diff(registry: ForgePromptRegistry) -> None:
    v0 = registry.register("line one\nline two\n", "planner", _edit(source="hand"), timestamp=TS0)
    v1 = registry.register(
        "line 1\nline two\nline three\n", "planner", _edit(), timestamp=TS1
    )

    diff = registry.diff(v0.version_id, v1.version_id)
    assert "--- v000" in diff
    assert "+++ v001" in diff
    assert "-line one" in diff
    assert "+line 1" in diff
    assert "+line three" in diff


def test_diff_identical_versions_empty(registry: ForgePromptRegistry) -> None:
    v0 = registry.register("same\n", "planner", _edit(source="hand"), timestamp=TS0)
    v1 = registry.register("same\n", "planner", _edit(), timestamp=TS1)
    assert registry.diff(v0.version_id, v1.version_id) == ""


# --------------------------------------------------------------------- #
# persistence / materialize
# --------------------------------------------------------------------- #


def test_persistence_across_reload(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    reg = ForgePromptRegistry(root)
    v0 = reg.register("base", "planner", _edit(source="hand"), timestamp=TS0)
    reg.record_validation(v0.version_id, {"score": 0.7}, accepted=True, timestamp=TS1)
    v1 = reg.register("edit", "planner", _edit(), parent_version="v000", timestamp=TS2)

    index_path = root / "registry.json"
    assert index_path.exists()
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    assert [v["version_id"] for v in raw["versions"]] == ["v000", "v001"]

    reloaded = ForgePromptRegistry(root)
    assert reloaded.best("planner").version_id == "v000"
    history = reloaded.history("planner")
    assert [v.version_id for v in history] == ["v000", "v001"]
    assert history[1].parent_version == "v000"
    assert history[1].status == "candidate"
    assert reloaded.text(v1.version_id) == "edit"

    # lineage continues seamlessly after reload
    reloaded.record_validation(v1.version_id, {"score": 0.9}, accepted=True, timestamp=TS3)
    assert reloaded.best("planner").version_id == "v001"
    assert [v.status for v in reloaded.history("planner")].count("best") == 1


def test_materialize_best_writes_file(
    registry: ForgePromptRegistry, tmp_path: Path
) -> None:
    v0 = registry.register("best prompt text", "planner", _edit(), timestamp=TS0)
    registry.record_validation(v0.version_id, {"score": 0.9}, accepted=True, timestamp=TS1)

    out_path = tmp_path / "out" / "best_planner_prompt.md"
    result = registry.materialize_best("planner", out_path)
    assert result == out_path
    assert out_path.read_text(encoding="utf-8") == "best prompt text"


def test_materialize_best_raises_without_best(
    registry: ForgePromptRegistry, tmp_path: Path
) -> None:
    with pytest.raises(LookupError):
        registry.materialize_best("planner", tmp_path / "best_planner_prompt.md")


def test_text_unknown_version_raises(registry: ForgePromptRegistry) -> None:
    with pytest.raises(KeyError, match="v000"):
        registry.text("v000")


# --------------------------------------------------------------------- #
# get
# --------------------------------------------------------------------- #


def test_get_returns_version_record(registry: ForgePromptRegistry) -> None:
    version = registry.register("some prompt", "planner", _edit(), timestamp=TS0)

    fetched = registry.get(version.version_id)
    assert fetched == version


def test_get_unknown_version_raises(registry: ForgePromptRegistry) -> None:
    with pytest.raises(KeyError, match="v999"):
        registry.get("v999")
