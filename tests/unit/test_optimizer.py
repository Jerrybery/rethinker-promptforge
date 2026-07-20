"""Unit tests for the forge optimizer LLM."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from forge.critic import StageEvaluation, StageScores
from forge.optimizer import OptimizerLLM, PromptEdit, apply_edits
from forge.registry import EditMetadata, PromptVersion


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _eval(stage: str = "episode", root_cause: str = "planner re-attempted failed grasp") -> StageEvaluation:
    return StageEvaluation(
        stage=stage,
        scores=StageScores(correctness=0.2, efficiency=0.3, safety=0.9),
        root_cause=root_cause,
        evidence="keyframe frame 2 (step 2, failure): grasp missed the block",
    )


def _rejected(reason: str, edit_type: str = "add", idx: int = 0) -> PromptVersion:
    return PromptVersion(
        version_id=f"v{idx:03d}",
        target_agent="planner",
        parent_version=None,
        edit=EditMetadata(edit_type=edit_type, reason=reason, source="optimizer"),
        validation=None,
        status="rejected",
        prompt_path=f"v{idx:03d}.md",
        registered_at="2026-07-20T00:00:00+00:00",
    )


def _edit_dict(new_text: str = "Never re-attempt a failed grasp more than once.", **kw) -> dict:
    base = {
        "target_agent": "planner",
        "edit_type": "add",
        "location": "Rules",
        "new_text": new_text,
        "reason": "critic observed a repeated failed grasp",
    }
    base.update(kw)
    return base


def _client(*payloads: str) -> MagicMock:
    client = MagicMock()
    client.model_id = "test/optimizer-1"
    client.temperature = 0.0
    client.max_tokens = 1024
    client.chat.side_effect = list(payloads)
    return client


def _optimizer(client, **kw) -> OptimizerLLM:
    kw.setdefault("budget_chars", 500)
    return OptimizerLLM(client, target_agent="planner", **kw)


PROMPT_TEXT = (
    "# Planner Prompt\n"
    "\n"
    "## Goal\n"
    "\n"
    "Pick the red block.\n"
    "\n"
    "## Rules\n"
    "\n"
    "- Rule one.\n"
    "- Rule two.\n"
    "\n"
    "## Output\n"
    "\n"
    "JSON only.\n"
)


# --------------------------------------------------------------------- #
# PromptEdit schema
# --------------------------------------------------------------------- #


def test_prompt_edit_valid_variants():
    add = PromptEdit(**_edit_dict())
    assert add.edit_type == "add"
    delete = PromptEdit(
        target_agent="planner", edit_type="delete", location="Output", new_text="", reason="stale"
    )
    assert delete.new_text == ""
    replace = PromptEdit(**_edit_dict(edit_type="replace"))
    assert replace.new_text


def test_prompt_edit_frozen():
    edit = PromptEdit(**_edit_dict())
    with pytest.raises(ValidationError):
        edit.new_text = "mutated"


def test_prompt_edit_delete_must_have_empty_new_text():
    with pytest.raises(ValidationError):
        PromptEdit(**_edit_dict(edit_type="delete"))


def test_prompt_edit_add_and_replace_require_new_text():
    for edit_type in ("add", "replace"):
        with pytest.raises(ValidationError):
            PromptEdit(**_edit_dict(edit_type=edit_type, new_text=""))
        with pytest.raises(ValidationError):
            PromptEdit(**_edit_dict(edit_type=edit_type, new_text="   "))


def test_prompt_edit_rejects_blank_fields_and_bad_type():
    with pytest.raises(ValidationError):
        PromptEdit(**_edit_dict(target_agent=""))
    with pytest.raises(ValidationError):
        PromptEdit(**_edit_dict(location=""))
    with pytest.raises(ValidationError):
        PromptEdit(**_edit_dict(edit_type="rewrite"))


# --------------------------------------------------------------------- #
# propose_edits: happy path + prompt construction
# --------------------------------------------------------------------- #


def test_propose_edits_valid_list():
    client = _client(json.dumps([_edit_dict()]))
    opt = _optimizer(client)
    edits = opt.propose_edits("prompt text", [_eval()], [])
    assert len(edits) == 1
    assert isinstance(edits[0], PromptEdit)
    assert edits[0].target_agent == "planner"
    assert edits[0].location == "Rules"
    assert client.chat.call_count == 1


def test_propose_edits_empty_list_is_valid():
    client = _client("[]")
    opt = _optimizer(client)
    assert opt.propose_edits("prompt text", [_eval()], []) == []


def test_propose_edits_parses_fenced_json():
    payload = "```json\n" + json.dumps([_edit_dict()]) + "\n```"
    client = _client(payload)
    opt = _optimizer(client)
    assert len(opt.propose_edits("prompt text", [_eval()], [])) == 1


def test_propose_edits_prompt_contains_context_and_budget():
    client = _client("[]")
    opt = _optimizer(client, budget_chars=432)
    opt.propose_edits(
        "THE BEST PROMPT BODY",
        [_eval(root_cause="unique root cause marker")],
        [_rejected("unique rejected marker")],
    )
    prompt = client.chat.call_args.args[0][0]["content"]
    assert "THE BEST PROMPT BODY" in prompt
    assert "unique root cause marker" in prompt
    assert "unique rejected marker" in prompt
    assert "432" in prompt
    assert "planner" in prompt


# --------------------------------------------------------------------- #
# propose_edits: parse retry path
# --------------------------------------------------------------------- #


def test_propose_edits_retries_on_malformed_json_then_raises():
    client = _client("not json at all", "still not json")
    opt = _optimizer(client, max_parse_attempts=2)
    with pytest.raises(ValueError):
        opt.propose_edits("prompt", [_eval()], [])
    assert client.chat.call_count == 2


def test_propose_edits_recovers_on_retry():
    client = _client("garbage", json.dumps([_edit_dict()]))
    opt = _optimizer(client, max_parse_attempts=2)
    edits = opt.propose_edits("prompt", [_eval()], [])
    assert len(edits) == 1
    assert client.chat.call_count == 2


# --------------------------------------------------------------------- #
# propose_edits: filtering
# --------------------------------------------------------------------- #


def test_wrong_target_edits_are_dropped():
    payload = json.dumps(
        [
            _edit_dict(target_agent="rethinker"),
            _edit_dict(new_text="Keep this one."),
        ]
    )
    client = _client(payload)
    opt = _optimizer(client)
    edits = opt.propose_edits("prompt", [_eval()], [])
    assert [e.new_text for e in edits] == ["Keep this one."]


def test_near_duplicate_of_rejected_edit_is_dropped():
    rejected = [_rejected("Never re-attempt a failed grasp more than once.")]
    payload = json.dumps(
        [
            _edit_dict(
                new_text="See rules.",
                reason="Never re-attempt a failed grasp more than once.",
            ),
            _edit_dict(
                new_text="State the target object color before each approach.",
                reason="critic saw wrong-object grasp at step 3",
            ),
        ]
    )
    client = _client(payload)
    opt = _optimizer(client)
    edits = opt.propose_edits("prompt", [_eval()], rejected)
    assert [e.new_text for e in edits] == ["State the target object color before each approach."]


def test_over_budget_edits_are_truncated_in_order():
    payload = json.dumps(
        [
            _edit_dict(new_text="A" * 30),
            _edit_dict(new_text="B" * 30, reason="second"),
            _edit_dict(new_text="C" * 10, reason="third"),
        ]
    )
    client = _client(payload)
    opt = _optimizer(client, budget_chars=50)
    edits = opt.propose_edits("prompt", [_eval()], [])
    assert [e.new_text for e in edits] == ["A" * 30]


def test_delete_edits_are_free_under_budget():
    payload = json.dumps(
        [
            _edit_dict(edit_type="delete", new_text="", location="Output"),
            _edit_dict(new_text="D" * 10),
        ]
    )
    client = _client(payload)
    opt = _optimizer(client, budget_chars=10)
    edits = opt.propose_edits("prompt", [_eval()], [])
    assert len(edits) == 2
    assert edits[0].edit_type == "delete"


def test_per_call_budget_overrides_default():
    payload = json.dumps([_edit_dict(new_text="E" * 20)])
    client = _client(payload)
    opt = _optimizer(client, budget_chars=500)
    assert opt.propose_edits("prompt", [_eval()], [], budget_chars=10) == []
    assert client.chat.call_count == 1


# --------------------------------------------------------------------- #
# Budget configuration
# --------------------------------------------------------------------- #


def _write_config(path: Path, optimizer_section: dict) -> Path:
    path.write_text(yaml.safe_dump({"optimizer": optimizer_section}), encoding="utf-8")
    return path


def test_budget_default_comes_from_config(tmp_path):
    cfg = _write_config(
        tmp_path / "models.yaml",
        {
            "model_id": "test/optimizer-1",
            "base_url": "http://localhost:9999/v1",
            "api_key": None,
            "temperature": 0.0,
            "max_tokens": 1024,
            "edit_budget_chars": 321,
        },
    )
    opt = OptimizerLLM(_client("[]"), target_agent="planner", config_path=cfg)
    assert opt.budget_chars == 321


def test_missing_budget_config_key_raises(tmp_path):
    cfg = _write_config(tmp_path / "models.yaml", {"model_id": "x"})
    with pytest.raises(ValueError, match="edit_budget_chars"):
        OptimizerLLM(_client("[]"), target_agent="planner", config_path=cfg)


def test_explicit_budget_skips_config(tmp_path):
    cfg = _write_config(tmp_path / "models.yaml", {"model_id": "x"})
    opt = OptimizerLLM(_client("[]"), target_agent="planner", budget_chars=77, config_path=cfg)
    assert opt.budget_chars == 77


def test_from_config_builds_client_and_reads_budget(tmp_path):
    cfg = _write_config(
        tmp_path / "models.yaml",
        {
            "model_id": "test/optimizer-1",
            "base_url": "http://localhost:9999/v1",
            "api_key": None,
            "temperature": 0.0,
            "max_tokens": 1024,
            "edit_budget_chars": 222,
        },
    )
    opt = OptimizerLLM.from_config("planner", config_path=cfg)
    assert opt.budget_chars == 222
    assert opt.target_agent == "planner"


def test_from_config_requires_model_id(tmp_path):
    cfg = _write_config(
        tmp_path / "models.yaml",
        {"model_id": None, "base_url": "http://localhost:9999/v1", "edit_budget_chars": 100},
    )
    with pytest.raises(ValueError, match="model_id"):
        OptimizerLLM.from_config("planner", config_path=cfg)


# --------------------------------------------------------------------- #
# apply_edits
# --------------------------------------------------------------------- #


def test_apply_edits_replace_section_body():
    edits = [
        PromptEdit(**_edit_dict(edit_type="replace", new_text="- Always verify the target before grasping."))
    ]
    out = apply_edits(PROMPT_TEXT, edits)
    assert out == (
        "# Planner Prompt\n"
        "\n"
        "## Goal\n"
        "\n"
        "Pick the red block.\n"
        "\n"
        "## Rules\n"
        "\n"
        "- Always verify the target before grasping.\n"
        "\n"
        "## Output\n"
        "\n"
        "JSON only.\n"
    )


def test_apply_edits_add_appends_within_section():
    edits = [PromptEdit(**_edit_dict(location="Goal", new_text="Avoid the occlusion zones."))]
    out = apply_edits(PROMPT_TEXT, edits)
    assert out == (
        "# Planner Prompt\n"
        "\n"
        "## Goal\n"
        "\n"
        "Pick the red block.\n"
        "\n"
        "Avoid the occlusion zones.\n"
        "\n"
        "## Rules\n"
        "\n"
        "- Rule one.\n"
        "- Rule two.\n"
        "\n"
        "## Output\n"
        "\n"
        "JSON only.\n"
    )


def test_apply_edits_delete_removes_section():
    edits = [PromptEdit(target_agent="planner", edit_type="delete", location="Output", new_text="", reason="stale")]
    out = apply_edits(PROMPT_TEXT, edits)
    assert out == (
        "# Planner Prompt\n"
        "\n"
        "## Goal\n"
        "\n"
        "Pick the red block.\n"
        "\n"
        "## Rules\n"
        "\n"
        "- Rule one.\n"
        "- Rule two.\n"
    )


def test_apply_edits_location_match_is_case_insensitive():
    edits = [PromptEdit(**_edit_dict(location="rules", new_text="- New rule."))]
    out = apply_edits(PROMPT_TEXT, edits)
    assert "- New rule." in out
    assert "- Rule one." in out  # add, not replace


def test_apply_edits_unknown_location_is_skipped_not_fatal():
    edits = [
        PromptEdit(**_edit_dict(location="Nonexistent", new_text="ghost text")),
        PromptEdit(**_edit_dict(location="Goal", new_text="Real addition.")),
    ]
    out = apply_edits(PROMPT_TEXT, edits)
    assert "ghost text" not in out
    assert "Real addition." in out
    assert "## Rules" in out


def test_apply_edits_round_trip_from_propose_edits():
    client = _client(json.dumps([_edit_dict(new_text="- Never re-attempt a failed grasp more than once.")]))
    opt = _optimizer(client)
    edits = opt.propose_edits(PROMPT_TEXT, [_eval()], [])
    out = apply_edits(PROMPT_TEXT, edits)
    assert "- Never re-attempt a failed grasp more than once." in out
    assert out.startswith("# Planner Prompt")
