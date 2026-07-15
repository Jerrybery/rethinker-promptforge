"""Unit tests for JSON extraction and Pydantic validation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from llm.parser import extract_json


class DummySchema(BaseModel):
    """Minimal schema for parser tests."""

    answer: int
    reason: str


def test_extract_json_bare_object() -> None:
    text = '{"answer": 42, "reason": "because"}'
    result = extract_json(text, DummySchema)
    assert result.answer == 42
    assert result.reason == "because"


def test_extract_json_markdown_fence() -> None:
    text = """
Some explanation.

```json
{"answer": 7, "reason": "lucky"}
```

More text.
"""
    result = extract_json(text, DummySchema)
    assert result.answer == 7
    assert result.reason == "lucky"


def test_extract_json_plain_fence() -> None:
    text = """
```
{"answer": 99, "reason": "plain"}
```
"""
    result = extract_json(text, DummySchema)
    assert result.answer == 99


def test_extract_json_inline_backtick() -> None:
    text = 'Use `{"answer": 1, "reason": "inline"}` for this.'
    result = extract_json(text, DummySchema)
    assert result.answer == 1


def test_extract_json_invalid_raises() -> None:
    with pytest.raises(ValueError):
        extract_json("not json", DummySchema)


def test_extract_json_validation_error_raises() -> None:
    text = '{"answer": "not an int", "reason": "bad"}'
    with pytest.raises(ValueError):
        extract_json(text, DummySchema)
