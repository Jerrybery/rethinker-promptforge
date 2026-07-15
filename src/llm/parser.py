"""JSON extraction and Pydantic validation helpers."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


_CODE_FENCE_PATTERNS = [
    r"```(?:json)?\s*(.*?)```",
    r"`([^`\n]+)`",
]


def extract_json(text: str, schema: type[T]) -> T:
    """Extract JSON from *text* and validate it against *schema*.

    Supports Markdown code fences (`` ```json ... ``` `` or `` ``` ... ``` ``)
    and bare JSON objects/arrays. Raises ``ValueError`` when no valid JSON
    matching the schema can be found.
    """
    stripped = text.strip()

    for pattern in _CODE_FENCE_PATTERNS:
        for match in re.finditer(pattern, stripped, re.DOTALL):
            inner = match.group(1).strip()
            try:
                return schema.model_validate(json.loads(inner))
            except (json.JSONDecodeError, ValidationError):
                pass

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = stripped.find(start_char)
        if start != -1:
            end = stripped.rfind(end_char)
            if end != -1 and end > start:
                try:
                    return schema.model_validate(json.loads(stripped[start : end + 1]))
                except (json.JSONDecodeError, ValidationError):
                    pass

    try:
        return schema.model_validate(json.loads(stripped))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Could not extract valid JSON for {schema.__name__}: {exc}"
        ) from exc
