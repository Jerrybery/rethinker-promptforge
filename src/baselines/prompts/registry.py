"""Minimal prompt registry for baseline prompts."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar


class PromptRegistry:
    """Versioned loader for baseline system/user prompts.

    New prompt versions are added as ``system_v{N}.md`` / ``user_v{N}.md``
    files under this package. The registry resolves them by version tag.
    """

    _PROMPT_DIR: ClassVar[Path] = Path(__file__).resolve().parent

    @classmethod
    def load(cls, version: str) -> tuple[str, str]:
        """Load the system and user prompt templates for *version*.

        Args:
            version: version tag such as ``"v0"``.

        Returns:
            A tuple ``(system_template, user_template)``.

        Raises:
            FileNotFoundError: if either prompt file is missing.
        """
        system_path = cls._PROMPT_DIR / f"system_{version}.md"
        user_path = cls._PROMPT_DIR / f"user_{version}.md"
        if not system_path.exists():
            raise FileNotFoundError(f"System prompt not found: {system_path}")
        if not user_path.exists():
            raise FileNotFoundError(f"User prompt not found: {user_path}")
        return system_path.read_text(encoding="utf-8"), user_path.read_text(
            encoding="utf-8"
        )

    @classmethod
    def versions(cls) -> list[str]:
        """Return all available version tags."""
        versions = set()
        for path in cls._PROMPT_DIR.glob("*.md"):
            stem = path.stem
            if stem.startswith("system_") or stem.startswith("user_"):
                versions.add(stem.split("_", 1)[1])
        return sorted(versions)
