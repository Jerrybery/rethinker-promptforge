"""Optimizer LLM: proposes bounded edits to the current best prompt.

Consumes the current best prompt text, the critic's :class:`StageEvaluation`
list (Task 3.4 contract) and the rejected-edit history from the forge
registry (Task 3.5 contract: ``history(target_agent, status="rejected")``),
and produces a bounded list of :class:`PromptEdit` proposals. The runner
(Task 3.8) materializes candidates with :func:`apply_edits`.

Budget semantics ("text learning rate"): the budget counts *inserted or
replacement characters only* — ``len(new_text)`` summed over kept edits in
model-proposed order. Deletions are free (they only shrink the prompt).
Edits are truncated to a prefix: the first edit that would push the running
total over the budget is dropped together with every edit after it. The
default budget lives in ``configs/models.yaml`` under
``optimizer.edit_budget_chars`` (no magic number in code); an explicit
``budget_chars`` argument always wins.

Rejected-edit dedup heuristic: an edit is dropped as a near-duplicate when
``difflib.SequenceMatcher`` ratio over the normalized (lowercased,
whitespace-collapsed) fingerprint ``"{edit_type}\\n{reason}\\n{new_text}"``
vs. the rejected version's ``"{edit_type}\\n{reason}"`` is >=
``similarity_threshold`` (default 0.8). The registry does not store the
rejected edit's text body, so the comparison is reason-text based.

LLM backend: any client with ``chat(messages) -> str`` plus
``model_id``/``temperature``/``max_tokens`` attributes. The intended backend
is a strong cloud model via :meth:`OptimizerLLM.from_config` (the
``optimizer`` section of ``configs/models.yaml``, mirroring
``cloud_critic``); the local ``VLLMClient`` (``vllm`` section) or
``CloudVLMClient`` are documented drop-in alternatives — pass them as
``client``.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from string import Template
from typing import Literal, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from forge.critic import StageEvaluation
from forge.registry import PromptVersion
from llm.parser import extract_json
from rethinker_promptforge.config import load_config

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"

#: Normalized-text similarity at/above which a proposed edit counts as a
#: near-duplicate of a rejected edit and is dropped.
DEFAULT_SIMILARITY_THRESHOLD = 0.8

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


class PromptEdit(BaseModel):
    """One bounded edit to a target agent's prompt.

    ``location`` is a section heading of the prompt text (matched
    case-insensitively, ``#`` markers stripped). ``new_text`` is the text to
    insert (``add``) or substitute for the section body (``replace``); it
    must be empty for ``delete`` and non-empty otherwise.
    """

    model_config = ConfigDict(frozen=True)

    target_agent: str = Field(..., min_length=1)
    edit_type: Literal["add", "delete", "replace"]
    location: str = Field(..., min_length=1)
    new_text: str = ""
    reason: str = ""

    @model_validator(mode="after")
    def _check_new_text(self) -> "PromptEdit":
        if self.edit_type == "delete" and self.new_text:
            raise ValueError("delete edits must have empty new_text")
        if self.edit_type in ("add", "replace") and not self.new_text.strip():
            raise ValueError(f"{self.edit_type} edits require non-empty new_text")
        return self


class _EditList(RootModel):
    """Parse wrapper: the LLM is instructed to output a bare JSON list."""

    root: list[PromptEdit]


class OptimizerLLM:
    """Proposes bounded prompt edits from critic evidence + reject history.

    Args:
        client: chat client (see module docstring for the contract).
        target_agent: the agent whose prompt this optimizer edits; edits
            proposed for any other agent are filtered out.
        prompt_dir: directory holding ``optimizer_{version}.md`` templates;
            defaults to the bundled prompts.
        prompt_version: prompt template version tag (fixed and logged).
        budget_chars: default text budget per proposal round. ``None``
            reads ``optimizer.edit_budget_chars`` from the models config.
        similarity_threshold: near-duplicate ratio for rejected-edit dedup.
        max_parse_attempts: LLM calls per round before giving up on
            unparseable output (>= 1).
        config_path: models config path (used when ``budget_chars`` is
            ``None`` and by :meth:`from_config`).

    Raises:
        ValueError: if ``target_agent`` is blank, ``max_parse_attempts`` < 1,
            or the budget default is missing from the config.
    """

    def __init__(
        self,
        client,
        target_agent: str,
        *,
        prompt_dir: str | Path | None = None,
        prompt_version: str = "v0",
        budget_chars: int | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_parse_attempts: int = 2,
        config_path: str | Path | None = None,
    ) -> None:
        if not target_agent.strip():
            raise ValueError("target_agent must be non-empty")
        if max_parse_attempts < 1:
            raise ValueError(f"max_parse_attempts must be >= 1, got {max_parse_attempts}")
        self._client = client
        self.target_agent = target_agent
        self.prompt_version = prompt_version
        self.similarity_threshold = similarity_threshold
        self.max_parse_attempts = max_parse_attempts
        if budget_chars is None:
            cfg = load_config(config_path or _DEFAULT_CONFIG).get("optimizer", {})
            budget_chars = cfg.get("edit_budget_chars")
            if budget_chars is None:
                raise ValueError(
                    "optimizer.edit_budget_chars is not configured; set it in "
                    "configs/models.yaml or pass budget_chars explicitly"
                )
        self.budget_chars = int(budget_chars)
        prompt_dir = Path(prompt_dir) if prompt_dir else _PROMPT_DIR
        template_path = prompt_dir / f"optimizer_{prompt_version}.md"
        if not template_path.is_file():
            raise FileNotFoundError(f"optimizer prompt template not found: {template_path}")
        self._template = Template(template_path.read_text(encoding="utf-8"))
        logger.info(
            "OptimizerLLM initialized: target_agent={}, model_id={}, "
            "temperature={}, max_tokens={}, prompt_version={}, budget_chars={}, "
            "similarity_threshold={}, max_parse_attempts={}",
            target_agent,
            getattr(client, "model_id", None),
            getattr(client, "temperature", None),
            getattr(client, "max_tokens", None),
            prompt_version,
            self.budget_chars,
            similarity_threshold,
            max_parse_attempts,
        )

    @classmethod
    def from_config(
        cls,
        target_agent: str,
        *,
        config_path: str | Path | None = None,
        **kwargs,
    ) -> "OptimizerLLM":
        """Build an optimizer on the ``optimizer`` config section (cloud LLM).

        Raises:
            ValueError: if ``optimizer.model_id`` is not configured.
        """
        from llm.vllm_client import VLLMClient

        client = VLLMClient(config_path=config_path, config_section="optimizer")
        if not client.model_id:
            raise ValueError(
                "optimizer.model_id is not configured; set it in "
                "configs/models.yaml to the optimizer model identifier"
            )
        return cls(client, target_agent, config_path=config_path, **kwargs)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def propose_edits(
        self,
        best_prompt: str,
        evaluations: Sequence[StageEvaluation],
        rejected_history: Sequence[PromptVersion] = (),
        budget_chars: int | None = None,
    ) -> list[PromptEdit]:
        """Propose bounded edits to *best_prompt*.

        Pipeline: render prompt -> call LLM (retrying on unparseable output
        up to ``max_parse_attempts``) -> parse -> drop edits for other
        target agents -> drop near-duplicates of rejected edits -> truncate
        to the text budget (prefix, in model-proposed order).

        Args:
            best_prompt: current best prompt text (e.g.
                ``registry.text(registry.best(agent).version_id)``).
            evaluations: critic stage evaluations for recent episodes.
            rejected_history: rejected versions from
                ``registry.history(agent, status="rejected")``.
            budget_chars: per-call budget override; defaults to the
                instance budget from construction/config.

        Raises:
            ValueError: if the LLM response stays unparseable after all
                attempts.
        """
        budget = self.budget_chars if budget_chars is None else int(budget_chars)
        prompt = self._render(best_prompt, evaluations, rejected_history, budget)
        edits = self._call_and_parse(prompt)
        kept = self._filter(edits, rejected_history, budget)
        logger.info(
            "propose_edits: target_agent={}, model_id={}, budget={}, "
            "proposed={}, kept={}",
            self.target_agent,
            getattr(self._client, "model_id", None),
            budget,
            len(edits),
            len(kept),
        )
        return kept

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _render(
        self,
        best_prompt: str,
        evaluations: Sequence[StageEvaluation],
        rejected_history: Sequence[PromptVersion],
        budget: int,
    ) -> str:
        evals_text = "\n".join(
            f"- stage {e.stage}: correctness={e.scores.correctness:.2f} "
            f"efficiency={e.scores.efficiency:.2f} safety={e.scores.safety:.2f}\n"
            f"  root_cause: {e.root_cause}\n"
            f"  evidence: {e.evidence}"
            for e in evaluations
        ) or "(no evaluations provided)"
        rejected_text = "\n".join(
            f"- [{v.edit.edit_type}] {v.edit.reason or '(no reason recorded)'} "
            f"(version {v.version_id})"
            for v in rejected_history
        ) or "(no rejected edits yet)"
        return self._template.substitute(
            target_agent=self.target_agent,
            best_prompt=best_prompt,
            evaluations=evals_text,
            rejected_edits=rejected_text,
            budget_chars=budget,
        )

    def _call_and_parse(self, prompt: str) -> list[PromptEdit]:
        last_error: ValueError | None = None
        for attempt in range(1, self.max_parse_attempts + 1):
            text = self._client.chat([{"role": "user", "content": prompt}])
            try:
                return extract_json(text, _EditList).root
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "optimizer output unparseable (attempt {}/{}): {}",
                    attempt,
                    self.max_parse_attempts,
                    exc,
                )
        raise ValueError(
            f"optimizer produced no parseable edit list after "
            f"{self.max_parse_attempts} attempts: {last_error}"
        )

    def _filter(
        self,
        edits: Sequence[PromptEdit],
        rejected_history: Sequence[PromptVersion],
        budget: int,
    ) -> list[PromptEdit]:
        on_target = []
        for edit in edits:
            if edit.target_agent != self.target_agent:
                logger.warning(
                    "dropping edit for wrong target agent {!r} (this optimizer owns {!r})",
                    edit.target_agent,
                    self.target_agent,
                )
                continue
            on_target.append(edit)

        fingerprints = [_fingerprint(v.edit.edit_type, v.edit.reason) for v in rejected_history]
        novel = []
        for edit in on_target:
            fp = _fingerprint(edit.edit_type, edit.reason, edit.new_text)
            if any(
                SequenceMatcher(None, fp, rejected_fp).ratio() >= self.similarity_threshold
                for rejected_fp in fingerprints
            ):
                logger.warning(
                    "dropping near-duplicate of a rejected edit: [{}] {}",
                    edit.edit_type,
                    edit.reason,
                )
                continue
            novel.append(edit)

        kept: list[PromptEdit] = []
        total = 0
        for edit in novel:
            cost = len(edit.new_text)  # deletes are free; see module docstring
            if total + cost > budget:
                logger.warning(
                    "text budget exhausted ({}/{} chars); truncating remaining {} edit(s)",
                    total,
                    budget,
                    len(novel) - len(kept),
                )
                break
            kept.append(edit)
            total += cost
        return kept


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _fingerprint(*parts: str) -> str:
    return _normalize("\n".join(parts))


def apply_edits(prompt_text: str, edits: Sequence[PromptEdit]) -> str:
    """Apply *edits* to *prompt_text*, returning the new prompt text.

    Pure and deterministic; edits apply sequentially in order. ``location``
    is matched against markdown section headings (``#``-prefixed lines,
    case-insensitive, markers stripped):

    - ``add``: append ``new_text`` at the end of the section body.
    - ``delete``: remove the section (heading + body).
    - ``replace``: replace the section body with ``new_text`` (heading kept).

    A location that cannot be found is skipped with a logged warning — one
    bad edit never crashes the round.
    """
    lines = prompt_text.split("\n")
    for edit in edits:
        lines = _apply_one(lines, edit)
    return "\n".join(lines)


def _apply_one(lines: list[str], edit: PromptEdit) -> list[str]:
    target = _normalize(edit.location)
    start = None
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match and _normalize(match.group(2)) == target:
            start = i
            break
    if start is None:
        logger.warning(
            "apply_edits: location {!r} not found; skipping {} edit",
            edit.location,
            edit.edit_type,
        )
        return lines
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _HEADING_RE.match(lines[j]):
            end = j
            break

    if edit.edit_type == "delete":
        return lines[:start] + lines[end:]

    body = edit.new_text.split("\n")
    tail = ([""] if end < len(lines) else []) + lines[end:]
    if edit.edit_type == "replace":
        return lines[: start + 1] + ["", *body] + tail
    # add: append to the section body, trailing blank lines collapsed
    section = list(lines[start + 1 : end])
    while section and not section[-1].strip():
        section.pop()
    return lines[: start + 1] + section + ["", *body] + tail
