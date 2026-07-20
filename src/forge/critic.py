"""Video-stage critic: two-level VLM evaluation of forge episodes.

Consumes an :class:`EpisodeRecording` (Task 3.3 contract, ``schema_version``
``"1.0"``) and produces structured :class:`StageEvaluation` objects:

1. **Pre-filter (local Qwen3-VL-2B)**: a cheap rule + local-model gate so
   only failure/borderline episodes hit the cloud critic. Episodes whose
   final feedback is success AND have no risk/failure keyframe events AND
   did not run near the step budget skip the cloud critic entirely
   (``filtered=True``, empty evaluations) when the local pre-filter agrees
   the run is clean.
2. **Global pass (cloud)**: a keyframe strip sampled across the episode
   plus the stage logs -> one episode-level evaluation with the root cause
   (``stage == "episode"``).
3. **Step pass (cloud)**: a short keyframe window around each marked event
   (``decision``/``failure``/``risk``/``success``; ``start``/``end`` are
   skipped) -> one step-level evaluation per event
   (``stage == "step:{step_index}"``).

Stage naming convention (contract for downstream tasks 3.6/3.7):
``"episode"`` for the single global evaluation, ``"step:{i}"`` (0-based
episode step index) for step-level evaluations, emitted in keyframe order,
one per marked keyframe event.

Video upload is not assumed: frames are decoded locally from the recorded
mp4 and sent as base64 image data URLs (the keyframe-strip fallback noted
in the Task 3.4 brief). Zero-frame episodes degrade to text-only prompts.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Literal, Sequence

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from forge.recorder import SCHEMA_VERSION, EpisodeRecording, KeyframeEvent
from llm.parser import extract_json

# Fraction of the step budget at which an episode counts as "near max
# rounds" and is escalated even on success (borderline case).
NEAR_MAX_ROUNDS_RATIO = 0.9

# Keyframe kinds that get a step-level evaluation (start/end are structural).
_STEP_EVENT_KINDS = ("decision", "failure", "risk", "success")

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class StageScores(BaseModel):
    """Fixed score set for one stage; all values in [0, 1]."""

    model_config = ConfigDict(frozen=True)

    correctness: float = Field(..., ge=0.0, le=1.0)
    efficiency: float = Field(..., ge=0.0, le=1.0)
    safety: float = Field(..., ge=0.0, le=1.0)


class StageEvaluation(BaseModel):
    """Structured critic output for one stage of an episode.

    ``stage`` is ``"episode"`` (global pass) or ``"step:{i}"`` (step pass).
    ``evidence`` must reference keyframe/step indices so downstream
    consumers can trace the judgement back to the recording.
    """

    model_config = ConfigDict(frozen=True)

    stage: str = Field(..., min_length=1)
    scores: StageScores
    root_cause: str
    evidence: str


class PrefilterVerdict(BaseModel):
    """Local pre-filter triage result."""

    model_config = ConfigDict(frozen=True)

    verdict: Literal["success", "borderline", "failure"]
    reason: str


class CriticModelMetadata(BaseModel):
    """Model/params provenance recorded on every critic result."""

    model_config = ConfigDict(frozen=True)

    cloud_model_id: str | None
    cloud_temperature: float | None
    cloud_max_tokens: int | None
    prefilter_model_id: str | None
    prompt_version: str
    recording_schema_version: str


class CriticResult(BaseModel):
    """Output of :meth:`VideoStageCritic.evaluate_episode`.

    ``filtered=True`` means the cloud critic was skipped (clean success);
    ``reason`` then explains why it was skipped. When ``filtered=False``,
    ``reason`` carries the escalation reason (see :func:`should_escalate`,
    or ``prefilter_verdict_{borderline,failure}`` when the local pre-filter
    overrode a clean rule gate).
    """

    model_config = ConfigDict(frozen=True)

    episode_id: str
    filtered: bool
    reason: str
    prefilter: PrefilterVerdict | None
    evaluations: list[StageEvaluation] = Field(default_factory=list)
    model_metadata: CriticModelMetadata


def should_escalate(
    recording: EpisodeRecording,
    final_success: bool,
    max_steps: int | None = None,
) -> tuple[bool, str]:
    """Rule gate: decide whether an episode must hit the cloud critic.

    Escalates iff ANY of:

    - ``final_success`` is False -> ``"episode_failed"``;
    - any keyframe has kind ``risk`` or ``failure``
      -> ``"risk_or_failure_events"``;
    - ``max_steps`` is given and the executed steps
      (``frame_count - 1``, frame 0 is the reset observation) reach
      ``NEAR_MAX_ROUNDS_RATIO * max_steps`` -> ``"near_max_rounds"``.

    Returns ``(escalate, reason)``; ``reason`` is ``""`` when not
    escalating. Borderline episodes (success with risk events, or
    near-max-rounds) escalate by design.
    """
    if not final_success:
        return True, "episode_failed"
    if any(kf.kind in ("risk", "failure") for kf in recording.keyframes):
        return True, "risk_or_failure_events"
    if max_steps is not None and max_steps > 0:
        executed_steps = max(recording.frame_count - 1, 0)
        if executed_steps >= NEAR_MAX_ROUNDS_RATIO * max_steps:
            return True, "near_max_rounds"
    return False, ""


class VideoStageCritic:
    """Two-level VLM critic over episode recordings.

    Args:
        cloud_client: OpenAI-compatible cloud VLM client
            (:class:`llm.cloud_critic.CloudVLMClient` or any object with
            ``chat(messages, images)`` plus ``model_id``/``temperature``/
            ``max_tokens`` attributes).
        prefilter_client: local VLM client (e.g. ``VLLMClient`` serving
            Qwen3-VL-2B) used for the cheap triage verdict.
        prompt_dir: directory holding ``critic_{prefilter,global,step}_
            {version}.md`` templates; defaults to the bundled prompts.
        prompt_version: prompt template version tag.
        step_window: frames on each side of an event keyframe included in
            the step-pass clip (window of ``2 * step_window + 1``, clipped
            to the episode bounds).
        max_strip_frames: cap on the global-pass keyframe strip; longer
            strips are evenly subsampled.
    """

    def __init__(
        self,
        cloud_client,
        prefilter_client,
        prompt_dir: str | Path | None = None,
        prompt_version: str = "v0",
        step_window: int = 1,
        max_strip_frames: int = 16,
    ) -> None:
        if step_window < 0:
            raise ValueError(f"step_window must be >= 0, got {step_window}")
        self._cloud = cloud_client
        self._prefilter = prefilter_client
        self._prompt_dir = Path(prompt_dir) if prompt_dir else _PROMPT_DIR
        self.prompt_version = prompt_version
        self.step_window = step_window
        self.max_strip_frames = max_strip_frames
        self._templates = {
            name: self._load_prompt(name)
            for name in ("critic_prefilter", "critic_global", "critic_step")
        }
        logger.info(
            "VideoStageCritic initialized: prompt_version={}, step_window={}, "
            "max_strip_frames={}",
            prompt_version,
            step_window,
            max_strip_frames,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate_episode(
        self,
        recording: EpisodeRecording,
        *,
        final_success: bool,
        max_steps: int | None = None,
        stage_logs: str | Sequence[str] = "",
    ) -> CriticResult:
        """Evaluate one episode recording, returning a :class:`CriticResult`.

        Raises:
            ValueError: if a VLM response cannot be parsed into the
                required schema.
        """
        if recording.schema_version != SCHEMA_VERSION:
            logger.warning(
                "Recording schema_version {} != expected {}",
                recording.schema_version,
                SCHEMA_VERSION,
            )
        logs_text = (
            stage_logs if isinstance(stage_logs, str) else "\n".join(stage_logs)
        )
        escalate, reason = should_escalate(recording, final_success, max_steps)
        verdict: PrefilterVerdict | None = None
        if not escalate:
            verdict = self._run_prefilter(recording, final_success)
            if verdict.verdict == "success":
                logger.info(
                    "Episode {} filtered (clean success); cloud critic skipped",
                    recording.episode_id,
                )
                return CriticResult(
                    episode_id=recording.episode_id,
                    filtered=True,
                    reason="clean_success: rule gate passed and pre-filter "
                    "verdict=success",
                    prefilter=verdict,
                    evaluations=[],
                    model_metadata=self._metadata(recording),
                )
            escalate = True
            reason = f"prefilter_verdict_{verdict.verdict}"

        logger.info(
            "Episode {} escalated to cloud critic: {}",
            recording.episode_id,
            reason,
        )
        evaluations = [self._global_pass(recording, final_success, logs_text)]
        for kf in recording.keyframes:
            if kf.kind not in _STEP_EVENT_KINDS:
                continue
            evaluations.append(self._step_pass(recording, kf))
        return CriticResult(
            episode_id=recording.episode_id,
            filtered=False,
            reason=reason,
            prefilter=verdict,
            evaluations=evaluations,
            model_metadata=self._metadata(recording),
        )

    # ------------------------------------------------------------------ #
    # Critic passes
    # ------------------------------------------------------------------ #

    def _run_prefilter(
        self, recording: EpisodeRecording, final_success: bool
    ) -> PrefilterVerdict:
        """Cheap local triage: metadata summary only, no images."""
        prompt = self._templates["critic_prefilter"].substitute(
            episode_id=recording.episode_id,
            final_success=final_success,
            frame_count=recording.frame_count,
            fps=recording.fps,
            keyframes=_format_keyframes(recording),
        )
        text = self._prefilter.chat([{"role": "user", "content": prompt}])
        return extract_json(text, PrefilterVerdict)

    def _global_pass(
        self, recording: EpisodeRecording, final_success: bool, logs_text: str
    ) -> StageEvaluation:
        indices = sorted({kf.frame_index for kf in recording.keyframes})
        frames = self._strip_frames(recording, indices)
        prompt = self._templates["critic_global"].substitute(
            episode_id=recording.episode_id,
            final_success=final_success,
            frame_count=recording.frame_count,
            fps=recording.fps,
            keyframes=_format_keyframes(recording),
            stage_logs=logs_text or "(no stage logs)",
        )
        text = self._cloud.chat(
            [{"role": "user", "content": prompt}], images=frames or None
        )
        evaluation = extract_json(text, StageEvaluation)
        return evaluation.model_copy(update={"stage": "episode"})

    def _step_pass(
        self, recording: EpisodeRecording, kf: KeyframeEvent
    ) -> StageEvaluation:
        if recording.frame_count > 0:
            lo = max(kf.frame_index - self.step_window, 0)
            hi = min(kf.frame_index + self.step_window, recording.frame_count - 1)
            frames = self._frames_at(recording, list(range(lo, hi + 1)))
        else:
            # Zero-frame episodes: marked events carry frame_index 0 even
            # though no frames exist (Task 3.3 caveat) — stay text-only.
            frames = []
        prompt = self._templates["critic_step"].substitute(
            episode_id=recording.episode_id,
            step_index=kf.step_index,
            kind=kf.kind,
            detail=kf.detail or "(no detail)",
            frame_index=kf.frame_index,
            timestamp_sec=f"{kf.timestamp_sec:.2f}",
        )
        text = self._cloud.chat(
            [{"role": "user", "content": prompt}], images=frames or None
        )
        evaluation = extract_json(text, StageEvaluation)
        return evaluation.model_copy(update={"stage": f"step:{kf.step_index}"})

    # ------------------------------------------------------------------ #
    # Frame decoding
    # ------------------------------------------------------------------ #

    def _strip_frames(
        self, recording: EpisodeRecording, indices: list[int]
    ) -> list[np.ndarray]:
        if recording.frame_count == 0 or not recording.video_path:
            return []
        if len(indices) > self.max_strip_frames:
            positions = np.linspace(0, len(indices) - 1, self.max_strip_frames)
            indices = [indices[round(p)] for p in positions]
        return self._frames_at(recording, indices)

    @staticmethod
    def _frames_at(
        recording: EpisodeRecording, indices: list[int]
    ) -> list[np.ndarray]:
        """Decode the requested frame indices from the recorded mp4 (RGB)."""
        if not indices or not recording.video_path:
            return []
        wanted = set(indices)
        cap = cv2.VideoCapture(recording.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open {recording.video_path}")
        frames: list[np.ndarray] = []
        try:
            for idx in sorted(wanted):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    logger.warning(
                        "Could not decode frame {} from {}", idx, recording.video_path
                    )
        finally:
            cap.release()
        return frames

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def _load_prompt(self, name: str) -> Template:
        path = self._prompt_dir / f"{name}_{self.prompt_version}.md"
        if not path.exists():
            raise FileNotFoundError(f"Critic prompt not found: {path}")
        return Template(path.read_text(encoding="utf-8"))

    def _metadata(self, recording: EpisodeRecording) -> CriticModelMetadata:
        return CriticModelMetadata(
            cloud_model_id=getattr(self._cloud, "model_id", None),
            cloud_temperature=getattr(self._cloud, "temperature", None),
            cloud_max_tokens=getattr(self._cloud, "max_tokens", None),
            prefilter_model_id=getattr(self._prefilter, "model_id", None),
            prompt_version=self.prompt_version,
            recording_schema_version=recording.schema_version,
        )


def _format_keyframes(recording: EpisodeRecording) -> str:
    if not recording.keyframes:
        return "(no keyframes)"
    return "\n".join(
        f"- frame {kf.frame_index} | step {kf.step_index} | {kf.kind} | "
        f"t={kf.timestamp_sec:.2f}s | {kf.detail}"
        for kf in recording.keyframes
    )
