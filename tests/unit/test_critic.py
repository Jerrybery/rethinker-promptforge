"""Unit tests for the forge video-stage critic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from forge.critic import (
    StageEvaluation,
    StageScores,
    VideoStageCritic,
    should_escalate,
)
from forge.recorder import EpisodeRecorder, EpisodeRecording


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _record(
    tmp_path: Path,
    episode_id: str = "ep",
    n_frames: int = 4,
    fps: float = 5.0,
    events: list[tuple[int, str, str]] | None = None,
) -> EpisodeRecording:
    """Record an episode of 8x8 frames; events are marked right after the
    frame whose index equals their step index (frame i <-> step i)."""
    rec = EpisodeRecorder()
    rec.start_episode(episode_id, tmp_path, fps=fps)
    for i in range(n_frames):
        rec.add_frame(np.full((8, 8, 3), 40 * i, dtype=np.uint8))
        for step, kind, detail in events or []:
            if step == i:
                rec.mark_event(step, kind, detail)
    return rec.finish()


GLOBAL_JSON = json.dumps(
    {
        "stage": "episode",
        "scores": {"correctness": 0.2, "efficiency": 0.4, "safety": 0.9},
        "root_cause": "planner selected the wrong target object",
        "evidence": "keyframe frame 2 (step 2, failure)",
    }
)


def _step_json(step: int) -> str:
    return json.dumps(
        {
            "stage": f"step:{step}",
            "scores": {"correctness": 0.5, "efficiency": 0.5, "safety": 1.0},
            "root_cause": f"step {step} issue",
            "evidence": f"window around frame {step}",
        }
    )


def _cloud_client(*responses_texts: str) -> MagicMock:
    cloud = MagicMock()
    cloud.model_id = "cloud-vlm-1"
    cloud.temperature = 0.0
    cloud.max_tokens = 256
    cloud.chat.side_effect = list(responses_texts)
    return cloud


def _prefilter_client(verdict: str = "success", reason: str = "clean run") -> MagicMock:
    pf = MagicMock()
    pf.model_id = "Qwen3-VL-2B"
    pf.chat.return_value = json.dumps({"verdict": verdict, "reason": reason})
    return pf


# --------------------------------------------------------------------- #
# StageEvaluation schema
# --------------------------------------------------------------------- #


def test_stage_evaluation_frozen_and_score_bounds() -> None:
    ev = StageEvaluation(
        stage="episode",
        scores=StageScores(correctness=0.1, efficiency=0.2, safety=0.3),
        root_cause="x",
        evidence="y",
    )
    with pytest.raises(ValidationError):
        ev.stage = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        StageScores(correctness=1.5, efficiency=0.0, safety=0.0)
    with pytest.raises(ValidationError):
        StageScores(correctness=0.0, efficiency=-0.1, safety=0.0)


# --------------------------------------------------------------------- #
# Pre-filter predicate
# --------------------------------------------------------------------- #


def test_should_escalate_on_failure(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=3)
    escalate, reason = should_escalate(rec, final_success=False)
    assert escalate is True
    assert reason == "episode_failed"


def test_should_escalate_clean_success_does_not(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=3)
    escalate, reason = should_escalate(rec, final_success=True)
    assert escalate is False
    assert reason == ""


def test_should_escalate_success_with_risk_event(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=3, events=[(1, "risk", "arm near obstacle")])
    escalate, reason = should_escalate(rec, final_success=True)
    assert escalate is True
    assert reason == "risk_or_failure_events"


def test_should_escalate_success_with_failure_event(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=3, events=[(2, "failure", "grasp failed")])
    escalate, _ = should_escalate(rec, final_success=True)
    assert escalate is True


def test_should_escalate_near_max_rounds(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=10)  # 9 executed steps (frame 0 = reset)
    escalate, reason = should_escalate(rec, final_success=True, max_steps=10)
    assert escalate is True
    assert reason == "near_max_rounds"


def test_should_escalate_below_max_rounds_does_not(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=5)  # 4 executed steps < 0.9 * 10
    escalate, _ = should_escalate(rec, final_success=True, max_steps=10)
    assert escalate is False


# --------------------------------------------------------------------- #
# VideoStageCritic flow
# --------------------------------------------------------------------- #


def test_filtered_clean_success_skips_cloud(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=3)
    cloud = _cloud_client()
    pf = _prefilter_client("success")
    critic = VideoStageCritic(cloud_client=cloud, prefilter_client=pf)
    result = critic.evaluate_episode(rec, final_success=True)
    assert result.filtered is True
    assert result.evaluations == []
    assert result.reason != ""
    assert result.episode_id == rec.episode_id
    cloud.chat.assert_not_called()
    pf.chat.assert_called_once()
    assert result.prefilter is not None
    assert result.prefilter.verdict == "success"


def test_prefilter_borderline_escalates_to_cloud(tmp_path: Path) -> None:
    # Clean rule gate (success, no risk/failure events) so the prefilter
    # verdict is what triggers escalation.
    rec = _record(tmp_path, n_frames=2, events=[(1, "decision", "retry grasp")])
    cloud = _cloud_client(GLOBAL_JSON, _step_json(1))
    pf = _prefilter_client("borderline", "possible wasted motion")
    critic = VideoStageCritic(cloud_client=cloud, prefilter_client=pf)
    result = critic.evaluate_episode(rec, final_success=True)
    assert result.filtered is False
    assert "borderline" in result.reason
    assert cloud.chat.call_count == 2  # global pass + one step event
    assert [e.stage for e in result.evaluations] == ["episode", "step:1"]


def test_failure_episode_two_level_evaluation(tmp_path: Path) -> None:
    events = [(0, "decision", "PICK_ONLY pick=red_block"), (2, "failure", "grasp failed")]
    rec = _record(tmp_path, n_frames=4, events=events)
    cloud = _cloud_client(GLOBAL_JSON, _step_json(0), _step_json(2))
    pf = _prefilter_client()
    critic = VideoStageCritic(cloud_client=cloud, prefilter_client=pf)
    result = critic.evaluate_episode(rec, final_success=False)
    assert result.filtered is False
    assert result.reason == "episode_failed"
    pf.chat.assert_not_called()  # rule gate already escalated
    stages = [e.stage for e in result.evaluations]
    assert stages == ["episode", "step:0", "step:2"]
    global_eval = result.evaluations[0]
    assert global_eval.root_cause == "planner selected the wrong target object"
    assert global_eval.scores == StageScores(
        correctness=0.2, efficiency=0.4, safety=0.9
    )
    assert "frame 2" in global_eval.evidence


def test_global_and_step_passes_send_video_frames(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=4, events=[(2, "failure", "boom")])
    cloud = _cloud_client(GLOBAL_JSON, _step_json(2))
    critic = VideoStageCritic(cloud_client=cloud, prefilter_client=_prefilter_client())
    critic.evaluate_episode(rec, final_success=False)
    global_call = cloud.chat.call_args_list[0]
    strip = global_call.kwargs["images"]
    # keyframes: start(0), failure(2), end(3) -> 3 unique frames
    assert len(strip) == 3
    assert all(img.shape == (8, 8, 3) for img in strip)
    step_call = cloud.chat.call_args_list[1]
    window = step_call.kwargs["images"]
    assert len(window) == 3  # frames 1..3 around frame 2 (step_window=1)
    assert all(img.shape == (8, 8, 3) for img in window)


def test_zero_frame_episode_is_text_only(tmp_path: Path) -> None:
    rec = EpisodeRecorder()
    rec.start_episode("zero", tmp_path, fps=5.0)
    rec.mark_event(0, "failure", "reset failed")
    recording = rec.finish()
    assert recording.video_path is None
    assert recording.frame_count == 0
    cloud = _cloud_client(GLOBAL_JSON, _step_json(0))
    critic = VideoStageCritic(cloud_client=cloud, prefilter_client=_prefilter_client())
    result = critic.evaluate_episode(recording, final_success=False)
    assert result.filtered is False
    for call in cloud.chat.call_args_list:
        assert call.kwargs["images"] is None
    assert [e.stage for e in result.evaluations] == ["episode", "step:0"]


def test_stage_strings_normalized_regardless_of_vlm_output(tmp_path: Path) -> None:
    wrong = json.dumps(
        {
            "stage": "banana",
            "scores": {"correctness": 0.1, "efficiency": 0.1, "safety": 0.1},
            "root_cause": "r",
            "evidence": "e",
        }
    )
    rec = _record(tmp_path, n_frames=2, events=[(1, "risk", "x")])
    cloud = _cloud_client(wrong, wrong)
    critic = VideoStageCritic(cloud_client=cloud, prefilter_client=_prefilter_client())
    result = critic.evaluate_episode(rec, final_success=False)
    assert [e.stage for e in result.evaluations] == ["episode", "step:1"]


def test_unparseable_global_response_raises(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=2)
    cloud = _cloud_client("this is not json at all")
    critic = VideoStageCritic(cloud_client=cloud, prefilter_client=_prefilter_client())
    with pytest.raises(ValueError):
        critic.evaluate_episode(rec, final_success=False)


def test_model_metadata_recorded(tmp_path: Path) -> None:
    rec = _record(tmp_path, n_frames=2)
    cloud = _cloud_client(GLOBAL_JSON)
    pf = _prefilter_client()
    critic = VideoStageCritic(cloud_client=cloud, prefilter_client=pf)
    result = critic.evaluate_episode(rec, final_success=False)
    md = result.model_metadata
    assert md.cloud_model_id == "cloud-vlm-1"
    assert md.cloud_temperature == 0.0
    assert md.cloud_max_tokens == 256
    assert md.prefilter_model_id == "Qwen3-VL-2B"
    assert md.prompt_version == "v0"
    assert md.recording_schema_version == "1.0"
