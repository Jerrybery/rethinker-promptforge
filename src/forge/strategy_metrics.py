"""Semantic strategy metrics for forge episodes.

Binary task success is dominated by physical execution noise (rim-catch
grasp ~50%), which drowns prompt-level effects at small sample sizes —
two forge accepts were later falsified by paired confirmation for exactly
this reason. These metrics measure the *strategy* instead, which is where
prompt differences actually live:

- ``move_aside_first``: on occlusion tasks (metadata ``hidden_by``), did
  the planner clear the declared occluder (``MOVE_ASIDE``) before its
  first manipulation attempt on the hidden target?
- ``presence_violations``: planner emitted a label outside the current
  detection set (recorded by ``rollout_episode``).
- ``action_count``: episode length; detects degenerate strategies (over-
  conservative loops vs. efficient sequences).
- ``decoy_picks``: manipulation attempts targeting a declared decoy label
  (metadata ``decoy_labels``) — picking the look-alike instead of the real
  target executes fine but can never satisfy ``check_success``.
"""

from __future__ import annotations

from typing import Any

from common.schema import Episode, MissionType
from tasks.schema import TaskDefinition

_TARGET_MISSIONS = (MissionType.PICK_AND_PLACE, MissionType.PICK_ONLY)


def episode_strategy_metrics(
    episode: Episode, task: TaskDefinition
) -> dict[str, Any]:
    """Compute per-episode strategy metrics.

    Returns a dict with ``action_count`` (int), ``presence_violations``
    (int), and ``move_aside_first`` (bool | None; None when the task is not
    an occlusion task or the hidden target was never manipulated).
    """
    actions = [
        step.planner_output for step in episode.steps if step.planner_output
    ]
    violations = int((episode.metadata or {}).get("presence_violations", 0))
    decoy_labels = set((task.metadata or {}).get("decoy_labels") or [])
    decoy_picks = sum(
        1 for a in actions if a.pick in decoy_labels
    )

    move_aside_first: bool | None = None
    hidden_by = (task.metadata or {}).get("hidden_by") or {}
    if hidden_by:
        hidden_labels = set(hidden_by.keys())
        occluders = {o for names in hidden_by.values() for o in names}
        first_target_idx = next(
            (
                i
                for i, action in enumerate(actions)
                if action.mission in _TARGET_MISSIONS and action.pick in hidden_labels
            ),
            None,
        )
        if first_target_idx is not None:
            move_aside_first = any(
                action.mission is MissionType.MOVE_ASIDE
                and action.pick in occluders
                for action in actions[:first_target_idx]
            )

    return {
        "action_count": len(actions),
        "presence_violations": violations,
        "move_aside_first": move_aside_first,
        "decoy_picks": decoy_picks,
    }


def aggregate_strategy_metrics(per_episode: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode strategy metrics into rates.

    ``move_aside_first_rate`` ignores episodes where the metric is not
    applicable (None); it is None when no episode qualified.
    """
    total_actions = sum(e["action_count"] for e in per_episode)
    total_violations = sum(e["presence_violations"] for e in per_episode)
    total_decoy_picks = sum(e.get("decoy_picks", 0) for e in per_episode)
    maf_values = [e["move_aside_first"] for e in per_episode]
    maf_valid = [v for v in maf_values if v is not None]
    return {
        "move_aside_first_rate": (
            sum(maf_valid) / len(maf_valid) if maf_valid else None
        ),
        "presence_violation_rate": (
            total_violations / total_actions if total_actions else 0.0
        ),
        "mean_action_count": (
            total_actions / len(per_episode) if per_episode else 0.0
        ),
        "presence_violations": total_violations,
        "decoy_pick_rate": (
            total_decoy_picks / total_actions if total_actions else 0.0
        ),
        "decoy_picks": total_decoy_picks,
    }
