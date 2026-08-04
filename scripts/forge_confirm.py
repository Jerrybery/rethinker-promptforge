#!/usr/bin/env python3
"""Paired A/B confirmation of a forge prompt acceptance (confirmation stage).

Independent of the forge gate: rolls two prompt versions on the validation
task set in interleaved order (A,B,A,B per task) so slow time drift (LLM
latency, planner/curobo variance trends) hits both versions symmetrically.

Usage (on the forge machine, proxies unset):

    PYTHONPATH=src python scripts/forge_confirm.py \
        --a results/forge/<run>/registry/v000.md \
        --b results/forge/<run>/registry/v001.md \
        --episodes-per-task 6 --out results/forge/<run>/confirm_v000_vs_v001.json

The report contains per-episode outcomes, per-task success rates, and the
paired delta. Judgement call is left to the reader (see
docs/forge-pipeline.md); as a rule of thumb the confirmation is positive
when B's overall rate exceeds A's and B is not worse on any task.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger

from forge.env import SimEnv
from forge.loader import load_forge_tasks
from forge.planner_agent import ForgePlannerAgent
from forge.strategy_metrics import (
    aggregate_strategy_metrics,
    episode_strategy_metrics,
)
from forge.validator import rollout_episode
from llm.vllm_client import VLLMClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True, help="baseline prompt path")
    p.add_argument("--b", type=Path, required=True, help="candidate prompt path")
    p.add_argument("--tasks", type=Path,
                   default=REPO_ROOT / "configs" / "forge_tasks_smoke.yaml")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs" / "models.local.yaml")
    p.add_argument("--episodes-per-task", type=int, default=6)
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    text_a = args.a.read_text(encoding="utf-8")
    text_b = args.b.read_text(encoding="utf-8")
    tasks = [t for t in load_forge_tasks(args.tasks)
             if (t.metadata or {}).get("split") == "val"]
    if not tasks:
        logger.error("no val tasks in {}", args.tasks)
        return 2

    env = SimEnv(repo_root=REPO_ROOT)
    planner = ForgePlannerAgent(vllm_client=VLLMClient(config_path=args.config))

    episodes: list[dict] = []
    t0 = datetime.now(timezone.utc).isoformat()
    for task in tasks:
        for k in range(args.episodes_per_task):
            for tag, text in (("A", text_a), ("B", text_b)):
                planner.set_prompt_text(text)
                violation = 0
                try:
                    ep = rollout_episode(env, planner, task, args.max_rounds)
                    success = bool((ep.metadata or {}).get("success", False))
                    steps = len(ep.steps)
                    termination = (ep.metadata or {}).get("termination_reason")
                except Exception as exc:
                    # A crashed rollout (e.g. planner emitted a label outside
                    # the detection set) counts as a failed episode, matching
                    # the forge runner's per-rollout guard.
                    logger.warning("rollout crashed ({} {} r{}): {}", tag, task.id, k, exc)
                    success, steps, termination = False, 0, "error"
                    if "not in the DINO label set" in str(exc):
                        violation = 1
                strat = (
                    episode_strategy_metrics(ep, task)
                    if termination != "error"
                    else {
                        "action_count": steps,
                        "presence_violations": violation,
                        "move_aside_first": None,
                    }
                )
                rec = {
                    "version": tag,
                    "task_id": task.id,
                    "round": k,
                    "success": success,
                    "steps": steps,
                    "termination": termination,
                    **strat,
                }
                episodes.append(rec)
                print(f"[{tag}] {task.id} r{k}: success={rec['success']} "
                      f"steps={rec['steps']}", flush=True)

    def rate(version: str, task_id: str | None = None) -> float:
        eps = [e for e in episodes if e["version"] == version
               and (task_id is None or e["task_id"] == task_id)]
        return sum(e["success"] for e in eps) / len(eps) if eps else 0.0

    def strat_of(version: str, task_id: str | None = None) -> dict:
        eps = [e for e in episodes if e["version"] == version
               and (task_id is None or e["task_id"] == task_id)]
        return aggregate_strategy_metrics(
            [
                {
                    "action_count": e["action_count"],
                    "presence_violations": e["presence_violations"],
                    "move_aside_first": e["move_aside_first"],
                }
                for e in eps
            ]
        )

    per_task = {
        t.id: {"A": rate("A", t.id), "B": rate("B", t.id),
               "delta": rate("B", t.id) - rate("A", t.id),
               "A_strategy": strat_of("A", t.id),
               "B_strategy": strat_of("B", t.id)}
        for t in tasks
    }
    summary = {
        "started": t0,
        "finished": datetime.now(timezone.utc).isoformat(),
        "prompts": {"A": str(args.a), "B": str(args.b)},
        "episodes_per_task": args.episodes_per_task,
        "overall": {"A": rate("A"), "B": rate("B"),
                    "delta": rate("B") - rate("A"),
                    "A_strategy": strat_of("A"),
                    "B_strategy": strat_of("B")},
        "per_task": per_task,
        "episodes": episodes,
    }
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"},
                     indent=1), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"report: {args.out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
