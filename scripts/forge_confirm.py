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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.schema import Episode
from loguru import logger

from forge.env import SimEnv
from forge.loader import load_forge_tasks
from forge.planner_agent import ForgePlannerAgent
from forge.strategy_metrics import (
    aggregate_strategy_metrics,
    episode_strategy_metrics,
)
from forge.validator import rollout_episode
from tasks.schema import TaskDefinition
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
    p.add_argument("--workers", type=int, default=2,
                   help="parallel rollout worker processes (1 = serial)")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _rollout_one(
    env: Any, planner: Any, task: Any, text: str, max_rounds: int
) -> tuple[Episode, bool, int, str, int]:
    """One guarded rollout; returns (episode|None, success, steps, termination, violation)."""
    planner.set_prompt_text(text)
    violation = 0
    try:
        ep = rollout_episode(env, planner, task, max_rounds)
        success = bool((ep.metadata or {}).get("success", False))
        steps = len(ep.steps)
        termination = (ep.metadata or {}).get("termination_reason") or ""
    except Exception as exc:
        logger.warning("rollout crashed ({}): {}", task.id, exc)
        success, steps, termination, ep = False, 0, "error", None
        if "not in the DINO label set" in str(exc):
            violation = 1
    return ep, success, steps, termination, violation


def _pair_record(
    ep: Any, tag: str, task: Any, k: int, success: bool, steps: int,
    termination: str, violation: int,
) -> dict:
    strat = (
        episode_strategy_metrics(ep, task)
        if ep is not None
        else {
            "action_count": steps,
            "presence_violations": violation,
            "move_aside_first": None,
        }
    )
    return {
        "version": tag,
        "task_id": task.id,
        "round": k,
        "success": success,
        "steps": steps,
        "termination": termination,
        **strat,
    }


def _worker(payload: dict) -> list[dict]:
    """Child-process worker: own SimEnv/planner, runs its (task, round) pairs.

    Within a pair the A rollout precedes the B rollout, preserving the
    interleaved A/B semantics of the serial version.
    """
    env = SimEnv(repo_root=REPO_ROOT)
    planner = ForgePlannerAgent(
        vllm_client=VLLMClient(config_path=payload["config_path"])
    )
    tasks = [TaskDefinition(**d) for d in payload["tasks"]]
    records: list[dict] = []
    for task_idx, k in payload["pairs"]:
        task = tasks[task_idx]
        for tag, text in (("A", payload["text_a"]), ("B", payload["text_b"])):
            ep, success, steps, termination, violation = _rollout_one(
                env, planner, task, text, payload["max_rounds"]
            )
            rec = _pair_record(ep, tag, task, k, success, steps, termination, violation)
            records.append(rec)
            print(
                f"[w{payload['worker_id']}][{tag}] {task.id} r{k}: "
                f"success={rec['success']} steps={rec['steps']}",
                flush=True,
            )
    env.close()
    return records


def main() -> int:
    args = _parse_args()
    text_a = args.a.read_text(encoding="utf-8")
    text_b = args.b.read_text(encoding="utf-8")
    tasks = [t for t in load_forge_tasks(args.tasks)
             if (t.metadata or {}).get("split") == "val"]
    if not tasks:
        logger.error("no val tasks in {}", args.tasks)
        return 2

    episodes: list[dict] = []
    t0 = datetime.now(timezone.utc).isoformat()
    n_workers = max(1, int(args.workers))
    if n_workers == 1:
        env = SimEnv(repo_root=REPO_ROOT)
        planner = ForgePlannerAgent(vllm_client=VLLMClient(config_path=args.config))
        for ti, task in enumerate(tasks):
            for k in range(args.episodes_per_task):
                for tag, text in (("A", text_a), ("B", text_b)):
                    ep, success, steps, termination, violation = _rollout_one(
                        env, planner, task, text, args.max_rounds
                    )
                    rec = _pair_record(ep, tag, task, k, success, steps, termination, violation)
                    episodes.append(rec)
                    print(f"[{tag}] {task.id} r{k}: success={rec['success']} "
                          f"steps={rec['steps']}", flush=True)
        env.close()
    else:
        import concurrent.futures as cf
        import multiprocessing as mp

        pairs = [(ti, k) for ti in range(len(tasks)) for k in range(args.episodes_per_task)]
        payloads = [
            {
                "worker_id": w,
                "pairs": pairs[w::n_workers],
                "tasks": [t.model_dump() for t in tasks],
                "text_a": text_a,
                "text_b": text_b,
                "config_path": args.config,
                "max_rounds": args.max_rounds,
            }
            for w in range(n_workers)
        ]
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            for worker_records in pool.map(_worker, payloads):
                episodes.extend(worker_records)
        episodes.sort(key=lambda e: (e["task_id"], e["round"], e["version"]))

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
                    "decoy_picks": e.get("decoy_picks", 0),
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
        "workers": n_workers,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
