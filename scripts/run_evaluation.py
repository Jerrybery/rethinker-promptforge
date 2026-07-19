#!/usr/bin/env python3
"""CLI to run the evaluation harness over serialized episodes.

This script expects the project to be installed in editable mode or to be run
with ``PYTHONPATH=src`` so that the top-level packages are importable.

Usage::

    PYTHONPATH=src python scripts/run_evaluation.py \\
        --tasks-path data/tasks/hello_tasks.yaml \\
        --episodes-jsonl /tmp/mock_episodes.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.schema import Episode
from evaluation.harness import EvaluationResult, evaluate_tasks
from tasks.loader import load_task_definitions


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate episodes against task success criteria."
    )
    parser.add_argument(
        "--tasks-path",
        default="data/tasks/hello_tasks.yaml",
        help="Path to a YAML task catalogue (default: data/tasks/hello_tasks.yaml).",
    )
    parser.add_argument(
        "--episodes-jsonl",
        required=True,
        help="Path to a JSON-lines file of serialized Episode objects.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to write the EvaluationResult.",
    )
    return parser.parse_args(argv)


def _load_episodes_jsonl(path: str | Path) -> list[Episode]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Episodes file not found: {file_path}")

    episodes: list[Episode] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(Episode.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid Episode on line {line_number}: {exc}"
                ) from exc
    return episodes


def _print_summary(result: EvaluationResult) -> None:
    print(f"Total episodes: {result.total}")
    print(f"Success count:  {result.success_count}")
    print(f"Success rate:   {result.success_rate:.2%}")
    print(f"Average steps:  {result.average_steps:.2f}")
    print(f"Min steps:      {result.min_steps}")
    print(f"Max steps:      {result.max_steps}")
    print(f"Failure count:  {result.failure_count}")
    print(f"Risky actions:  {result.risky_action_count}")
    print(f"Reflections:    {result.reflection_count}")
    if result.average_runtime_seconds is not None:
        print(f"Avg runtime:    {result.average_runtime_seconds:.2f}s")
        print(f"Min runtime:    {result.min_runtime_seconds:.2f}s")
        print(f"Max runtime:    {result.max_runtime_seconds:.2f}s")
    print("Termination reason counts:")
    for reason, count in sorted(result.termination_reason_counts.items()):
        print(f"  {reason}: {count}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tasks = load_task_definitions(args.tasks_path)
    episodes = _load_episodes_jsonl(args.episodes_jsonl)
    result = evaluate_tasks(tasks, episodes)
    _print_summary(result)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote evaluation result to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
