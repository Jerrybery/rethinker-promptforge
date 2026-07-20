#!/usr/bin/env python3
"""Forge training loop CLI (Task 3.8).

Runs the EmbodiedPromptForge training loop: roll out the current best
planner prompt on the fixed train task set (recording every episode), let
the video-stage critic evaluate failures, let the optimizer propose bounded
prompt edits, then validate the candidate on the fixed held-out task set
and accept only strict improvements.

Artifacts land in ``<out>/YYYYMMDD_HHMMSS/`` (default
``results/forge/...``): ``run_config.json`` (full reproducibility
snapshot), ``forge_log.json``, per-epoch ``epochs/epoch_NNN.json``,
``metrics.jsonl`` (one line per epoch), best-prompt snapshots under
``prompts/``, episode recordings under ``recordings/``, and the prompt
registry under ``registry/``.

Real runs need the local vLLM endpoint (planner), the ``optimizer`` section
(a strong cloud/LLM endpoint), and — unless ``--no-critic`` — the
``cloud_critic`` section of the models config. ``--stub-llm`` replaces all
LLM clients with deterministic scripted responses so the full loop can be
smoke-tested offline (CI); critic clients are scripted too when combined
with ``--use-critic``.

Usage::

    PYTHONPATH=src python scripts/run_forge.py \
        --tasks data/tasks/hello_tasks.yaml --epochs 3 --no-critic

    PYTHONPATH=src python scripts/run_forge.py --stub-llm --no-critic \
        --epochs 1 --out results/forge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger

from evaluation.batch import make_run_dir
from forge.loader import load_forge_tasks
from forge.runner import ForgeRunner
from rethinker_promptforge.config import load_config
from tasks.schema import TaskDefinition

DEFAULT_TASKS_PATH = REPO_ROOT / "data" / "tasks" / "hello_tasks.yaml"
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"

_PLANNER_STUB_RESPONSES = [
    json.dumps(
        {
            "plan_id": "stub-act",
            "mission": "PICK_AND_PLACE",
            "pick": "mock_object",
            "place": "mock_object",
        }
    ),
    json.dumps({"plan_id": "stub-stop", "mission": "STOP", "pick": "none"}),
]

_OPTIMIZER_STUB_RESPONSE = json.dumps(
    [
        {
            "target_agent": "planner",
            "edit_type": "add",
            "location": "Constraints (hard rules)",
            "new_text": (
                "6. When feedback confirms the task is complete, "
                "prefer STOP over further actions."
            ),
            "reason": "stub: reinforce stopping on confirmed success",
        }
    ]
)

_CRITIC_STAGE_STUB_RESPONSE = json.dumps(
    {
        "stage": "episode",
        "scores": {"correctness": 0.4, "efficiency": 0.3, "safety": 0.9},
        "root_cause": "stub root cause (scripted smoke response)",
        "evidence": "frame 0",
    }
)

_CRITIC_PREFILTER_STUB_RESPONSE = json.dumps(
    {"verdict": "borderline", "reason": "stub prefilter (scripted)"}
)


class _ScriptedClient:
    """Deterministic offline chat client cycling scripted responses."""

    def __init__(
        self,
        responses: list[str],
        model_id: str = "stub-scripted",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        if not responses:
            raise ValueError("responses must be non-empty")
        self._responses = list(responses)
        self._index = 0
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict], images: list | None = None) -> str:
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forge training loop: rollout -> critic -> optimizer -> validate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODELS_CONFIG,
        help="Models YAML config path (default: configs/models.yaml).",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help=(
            "Task catalogue YAML (default: data/tasks/hello_tasks.yaml). "
            "Without --val-tasks the last third (min 1) is held out for "
            "validation."
        ),
    )
    parser.add_argument(
        "--val-tasks",
        type=Path,
        default=None,
        help="Optional separate held-out task catalogue YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Maximum forge epochs (default: 3).",
    )
    parser.add_argument(
        "--edit-budget",
        type=int,
        default=None,
        help=(
            "Optimizer text budget in chars per epoch (default: "
            "optimizer.edit_budget_chars from the models config)."
        ),
    )
    parser.add_argument(
        "--use-critic",
        dest="use_critic",
        action="store_true",
        default=True,
        help=(
            "Run the video-stage critic on train episodes (default: on; "
            "requires cloud_critic config or --stub-llm)."
        ),
    )
    parser.add_argument(
        "--no-critic",
        dest="use_critic",
        action="store_false",
        help="Skip the cloud critic entirely (validation gate works without it).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum planner steps per episode (default: 10).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/forge"),
        help="Base output directory; a YYYYMMDD_HHMMSS run dir is created inside.",
    )
    parser.add_argument(
        "--initial-prompt",
        type=Path,
        default=None,
        help=(
            "Seed prompt text path (default: packaged planner system_v0.md). "
            "Used only when the registry has no accepted best yet."
        ),
    )
    parser.add_argument(
        "--stub-llm",
        action="store_true",
        help=(
            "Replace all LLM clients (planner, optimizer, critic) with "
            "deterministic scripted responses for offline smoke runs/CI."
        ),
    )
    return parser.parse_args(argv)


def _split_tasks(
    args: argparse.Namespace,
) -> tuple[list[TaskDefinition], list[TaskDefinition]] | None:
    """Return (train, val); None + logged error on invalid input."""
    if not args.tasks.exists():
        logger.error("Task catalogue not found: {}", args.tasks)
        return None
    tasks = load_forge_tasks(args.tasks)
    if args.val_tasks is not None:
        if not args.val_tasks.exists():
            logger.error("Validation catalogue not found: {}", args.val_tasks)
            return None
        val_tasks = load_forge_tasks(args.val_tasks)
        if not tasks or not val_tasks:
            logger.error("Train/validation catalogues must both be non-empty")
            return None
        return tasks, val_tasks
    if len(tasks) < 2:
        logger.error(
            "Need at least 2 tasks in {} to hold out a validation set "
            "(or pass --val-tasks)",
            args.tasks,
        )
        return None
    val_count = max(1, len(tasks) // 3)
    return tasks[:-val_count], tasks[-val_count:]


def _initial_prompt_text(args: argparse.Namespace) -> str:
    if args.initial_prompt is not None:
        return args.initial_prompt.read_text(encoding="utf-8")
    from planner.prompts.registry import PromptRegistry

    system_template, _ = PromptRegistry.load("v0")
    return system_template


def _build_llm_stack(args: argparse.Namespace):
    """Build (planner_client, optimizer, critic|None); None on config error."""
    from forge.critic import VideoStageCritic
    from forge.optimizer import OptimizerLLM
    from llm.vllm_client import VLLMClient

    optimizer_kwargs = (
        {} if args.edit_budget is None else {"budget_chars": args.edit_budget}
    )
    critic = None
    if args.stub_llm:
        planner_client = _ScriptedClient(list(_PLANNER_STUB_RESPONSES))
        optimizer = OptimizerLLM(
            client=_ScriptedClient([_OPTIMIZER_STUB_RESPONSE]),
            target_agent="planner",
            config_path=args.config,
            **optimizer_kwargs,
        )
        if args.use_critic:
            critic = VideoStageCritic(
                cloud_client=_ScriptedClient(
                    [_CRITIC_STAGE_STUB_RESPONSE], model_id="stub-cloud-critic"
                ),
                prefilter_client=_ScriptedClient(
                    [_CRITIC_PREFILTER_STUB_RESPONSE], model_id="stub-prefilter"
                ),
            )
        return planner_client, optimizer, critic

    planner_client = VLLMClient(config_path=args.config)
    try:
        optimizer = OptimizerLLM.from_config(
            "planner", config_path=args.config, **optimizer_kwargs
        )
    except ValueError as exc:
        logger.error(
            "Optimizer unavailable: {}. Configure optimizer.model_id in "
            "the models config or run with --stub-llm.",
            exc,
        )
        return None
    if args.use_critic:
        from llm.cloud_critic import CloudVLMClient

        try:
            cloud_client = CloudVLMClient(config_path=args.config)
        except ValueError as exc:
            logger.error(
                "Cloud critic unavailable: {}. Use --no-critic or "
                "--stub-llm, or configure cloud_critic.model_id.",
                exc,
            )
            return None
        critic = VideoStageCritic(
            cloud_client=cloud_client,
            prefilter_client=VLLMClient(config_path=args.config),
        )
    return planner_client, optimizer, critic


def _write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    train_tasks: list[TaskDefinition],
    val_tasks: list[TaskDefinition],
    seed_source: str,
) -> Path:
    """Snapshot every run input for reproducibility."""
    snapshot = {
        "argv": sys.argv,
        "config_path": str(args.config),
        "models_config": load_config(args.config),
        "tasks_path": str(args.tasks),
        "val_tasks_path": str(args.val_tasks) if args.val_tasks else None,
        "train_task_ids": [t.id for t in train_tasks],
        "val_task_ids": [t.id for t in val_tasks],
        "epochs": args.epochs,
        "edit_budget": args.edit_budget,
        "use_critic": args.use_critic,
        "max_rounds": args.max_rounds,
        "stub_llm": args.stub_llm,
        "seed_prompt_source": seed_source,
    }
    path = run_dir / "run_config.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.epochs < 1:
        logger.error("--epochs must be >= 1, got {}", args.epochs)
        return 2

    split = _split_tasks(args)
    if split is None:
        return 2
    train_tasks, val_tasks = split

    stack = _build_llm_stack(args)
    if stack is None:
        return 2
    planner_client, optimizer, critic = stack

    from forge.env import SimEnv
    from forge.planner_agent import ForgePlannerAgent
    from forge.registry import ForgePromptRegistry

    run_dir = make_run_dir(args.out)
    seed_source = (
        str(args.initial_prompt) if args.initial_prompt else "planner system_v0.md"
    )
    config_path = _write_run_config(run_dir, args, train_tasks, val_tasks, seed_source)

    registry = ForgePromptRegistry(run_dir / "registry")
    runner = ForgeRunner(
        registry=registry,
        env=SimEnv(repo_root=REPO_ROOT),
        planner=ForgePlannerAgent(vllm_client=planner_client),
        optimizer=optimizer,
        critic=critic,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        max_rounds=args.max_rounds,
        run_dir=run_dir,
        initial_prompt_text=_initial_prompt_text(args),
    )

    logger.info(
        "Starting forge run: epochs={} train={} val={} critic={} stub_llm={} "
        "run_dir={}",
        args.epochs,
        [t.id for t in train_tasks],
        [t.id for t in val_tasks],
        args.use_critic,
        args.stub_llm,
        run_dir,
    )
    forge_log = runner.run(args.epochs)

    for epoch in forge_log.epochs:
        print(
            f"epoch {epoch.epoch_index}: candidate={epoch.candidate_version_id} "
            f"accepted={epoch.accepted} "
            f"val_success_rate={epoch.validation_success_rate} "
            f"reason={epoch.validation_reason}"
        )
    print(f"Final best prompt version: {forge_log.final_best_version_id}")
    print(f"Run config: {config_path}")
    print(f"Forge log:  {run_dir / 'forge_log.json'}")
    print(f"Metrics:    {run_dir / 'metrics.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
