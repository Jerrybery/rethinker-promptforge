"""Unit tests for the forge occlusion task suite (configs/forge_tasks.yaml).

The suite is the task set the forge experiments (Task 3.10) run against:
pick-place-style RoboTwin tasks with occlusion variants, split into a train
set and a validation set whose occlusion patterns are UNSEEN in train
(split by pattern, not just by seed).
"""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import ModuleType

from forge.loader import load_forge_tasks
from tasks.schema import TaskDefinition

REPO_ROOT = Path(__file__).resolve().parents[2]
FORGE_TASKS_PATH = REPO_ROOT / "configs" / "forge_tasks.yaml"
HELLO_TASKS_PATH = REPO_ROOT / "data" / "tasks" / "hello_tasks.yaml"

EXPECTED_TRAIN_IDS = {
    "forge-train-a2b-right-distractor",
    "forge-train-a2b-left-distractor",
    "forge-train-can-pot-distractor",
    "forge-train-bread-skillet-distractor-randomlight",
    "forge-train-container-plate-distractor-randomlight",
}

EXPECTED_VAL_IDS = {
    "forge-val-object-basket-randomcam",
    "forge-val-can-basket-randomcam-randomlight",
    "forge-val-empty-cup-noclutter-randomcam-randomlight",
}

EXPECTED_TRAIN_PATTERNS = {
    "occ-distractor-staticcam-fixedlight",
    "occ-distractor-staticcam-randomlight",
}

EXPECTED_VAL_PATTERNS = {
    "occ-distractor-randomcam-fixedlight",
    "occ-distractor-randomcam-randomlight",
    "occ-nodistractor-randomcam-randomlight",
}

# Valid RoboTwin task names, mirroring how robot/robottwin_env.py resolves
# tasks (importlib.import_module(f"envs.{task_name}")). Static snapshot of the
# fork's envs/ directory (submodule @ c3ddfa8); regenerate with:
#   ls third_party/RoboTwin/envs/*.py | xargs -n1 basename \
#       | sed 's/\.py$//' | grep -v '^_' | sort
ROBOTTWIN_TASK_NAMES = frozenset(
    {
        "adjust_bottle",
        "beat_block_hammer",
        "blocks_ranking_rgb",
        "blocks_ranking_size",
        "click_alarmclock",
        "click_bell",
        "dump_bin_bigbin",
        "grab_roller",
        "handover_block",
        "handover_mic",
        "hanging_mug",
        "lift_pot",
        "move_can_pot",
        "move_pillbottle_pad",
        "move_playingcard_away",
        "move_stapler_pad",
        "open_laptop",
        "open_microwave",
        "pick_diverse_bottles",
        "pick_dual_bottles",
        "place_a2b_left",
        "place_a2b_right",
        "place_bread_basket",
        "place_bread_skillet",
        "place_burger_fries",
        "place_can_basket",
        "place_cans_plasticbox",
        "place_container_plate",
        "place_dual_shoes",
        "place_empty_cup",
        "place_fan",
        "place_mouse_pad",
        "place_object_basket",
        "place_object_scale",
        "place_object_stand",
        "place_phone_stand",
        "place_shoe",
        "press_stapler",
        "put_bottles_dustbin",
        "put_object_cabinet",
        "rotate_qrcode",
        "scan_object",
        "shake_bottle",
        "shake_bottle_horizontally",
        "stack_blocks_three",
        "stack_blocks_two",
        "stack_bowls_three",
        "stack_bowls_two",
        "stamp_seal",
        "turn_switch",
    }
)

ROBOTTWIN_TASK_CONFIGS = frozenset({"demo_clean", "demo_randomized"})


def _load_run_forge_module() -> ModuleType:
    """Import scripts/run_forge.py (not a package) for _split_tasks tests."""
    script = REPO_ROOT / "scripts" / "run_forge.py"
    spec = importlib.util.spec_from_file_location("run_forge", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_of(task: TaskDefinition) -> str | None:
    return (task.metadata or {}).get("split")


def test_forge_catalogue_loads_and_validates() -> None:
    """configs/forge_tasks.yaml loads through the forge loader."""
    tasks = load_forge_tasks(FORGE_TASKS_PATH)
    assert len(tasks) == 8
    assert all(isinstance(t, TaskDefinition) for t in tasks)
    assert all(t.id.startswith("forge-") for t in tasks)
    assert len({t.id for t in tasks}) == 8


def test_forge_catalogue_exact_split_counts() -> None:
    """Exactly 5 train and 3 val tasks, by metadata.split."""
    tasks = load_forge_tasks(FORGE_TASKS_PATH)
    train = {t.id for t in tasks if _split_of(t) == "train"}
    val = {t.id for t in tasks if _split_of(t) == "val"}
    assert train == EXPECTED_TRAIN_IDS
    assert val == EXPECTED_VAL_IDS


def test_forge_tasks_reference_valid_robottwin_tasks() -> None:
    """Every initial_scene/metadata RoboTwin task name exists in the fork."""
    tasks = load_forge_tasks(FORGE_TASKS_PATH)
    for task in tasks:
        assert task.initial_scene is not None, f"{task.id} missing initial_scene"
        scene_name = task.initial_scene.get("task_name")
        assert scene_name in ROBOTTWIN_TASK_NAMES, (
            f"{task.id}: initial_scene.task_name {scene_name!r} not in the "
            "RoboTwin fork's envs/ registry"
        )
        metadata = task.metadata or {}
        rt_name = metadata.get("robottwin_task_name")
        assert rt_name in ROBOTTWIN_TASK_NAMES, (
            f"{task.id}: metadata.robottwin_task_name {rt_name!r} not in the "
            "RoboTwin fork's envs/ registry"
        )
        assert metadata.get("robottwin_task_config") in ROBOTTWIN_TASK_CONFIGS


def test_forge_val_patterns_disjoint_from_train() -> None:
    """Validation occlusion patterns must be UNSEEN: disjoint pattern ids."""
    tasks = load_forge_tasks(FORGE_TASKS_PATH)
    train_patterns = {
        (t.metadata or {}).get("occlusion_pattern")
        for t in tasks
        if _split_of(t) == "train"
    }
    val_patterns = {
        (t.metadata or {}).get("occlusion_pattern")
        for t in tasks
        if _split_of(t) == "val"
    }
    assert train_patterns == EXPECTED_TRAIN_PATTERNS
    assert val_patterns == EXPECTED_VAL_PATTERNS
    assert train_patterns.isdisjoint(val_patterns), (
        f"val occlusion patterns leak into train: "
        f"{train_patterns & val_patterns}"
    )


def test_forge_seed_ranges_disjoint_across_splits() -> None:
    """Held-out seed ranges: no train/val seed-range overlap."""
    tasks = load_forge_tasks(FORGE_TASKS_PATH)

    def ranges(split: str) -> list[tuple[int, int]]:
        out = []
        for t in tasks:
            if _split_of(t) == split:
                lo, hi = (t.metadata or {})["seed_range"]
                out.append((int(lo), int(hi)))
        return out

    train_ranges = ranges("train")
    val_ranges = ranges("val")
    assert train_ranges and val_ranges
    for t_lo, t_hi in train_ranges:
        for v_lo, v_hi in val_ranges:
            assert t_hi < v_lo or v_hi < t_lo, (
                f"seed range overlap: train [{t_lo},{t_hi}] vs val [{v_lo},{v_hi}]"
            )


def test_forge_tasks_required_metadata() -> None:
    """Every task carries split/pattern/role/max_rounds/occlusion metadata."""
    tasks = load_forge_tasks(FORGE_TASKS_PATH)
    for task in tasks:
        metadata = task.metadata or {}
        assert metadata.get("split") in {"train", "val"}, task.id
        pattern = metadata.get("occlusion_pattern")
        assert isinstance(pattern, str) and pattern.strip(), task.id
        assert metadata.get("forge_role") in {
            "prompt_training",
            "prompt_validation",
        }, task.id
        max_rounds = metadata.get("max_rounds")
        assert isinstance(max_rounds, int) and max_rounds > 0, task.id
        sources = metadata.get("occlusion_sources")
        assert isinstance(sources, list) and len(sources) > 0, task.id
        seed_range = metadata.get("seed_range")
        assert (
            isinstance(seed_range, list)
            and len(seed_range) == 2
            and seed_range[0] <= seed_range[1]
        ), task.id
        seed = (task.initial_scene or {}).get("seed")
        assert isinstance(seed, int) and seed_range[0] <= seed <= seed_range[1], (
            task.id
        )
        assert task.instruction.strip(), task.id
        assert task.success_criteria, task.id


def test_load_forge_tasks_split_filter() -> None:
    """load_forge_tasks(split=...) filters by metadata.split."""
    train = load_forge_tasks(FORGE_TASKS_PATH, split="train")
    val = load_forge_tasks(FORGE_TASKS_PATH, split="val")
    assert {t.id for t in train} == EXPECTED_TRAIN_IDS
    assert {t.id for t in val} == EXPECTED_VAL_IDS
    assert load_forge_tasks(FORGE_TASKS_PATH, split="nonexistent") == []


def test_run_forge_split_tasks_uses_metadata_split() -> None:
    """run_forge --tasks on the suite auto-splits by metadata.split."""
    module = _load_run_forge_module()
    args = Namespace(tasks=FORGE_TASKS_PATH, val_tasks=None)
    split = module._split_tasks(args)
    assert split is not None
    train, val = split
    assert {t.id for t in train} == EXPECTED_TRAIN_IDS
    assert {t.id for t in val} == EXPECTED_VAL_IDS


def test_run_forge_split_tasks_fallback_holdout_unchanged() -> None:
    """Catalogues without metadata.split keep the last-third holdout."""
    module = _load_run_forge_module()
    args = Namespace(tasks=HELLO_TASKS_PATH, val_tasks=None)
    split = module._split_tasks(args)
    assert split is not None
    train, val = split
    assert len(train) == 1
    assert len(val) == 1
    assert train[0].id == "hello-place-a2b-right"
    assert val[0].id == "hello-pick-only"
