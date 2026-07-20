# Task Definitions

This document describes the 4 real-robot evaluation tasks defined in
`data/tasks/real_tasks.yaml`, plus the sim-mapping caveats for each. The
hello-loop smoke tasks live separately in `data/tasks/hello_tasks.yaml` (see
the bottom of this document).

All real tasks target the **Realman-65B** arm. Because no exact RoboTwin sim
counterpart exists for these tasks, each entry also maps to the closest
RoboTwin task (`metadata.robottwin_task_name` / `initial_scene.task_name`,
embodiment `aloha-agilex`) so the closed-loop stack can be exercised in
simulation. Fields without a dedicated schema slot (`max_rounds`,
`occlusion_sources`, `failure_criteria`, `allowed_actions`, `difficulty`)
live in `metadata`; see `src/tasks/schema.py` (`TaskDefinition`).

## Running tasks by name

```bash
# Hello-loop CLI (stubbed reasoning agents, RoboTwin sim backend)
PYTHONPATH=src python scripts/run_hello_loop.py \
    --task-id <task-id> --tasks-path data/tasks/real_tasks.yaml

# Rethinker CLI (real agents), loading the task from the catalogue
PYTHONPATH=src python scripts/run_rethinker.py \
    --config-path configs/models.yaml \
    --catalogue data/tasks/real_tasks.yaml --task-id <task-id>
```

## 1. clear_cluttered_plate_easy

- **Mission type:** `PICK_AND_PLACE` (repeated)
- **Layout:** A plate on the table cluttered with small objects (cup, spoon);
  free table space next to the plate.
- **Goal:** Remove every object from the plate and place it stably on the
  table beside the plate.
- **Occlusion sources:** none (easy baseline; all objects fully visible).
- **Success criteria:** plate empty; removed objects resting stably on the
  table next to the plate; plate not knocked over or displaced.
- **Failure criteria:** object dropped outside the workspace; plate knocked
  over or pushed off its spot; `max_rounds` (10) exceeded.
- **Allowed actions:** `PICK_AND_PLACE`, `REOBSERVE`, `STOP`.
- **Max rounds:** 10.
- **Sim mapping:** RoboTwin `place_container_plate`. Caveat: the sim task
  places a container *onto* a plate (the inverse direction) with a single
  object, so it approximates the pick-and-place skill but not multi-object
  decluttering.

## 2. get_hidden_pliers

- **Mission type:** `PICK_ONLY` (with `MOVE_ASIDE` / `REOBSERVE` sub-actions)
- **Layout:** Workbench with a cloth fully covering the pliers; distractor
  tools (screwdriver, wrench) nearby.
- **Goal:** Move the cloth aside, re-observe, then grasp and lift the pliers.
- **Occlusion sources:** cloth fully covers the pliers at episode start;
  distractor tools partially occlude the pliers from the head camera.
- **Success criteria:** cloth moved aside so pliers are fully visible; pliers
  grasped and lifted clear of the workbench; distractors left on the bench.
- **Failure criteria:** grasped a non-pliers object; cloth knocked off the
  workbench; pliers pushed out of reach; `max_rounds` (12) exceeded.
- **Allowed actions:** `REOBSERVE`, `MOVE_ASIDE`, `PICK_ONLY`, `STOP`.
- **Max rounds:** 12.
- **Sim mapping:** RoboTwin `find_the_hidden_apple`. Caveat: closest existing
  hidden-object discovery task; the target object and scene contents differ
  (apple vs. pliers, no tool distractors).

## 3. fruit_put_pot

- **Mission type:** `PICK_AND_PLACE`
- **Layout:** A fruit on the table and an upright pot within reach.
- **Goal:** Pick the fruit and put it into the pot.
- **Occlusion sources:** the pot rim partially occludes the fruit from the
  head camera during approach; the gripper occludes the fruit during grasping.
- **Success criteria:** fruit inside the pot; pot upright at its original
  position; gripper open and clear of the pot at episode end.
- **Failure criteria:** fruit dropped outside the pot; pot tipped over or
  displaced; `max_rounds` (8) exceeded.
- **Allowed actions:** `PICK_AND_PLACE`, `REOBSERVE`, `STOP`.
- **Max rounds:** 8.
- **Sim mapping:** RoboTwin `move_can_pot`. Caveat: sim moves a rigid can into
  a pot; a real fruit differs in shape, compliance, and grasp tolerance.

## 4. cookie_put_pot

- **Mission type:** `PICK_AND_PLACE` (with `MOVE_ASIDE` / `REOBSERVE`
  sub-actions)
- **Layout:** A cookie on the table partially covered by a napkin, next to an
  upright pot.
- **Goal:** Uncover the cookie if needed, pick it up intact, and put it into
  the pot.
- **Occlusion sources:** napkin partially covers the cookie at episode start;
  the pot body occludes the cookie from the head camera once the cookie is
  near the rim.
- **Success criteria:** cookie inside the pot and intact; pot upright at its
  original position; gripper open and clear of the pot at episode end.
- **Failure criteria:** cookie dropped outside the pot; cookie broken or
  crumbled during grasping; pot tipped over or displaced; `max_rounds` (8)
  exceeded.
- **Allowed actions:** `REOBSERVE`, `MOVE_ASIDE`, `PICK_AND_PLACE`, `STOP`.
- **Max rounds:** 8.
- **Sim mapping:** RoboTwin `place_bread_skillet`. Caveat: closest flat-food
  placement task (bread into skillet); the sim has no napkin occlusion and
  bread is less fragile than a cookie.

## Hello-loop smoke tasks

`data/tasks/hello_tasks.yaml` contains the Milestone-1 smoke tasks
(`hello-place-a2b-right`, `hello-pick-only`) used to validate the closed-loop
wiring against RoboTwin with stub agents. They are not part of the real
evaluation suite.

## Forge occlusion task suite

`configs/forge_tasks.yaml` (Milestone 3, Task 3.9) is the task set the forge
experiments run against: 8 pick-place-style RoboTwin tasks with occlusion
variants, split 5 train / 3 val where **val contains UNSEEN occlusion
patterns** — the split is by occlusion *pattern*, not just by seed. All
variants are expressed through existing RoboTwin knobs
(`domain_randomization` flags + seed ranges); RoboTwin itself is unmodified.

### Occlusion pattern taxonomy

Three randomization axes, combined into pattern ids
(`metadata.occlusion_pattern`, mirrored in `metadata.occlusion_axes`):

| Axis | RoboTwin knob | Effect |
| --- | --- | --- |
| `clutter` | `domain_randomization.cluttered_table` | distractor objects on the table partially occlude the target from the head camera |
| `lighting` | `domain_randomization.random_light` + `crazy_random_light_rate` | illumination-driven obscuration (shadows, glare, low visibility) |
| `camera` | `domain_randomization.random_head_camera_dis > 0` | head-camera viewpoint randomization; changes the occlusion geometry per episode |

- **Train pattern set A** (camera axis never varied):
  `occ-distractor-staticcam-fixedlight`,
  `occ-distractor-staticcam-randomlight`.
- **Validation pattern set B** (camera axis always varied — unseen in train):
  `occ-distractor-randomcam-fixedlight`,
  `occ-distractor-randomcam-randomlight`,
  `occ-nodistractor-randomcam-randomlight` (clutter off: pure viewpoint +
  lighting occlusion).

Split rationale: the forge loop optimizes the planner prompt against train
and accepts candidates only on strict val improvement, so val must measure
*generalization to unseen occlusion*, not memorized pattern fixes. Train and
val pattern ids are disjoint, and seed ranges are held out (train 100–599,
val 900–989). `random_background`, `clean_background_rate`, and
`random_table_height` stay fixed across the suite (future extension axes).

### Tasks

| id | RoboTwin task | split | occlusion_pattern | seed range |
| --- | --- | --- | --- | --- |
| forge-train-a2b-right-distractor | place_a2b_right | train | occ-distractor-staticcam-fixedlight | 100–199 |
| forge-train-a2b-left-distractor | place_a2b_left | train | occ-distractor-staticcam-fixedlight | 200–299 |
| forge-train-can-pot-distractor | move_can_pot | train | occ-distractor-staticcam-fixedlight | 300–399 |
| forge-train-bread-skillet-distractor-randomlight | place_bread_skillet | train | occ-distractor-staticcam-randomlight | 400–499 |
| forge-train-container-plate-distractor-randomlight | place_container_plate | train | occ-distractor-staticcam-randomlight | 500–599 |
| forge-val-object-basket-randomcam | place_object_basket | val | occ-distractor-randomcam-fixedlight | 900–929 |
| forge-val-can-basket-randomcam-randomlight | place_can_basket | val | occ-distractor-randomcam-randomlight | 930–959 |
| forge-val-empty-cup-noclutter-randomcam-randomlight | place_empty_cup | val | occ-nodistractor-randomcam-randomlight | 960–989 |

Each task carries `metadata.split` (`train`/`val`), `metadata.forge_role`
(`prompt_training`/`prompt_validation`), `metadata.occlusion_pattern`,
`metadata.occlusion_axes`, `metadata.occlusion_sources` (human-readable),
`metadata.seed_range` (`[lo, hi]`; `initial_scene.seed` is the per-task base
seed inside it), and `metadata.max_rounds` (8).

### Running the suite

```bash
# Single catalogue: run_forge auto-splits by metadata.split (train/val)
PYTHONPATH=src python scripts/run_forge.py --tasks configs/forge_tasks.yaml --epochs 3

# Programmatic split filtering
from forge.loader import load_forge_tasks
train = load_forge_tasks("configs/forge_tasks.yaml", split="train")
val = load_forge_tasks("configs/forge_tasks.yaml", split="val")
```

Explicit `--tasks/--val-tasks` file pairs keep their previous behavior (files
used as-is), as do catalogues without `metadata.split` (last-third holdout).
