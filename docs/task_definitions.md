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
