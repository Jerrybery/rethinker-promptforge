# EmbodiedPromptForge Pipeline Runbook

The five-layer pipeline for prompt optimization under occlusion, and the
governance rules that keep its results honest.

## Layer 1 — Environment (RoboTwin sim, single-arm)

- Embodiment: `franka-panda` (single arm, mirrors the Realman-65B target).
  `aloha-agilex` is retired (dual-arm).
- Every task must pass an **actor-mapping audit**: `object_actors` maps each
  semantic label to a real env actor attribute; grasps/places go through the
  RoboTwin actor-level skills (`grasp_actor`/`place_actor`/
  `move_by_displacement`) via `RoboTwinBackend`.
- Execution honesty: `lift()` physically verifies the carry (gripped object
  must follow in z); a kinematically successful but empty grasp is retried
  once with the alternate contact mode, then reported as failure.
- Visibility: detection is NOT all-knowing. `RaycastVisibilityProvider`
  (PhysX raycasts from the head camera) plus the declarative `hidden_by`
  cover rule determine which labels appear in observations.

## Layer 2 — Task-set governance

Admission criteria for any task (train or val):

1. **Reachability smoke**: the intended strategy executed manually through
   `SimEnv` yields `check_success=True` (2 of 3 runs minimum — curobo and
   friction have variance).
2. **Stable scene generation**: no `UnStableError` at the chosen seed;
   scan seeds when in doubt.
3. **Discrimination**: the task must not be trivially 1-step successful
   for the seed prompt, nor physically impossible (0 forever). It should
   sit in the "strategy-dependent" band.
4. **Held-out separation (val)**: different seed and/or variant from any
   train task of the same family; val randomization flags per convention.

Current set (train 4 / val 4): train = a2b-right, a2b-left, can-pot,
can-pot-occluded@301; val = empty-cup@960 (easy anchor),
container-plate@901 (physical medium), object-stand@922 (physical hard),
can-pot-occluded@310 (semantic occlusion).

## Layer 3 — Forge loop (runner)

Per epoch: train rollouts (1 per train task) → video-stage critic on
failures → optimizer bounded edits → candidate registered → validation
gate. `faulthandler` watchdog dumps thread stacks every 15 min into the
run log; SAPIEN/GPU contention (vLLM colocated) previously wedged camera
readback — keep GPU residents minimal.

## Layer 4 — Quality gate (validator)

- **4 episodes per val task** (16 per validation, ~25 min). Never accept
  on single-episode deltas.
- Composite: lexicographic `(success_rate, -average_steps)`;
  `average_steps` is computed over SUCCESSFUL episodes only (a candidate
  must not "improve" by failing fast); a fully-failed candidate gets the
  worst-case `max_rounds`.
- Accept only on strict improvement over the incumbent's recorded metrics,
  same fixed task set, same episode budget.

## Layer 5 — Confirmation & promotion

A gate `accepted` is a **nomination**, not a result. Before a prompt
version is cited as an improvement:

1. Run the paired confirmation:
   `scripts/forge_confirm.py --a <best>.md --b <candidate>.md
   --episodes-per-task 6` (48 interleaved A/B episodes).
2. Confirmation is positive when B's overall rate exceeds A's and B is not
   worse on any single task. Physical-variance bands overlap easily at
   small N — demand a delta visible in at least two tasks, not one.
3. Only after a positive confirmation may the version be archived/reported
   as an improvement; the confirmation JSON lives next to the run's
   `forge_log.json` (`confirm_<A>_vs_<B>.json`).

## Operational notes

- Planner LLM: DeepSeek via OpenRouter (`vllm:` section of
  `configs/models.local.yaml`; the direct DeepSeek account ran out of
  credit 2026-08 — switch back when refilled).
- tmux `forge-fix`: window 2 = forge runs, window `confirm` = confirmation
  experiments. Never run two SAPIEN experiments in the same window.
- Backups: every manual config/code edit keeps a `.bak-stepN` sibling.
