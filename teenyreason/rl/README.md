# RL Maintainer Map

## What This Folder Owns

`teenyreason/rl/` owns the downstream control stack for the latent-belief experiments:

- plain PPO baselines
- probe-conditioned PPO
- shared rollout and optimizer primitives

## Main Entrypoints

- `core/`
  Shared PPO models and optimizer utilities. Start here when the change is about rollout packing, actor/value heads, or PPO updates.
- `probe_policy/`
  Probe-conditioned PPO entrypoints and probe-policy helpers. Start here when the change is about env-expression handoff, probe-family selection, or matched no-expression benchmarking.
- `probe_policy/budgeting/`
  Probe-family economics, standard/fair selection, and budget coverage rules.
- `probe_policy/handoff/`
  Solver-facing env-expression construction, message audits, and handoff diagnostics.
- `probe_policy/training/`
  PPO training loops, matched evaluation, rollout helpers, logging, and reporting.

## Public Imports

Use the subpackage entrypoints directly:

- `teenyreason.rl.core`
- `teenyreason.rl.probe_policy`

## Where To Add New Code

- Add PPO-core math or models under `core/`.
- Add probe-conditioned policy behavior under `probe_policy/`.
- Add crawler support-collection logic under `teenyreason/crawler/runtime/`.

## Where Not To Add New Code

- Do not reintroduce flat PPO shim files under `teenyreason/rl/`.
- Do not reintroduce the retired `full_system` planner/controller stack under `teenyreason/rl/`.
- Do not hide new behavior in `__init__.py` files.

## Metrics And Tests That Matter

- Probe path: solve episode, solve env steps, `probe_ready_frac`, `probe_muted_frac`, env-expression-off matched eval, probe-family selection fractions.
- Representation-sensitive downstream checks: learned vs zero/shuffled/stale/no-expression handoff controls.
- Tests:
  - `tests/crawler_probes/`
  - `tests/rl/`
  - `tests/api/test_import_compatibility.py`
