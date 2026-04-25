# RL Maintainer Map

## What This Folder Owns

`teenyreason/rl/` owns the downstream control stack for the latent-belief experiments:

- plain PPO baselines
- probe-conditioned PPO
- full-system controller/planner training
- simulator fanout
- shared rollout and optimizer primitives

## Main Entrypoints

- `core/`
  Shared PPO models and optimizer utilities. Start here when the change is about rollout packing, actor/value heads, or PPO updates.
- `probe_policy/`
  Probe-conditioned PPO entrypoints and probe-policy helpers. Start here when the change is about env-expression handoff, probe-family selection, or matched no-expression benchmarking.
- `full_system/`
  Belief-controller, planner, simulator fanout, and the shared curriculum/context helpers used by those systems.

## Public Imports

Use the subpackage entrypoints directly:

- `teenyreason.rl.core`
- `teenyreason.rl.probe_policy`
- `teenyreason.rl.full_system`

`teenyreason.rl.full_system` exports the common planner, controller, fanout,
train, and evaluation entrypoints. For narrower ownership, import from the
specific files under `full_system/`, such as `planner_train.py`,
`affordance_train.py`, or `simulator_fanout.py`.

## Where To Add New Code

- Add PPO-core math or models under `core/`.
- Add probe-conditioned policy behavior under `probe_policy/`.
- Add planner/controller/fanout behavior under `full_system/`.
- Add shared full-system support logic under `full_system/context_support.py` or `full_system/curriculum.py` before inventing new cross-module private imports.

## Where Not To Add New Code

- Do not reintroduce flat PPO shim files under `teenyreason/rl/`.
- Do not reintroduce flat full-system shim files under `teenyreason/rl/`.
- Do not import private helpers out of full-system training modules into other modules.
- Do not hide new behavior in `__init__.py` files.

## Metrics And Tests That Matter

- Probe path: solve episode, solve env steps, `probe_ready_frac`, `probe_muted_frac`, env-expression-off matched eval, probe-family selection fractions.
- Representation-sensitive downstream checks: learned vs `state-only`, `zero`, `shuffled`, `stale`, `no-refresh`, `frozen`.
- Tests:
  - `tests/probe/`
  - `tests/rl/`
  - `tests/api/test_import_compatibility.py`
