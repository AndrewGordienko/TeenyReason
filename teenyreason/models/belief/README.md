# Belief Model Maintainer Map

## What This Folder Owns

`teenyreason/models/belief/` owns the learned world-belief stack:

- encoder / projector / predictor modules
- belief losses and targets
- training loops and env-belief phase schedules

## Main Entrypoints

- `core/`
  Model components and common tensor helpers.
- `objectives/`
  Losses and supervised targets.
- `training/`
  Training loops, environment-specific schedules, and shared training helpers.

## Compatibility Shims

The older flat modules still work during the first pass:

- `belief_components.py`
- `belief_losses.py`
- `belief_targets.py`
- `belief_training.py`
- `belief_training_env.py`
- `belief_training_env_config.py`
- `belief_training_window.py`
- `belief_training_common.py`

Use the subpackages for new imports, but do not delete the old modules yet.

## Where To Add New Code

- Add new encoder/projector modules under `core/`.
- Add new training losses under `objectives/losses.py` or a sibling objective module.
- Add training schedule or loop code under `training/`.

## Where Not To Add New Code

- Do not keep extending the old flat modules when the new subpackage has an obvious home.
- Do not put dashboard or benchmark-only logic here.

## Metrics And Tests That Matter

- mechanics fit
- neighbor alignment
- split retrieval
- gap ratio
- probe leakage
- uncertainty-error correlation
- Tests:
  - `tests/rl/test_benchmark_metrics.py`
  - `tests/probe/test_probe_logic.py`
  - `tests/rl/test_belief_targets.py`

