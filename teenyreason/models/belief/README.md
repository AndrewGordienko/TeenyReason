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

## Where To Add New Code

- Add new encoder/projector modules under `core/`.
- Add new training losses under `objectives/losses.py` or a sibling objective module.
- Add training schedule or loop code under `training/`.

## Public Imports

- Prefer direct imports from `core/`, `objectives/`, and `training/`.
- `teenyreason.models.belief` is the public package for belief world-model code.
- `teenyreason.models.envbelief` is the public package for env-belief aggregation code.

## Where Not To Add New Code

- Do not reintroduce flat compatibility modules beside this README.
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
  - `tests/crawler_probes/test_probe_logic.py`
  - `tests/rl/test_belief_targets.py`
