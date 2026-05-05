# Crawler Probes Maintainer Map

## What This Folder Owns

`teenyreason/crawler/probes/` owns the environment-facing probing side of the crawler:

- probe data collection
- active probe selection / exploration policy glue
- latent aggregation helpers used by crawler runtime and probe-conditioned PPO

## Main Entrypoints

- `data/`
  Canonical probe data collection helpers.
- `latent/`
  Canonical latent update and aggregation helpers.
- `explorer.py`
  The active probe planner currently used by the RL path.

## Where To Add New Code

- Add environment interaction / rollout gathering code under `data/`.
- Add belief update and latent aggregation code under `latent/`.
- Add probe-planner logic in `explorer.py` or a sibling module if it clearly owns a distinct planner responsibility.

## Where Not To Add New Code

- Do not put new downstream PPO/controller behavior here; that belongs in `teenyreason/rl/`.
- Do not reintroduce top-level `teenyreason/probe`; probes are crawler internals.

## Metrics And Tests That Matter

- support diversity
- probe leakage
- future-probe gain
- uncertainty-drop after probing
- Tests:
  - `tests/crawler_probes/`
  - `tests/crawler/` when runtime data-collection contracts move
