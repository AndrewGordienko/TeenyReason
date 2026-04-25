# Probe Maintainer Map

## What This Folder Owns

`teenyreason/probe/` owns the environment-facing side of the latent-belief story:

- probe data collection
- active probe selection / exploration policy glue
- latent aggregation facades used by the rest of the repo

## Main Entrypoints

- `data/`
  Canonical probe data collection helpers.
- `latent/`
  Canonical latent update and aggregation helpers.
- `explorer.py`
  The active probe planner currently used by the RL path.

## Compatibility Shims

These facades stay alive during the first reorg pass:

- `probe_data.py`
- `probe_latent.py`

Do not delete them until the audit script says they are truly unreferenced and all internal imports have migrated.

## Where To Add New Code

- Add environment interaction / rollout gathering code under `data/`.
- Add belief update and latent aggregation code under `latent/`.
- Add probe-planner logic in `explorer.py` or a sibling module if it clearly owns a distinct planner responsibility.

## Where Not To Add New Code

- Do not put new downstream PPO/controller behavior here; that belongs in `teenyreason/rl/`.
- Do not grow `probe_data.py` or `probe_latent.py` into new primary implementation homes; keep them as compatibility surfaces.

## Metrics And Tests That Matter

- support diversity
- probe leakage
- future-probe gain
- uncertainty-drop after probing
- Tests:
  - `tests/probe/`
  - `tests/crawler/` when data-collection contracts move

