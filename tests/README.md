# Test Layout

The test suite is grouped by subsystem so it is obvious where new coverage belongs.

## Folders

- `api/`
  Public API and import-compatibility coverage.
- `crawler/`
  Crawler interfaces, library utilities, and recipe-level guardrails.
- `probe/`
  Probe collection, probe-family logic, env-expression handoff, and phase gating.
- `rl/`
  Planner/controller behavior, benchmark metrics, and belief-target training helpers.
- `viz/`
  Dashboard payloads and live dashboard behavior.
- `multidomain/`
  Multidomain benchmark summaries.

## Where To Add New Tests

- Put benchmark metric and gate tests in `rl/`.
- Put env-expression and probe-family tests in `probe/`.
- Put compatibility and re-export tests in `api/`.
- Put dashboard payload and artifact surfacing tests in `viz/`.

## Where Not To Add New Tests

- Do not add new top-level `tests/test_*.py` files unless there is a very strong reason.
- Do not mix subsystem concerns in one test file when a clearer folder already exists.

## Default Command

From the repo root:

```bash
python3 -m unittest discover -s tests
```

