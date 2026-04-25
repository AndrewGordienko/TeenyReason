# Benchmark Harness

This file describes the current app-layer benchmark harness.

It is intentionally not the crawler API spec. The harness exists to exercise
the current research stack, record artifacts, and feed the dashboard while the
library refactor is underway.

## What Lives Here

The benchmark harness owns:

- seed loops
- artifact writing
- dashboard context updates
- benchmark profile selection
- side-by-side downstream comparisons
- compatibility glue for the current RL stack

The benchmark harness does not define the crawler core.

## Current Entry Points

- `main.py`
  Composes `run(...)` and `ppo()`. The local default is `seeds=2`; use `5`
  seeds only for confirmation runs.
- `teenyreason/app/benchmark.py`
  Runs the current RL benchmark harness.
- `teenyreason/app/multidomain_suite.py`
  Runs the current RL, image, and language sample-efficiency suite.

## Current Compatibility Boundary

Right now the RL benchmark still consumes legacy crawler objects such as:

- `PredictiveBelief`
- `MetricBelief`
- `UncertaintyEstimate`
- `EnvExpression`

Those remain valid during migration, but they are now compatibility adapters on
top of the generic crawler contract in `teenyreason/crawler/types.py`.

## What Should Stay Out Of The Crawler Core

These belong to the benchmark/app layer or downstream consumer layer:

- PPO rollout/update policy
- fair handoff policy
- solver expression scaling
- shadow/oracle benchmark variants
- dashboard payload shaping
- saved artifact packaging

The crawler core should only know how to gather evidence, update belief, score
queries, stop, and emit a generic message.
