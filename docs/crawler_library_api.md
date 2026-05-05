# Crawler Library API

This file describes the public crawler library contract.

It is intentionally narrower and more implementation-facing than
[core_idea.md](./core_idea.md) or
[architecture_and_training.md](./architecture_and_training.md).

## Purpose

The crawler library owns the path from:

`evidence -> belief -> message`

Downstream algorithms should consume that output instead of quietly rebuilding
their own belief format inside one benchmark-specific learner.

## Canonical Public Interfaces

The stable interfaces live in `teenyreason/crawler/core.py`.

- `WorldAdapter`
  Exposes available queries and executes one query into an evidence slice.
- `BeliefBackend`
  Updates the current belief from evidence.
- `QueryPolicy`
  Chooses the next query.
- `StopPolicy`
  Decides when the crawler should stop collecting evidence.
- `MessageProjector`
  Turns a belief state into a downstream-facing message.
- `Crawler`
  Runs the generic crawl-update-stop-message loop.

## Canonical Public Types

The stable runtime types live in `teenyreason/crawler/types.py`.

- `EvidenceSlice`
  One generic evidence slice with an opaque payload and lightweight metadata.
- `BeliefState`
  The current latent belief, uncertainty, support size, and diagnostics
  metadata.
- `CrawlerMessage`
  The downstream-facing message: vector, confidence, ready flag, uncertainty,
  and metadata.
- `CrawlerStep`
  One crawler update after collecting one evidence slice.
- `CrawlerRunResult`
  The final crawler output before a downstream consumer takes over.

## Tiny Public Surface

The intended top-level user API is deliberately small:

- `run(env, algo=ppo(), seeds=2, profile="fast")`
- `probe_ppo(profile="fast")`
- `ppo(**kwargs)`
- `crawler(env)`
- `make(env)`
- `load_crawler(env, checkpoint_path=None)`
- `Crawler`, `Evidence`, `Belief`, `Message`, `Step`, `Run`

Use `crawler(...)` and `probe_ppo(...)` separately when you want one
crawler-backed PPO training run by itself, without the baseline,
no-expression, controller, and sim-fanout comparison tracks. Pass `seeds=2` or
a seed list only when you intentionally want repeated runs. Use a Gym id string
for standard environments, for example
`run("BipedalWalker-v3", ppo(), seeds=2, profile="fast")`. Special local envs
may still pass a custom factory/class.

The rest of the package is still importable, but these are the names docs and
examples should teach first.

## Recipes

Recipes live in `teenyreason/recipes/`.

Recipes are user-facing compositions, not crawler-core abstractions. They are
allowed to choose:

- target builders
- intervention catalogs
- intervention scorers
- benchmark compatibility metadata

They are not allowed to make the crawler core branch on a specific world.

## Downstream Consumers

Downstream consumers live in `teenyreason/consumers.py`.

Examples still exist internally, but the public path should usually go through
`run(...)` and `ppo()` instead of teaching the concrete consumer class names first.

The consumer decides how to use the crawler message. The crawler does not own
PPO handoff rules, fair benchmark policy, or dashboard packaging.

## Runtime Crawler

Use `load_crawler(...)` when you want the crawler to be a separate entity:

```python
import teenyreason as tr

env_id = "ContinuousCartPole-v0"
crawler = tr.load_crawler(env_id)
result = crawler(seed=0)
algo_context = result.as_algo_context()
```

The returned context includes the `EnvExpression`, optional
`ControllerBeliefContext`, flat expression vector, confidence, readiness,
uncertainty, and probe-cost metadata. Downstream algorithms can consume that
payload directly or expose a `set_crawler_context(...)` /
`with_crawler_context(...)` hook for `crawler.run_into(env, algo)`.

## Compatibility Layer

The current RL benchmark still uses legacy compatibility objects:

- `PredictiveBelief`
- `MetricBelief`
- `UncertaintyEstimate`
- `EnvExpression`
- `ControllerBeliefContext`

Those remain exported during migration, but they are not the canonical library
API anymore.

## Main Entry Point

`main.py` should stay one screen and read like:

1. `env = make("ContinuousCartPole-v0")`
2. `crawler = crawler(env)`
3. `ppo = probe_ppo(profile="fast")`
4. `ppo.train(crawler)`

That keeps the library boundary explicit even when the current benchmark still
uses compatibility adapters under the hood.

Use the default single probe PPO run for local iteration. Move to repeated
seeds only when you want a confirmation run.
