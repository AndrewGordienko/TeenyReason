# TeenyReason

Small research experiments around latent environment beliefs for reinforcement
learning.

The core question is:

How do we let an agent interact with an environment for a short time, build a
compact belief about the hidden rules of that world, and then use that belief
to learn or control much faster afterward?

This repo is no longer best thought of as just "probe-conditioned PPO." That is
only one downstream testbed. The main object of study is the latent space.

## Research Framing

Read [docs/README.md](docs/README.md) first for the full docs map.

Then read [docs/core_idea.md](docs/core_idea.md) first.

Then read [docs/research_manifesto.md](docs/research_manifesto.md) for the
current thesis, intended architecture, target losses, and the metrics that are
actually supposed to matter.

Then read [docs/general_crawler_belief.md](docs/general_crawler_belief.md) and
[docs/language_domain_synthesis.md](docs/language_domain_synthesis.md) for the
broader cross-domain version of the idea. The current code is mostly an RL
testbed, but the intended latent-belief mechanism is supposed to generalize to
language, images, and other structured environments too.

Then read [docs/image_domain_synthesis.md](docs/image_domain_synthesis.md) and
[docs/image_belief_design.md](docs/image_belief_design.md) for the image-side
version of the same thesis: few-shot concept learning, object-centric visual
beliefs, and support-set-efficient downstream solving.

Then read [docs/human_learning_synthesis.md](docs/human_learning_synthesis.md)
and [docs/human_learning_design.md](docs/human_learning_design.md) for the
developmental-science and neuroscience bridge: core knowledge, surprise-driven
exploration, predictive maps, and what "child-like" sample efficiency should
mean in code rather than only as a metaphor.

The latent in this repo is intended to be:

- predictive
- task-relevant
- uncertainty-aware
- reusable across downstream control

More generally, the repo is trying to study a reusable
`crawler -> belief -> solver` pattern, where the crawler learns the hidden
constraints of a new environment quickly and the solver then focuses on the
task instead of first having to infer the world from scratch.

## Mental Model

The base object is the crawler. A recipe tells the crawler what world to probe,
which evidence windows to collect, and how to summarize those windows into a
compact belief with uncertainty. The env expression is the explicit handoff
from crawler to solver: it carries the belief vector, confidence, readiness,
mute reason, source, and scale used by the downstream model.

PPO, the language Transformer, and the image classifier are consumers of that
handoff. They are not the crawler core. A small public run should read like:

```python
import teenyreason as tr

cartpole = tr.experiment("cartpole")
tr.run(cartpole, tr.ppo(), seeds=2, profile="fast")
```

`tr.experiment(...)` and the shorter recipe builder are the same composition
entrypoint; use whichever reads better in local research scripts.

## Progress Metrics

There are now two explicit progress targets instead of a vague “this run feels
better” heuristic.

- `Belief Progress Index (BPI)`
  `0.30*mechanics_fit + 0.20*neighbor_alignment + 0.15*split_retrieval + 0.15*(1-heldout_probe_error) + 0.10*max(uncert_error_corr, 0) + 0.10*(1-probe_leakage)`
- `Latent Win Gate`
  The benchmark only counts as cracked on the `full` 5-seed run when the probe
  path beats the baseline and the matched no-expression arm, clears the
  representation floors, and the full-system learned context materially beats
  the context ablations.

If downstream return rises while `BPI` falls, treat that run as suspect. The
representation is the primary object of study, and the controller only gets to
win if the latent-sensitive ablations agree.

## Repo Layout

- `main.py`
  Small composition entrypoint. Pick a recipe and a downstream consumer here.
- `serve_latent_dashboard.py`
  Root wrapper for the local Flask dashboard.
- `teenyreason/crawler/`
  Canonical world-agnostic crawler interfaces, runtime types, and compatibility
  adapters for the current RL stack.
- `teenyreason/recipes/`
  Concrete recipe compositions such as CartPole, MNIST, and language. These
  choose capabilities without making them part of the crawler core.
- `teenyreason/consumers.py`
  Small downstream benchmark consumers used by the public `run(...)` and
  `ppo(...)` helpers.
- `teenyreason/representation/`
  Stable entrypoint for the current latent-belief system and latent snapshot
  analysis helpers.
- `teenyreason/crawler/probes/`
  Probe data collection, active probing, belief aggregation, and online belief
  updates. See [teenyreason/crawler/probes/README.md](teenyreason/crawler/probes/README.md).
- `teenyreason/rl/`
  Downstream control code. The first reorg pass now splits this into:
  `rl/core/`, `rl/probe_policy/`, and `rl/full_system/`. See
  [teenyreason/rl/README.md](teenyreason/rl/README.md).
- `teenyreason/viz/`
  Local dashboard for latent snapshots and benchmark summaries.
- `teenyreason/models/belief/`
  Belief encoder stack. The first reorg pass now splits this into `core/`,
  `objectives/`, and `training/`. See
  [teenyreason/models/belief/README.md](teenyreason/models/belief/README.md).
- `tests/`
  Subsystem-oriented unit tests. See [tests/README.md](tests/README.md).
- `papers/`
  Local paper library for the repo. See [papers/README.md](papers/README.md)
  for the categorized reading map.

## Maintainer Notes

- New imports should prefer the organized subpackages under `teenyreason/rl/`
  and `teenyreason/models/belief/`.
- Old flat module paths are still kept alive as compatibility shims during the
  first pass.
- Conservative dead-file review now lives in:

```bash
python3 scripts/audit_python_files.py
```

That script reports orphan candidates, but it does not delete anything.

## Current Flow

1. Collect probe trajectories across a family of environments.
2. Turn those into fixed-length windows.
3. Train a latent encoder and structured prediction heads.
4. Save a latent snapshot artifact for visualization.
5. Train a baseline PPO agent and a probe-conditioned PPO agent.
6. Compare not just return, but solve speed and environment-step cost.

The current fair-mode controller contract is intentionally small and explicit:

- the crawler gathers a short support set first
- it hands PPO one compact `EnvExpression` for the episode
- PPO consumes `state + env_expression.vector + confidence + uncertainty`
- the actor/value heads use a plain state backbone plus a small
  confidence-scaled env-expression residual
- fair mode freezes the env expression for the whole control episode
- fair mode never buys a third probe; it either hands off early when the
  expression is ready or hands off after probe two with the env expression
  muted if it still has not earned trust
- fair mode always starts with a passive identification probe and uses a
  deterministic second active probe when one more world-identification step is
  needed
- probe-conditioned PPO is trained with occasional env-expression muting so it
  keeps a healthy state-only fallback instead of becoming brittle; after the
  early warmup that muting becomes scale jitter instead of hard zeroing
- any factorized latent machinery is now considered internal training support,
  not the public controller contract

The benchmark now records three matched downstream arms:

- baseline PPO
- probe + env expression
- probe + no env expression

That is intentional. A benchmark win only counts as a thesis win when the
env-expression arm beats both the baseline and the matched no-expression arm.
If probing helps but muting the env expression leaves the result mostly
unchanged, the artifact should be read as a protocol win, not a latent win.

## Run Training

From the repo root:

```bash
python3 main.py
```

`main.py` is the local live-dashboard comparison entrypoint. For the current
representation-repair pass, use the matched CartPole benchmark runner instead:

```bash
python3 scripts/run_cartpole_repr_repair.py
```

That run writes a fresh `cartpole_repr_repair_v2` snapshot and benchmark from
the same code revision, with three CartPole seeds, fair two-probe handoff,
particle sysid enabled, and the representation-repair gate active. By default
it stops after the latent snapshot if paired/cross split retrieval and local
geometry are still too weak for a fair PPO handoff.

The tiny public API example is still:

```python
from teenyreason import ppo, run

run("ContinuousCartPole-v0", ppo(), seeds=2, profile="fast")
```

Use `seeds=2` for local iteration. Save `5`-seed runs for confirmation once the
message path is actually live.

The default composition still runs the existing PPO benchmark harness, so it
will:

- collect probe data
- train the current latent encoder
- save checkpoint artifacts
- save a latent snapshot under `artifacts/*_latent_snapshot.npz`
- run the baseline/probe/probe-no-expression benchmark

## Run Playback

To render a saved probe-conditioned policy:

```bash
python3 play_probe_policy.py --checkpoint artifacts/continuous_cartpole_ppo_seed_0_probe_ppo_checkpoint.pt --episodes 3
```

Recent checkpoints can carry three policy snapshots:

- final policy
- best single-run snapshot
- solve-verified snapshot

Playback will prefer the solve-verified snapshot when available.

## Run The Local Dashboard

Start the localhost dashboard:

```bash
python3 serve_latent_dashboard.py
```

Then open:

```text
http://127.0.0.1:5050
```

The dashboard currently shows:

- saved latent-space snapshots
- a 2D PCA projection of latent means
- reward / uncertainty coloring
- probe-mode counts
- benchmark summary tables
- tri-domain CartPole, Shakespeare, and MNIST latent-expression diagnostics
- readiness blockers, muting, bitrate, ablation gaps, and claim gates

## Cleanup Ledger

Files split or added in the current cleanup pass:

- `pyproject.toml`
  Local editable-install metadata, dashboard package data, and console scripts.
- `teenyreason/public_suite.py`
  Small public `run_suite(...)` wrapper for the tri-domain suite.
- `teenyreason/viz/suite_payloads.py`
  Dashboard payload helpers for tri-domain suite artifacts.
- `teenyreason/multidomain/suite_cartpole.py`
  CartPole domain normalization for suite payloads.
- `teenyreason/multidomain/suite_acceptance.py`
  Shared conservative claim gates for CartPole, language, and image domains.
- `teenyreason/multidomain/image_models.py`
  Image solver model helpers separated from benchmark orchestration.
- `teenyreason/multidomain/language_models.py`
  Language solver model helpers separated from benchmark orchestration.
- `teenyreason/recipes/evidence.py`
  Shared evidence-window recipe helpers.

Deleted files:

- None. Deletion remains gated by `python3 scripts/audit_python_files.py` plus
  a follow-up reference check.

Compatibility paths intentionally retained:

- `teenyreason.models.belief_world_model`
- `teenyreason.models.env_belief`
- `teenyreason.crawler.probes.data`
- `teenyreason.crawler.probes.latent`

## Notes

- `artifacts/` contains generated outputs from training and evaluation.
- `.gitignore` now ignores future generated artifacts and bytecode files.
- The representation package exists to make the repo easier to reason about
  from the latent-belief angle without breaking the current training path.
