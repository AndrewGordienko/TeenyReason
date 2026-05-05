# Core Idea

For the concrete research thesis, architecture, losses, and success criteria,
read [research_manifesto.md](./research_manifesto.md) after this page.

This repo is about one question:

How can an agent build a compact, reusable, uncertainty-aware belief about an
environment's hidden rules from a small amount of interaction, and then use
that belief to learn or control much faster later on?

That is a stronger goal than "add a probe vector to PPO."

The latent should not just be a compressed history. It should act like a belief
over hidden causal structure.

The repo is also explicitly guided by three external sample-efficiency stories:

- children building fast world beliefs from play and surprise
- language learners extracting grammar and meaning from small but structured
  evidence
- visual learners building object- and concept-level structure from limited
  examples

Those bridges are written out in:

- [language_domain_synthesis.md](./language_domain_synthesis.md)
- [image_domain_synthesis.md](./image_domain_synthesis.md)
- [human_learning_synthesis.md](./human_learning_synthesis.md)

## What The Latent Should Mean

A good latent in this repo should be:

- Predictive: it should help forecast what happens after actions.
- Causal-ish: it should align with hidden mechanics like gravity, friction,
  delay, contacts, or actuation limits rather than only surface correlations.
- Task-relevant: it should keep what matters for control and throw away what
  does not.
- Uncertainty-aware: it should represent what the agent still does not know.
- Reusable: it should transfer across downstream tasks in the same family of
  environments.

In practice that can mean two closely related env-level representations:

- a predictive belief that carries hidden mechanics and held-out probe
  forecasting
- a metric projection that is optimized for retrieval, neighborhood structure,
  and geometry diagnostics

The key implementation detail is that the predictive belief should behave more
like a posterior update than a pooled summary. Each probe family should update
the current world hypothesis in a family-conditioned way, carrying forward both
factor means and factor uncertainty.

That posterior should drive probe choice directly. The crawler should rank
probe families by how much they reduce posterior entropy and how strongly they
separate the remaining plausible world hypotheses, not only by latent-space
disagreement.

The downstream solver should not consume the raw predictive belief directly by
default. The current contract is:

- `PredictiveBelief`
  the canonical world model object used for mechanics decode and held-out probe
  prediction
- `MetricBelief`
  the auxiliary object used for retrieval and geometry diagnostics
- `UncertaintyEstimate`
  the operational uncertainty object that decides whether another probe is
  worth paying for
- `EnvExpression`
  the compact solver-facing handoff, optionally compressed, that becomes the
  actual controller input alongside one confidence scalar and one uncertainty
  scalar

In the current fair CartPole path, that handoff is frozen for one control
episode at a time. The crawler probes first, emits one compact env
expression, and PPO then spends the episode turning that world guess into
behavior instead of trying to re-infer the mechanics from scratch inside the
same rollout.

## Working Hypothesis

The agent should have two phases:

1. Explore and identify the environment.
2. Control using the inferred belief.

The current probe-conditioned PPO experiments are only one early testbed for
that larger idea.

## Research Direction

The repo is moving toward a "latent environment belief lab" with these pieces:

- Active probing:
  choose experiments that reduce uncertainty instead of replaying fixed scripts
- Structured posterior:
  represent mean belief plus uncertainty, not only one point embedding
- Internal factor support only:
  if factorization is still useful for training or uncertainty estimation, keep
  it behind the env-expression interface instead of making it the main public
  story
- Task-relevant supervision:
  predict not just next state, but controllability, recoverability, risk, and
  return-relevant summaries
- Belief evaluation:
  inspect calibration, identifiability, transfer value, and nearest-neighbor
  semantics instead of only raw PPO score
- Predictive reuse:
  require the env belief to predict what a held-out future probe would reveal,
  not only decode hidden parameters after the fact
- Library boundary:
  treat the crawler and belief builder as a reusable library whose outputs can
  feed RL, language, or vision downstream learners instead of living inside one
  PPO-specific training loop
- Fair downstream comparison:
  benchmark the belief against a matched downstream learner with the same
  actor-critic capacity and rollout/update schedule, then keep extra adaptive
  controller tricks in a separate full-system mode
- Probe-budget honesty:
  keep fair mode intentionally conservative: hand off early when the env
  expression is ready, otherwise allow at most two probes before control
- Visualization:
  make the latent space and belief trajectories visible during research

## Current Repo Layout

- `teenyreason/representation/`
  The stable entrypoint for representation learning, latent analysis, and saved
  latent-space artifacts.
- `teenyreason/crawler/`
  Library-facing bundle, public crawler data types, predictive-target helpers,
  env-expression construction, and family-level probe scoring.
- `teenyreason/crawler/probes/`
  Probe collection and online belief update utilities.
- `teenyreason/rl/`
  Downstream RL code that consumes the latent belief.
- `teenyreason/viz/`
  Local dashboard code for browsing latent snapshots and benchmark summaries.

## Short-Term Questions

- Does the latent cluster environments by true hidden mechanics?
- Does uncertainty shrink when the probe policy gathers informative evidence?
- Does downstream control improve in environment steps, not just episodes?
- Does the solver-facing message still preserve mechanics and prediction when
  we compress it to a smaller bitrate?
- What parts of the latent help prediction vs. control vs. uncertainty?
- When does a fixed-history state encoder already solve the task, making the
  latent unnecessary?

## What Counts As Success

Success is not one lucky high-return run.

Success is when:

- latent snapshots show meaningful structure
- belief uncertainty is calibrated
- similar latents correspond to similar mechanics and affordances
- downstream control solves faster in total interaction cost
- the learned belief generalizes beyond one benchmark environment

For the current CartPole benchmark, that downstream win has to be interpreted
carefully:

- `latent_win`
  the probe + env-expression arm beats both baseline PPO and the matched probe
  + no-expression arm
- `protocol_win`
  probing helps, but muting the env expression leaves most of the gain behind
- `controller_compensation`
  the controller can still stabilize or regularize itself, but the env
  expression is not yet carrying the win

The child-like target in the docs is specifically the first case: a tiny,
diverse support set produces a compact world belief that normal PPO actually
uses to zoom.
