# Core Idea

This repo is about one question:

How can an agent build a compact, reusable, uncertainty-aware belief about an
environment's hidden rules from a small amount of interaction, and then use
that belief to learn or control much faster later on?

That is a stronger goal than "add a probe vector to PPO."

The latent should not just be a compressed history. It should act like a belief
over hidden causal structure.

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
- Task-relevant supervision:
  predict not just next state, but controllability, recoverability, risk, and
  return-relevant summaries
- Belief evaluation:
  inspect calibration, identifiability, transfer value, and nearest-neighbor
  semantics instead of only raw PPO score
- Visualization:
  make the latent space and belief trajectories visible during research

## Current Repo Layout

- `teenyreason/representation/`
  The stable entrypoint for representation learning, latent analysis, and saved
  latent-space artifacts.
- `teenyreason/probe/`
  Probe collection and online belief update utilities.
- `teenyreason/rl/`
  Downstream RL code that consumes the latent belief.
- `teenyreason/viz/`
  Local dashboard code for browsing latent snapshots and benchmark summaries.

## Short-Term Questions

- Does the latent cluster environments by true hidden mechanics?
- Does uncertainty shrink when the probe policy gathers informative evidence?
- Does downstream control improve in environment steps, not just episodes?
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
