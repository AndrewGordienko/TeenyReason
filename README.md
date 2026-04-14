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

## Repo Layout

- `main.py`
  Root benchmark entrypoint. Pick the environment and seeds here.
- `serve_latent_dashboard.py`
  Root wrapper for the local Flask dashboard.
- `teenyreason/representation/`
  Stable entrypoint for the current latent-belief system and latent snapshot
  analysis helpers.
- `teenyreason/probe/`
  Probe data collection, active probing, belief aggregation, and online belief
  updates.
- `teenyreason/rl/`
  PPO code that consumes the learned belief.
- `teenyreason/viz/`
  Local dashboard for latent snapshots and benchmark summaries.
- `teenyreason/models/belief_world_model.py`
  Current recurrent posterior encoder and structured latent supervision.
- `teenyreason/models/world_model.py`
  Older simpler prototype kept around as a reference point.
- `papers/`
  Local paper library for the repo. See [papers/README.md](papers/README.md)
  for the categorized reading map.

## Current Flow

1. Collect probe trajectories across a family of environments.
2. Turn those into fixed-length windows.
3. Train a latent encoder and structured prediction heads.
4. Save a latent snapshot artifact for visualization.
5. Train a baseline PPO agent and a probe-conditioned PPO agent.
6. Compare not just return, but solve speed and environment-step cost.

## Run Training

From the repo root:

```bash
python3 main.py
```

That will:

- collect probe data
- train the current latent encoder
- save checkpoint artifacts
- save a latent snapshot under `artifacts/*_latent_snapshot.npz`
- run the baseline/probe benchmark

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

## Notes

- `artifacts/` contains generated outputs from training and evaluation.
- `.gitignore` now ignores future generated artifacts and bytecode files.
- The representation package exists to make the repo easier to reason about
  from the latent-belief angle without breaking the current training path.
