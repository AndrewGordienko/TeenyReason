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

Read [docs/core_idea.md](docs/core_idea.md) first.

The latent in this repo is intended to be:

- predictive
- task-relevant
- uncertainty-aware
- reusable across downstream control

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
