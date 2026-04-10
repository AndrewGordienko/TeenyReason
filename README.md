# TeenyReason

Small experiments around probe-conditioned reinforcement learning.

The rough idea is:

- run short scripted probe episodes in an environment
- train an encoder to turn those probe windows into a compact latent "environment fingerprint"
- train a control policy that gets both the normal state and that latent belief
- compare that against a plain PPO baseline

Right now the main pipeline is set up around `BipedalWalker` and benchmarks:

- a baseline PPO run
- a probe-conditioned PPO run
- multiple seeds

## Files

- `main.py`: benchmark entrypoint
- `probe_data.py`: probe crawler and scripted probe policies
- `world_model.py`: latent encoder and prediction targets
- `probe_latent.py`: latent aggregation, belief updates, novelty, uncertainty
- `probe_ppo.py`: baseline PPO and probe-conditioned PPO training loops
- `ppo_core.py`: shared PPO models and update code

## How it works

1. Collect short probe trajectories with fixed action patterns.
2. Build windowed training data from those trajectories.
3. Train an encoder that maps probe windows to a latent vector.
4. Use one or more probe latents to build a belief vector.
5. Train PPO with that belief as extra conditioning input.
6. Compare solve speed and returns against plain PPO.

## Run

From the repo root:

```bash
python3 main.py
```

The script will:

- collect probe data
- train the encoder
- train baseline PPO
- train probe-conditioned PPO
- save returns and benchmark artifacts under `artifacts/`

## Notes

- The code currently uses local hardcoded experiment settings in `main.py`.
- Recent logs include the benchmark run number, whether the trainer is `baseline` or `probe`, and a short solve-status summary under each episode line.
- Generated artifacts and `__pycache__` are currently present in the repo.

