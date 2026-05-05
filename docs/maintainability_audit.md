# Maintainability Audit

This repo is small enough that bugs should usually be findable by following one
behavioral path at a time: app wiring, benchmark reporting, crawler/probe data,
belief models, PPO controllers, and visualization payloads. The main thing that
hurts that workflow right now is not dead code; it is large files that mix
several responsibilities.

## Current Inventory

`scripts/audit_python_files.py` reports no orphan candidates. The repo still has
large active modules, so the next cleanup should be split-by-ownership rather
than deletion.

Largest active files after the PPO core split:

| File | Lines | Main Responsibility | Suggested Split |
| --- | ---: | --- | --- |
| `teenyreason/rl/probe_policy/_train_impl.py` | 2719 | Plain/probe PPO loops, rollout control, metrics | `train_plain.py`, `train_probe.py`, `probe_rollout_loop.py`, `probe_eval_hooks.py` |
| `teenyreason/rl/full_system/_affordance_train_impl.py` | 2687 | Belief-controller training and eval attribution | `affordance_train_loop.py`, `affordance_eval.py`, `affordance_metrics.py` |
| `teenyreason/viz/payloads.py` | 2103 | Snapshot/benchmark/live payload assembly | `snapshot_payload.py`, `benchmark_payload.py`, `live_payload.py`, `series_payload.py` |
| `teenyreason/rl/probe_policy/_messages_impl.py` | 1449 | Env-expression/message construction and gates | `message_builders.py`, `message_readiness.py`, `message_ablation.py` |
| `teenyreason/rl/full_system/_planner_train_impl.py` | 1414 | Planner training, replay, checkpoint selection | `planner_train_loop.py`, `planner_eval.py`, `planner_replay.py` |
| `teenyreason/app/benchmark/runner.py` | 1814 | CLI benchmark orchestration | Keep under `app/benchmark/`; split run phases only when behavior separates cleanly |
| `teenyreason/crawler/library.py` | 1192 | Crawler library and probe family bookkeeping | `library_store.py`, `library_sampling.py`, `library_scores.py` |
| `teenyreason/models/belief/belief_training_env.py` | 1046 | Belief training env construction | `env_dataset.py`, `env_support.py`, `env_metrics.py` |
| `teenyreason/app/benchmark/reporting.py` | 1209 | Console/UI benchmark summaries | `summary_rows.py`, `summary_gates.py`, `summary_text.py` |
| `teenyreason/app/benchmark/artifacts.py` | 1088 | Artifact loading, decoding, and dashboard shaping | `artifact_loader.py`, `snapshot_decode.py`, `benchmark_decode.py` |
| `teenyreason/app/ppo/comparison.py` | 533 | Standard PPO vs probe-PPO comparison runner | Split only if comparison modes grow |

## Completed In This Pass

The old monolithic RL core was split into concrete modules:

| New Module | Job |
| --- | --- |
| `teenyreason/rl/core/types.py` | Batch data contract |
| `teenyreason/rl/core/numerics.py` | Sanitization, action squashing, GAE |
| `teenyreason/rl/core/normalization.py` | Running mean/variance normalizer |
| `teenyreason/rl/core/models.py` | PPO actor-critic models |
| `teenyreason/rl/core/batches.py` | Rollout packing and recurrent minibatches |
| `teenyreason/rl/core/optim.py` | PPO optimizer update |

Import sites now use `teenyreason.rl.core` or one of these concrete modules. The
old `teenyreason.rl.core.ppo_core` module was removed instead of being kept as a
blank compatibility shim.

The app benchmark harness was also moved out of the flat `teenyreason/app/`
directory into `teenyreason/app/benchmark/`. The public import
`teenyreason.app.benchmark` is now a small package facade over runner,
reporting, support, health, diagnostics, and artifact helpers.

The cross-domain suite no longer lives in `teenyreason/app/`. It is now under
`teenyreason/multidomain/suite/`, split into config, runner, JSON helpers,
domain payloads, metric attachment, and final dashboard payload assembly. Live
dashboard trace state moved to `teenyreason/viz/live/`, and PPO experiment
entrypoints moved to `teenyreason/app/ppo/`.

## Performance Notes Found During Audit

- PPO update repeatedly rebuilt action-bound tensors inside every minibatch.
  `update_ppo_policy` now caches scale/bias once per optimizer call.
- Controller-context corruption used per-sequence random `.item()` calls inside
  the recurrent PPO minibatch path. It is now vectorized to avoid accidental GPU
  synchronization and reduce Python loop overhead.
- The next large likely win is `viz/payloads.py`: it assembles many independent
  views from the same `.npz` artifacts. Splitting it should make repeated decode
  and normalization work easier to spot and cache.
- The probe/full-system training files have multiple evaluation and attribution
  passes interleaved with the training loop. Splitting those paths should make it
  clearer where expensive matched evals, shadow evals, and artifact writes are
  triggered.

## Next Cleanup Order

1. Split `viz/payloads.py` by payload type first. It has low training-risk and a
   high debugging payoff for the UI.
2. Split `app/benchmark/artifacts.py` and `app/benchmark/reporting.py` after
   payloads so artifact decode and presentation stay aligned.
3. Split `rl/probe_policy/_messages_impl.py` around message construction,
   readiness gates, and matched ablations.
4. Split the training loops last. They have the highest regression risk, so the
   smaller supporting modules should be stable before moving them.
