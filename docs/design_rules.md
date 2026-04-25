# Design Rules

This repo should feel closer to PyTorch or tinygrad than to a framework stack.

## Public Taste

- prefer short public names
- prefer `run(...)` and `ppo()`
- keep internal compatibility names off the main user path

## Code Taste

- prefer direct data flow
- prefer small helpers over abstraction towers
- prefer explicit inputs and outputs over hidden config
- prefer one file doing one job

## Bloat Rules

- target roughly `300-500` lines per file when practical
- if a file grows much past that, split by behavior and ownership
- do not split just to create more layers

## Benchmark Boundary

- the crawler core should stay world-agnostic
- benchmark policy, PPO plumbing, dashboard shaping, and artifact formatting do not belong on the public crawler path
- compatibility glue can stay during migration, but it should be boring and easy to remove later
