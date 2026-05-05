# Docs Index

This folder is the operating manual for the repo.

If you are new to the project, read in this order:

1. [core_idea.md](./core_idea.md)
2. [general_crawler_belief.md](./general_crawler_belief.md)
3. [general_program_plan.md](./general_program_plan.md)
4. [value_aware_crawler_strategy.md](./value_aware_crawler_strategy.md)
5. [architecture_and_training.md](./architecture_and_training.md)
6. [crawler_library_api.md](./crawler_library_api.md)
7. [benchmark_harness.md](./benchmark_harness.md)
8. [design_rules.md](./design_rules.md)
9. [evaluation_and_diagnostics.md](./evaluation_and_diagnostics.md)
10. [paper_synthesis.md](./paper_synthesis.md)
11. [language_domain_synthesis.md](./language_domain_synthesis.md)
12. [language_belief_design.md](./language_belief_design.md)
13. [image_domain_synthesis.md](./image_domain_synthesis.md)
14. [image_belief_design.md](./image_belief_design.md)
15. [human_learning_synthesis.md](./human_learning_synthesis.md)
16. [human_learning_design.md](./human_learning_design.md)
17. [research_manifesto.md](./research_manifesto.md)
18. [agent_guide.md](./agent_guide.md)
19. [research/world_model_handoff_review.md](./research/world_model_handoff_review.md)

Each file has one job.

- `core_idea.md`
  Short framing for the research question.
- `general_crawler_belief.md`
  The cross-domain version of the thesis: how the crawler-belief-solver pattern
  should apply not just to physical RL but to language and other environments.
- `general_program_plan.md`
  The staged rollout and operating rules for turning the current code into a
  real crawler library.
- `value_aware_crawler_strategy.md`
  The current research bet: active causal experiments should create compact
  beliefs whose solver value exceeds their sample cost.
- `architecture_and_training.md`
  The intended library architecture, the current migration seams, and what
  still belongs to the RL benchmark compatibility path.
- `crawler_library_api.md`
  The public crawler contract: generic runtime types, interfaces, recipes,
  consumers, and compatibility adapters.
- `benchmark_harness.md`
  The app-layer benchmark runner, artifact flow, dashboard linkage, and what
  still belongs to the RL harness instead of the crawler core.
- `evaluation_and_diagnostics.md`
  Which metrics matter, what they mean, and how to read the dashboard without
  over-claiming what the latent has learned.
- `design_rules.md`
  The code-taste rules for keeping the public API small and the codebase
  readable.
- `paper_synthesis.md`
  Paper-by-paper extraction of ideas and what each one means for this repo.
- `language_domain_synthesis.md`
  The language-side paper bridge.
- `language_belief_design.md`
  The concrete language-side design.
- `image_domain_synthesis.md`
  The image-side paper bridge.
- `image_belief_design.md`
  The concrete image-side design.
- `human_learning_synthesis.md`
  The developmental-science and neuroscience bridge.
- `human_learning_design.md`
  The operational companion to the human-learning bridge.
- `research_manifesto.md`
  The wider thesis and the guardrails for what the project is not trying to
  become.
- `agent_guide.md`
  The writing and coding guide for future agents.
- `research/world_model_handoff_review.md`
  Current world-model construction and solver-handoff review, including
  V-JEPA 2 and the repo's latest sample-efficiency bottlenecks.

## The Four Reading Buckets

- Thesis:
  `core_idea.md`, `general_crawler_belief.md`, `general_program_plan.md`,
  `value_aware_crawler_strategy.md`
- Public API:
  `architecture_and_training.md`, `crawler_library_api.md`, `design_rules.md`
- Benchmark Harness:
  `benchmark_harness.md`, `evaluation_and_diagnostics.md`
- Research Background:
  `paper_synthesis.md`, `language_domain_synthesis.md`,
  `image_domain_synthesis.md`, `human_learning_synthesis.md`,
  `research/world_model_handoff_review.md`

## Why This Exists

This repo has a genuine research idea behind it, but it is easy for day-to-day
implementation work to drift into:

- chasing benchmark wins
- over-reading PCA plots
- optimizing metrics that can be gamed
- letting the controller compensate for a weak latent
- mixing together local evidence, env-level belief, and benchmark-specific glue

These docs exist to keep the thesis, library boundary, and benchmark harness
distinct from one another.

The local paper library that backs these docs lives in
[`papers/README.md`](../papers/README.md).

## The Short Version

We are trying to build:

- a crawler that actively experiments on a new environment
- a belief model that infers hidden mechanics from a small amount of evidence
- a solver that consumes a compact belief message instead of rebuilding its own
  world model from scratch

The main object is the env-level belief, not the policy.
