# Docs Index

This folder is the operating manual for the repo.

If you are new to the project, read in this order:

1. [core_idea.md](./core_idea.md)
2. [research_manifesto.md](./research_manifesto.md)
3. [general_crawler_belief.md](./general_crawler_belief.md)
4. [general_program_plan.md](./general_program_plan.md)
5. [paper_synthesis.md](./paper_synthesis.md)
6. [language_domain_synthesis.md](./language_domain_synthesis.md)
7. [language_belief_design.md](./language_belief_design.md)
8. [architecture_and_training.md](./architecture_and_training.md)
9. [evaluation_and_diagnostics.md](./evaluation_and_diagnostics.md)
10. [agent_guide.md](./agent_guide.md)

Each file has one job.

- `core_idea.md`
  Short framing for the research question.
- `research_manifesto.md`
  The thesis, intended architecture, target losses, and what the project is
  explicitly not trying to do.
- `general_crawler_belief.md`
  The cross-domain version of the thesis: how the crawler-belief-solver pattern
  should apply not just to physical RL but to language and other environments.
- `general_program_plan.md`
  The operational companion to the cross-domain thesis: staged rollout,
  artifact expectations, metric families, and expansion rules.
- `paper_synthesis.md`
  Paper-by-paper extraction of ideas and what each one means for this repo.
- `language_domain_synthesis.md`
  The language-side paper bridge: what one-book or small-corpus learning can
  and cannot mean, grounded in the cited language papers.
- `language_belief_design.md`
  The concrete language-side design: latent factorization, crawler behavior,
  implementation phases, metrics, and module boundaries.
- `architecture_and_training.md`
  Concrete system design, module ownership, data flow, artifact flow, and
  training-stage responsibilities.
- `evaluation_and_diagnostics.md`
  Which metrics matter, what they mean, what can go wrong, and how to read the
  dashboard honestly.
- `agent_guide.md`
  The writing and coding guide for future agents so they push the same research
  agenda instead of chasing whichever metric moved last.

## Why This Exists

This repo has a genuine research idea behind it, but it is easy for day-to-day
implementation work to drift into:

- chasing benchmark wins
- over-reading PCA plots
- optimizing metrics that can be gamed
- letting the controller compensate for a weak latent
- mixing together local probe evidence and env-level belief

These docs are here to make the idea concrete enough that future edits can be
judged against the actual thesis, not just against the latest run.

The newer cross-domain docs also exist to keep the repo from accidentally
shrinking its own idea. The current codebase is mostly a physical-control
testbed, but the larger research question is about a general crawler that can
infer hidden rules in many kinds of environments.

## The Short Version

We are trying to build:

- a crawler that actively experiments on a new environment
- a belief model that infers hidden mechanics from a small amount of evidence
- a controller that uses that belief to solve tasks much faster

The main object is the env-level belief, not the policy.
