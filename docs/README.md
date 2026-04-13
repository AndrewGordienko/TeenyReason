# Docs Index

This folder is the operating manual for the repo.

If you are new to the project, read in this order:

1. [core_idea.md](./core_idea.md)
2. [research_manifesto.md](./research_manifesto.md)
3. [paper_synthesis.md](./paper_synthesis.md)
4. [architecture_and_training.md](./architecture_and_training.md)
5. [evaluation_and_diagnostics.md](./evaluation_and_diagnostics.md)
6. [agent_guide.md](./agent_guide.md)

Each file has one job.

- `core_idea.md`
  Short framing for the research question.
- `research_manifesto.md`
  The thesis, intended architecture, target losses, and what the project is
  explicitly not trying to do.
- `paper_synthesis.md`
  Paper-by-paper extraction of ideas and what each one means for this repo.
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

## The Short Version

We are trying to build:

- a crawler that actively experiments on a new environment
- a belief model that infers hidden mechanics from a small amount of evidence
- a controller that uses that belief to solve tasks much faster

The main object is the env-level belief, not the policy.
