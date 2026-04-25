# Research Manifesto

This repo is about one problem:

How can an agent actively probe a new environment, infer a compact belief about
its hidden mechanics from only a small amount of interaction, and then solve a
downstream control task much faster because it already understands the world?

That is the north star.

For the full companion docs, see [README.md](./README.md).

This is not a project about making PPO look good with an extra feature vector.
It is not a project about getting a pretty PCA plot. It is not a project about
maximizing one benchmark score by any means necessary.

We are trying to build a small "scientist + controller" system:

1. A crawler that runs experiments.
2. A belief model that infers hidden mechanics.
3. A controller that uses that belief to solve tasks quickly.

## The Core Object

The main object in the repo is not the policy. It is the environment belief.

That belief should answer:

- What world am I in?
- What hidden mechanics seem to govern it?
- What can I already predict well?
- What am I still uncertain about?

In code terms, that means we want to cleanly separate:

- `z_window`
  Local evidence from one intervention window.
- `z_env`
  An env-level belief pooled across many windows from the same sampled world.
- `u_env`
  Uncertainty over the env-level belief, ideally tied to mechanics error and
  subset disagreement.

The controller should consume `state + z_env + u_env`, not raw probe history.

## What We Are Trying To Build

The desired pipeline is:

1. Sample a hidden environment instance.
2. Let the crawler run a small number of informative interventions.
3. Encode each intervention into a local window-level evidence representation.
4. Aggregate those windows into one env-level belief.
5. Use that belief for prediction and downstream control.
6. Optionally refine the belief online as more task interaction arrives.

In the current repo this is implemented as:

1. the crawler scores probe families by expected belief improvement
2. the predictive belief is trained to forecast held-out probe summaries
3. a separate metric projector owns retrieval and local geometry
4. an env-expression projector turns the predictive belief into the solver input
5. the benchmark runs in explicit `fair` and `adaptive` modes

The ideal behavior is:

- a few probes are enough
- the belief becomes stable quickly
- the belief reflects hidden mechanics, not probe identity
- the controller adapts fast because physics discovery is mostly done

## The Paper Synthesis

These papers do not all say the same thing, but together they point toward a
consistent design.

### Belief, not embedding

`PEARL` and `VariBAD` say the agent should maintain a posterior belief over
hidden task or dynamics variables. The control policy should act on that
belief, not on a plain history summary.

Repo consequence:

- we should think in terms of inference over hidden mechanics
- uncertainty matters
- the inference object should be reusable across tasks

### Contrastive sufficiency, not full reconstruction

`ContraBAR` says we do not need to reconstruct every detail of the world. The
representation only needs to preserve the information sufficient for
Bayes-adaptive behavior.

Repo consequence:

- we should not reward the latent for memorizing probe texture
- we should use multi-view and contrastive objectives around the hidden world

### World-model usefulness

`DreamerV3` and `TD-MPC2` are reminders that a latent becomes valuable when it
supports prediction and control-relevant counterfactual reasoning.

Repo consequence:

- belief quality should be judged partly by predictive usefulness
- the controller should treat the belief as first-class, not a small residual

### Minimal causal abstraction

`CBM` argues for minimal, reusable, causal abstractions instead of large,
surface-level state summaries.

Repo consequence:

- the best latent is not the richest latent
- we want a compact belief that preserves mechanics relevant for control

### Belief as communication channel

`TurboQuant` is a useful reminder that a latent is also a message passed from
one subsystem to another. In this repo, that means the crawler and belief stack
must send the solver a compact code that preserves what the solver actually
needs.

Repo consequence:

- we should think in rate-distortion terms, not only embedding terms
- `z_env` should eventually be evaluated under explicit bottlenecks
- a good belief should survive meaningful compression without losing its core
  mechanics information

### Active system identification

`ASID` pushes the crawler toward actions that improve identification, not just
coverage or random motion.

Repo consequence:

- the crawler is an experiment designer
- probe reward should be about information gain, disagreement reduction, or
  mechanics-belief improvement
- the unit of planning should be the probe family, not only the next
  low-level action
- each family should carry an explicit estimate of:
  - predicted mechanics uncertainty reduction
  - predicted held-out future-probe improvement
  - predicted split-half mismatch reduction

### History as prompt / context

`Dynamics as Prompts` supports using interaction history as context for
dynamics adaptation.

Repo consequence:

- we want online belief updates
- the belief should be able to sharpen during the first phase of task control

### Human-like priors and intuitive physics

`Lake et al.` and `PLATO` point toward causal structure, compositionality,
object-level reasoning, and strong priors as the reason humans can learn from
limited experience.

Repo consequence:

- the goal is not "less data with the same old objective"
- the goal is better structure, better questions, and better abstractions

## The Actual Research Thesis

The repo thesis should be:

Active probing can produce a stable, uncertainty-aware belief over hidden
environment mechanics, and that belief can materially reduce downstream task
learning cost by separating physics discovery from task optimization.

An equivalent way to say the same thing is:

the crawler should discover a small, information-dense message about the world
that the downstream solver can use directly.

That is the sentence all code paths should serve.

In the current fair CartPole path, that means the benchmark has to separate
three stories:

- a baseline controller that never sees the env belief
- a probe-conditioned controller that does see the env expression
- a matched probe-conditioned controller with the env expression muted

If probing wins while the muted arm keeps most of the gain, that is still
useful progress, but it is a protocol win rather than the intended child-like
world-belief win.

## The Intended Architecture

The system should have four explicit pieces.

### 1. Crawler

Job:

- interact before the task controller takes over
- choose informative interventions
- stop once the belief is sharp enough

Desired outputs:

- windows of `(s, a, r, s')`
- probe metadata
- env-instance grouping during training
- per-family expected information-gain estimates
- per-family realized uncertainty reduction
- a stop reason saying why the crawler handed off when it did

### 2. Window Evidence Encoder

Job:

- turn one probe window into local evidence about the environment

Desired properties:

- predictive of short-horizon consequences
- less sensitive to probe-script identity
- useful as a building block, but not itself the final env belief

### 3. Env Belief Aggregator

Job:

- combine many windows from one world into a stable belief `z_env`

Desired properties:

- same env + different probe modes -> nearby beliefs
- different envs -> separated beliefs
- small subsets of windows should already produce similar beliefs

### 4. Controller

Job:

- solve the downstream task using `state + belief`

Desired properties:

- benefit from the belief quickly
- not need to rediscover mechanics from scratch
- optionally refine the belief online during early task execution

## The Current Code Ownership

The repo should map onto that architecture cleanly.

- `teenyreason/probe/probe_data.py`
  Probe crawler and environment perturbation data collection.
- `teenyreason/models/belief_world_model.py`
  Window-level evidence encoder and training losses tied to prediction and
  supervised structure.
- `teenyreason/models/env_belief.py`
  Env-level belief aggregation, subset pooling, and uncertainty construction.
- `teenyreason/probe/probe_latent.py`
  Online belief helpers and active probe action selection.
- `teenyreason/rl/probe_policy/`
  Downstream control using the env-level belief.
- `teenyreason/representation/analysis.py`
  Artifact construction for env-belief analysis.
- `teenyreason/viz/dashboard.py`
  Diagnostics and dashboards for representation quality.

## The Losses We Actually Want

These are the loss families that fit the thesis.

### Window-level losses

- next-step or delta prediction
- short-horizon predictive consistency
- task-relevant auxiliary targets
- contrastive future-summary prediction

Purpose:

- make local evidence useful and predictive

### Env-belief losses

- same-env multi-view consistency
- subset consistency
- within-vs-between env metric loss
- env-parameter decoding loss
- geometry loss based on env-parameter distance
- probe-mode invariance pressure

Purpose:

- make `z_env` stable, mechanical, and reusable

### Communication losses or constraints

- explicit belief bottlenecks
- compressed-belief consistency
- coarse-plus-residual belief coding
- distortion penalties targeted at mechanics decode and control usefulness

Purpose:

- make `z_env` small enough to test sufficiency
- treat the belief as a message, not an unconstrained feature blob

### Uncertainty losses

- subset disagreement calibration
- decoder disagreement calibration
- belief error calibration

Purpose:

- make uncertainty mean "I do not know this world well yet"

### Crawler objectives

- posterior shrinkage
- disagreement reduction
- env-belief improvement
- counterfactual identification value

Purpose:

- choose better experiments, not prettier trajectories

## The Metrics That Matter

The main metrics should reflect the thesis, not convenience.

### Representation metrics

- mechanics fit from `z_env`
- per-parameter `R²`
- same-env subset spread
- nearest-neighbor env alignment
- probe leakage
- uncertainty calibration against actual mechanics error
- few-probe sufficiency: how many windows are needed before belief stabilizes
- belief rate-distortion: how much performance survives when `z_env` is
  aggressively compressed

### Downstream metrics

- solve episodes
- solve environment steps
- success rate across seeds
- performance as a function of probe budget
- performance as a function of belief bitrate

### Combined metric

The best score is not "highest return".

The best score is something like:

- few probes
- stable belief
- strong mechanics decode
- low required belief bitrate
- fast downstream solve

## What We Are Not Doing

We are not trying to:

- maximize PCA coverage
- maximize latent cluster prettiness
- make probe mode easy to decode
- win one benchmark by leaking shortcuts into the latent
- let PPO compensate for a weak belief and then call the belief good
- treat a lucky high-return seed as proof of representation quality
- optimize dashboards instead of the actual underlying belief

If a metric improves but the latent becomes more probe-shaped, less stable, or
less mechanical, that is a loss, not a win.

## The Practical Design Rule

Before adding a new loss, metric, or dashboard view, ask:

Does this help us answer whether the agent has learned the hidden mechanics of
the environment from a small number of interventions, in a way that a
downstream controller can use quickly?

If the answer is no, it is likely side noise.

## Current Working Direction

The repo is currently closest to this design:

- window evidence encoder
- env-level belief aggregator
- subset-based consistency and uncertainty
- modified PPO consuming env belief

That is the right direction.

The next improvements should mostly tighten:

- local geometry of env beliefs
- few-probe sufficiency
- uncertainty calibration
- active crawler objectives
- belief communication efficiency

Not add more unrelated benchmark machinery.

## Companion Docs

This manifesto is the thesis file. The companion docs are:

- [paper_synthesis.md](./paper_synthesis.md)
  Which ideas the papers contribute and how they map into code.
- [architecture_and_training.md](./architecture_and_training.md)
  What the intended machine looks like and which module owns what.
- [evaluation_and_diagnostics.md](./evaluation_and_diagnostics.md)
  Which metrics actually matter and how to interpret them honestly.
- [agent_guide.md](./agent_guide.md)
  How future agents should work in the repo without drifting away from the
  research question.
