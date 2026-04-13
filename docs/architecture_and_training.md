# Architecture And Training

This file is the concrete system design for the repo.

It answers:

- what each subsystem is supposed to do
- how data should move through the system
- what each module owns
- what gets trained, when, and why
- which abstractions are canonical and which are only helper artifacts

If `research_manifesto.md` explains the thesis, this file explains the
intended machine.

## Canonical High-Level Architecture

The intended architecture is:

1. sample a hidden environment instance
2. run the crawler to gather informative interventions
3. encode each intervention window into local evidence
4. aggregate those windows into an env-level belief
5. decode mechanics and predictive targets from that belief
6. condition the downstream controller on the belief
7. optionally refine the belief online during task execution

This should be understood as a pipeline, not as a loose collection of helpers.

## The Four Main Objects

There are four canonical object types in the project.

### 1. Hidden Environment Instance

Meaning:

- one sampled world with fixed hidden mechanics
- examples: different gravity, pole length, mass, or actuation constants

Why it matters:

- this is the thing the agent is trying to infer
- every env-level belief should correspond to one hidden environment instance

Canonical identifiers:

- `env_instance_id`
- `env_params`

### 2. Probe Window

Meaning:

- one intervention sequence of short horizon
- one slice of evidence collected from a single environment instance

Why it matters:

- this is the unit of local evidence
- it is not yet the env belief

Canonical fields:

- `states`
- `actions`
- `rewards`
- `terminated`
- `truncated`
- `probe_mode`
- `env_instance_id`

### 3. Window Evidence Latent

Meaning:

- the local latent representation extracted from one probe window

Why it matters:

- useful for local prediction
- useful as an ingredient for the env-level belief
- should not be treated as the final environment representation

Canonical name:

- `z_window`

### 4. Env-Level Belief

Meaning:

- the pooled belief over hidden mechanics for one environment instance

Why it matters:

- this is the main research object
- this is what the controller should consume
- this is what the dashboard should primarily visualize

Canonical names:

- `z_env`
- `u_env`

## Current Module Ownership

The current repo should be mentally organized like this.

### `teenyreason/probe/probe_data.py`

Owns:

- environment randomization
- probe crawler execution
- collection of transitions and windows
- probe mode metadata

Should not own:

- env-level belief aggregation
- downstream control logic
- dashboard interpretation logic

### `teenyreason/models/belief_world_model.py`

Owns:

- window-level evidence encoder
- local predictive heads
- local contrastive and supervised losses
- training of the evidence encoder

Should not own:

- dashboard presentation
- benchmark orchestration
- broad repo-level evaluation policy

### `teenyreason/models/env_belief.py`

Owns:

- env-level aggregation from many windows
- subset pooling
- env-level uncertainty construction
- env-level predictor ensembles

This file is the clearest boundary between:

- local probe evidence
- actual environment belief

### `teenyreason/probe/probe_latent.py`

Owns:

- belief helper functions
- active probe scoring
- online belief updates
- controller-facing belief construction

Should not own:

- long training loops
- benchmark summary policy

### `teenyreason/rl/probe_ppo.py`

Owns:

- downstream task training
- how the controller consumes the belief
- solve checks and benchmark-side training behavior

Should not redefine:

- what the env belief means
- which representation metrics count as success

### `teenyreason/representation/analysis.py`

Owns:

- conversion from saved training artifacts into analysis artifacts
- latent snapshot construction

Should be representation-first:

- save the fields needed to inspect env beliefs honestly
- avoid collapsing env-level and window-level quantities into one number

### `teenyreason/viz/dashboard.py`

Owns:

- reading artifacts
- metric computation for dashboard interpretation
- JSON payload building

Should not quietly redefine the thesis.

If the dashboard adds a metric, that metric should correspond to a real
research question and have an interpretation written down elsewhere.

## Intended Data Flow

The intended data flow is:

1. `ProbeCrawler.collect(...)`
   gathers windows for many env instances
2. `build_training_tensors(...)`
   turns them into model-ready tensors
3. `WorldEncoder`
   maps each probe window to local evidence
4. `EnvBeliefAggregator`
   pools many windows from the same env into `z_env`
5. `EnvParamPredictorEnsemble`
   decodes hidden mechanics and disagreement
6. `train_probe_conditioned_ppo(...)`
   uses `z_env + u_env` during downstream control
7. `build_latent_snapshot(...)`
   saves env-level and window-level diagnostics for analysis
8. `dashboard.py`
   visualizes those artifacts

That flow should stay easy to explain in one whiteboard sketch.

## The Training Stages

The intended training stages are:

### Stage 1. Generate Probe Data

Input:

- environment family
- probe configuration
- randomization range

Output:

- probe windows grouped by env instance

What can go wrong:

- too few env instances
- too many windows from each env but too little env diversity
- probe modes that dominate the data distribution
- windows that mostly encode probe style instead of mechanics

### Stage 2. Train Window Evidence Encoder

Input:

- probe windows

Output:

- local window evidence representation

Objective:

- capture short-horizon consequences and task-relevant structure from each
  intervention

What can go wrong:

- local latent becomes probe-mode classifier
- local latent becomes motion-phase summary
- predictive losses encourage superficial rollout texture instead of mechanics

### Stage 3. Train Env Belief Aggregation

Input:

- window-level evidence grouped by env instance

Output:

- one env-level belief per hidden world

Objective:

- different probes from the same world should converge to one stable belief
- different worlds should separate
- belief should decode mechanics
- small subsets of probes should already recover a similar belief

What can go wrong:

- pooled belief just averages incompatible local encodings
- same-env subset beliefs disagree wildly
- geometry is decodable globally but incoherent locally
- uncertainty collapses to near-zero because the subset construction is bad

### Stage 4. Train Controller

Input:

- current state
- env-level belief
- uncertainty

Output:

- task policy

Objective:

- solve the downstream task quickly because inference is already mostly done

What can go wrong:

- controller learns to ignore the belief
- belief acts only as weak regularizer
- controller succeeds despite poor local belief geometry

### Stage 5. Analyze Artifacts

Input:

- latent snapshot
- benchmark summary

Output:

- diagnostic read on representation quality

Objective:

- determine whether the latent really encodes hidden mechanics
- determine whether it is stable and uncertainty-aware

What can go wrong:

- reading too much into PCA
- mixing window-level and env-level quantities
- trusting a benchmark win more than the representation evidence

## The Canonical Interfaces

These interfaces should remain conceptually stable even if the code evolves.

### Probe Window Interface

Minimum contents:

- states
- actions
- rewards
- env_instance_id
- probe_mode
- terminated/truncated flags

### Window Evidence Interface

Minimum contents:

- posterior mean
- posterior uncertainty
- enough local predictive content to support aggregation

### Env Belief Interface

Minimum contents:

- env belief mean
- env uncertainty
- decoded mechanics
- same-env subset disagreement

This interface is what the controller and dashboard should care about.

### Controller Interface

Minimum contents:

- current task state
- env belief mean
- env uncertainty

The controller should not need raw probe trajectories.

## The Intended Meaning Of Uncertainty

Uncertainty is one of the easiest places to drift.

We should treat uncertainty as:

- uncertainty about hidden mechanics
- uncertainty about what belief would be inferred from a small new subset of
  probes
- uncertainty that should correlate with mechanics prediction error

We should not treat uncertainty as:

- just posterior variance because the model emitted one
- generic noise magnitude
- a dashboard ornament with unclear semantics

The best uncertainty object in this repo should answer:

If I stopped probing now, how wrong might my world-model be?

## The Intended Meaning Of Geometry

Latent geometry matters, but only in the right sense.

Good geometry means:

- same env subsets are near each other
- nearby env beliefs correspond to nearby mechanics
- distances have operational meaning for transfer and retrieval

Bad geometry can still look nice:

- very high PCA coverage from a collapsed manifold
- a neat cloud whose nearest neighbors do not match mechanics
- clusters driven by probe mode or termination style

So geometry should be treated as a semantic property, not just a visual one.

## Target Failure Mode Checklist

Whenever a new run looks strong, check whether it is actually failing in one of
these ways:

- probe identity leakage
- phase-of-trajectory encoding
- global decode but poor local geometry
- belief collapse with fake low uncertainty
- controller compensating for a weak belief
- benchmark win caused by regularization rather than environment inference

If any of those are happening, the latent story is weaker than the controller
story.

## The Near-Term Design Priorities

The current near-term priorities should be:

1. few-probe belief stability
2. same-env subset consistency
3. neighbor geometry aligned with mechanics
4. uncertainty calibrated to mechanics error
5. active crawler objectives tied to belief improvement

These priorities are more important than adding new benchmark environments or
new downstream policy tricks.

## The One-Sentence Design Rule

Before changing the code, ask:

Does this make the env-level belief more accurate, more stable, more
mechanical, or more useful to a downstream controller after only a small amount
of probing?

If not, it is probably not central to the project.
