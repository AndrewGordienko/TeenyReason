# Agent Guide

This file is for future coding agents working in this repo.

Its purpose is simple:

do not let implementation drift away from the actual research idea.

This repo is especially vulnerable to drift because small code changes can
improve:

- benchmark results
- dashboard plots
- uncertainty numbers
- PPO behavior

without improving the thing we actually care about:

the quality of the env-level belief over hidden mechanics.

If you are writing code here, read this file before making nontrivial changes.

## The One Job

Your job is not to "improve PPO."

Your job is to help build:

- a crawler that actively probes a new world
- a belief model that infers hidden mechanics from a small number of probes
- a controller that uses that belief to solve tasks quickly

If your change does not clearly help one of those three pieces, it is probably
not central.

## The Main Research Object

The main research object is `z_env`, the env-level belief.

Everything else is in service of that:

- crawler quality matters because it affects the belief
- window-level evidence matters because it feeds the belief
- controller quality matters because it tests whether the belief is useful
- dashboard quality matters because it helps inspect the belief

Do not confuse any of these helper systems with the belief itself.

## The Canonical Distinctions

Future changes must preserve these distinctions.

### Window-level evidence is not env belief

`z_window`:

- one probe window
- local evidence
- may encode motion style, phase, and local response

`z_env`:

- pooled across many windows from one hidden world
- the object that should represent mechanics
- the object the controller should consume

Never silently collapse those two concepts together.

### Uncertainty is not generic variance

Good uncertainty should mean:

- the model is still unsure about hidden mechanics
- different small evidence subsets produce meaningfully different beliefs
- decoded mechanics vary because the world is not yet pinned down

Bad uncertainty:

- arbitrary posterior variance with no calibration
- a scalar that looks nice in logs but does not track error

### Benchmark wins are not proof of belief quality

A controller can improve because of:

- regularization
- extra context
- lucky architectural bias
- better optimization

without the belief actually representing hidden mechanics well.

Always check representation metrics first.

## The Questions To Ask Before Coding

Before making any substantial change, answer these questions in your head:

1. Which failure mode am I addressing?
2. Does this change target the crawler, the belief, or the controller?
3. How would I know if the change improved the env belief itself?
4. What metric should move if the change worked for the right reason?
5. What shortcut or fake win could still produce an apparent improvement?

If you cannot answer those, stop and write the change down more clearly first.

## The Main Failure Modes

These are the most important failure modes in the repo.

### 1. Probe Leakage

The latent encodes which probe routine ran, not which world generated the
trajectory.

Typical symptoms:

- strong probe-mode separability
- weak mechanics fit
- neat PCA clusters by probe family

### 2. Trajectory Texture

The latent encodes local style or motion phase rather than hidden mechanics.

Typical symptoms:

- visually structured manifold
- weak same-env agreement
- poor env-parameter decode

### 3. Global Decode, Bad Local Geometry

The belief is linearly decodable but nearest neighbors do not match mechanics.

Typical symptoms:

- decent mechanics fit
- poor neighbor alignment
- downstream controller still improves

### 4. Collapsed Uncertainty

The uncertainty object becomes almost constant or near-zero and stops being
informative.

Typical symptoms:

- flat uncertainty
- poor calibration
- no meaningful error tracking

### 5. Controller Compensation

The controller benefits even though the belief is weak.

Typical symptoms:

- benchmark gains
- muddy representation metrics
- latent still not clearly mechanical

## What Types Of Changes Are Good

These are usually aligned with the project:

- making same-env beliefs more stable across probe subsets
- making nearby env beliefs correspond to nearby mechanics
- improving mechanics decode from env beliefs
- improving uncertainty calibration against real mechanics error
- making the crawler choose more informative interventions
- reducing probe leakage at the env-belief level
- making the controller use the env belief more directly and honestly

## What Types Of Changes Are Suspicious

These are not automatically bad, but they need strong justification:

- adding new benchmark metrics without a clear research decision they support
- adding more PPO complexity when the belief is weak
- changing dashboard visuals without improving artifact semantics
- optimizing reward-colored plots
- mixing window-level and env-level metrics in the same readout
- adding losses that do not map to a clear failure mode

## The Coding Rules For This Repo

When editing code, follow these design rules.

### Preserve explicit boundaries

Keep these boundaries explicit:

- data collection
- local evidence encoding
- env belief aggregation
- controller consumption
- diagnostics

Do not smear all of this into one giant abstraction.

### Prefer one obvious meaning per tensor

Each important tensor should have one conceptual role.

Examples:

- window evidence
- env belief
- subset disagreement
- decoded env params

Avoid generic "latent" variables where the conceptual meaning is unclear.

### Do not add metrics casually

Every metric should answer a real question:

- does the belief encode mechanics?
- is it stable?
- is uncertainty calibrated?
- does it help downstream control?

If a metric does not change a real research decision, it probably does not
belong.

### Keep dashboard semantics honest

Never let the dashboard imply something stronger than the artifact supports.

Examples:

- do not label a generic variance term as "uncertainty" if it is not
  calibrated
- do not present a window-level quantity as if it were an env-level quantity
- do not let mixed semantics hide representation problems

### Do not let PPO become the explanation for everything

If the controller improves, ask why.

Possible explanations:

- the env belief got better
- the controller got easier optimization
- extra context regularized the policy
- randomness helped

The first explanation is the one we want. Do not assume it automatically.

## The Writing Rules For This Repo

When updating docs or comments:

- define what each latent or uncertainty object means
- distinguish clearly between current behavior and intended behavior
- state what a metric does and does not imply
- avoid aspirational wording that overstates what the system currently does
- keep the research question visible

Good writing here should reduce ambiguity, not create momentum through hype.

## The Preferred Change Process

For substantial research changes, prefer this process:

1. identify the failure mode
2. write down the intended fix in one paragraph
3. name the exact metric that should move
4. implement the smallest coherent version
5. update docs if semantics changed
6. evaluate representation metrics before celebrating benchmark gains

This process matters because the repo is easy to "improve" for the wrong
reason.

## The Questions To Ask After A Run

After a run, answer these:

1. What did the belief get better at?
2. What did it still fail to capture?
3. Did uncertainty become more meaningful?
4. Did same-env agreement improve?
5. Did local geometry improve?
6. Did downstream control improve for the same reason?

If those questions do not have clear answers, the next step should probably be
better diagnostics before more algorithmic complexity.

## The Current Priorities

At the current stage of the repo, agents should prioritize:

1. env-belief stability from small probe subsets
2. mechanics-aligned local geometry
3. calibrated uncertainty
4. active crawler objectives
5. controller usage of the belief

These priorities are more important than:

- adding more environments
- tuning PPO in isolation
- polishing visuals
- inventing new aggregate scores

## The Final Standard

A successful change in this repo should move us closer to this statement:

After only a small number of active probes, the agent forms a stable,
mechanics-aware, uncertainty-calibrated env belief that a downstream
controller can use to solve the task much faster than if it had to infer the
world and the task at the same time.

If your change clearly serves that sentence, it probably belongs.
