# Evaluation And Diagnostics

This file defines how to evaluate the project honestly.

It exists because this repo is unusually easy to fool:

- the controller can improve while the latent stays weak
- the latent can produce attractive plots while encoding the wrong thing
- uncertainty can collapse and still look "confident"
- one benchmark can reward regularization more than true environment inference

The purpose of evaluation is to answer:

Did the agent actually learn a useful belief about hidden mechanics from a
small amount of probing?

## The Evaluation Hierarchy

Not all metrics are equally important.

The correct order is:

1. representation meaning
2. representation stability
3. uncertainty calibration
4. downstream adaptation
5. cosmetic geometry or dashboard appeal

If a lower-priority metric improves while a higher-priority one gets worse, the
run should usually be treated as regression.

## The Main Questions

Every run should be interpreted through these questions.

### 1. Does the belief encode hidden mechanics?

This is the most important question.

Relevant metrics:

- env-level mechanics fit
- per-parameter `R²`
- env-parameter prediction error

Interpretation:

- if these are weak, then the belief is not really learning the world

### 2. Is the belief stable across different probes of the same world?

Relevant metrics:

- same-env subset spread
- subset-to-subset agreement
- within-vs-between distance ratio

Interpretation:

- if these are weak, then the belief is not a world model, it is a view model

### 3. Does local geometry mean anything?

Relevant metrics:

- neighbor alignment
- retrieval quality across env beliefs
- hard-negative separation

Interpretation:

- if nearby beliefs do not correspond to nearby mechanics, the space is not
  semantically well organized

### 4. Is uncertainty useful?

Relevant metrics:

- uncertainty vs mechanics prediction error
- uncertainty vs subset disagreement
- uncertainty vs downstream failure or brittleness

Interpretation:

- if uncertainty does not track actual error, it is decorative

### 5. Does downstream control benefit?

Relevant metrics:

- solve episodes
- solve env steps
- success rate across seeds
- performance as a function of probe budget

Interpretation:

- this matters, but it is downstream evidence, not the whole story

## The Representation Metrics

These metrics should be treated as the primary readout.

### Mechanics Fit

Definition:

- how well the env belief linearly predicts hidden env parameters

Why it matters:

- it is the fastest sanity check for whether mechanics are in the latent at all

What a good result looks like:

- multiple hidden parameters are decodable above noise
- not just one easy parameter

What can fool it:

- global decode can improve while local geometry is still bad

### Per-Parameter `R²`

Definition:

- one decode score per hidden mechanics variable

Why it matters:

- the belief may learn some mechanics but not others
- aggregate metrics can hide that

What to look for:

- which variables are easy
- which variables are absent
- whether one parameter dominates all gains

### Same-Env Spread

Definition:

- disagreement between env beliefs rebuilt from different subsets of probe
  windows from the same env instance

Why it matters:

- this is one of the cleanest tests of whether a few probes are enough

What a good result looks like:

- low mean spread
- low high-percentile spread
- decreasing spread as probe budget grows

What can fool it:

- if subsets are badly sampled and always include nearly all views
- if the belief simply collapses to a constant vector

### Neighbor Alignment

Definition:

- whether nearby env beliefs correspond to nearby hidden mechanics

Why it matters:

- this tests local geometry, not just global decode

What a good result looks like:

- nearest neighbors in latent space are also close in mechanics space

Why it matters so much:

- a representation can be globally decodable but locally useless
- local geometry is what makes transfer, retrieval, and interpolation credible

### Probe Leakage

Definition:

- how much of the representation variance is explained by probe identity or
  probe-style structure instead of environment mechanics

Why it matters:

- the easiest shortcut is often probe-script memorization

What a good result looks like:

- low probe leakage at the env-belief level

What can fool it:

- low leakage alone does not mean mechanics are captured well

## The Uncertainty Metrics

Uncertainty should be treated as a real prediction target.

### Mechanics Error Calibration

Question:

- do high-uncertainty beliefs actually have larger env-parameter prediction
  errors?

This is the most important uncertainty metric.

### Subset Disagreement

Question:

- if we rebuild the belief from different small subsets, how much does it move?

This is the operational "few probes are enough" uncertainty metric.

### Failure Lift

Question:

- do high-uncertainty beliefs correspond to more brittle or failure-prone task
  behavior?

Usefulness:

- secondary
- helpful, but less central than mechanics-error calibration

Important caution:

- failure lift can be weak even if the belief is good, especially on easy
  tasks

## The Downstream Metrics

These are necessary, but secondary to the representation story.

### Solve Episodes

Question:

- how many episodes until the controller solves the task?

Why it matters:

- it is the cleanest adaptation-speed metric

Limit:

- it can hide probe overhead

### Solve Environment Steps

Question:

- how many real env interactions were needed end to end?

Why it matters:

- it is the honesty check against front-loading cost into probing

Limit:

- on some research questions, probe cost is intentional and part of the design

### Success Rate

Question:

- how reliably does the method solve across seeds?

Why it matters:

- a strong belief may mainly help hard seeds
- reliability is an important downstream benefit

## The Correct Reading Order For A Run

When looking at one run, use this order:

1. mechanics fit and per-parameter `R²`
2. same-env spread
3. neighbor alignment
4. uncertainty calibration
5. downstream benchmark

Do not start with benchmark score.

If the benchmark looks strong but the first four items look weak, then the
controller is probably compensating for a weak belief.

## The Dashboard Rules

The dashboard is helpful, but it can mislead.

### Use The Dashboard For

- reading representation diagnostics
- comparing env-level beliefs
- spotting obvious failure modes
- checking whether the same metrics move together

### Do Not Use The Dashboard As

- proof by visualization
- evidence that one PCA cloud is "better looking"
- an excuse to ignore the underlying artifact definitions

### Preferred Dashboard Views

- env-level belief points, not raw window points
- per-parameter decode rows
- same-env subset spread
- uncertainty vs actual error
- mechanics-based coloring

### Views To Treat Cautiously

- reward coloring
- PCA coverage
- any mixed window/env uncertainty display

## The Anti-Metrics

These are metrics that can be useful as secondary context but should never lead
the research story.

- PCA coverage
- cluster prettiness
- raw latent norm
- generic entropy values
- probe reward without belief interpretation
- average return on a single seed

If one of these moves while the real representation metrics do not, do not let
it drive major design changes.

## The Main Failure Modes

These are the failure modes most likely to fool us.

### 1. Probe-Style Latent

Symptoms:

- low mechanics fit
- high probe leakage
- neat clusters driven by probe mode

### 2. Trajectory-Texture Latent

Symptoms:

- structured PCA cloud
- weak mechanics decode
- weak same-env agreement

### 3. Globally Decodable, Locally Bad

Symptoms:

- decent mechanics fit
- poor neighbor alignment
- controller benefits but manifold is not coherent

### 4. Collapsed Belief With Fake Confidence

Symptoms:

- low uncertainty
- low spread
- poor mechanics fit
- nearly constant belief output

### 5. Controller Compensation

Symptoms:

- benchmark improves
- representation metrics stay muddy
- belief is more like a regularizer than a mechanics belief

## The Evaluation Checklist For Any New Change

When a new change lands, ask:

1. Did mechanics fit move?
2. Did per-parameter decode improve or degrade?
3. Did same-env spread improve?
4. Did neighbor alignment improve?
5. Did uncertainty calibration improve?
6. Did downstream solve speed improve?
7. Are the improvements coherent, or is one subsystem compensating for another?

If only item 6 moved, the result is incomplete.

## The Gold-Standard Experiment

The cleanest experiment in this repo is:

1. sample many hidden env instances
2. give the system only a small fixed probe budget
3. build env beliefs from those probes
4. measure:
   - mechanics decode
   - same-env stability
   - uncertainty calibration
5. then train or run downstream control
6. compare adaptation cost to a baseline without env belief

That experiment is closer to the real thesis than simply running PPO longer.

## The One-Sentence Rule

The benchmark tells us whether the belief pays off.
The representation metrics tell us whether the belief is actually the thing we
claim it is.

Always read them in that order.
