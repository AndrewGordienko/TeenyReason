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
- held-out probe prediction error

Interpretation:

- if these are weak, then the belief is not really learning the world
- if mechanics decode is decent but held-out probe prediction is weak, the
  belief may be acting more like a compressed control hint than a predictive map

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

Important refinement:

- subset disagreement only means something if the subsets are genuinely small
  and diverse
- if the support set itself is narrow, low disagreement can be fake

For the current CartPole path, the intended interpretation is:

- canonical support is roughly `4` windows
- diagnostic disagreement should come mainly from two disjoint support halves
- disagreement should also be tested under leave-one-goal-out ablations

Important:

- overlapping random subsets can make uncertainty and same-env spread look
  better than they really are
- disjoint splits are the more honest "would a different small experiment set
  infer the same world?" test

### Support Diversity

Question:

- do the windows used to build the env belief come from distinct experiment
  families?

Why it matters:

- a low-disagreement belief built from repeated copies of one easy probe is not
  evidence of broad world understanding

What a good result looks like:

- high support diversity ratio
- multiple goal families represented inside the actual support set

What can fool it:

- many total windows per env but only a narrow support subset actually used
- a crawler that repeats one safe experiment and only occasionally touches the
  rest

Current support-mix diagnostics:

- `center_window_share` asks whether a stable no-op style probe is still
  dominating the saved evidence, even if online probe selection looks diverse.
- `directional_window_share` and `effective_window_families` show whether the
  support set is actually intervention-balanced or only nominally varied.
- `window_mode_leakage` measures probe-family leakage before pooling, while
  `env_mode_leakage` measures how much of that nuisance survives in the final
  env belief.
- `nearest_between_median`, `pairwise_between_mean`, and `belief_norm_std` keep
  support diversity tied to geometry collapse instead of treating coverage as a
  separate checkbox.

Interpretation:

- if online probe choice is directional but `center_window_share` stays high,
  the offline support collector is retrying or retaining the wrong evidence
- if `window_mode_leakage` is high but `env_mode_leakage` is low, aggregation is
  helping but the window encoder still carries probe script identity
- if both leakage values are high, increase probe-invariance pressure only
  after checking support balance, because leakage penalties can otherwise erase
  useful mechanics signal from an undercovered support set

### Uncertainty vs Actual Mechanics Error

Question:

- when uncertainty is high, is mechanics prediction error also high?

This should be treated as the main honesty check for uncertainty.

What a good result looks like:

- high-uncertainty env beliefs have larger actual mechanics decode error
- low-uncertainty env beliefs are genuinely easier and more stable

What can fool it:

- uncertainty that only tracks support count
- uncertainty that collapses because all subsets are too similar or too
  overlapping
- uncertainty that is measured on window latents while being interpreted as an
  env-belief quantity
- uncertainty that is free to invert the ordering because the calibration head
  is unconstrained

Preferred current implementation:

- use a monotone learned uncertainty head over explicit disagreement features
- train against normalized mechanics-error targets plus ranking and
  high-error-vs-low-error separation losses
- inspect the learned feature weights in the dashboard so we can see which
  uncertainty ingredients the model is actually using
- if geometry-heavy features become anti-correlated with mechanics error, keep
  them as diagnostics but reduce their role inside the uncertainty object
- inspect the raw uncertainty distribution too, because a nearly constant
  uncertainty signal can still look acceptable under some normalized losses

### Split Retrieval

Question:

- if one disjoint support half describes a world, can it retrieve the matching
  other half?

Why it matters:

- this is a direct local-geometry check
- it is often more operational than a PCA plot

What a good result looks like:

- high top-1 retrieval between split A and split B
- rising mean reciprocal rank or improving median match rank, not only one
  noisy top-1 number
- retrieval improves alongside neighbor alignment, not instead of it

What can fool it:

- retrieval on projected auxiliary embeddings while the raw env belief remains
  muddy
- easy retrieval only because the environment set is tiny or trivially
  separated

Additional honesty check:

- compare same-world split disagreement against nearest different-world latent
  distance
- if same-world disagreement is a large fraction of nearest-between distance,
  the belief space is still locally weak even if raw disagreement looks small

Current dashboard expectation:

- show both raw same-world gap and nearest-between distance
- plot these against each other so the ratio is visually obvious
- treat gap ratio as more trustworthy than rounded same-env spread alone
- also show nearest-between-distance distribution, because a tiny absolute
  belief scale can make ratios look confusing until the collapse is visible
- also show full pairwise different-world distance distribution, because
  nearest-neighbor views alone can hide global codebook collapse
- also show split retrieval-rank distribution, because median rank movement is
  often more informative than noisy top-1 retrieval alone
- also show belief-norm distribution and raw-vs-unit distance summaries, so we
  can tell whether collapse lives in the raw belief or only in the normalized
  comparison view

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

For the current fair CartPole benchmark, downstream evaluation should always
track three matched arms:

- baseline PPO
- probe plus env expression
- probe plus no env expression

Why this matters:

- if probe plus env expression beats baseline and also beats the matched
  no-expression arm, that is the intended latent-driven win
- if probe plus env expression beats baseline but not the no-expression arm,
  that is a protocol win rather than proof that the env belief itself is doing
  the work
- if the probe arm wins overall while the no-expression arm is just as good or
  better, the controller or schedule is compensating for a weak latent

The current artifact language should use:

- `latent_win`
- `protocol_win`
- `controller_compensation`

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

### Post-Expression Latency

Question:

- once the crawler has handed off an env expression, how many control-only
  episodes or env steps does the controller still need to solve?

Why it matters:

- this is the cleanest check on the actual value of the env expression itself
- it separates "the controller used the expression well" from "probing was
  cheap enough overall"

Interpretation:

- if post-expression latency is good but total env steps are bad, the
  expression may still be useful while the budget policy is overpaying
- if post-expression latency is also bad, the expression handoff itself is not
  strong enough yet

### No-Expression Matched Arm

Question:

- if we keep the same probe schedule and same downstream PPO but mute the env
  expression, how much performance disappears?

Why it matters:

- this is the cleanest current check on whether the env belief itself is
  driving the benchmark win

Interpretation:

- if probe plus env expression clearly beats probe plus no env expression, the
  latent is contributing real downstream value
- if the two are similar, the run is mostly a protocol win
- if no-expression wins, the controller is still being helped more by the
  schedule than by the env belief

### Probe Budget Honesty

Question:

- why did probing stop, and how often is the system still buying a third probe?

Why it matters:

- episode speed can improve while env-step cost still regresses if probe three
  is too cheap
- a fair-mode run should now end through `expression_ready`,
  `fair_two_probe_handoff`, or `probe_failure` rather than through economic
  stop reasons

Relevant diagnostics:

- final stop reason
- episode-level stop-reason counts
- probe-count distribution
- probe cost split: encoder vs online probing vs downstream control

Interpretation:

- if probe wins on episodes but loses badly on env steps, inspect probe-count
  and stop-reason traces before blaming the belief itself
- if fair mode rarely reaches `expression_ready`, the latent still is not
  earning early trust
- if `fair_two_probe_handoff` dominates and the env expression is then muted by
  policy, the protocol may be working while the latent still is not
- if third probes are common while next-family value is only barely positive,
  the controller is still paying for weak evidence

### Controller Robustness

Question:

- does the downstream solver still behave sensibly when the belief is weak,
  misleading, or absent?

Why it matters:

- a belief can contain some mechanics signal globally while still having bad
  local geometry
- negative env-expression ablations usually mean the controller trusts the
  wrong latent too much, not just that probing spent too much

Relevant diagnostics:

- env-expression ablation delta
- expression-scale trace
- median expression scale during control
- fraction of control episodes with expression scale above `0.1`
- fraction of fair handoffs that were force-muted by policy
- solve behavior on seeds with weak neighbor alignment or poor split retrieval

Interpretation:

- if env-expression ablation turns negative on multiple seeds, fix controller trust
  before adding more encoder complexity
- if expression scale stays high while local geometry metrics are weak, trust
  gating is too permissive
- if expression scale stays near zero on winning runs, the benchmark is
  probably a protocol win instead of a latent win
- if the controller fully collapses when the expression is ablated, it has not
  learned a healthy state-only fallback yet

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
5. probe plus env expression versus probe plus no env expression
6. downstream benchmark
7. probe-budget honesty
8. controller robustness

Do not start with benchmark score.

If the benchmark looks strong but the first four items look weak, then the
controller is probably compensating for a weak belief.

If the benchmark looks strong and the no-expression matched arm is nearly as
strong, classify the run as a protocol win rather than a thesis-complete
latent win.

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
- support diversity ratio
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
