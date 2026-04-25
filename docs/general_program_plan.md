# General Program Plan

This file is the operational companion to
[general_crawler_belief.md](./general_crawler_belief.md).

That file defines the general crawler-belief-solver idea.

This file answers the practical follow-up questions:

- what artifacts should the system save?
- how should the repo expand without losing the plot?
- which staged implementation path makes sense?
- which metric families matter across domains?
- what does success actually look like?

## Canonical Artifact Set

If we keep pushing the repo toward a broader cross-domain system, the saved
artifacts should become more standardized.

For every domain, the system should try to save:

- raw evidence windows
- world or environment grouping ids
- local evidence latents
- env-level or domain-level beliefs
- uncertainty estimates
- decoded hidden-structure targets
- subset-agreement statistics
- downstream solver results

The artifact contract matters because a large part of the research is not just
training the belief but inspecting whether it actually means what we think it
means.

## A Concrete Staged Implementation Path

The repo should not jump straight from physical control into a full universal
agent. The transition should be staged.

### Stage 1. Stabilize the physical belief stack

Goal:

- make the current control testbeds honest
- ensure `z_env` is not probe identity
- ensure uncertainty is meaningful

Deliverables:

- stable env-level aggregation
- subset disagreement metrics
- mechanics decode metrics
- cleaner crawler objectives

### Stage 2. Introduce domain-agnostic interfaces

Goal:

- define abstractions that survive outside control

Deliverables:

- a domain-agnostic notion of evidence window
- a domain-agnostic belief update interface
- a domain-agnostic uncertainty interface
- solver inputs defined as `local_state + belief + uncertainty`

Important rule:

Do not destroy the current physical code to do this. Add clear abstractions and
adapters rather than hiding everything behind one large generic layer.

Current status:

- this stage is partially implemented through a new generic crawler core plus
  compatibility adapters for the current RL benchmark
- the canonical library objects are now `EvidenceSlice`, `BeliefState`,
  `CrawlerMessage`, `CrawlerStep`, and `CrawlerRunResult`
- the downstream fair benchmark still consumes a solver-facing compatibility
  message while the app layer migrates
- future image and language recipes should plug into the same crawler API
  rather than invent a second belief contract

### Stage 3. Add a passive language or image testbed

Goal:

- test whether the same belief machinery can learn structure from small passive
  datasets outside control

Candidate tasks:

- cloze and infill
- next-sentence selection
- discourse tracking
- style adaptation
- small-corpus continuation
- one-shot image classification
- support-set retrieval
- object-centric consistency across views

Why this stage matters:

It tests whether the crawler-belief pattern works in passive environments where
intervention mostly means active hypothesis-testing over fixed evidence.

### Stage 4. Add an interactive language, image, or tool-using testbed

Goal:

- test whether the crawler can ask questions, request clarifications, or use
  tools to sharpen the belief

Candidate tasks:

- constrained dialogue with hidden partner rules
- tool-augmented QA with hidden schemas
- symbolic games with textual interfaces
- moving-camera object recognition
- controllable multi-view visual identification

Why this stage matters:

It is the true cross-domain analogue of active system identification.

### Stage 5. Unify solver-side evaluation

Goal:

- compare whether the belief helps multiple downstream solvers, not just PPO

Candidate solvers:

- PPO or policy gradient
- model-predictive control
- decoder-only generator
- parser / structured predictor
- reasoning or tool-use policy

This is where the repo graduates from "RL testbed with ideas about language" to
"general crawler-belief-solver system."

## Cross-Domain Metrics That Matter

The same metric families should survive across domains.

### Belief quality

- same-world subset consistency
- between-world separation
- hidden-rule decode quality
- uncertainty calibration

### Geometry quality

- local neighbor alignment
- retrieval quality
- cluster purity when there are known world families

### Solver value

- few-shot adaptation speed
- total interaction cost
- robustness across seeds or world instances
- performance under partial evidence

### Crawler value

- uncertainty reduction per intervention
- belief improvement per intervention
- early stopping quality

### Communication value

- mechanics fit versus belief bitrate
- downstream solve quality versus belief bitrate
- retrieval or neighbor quality versus belief bitrate
- uncertainty quality versus belief bitrate

For the current benchmark compatibility layer, the repo already saves:

- mechanics-fit versus bits
- split-retrieval versus bits
- solver-message norm versus bits

The next cross-artifact comparison step is downstream solve cost versus bits.

If a metric does not fit one of these categories, it should need strong
justification before being added.

## Belief Communication Experiments

The repo should explicitly treat `z_env` as a communication channel between the
crawler and the downstream solver.

That means we should eventually run experiments like:

### 1. Fixed-bitrate belief handoff

Procedure:

- build `z_env`
- rotate or normalize it if needed
- quantize it to a chosen budget
- feed only the compressed belief to the solver

Question:

- how many bits are needed before the solver still adapts quickly?

### 2. Rate-distortion curves

Measure:

- mechanics decode versus bits
- same-env agreement versus bits
- local geometry versus bits
- downstream task performance versus bits

Question:

- what structure survives compression, and what collapses first?

### 3. Coarse belief plus residual sketch

Procedure:

- send a coarse compressed belief
- add a tiny residual code for what the coarse code misses

Question:

- can the belief be decomposed into a cheap global world summary plus a tiny
  correction?

### 4. Adaptive bitrate by uncertainty

Procedure:

- use lower bitrates when the world is easy and belief is sharp
- spend more bits when the world is still uncertain

Question:

- can uncertainty control communication cost intelligently?

These experiments matter because they test whether the latent is truly a small
sufficient message, not just a large helpful feature vector.

## What Counts As Success

The general success criteria are:

- a few pieces of evidence are enough
- same-environment beliefs agree
- different-environment beliefs separate
- hidden rules are decodable from the belief
- uncertainty tracks unresolved structure
- a compressed version of the belief still preserves most of that value
- the downstream solver improves because the belief is informative

Importantly, success is not domain-specific.

In physical control, that means faster adaptation.

In language, that means learning grammar, discourse regularities, and local
world assumptions much faster than a solver that only sees raw token history.

## What This Means For Language

The language case should not be treated as an afterthought.

If the broader thesis is real, then the same architecture should apply to text:

- local text windows become evidence
- those windows update a belief over the latent rules of the corpus or
  language-like environment
- the downstream generator or reasoner uses that belief to act efficiently

However, the language case also imposes a key caution:

text-only environments may be sufficient to learn syntax, distributional
regularities, discourse flow, and style, while still being insufficient to
ground full semantics.

That means the general program must distinguish between:

- what can be inferred from form alone
- what requires interaction or grounding

This is not a bug in the program. It is part of the design boundary.

## What This Means For The Current Repo

Right now the repo is still organized around physical RL.

That is fine as long as we are honest:

- the current code is a proving ground
- the physical environments are a convenient first domain
- the deeper research object is the general crawler-belief mechanism

Future code should therefore preserve two levels of abstraction:

1. domain-agnostic concepts
   - evidence windows
   - beliefs
   - subset disagreement
   - predictive heads
   - solver inputs
2. domain-specific adapters
   - continuous control probes
   - text probing routines
   - dialogue or tool use actions

Do not hard-code the belief semantics to "physics only" if the real thesis is
broader than that.

## Anti-Goals

The broader program is not:

- "make one latent that solves everything with no structure"
- "pretend text-only corpora automatically give grounded meaning"
- "collapse all domains into identical observation tensors"
- "force every task into PPO"

The cross-domain claim is architectural, not naive.

The claim is:

there exists a reusable pattern of active evidence gathering, belief formation,
uncertainty tracking, and downstream solving that can apply across many
environments, even though the domain adapters differ.

## Working Motto

If we need one sentence to guide future code and writing, it should be:

Build a general crawler that learns the hidden constraints of a new world
quickly, then let a downstream solver focus on the task instead of first having
to rediscover the world.
