# Image Belief Design

This file is the design companion to
[image_domain_synthesis.md](./image_domain_synthesis.md).

That file explains what the image papers imply.

This file answers the follow-up question:

if we took those implications seriously, what would the image-side version of
the crawler-belief-solver architecture actually look like?

## The Image Version Of The Latent

The image analogue of the repo's env belief should not be one undifferentiated
embedding used only for classification.

It should be thought of as a structured visual belief with parts like:

- `z_objects`
  Belief over object-like slots or entities in the scene.
- `z_parts`
  Belief over reusable parts, shapes, textures, and local composition.
- `z_scene`
  Belief over the global scene layout and relations.
- `z_category`
  Belief over concept identity or prototype family.
- `z_affordances`
  Belief over what transformations or actions preserve concept identity.
- `z_surface`
  Nuisance information like background style, lighting, or viewpoint details.
- `u_vision`
  What is still unresolved.

Mapped back to the repo's general notation:

- `z_rules`
  visual regularities, category structure, transformation rules
- `z_entities`
  objects and parts
- `z_affordances`
  allowed transformations and viewpoint-invariant structure
- `u`
  what the system still cannot identify confidently

The key idea is:

the image latent should represent reusable visual structure that survives across
views, crops, examples, and tasks.

## The Image Crawler

The image-side crawler should not be understood as "just data augmentation."

The crawler's job is to choose informative visual evidence and transformations.

Depending on the setting, this means different things.

### Passive image crawler

For a fixed image corpus, the crawler can choose:

- which images to inspect first
- which crops or patches to compare
- which masked regions to predict
- which augmentations preserve identity
- which pairs are likely to show the same object or class
- which examples are most informative for prototype formation

This is active inference over a passive dataset.

### Interactive visual crawler

For embodied or controllable image environments, the crawler can also choose:

- camera angle
- zoom or crop
- movement around an object
- object manipulation
- lighting or occlusion changes
- interventions that reveal hidden parts

This is the clean vision analogue of active system identification.

## Concrete Image Probes

If we ever build the image side in code, these are the kinds of probes that
best match the repo thesis.

### Multi-view probes

- compare two augmented views of the same image
- compare two nearby crops
- compare multiple viewpoints of the same object

These are the direct SimCLR / BYOL / DINO-style evidence units.

### Masked-structure probes

- hide patches and predict them
- infer missing parts from visible context
- compare which completions preserve concept identity

These are the MAE-style probes.

### Prototype probes

- choose support examples for a new class
- estimate whether a new example should update or split a prototype
- compare class-center stability under few examples

These are the Matching Nets / ProtoNets style probes.

### Object-centric probes

- decompose scenes into slots
- track whether slots stay stable across views
- compare part reuse across examples

These are the Slot Attention style probes.

### Language-anchored probes

- attach names or text descriptions to visual examples
- compare whether textual and visual neighbors agree
- test if a textual cue sharpens the visual belief

These are the CLIP-style probes.

## The Solver In Images

The image-side solver does not have to be a plain classifier.

The broader solver could be:

- few-shot classifier
- retrieval system
- segmentation head
- detector
- embodied visual controller
- multimodal reasoner

The shared idea is that the solver receives a structured belief and solves a
task more efficiently because it does not have to infer the visual world from
scratch.

That is the direct analogue of:

first understand the pendulum, then balance it.

For images the analogue is:

first infer the object, part, and concept structure, then classify, retrieve,
segment, or act.

## The General Cross-Domain Thesis

This repo should therefore treat images, language, and physical control as
three instances of the same broader pattern.

The latent mechanism should be able to support:

- physical dynamics discovery
- language-structure discovery
- visual concept discovery
- other symbolic or interactive domains

That does not mean the same encoder should be reused blindly.

It means the same architecture pattern should survive:

1. local evidence windows
2. belief updates over hidden structure
3. calibrated uncertainty
4. downstream solver conditioned on belief

## What We Should Import Into Code

If future agents extend the repo toward a more domain-general system, these are
the ideas worth importing from the image literature.

### 1. Prototype-friendly geometry

The latent should support:

- few-example class summaries
- fast nearest-neighbor comparisons
- stable concept centers

This is the key few-shot geometry lesson.

### 2. Self-supervised multi-view consistency

The crawler should rely heavily on:

- cross-view agreement
- masked prediction
- self-distillation

before large amounts of label supervision are needed.

### 3. Object-centric decomposition

The image latent should separate:

- objects
- parts
- scene layout
- nuisance background or style

This is likely essential for child-like efficiency.

### 4. Support-set conditioning

The solver should be built to adapt from a support belief, not just to emit
fixed logits.

### 5. Confidence-gated self-training

The system should exploit weak supervision or unlabeled images only when the
belief is sharp enough to justify it.

### 6. Vision-language concept priors

The cross-domain system should eventually allow language to help shape visual
beliefs and visual evidence to help shape language beliefs.

## A Concrete Image Track For This Repo

If we actually wanted to prototype the image side in this repo, the first goal
should not be "beat ImageNet."

The first goal should be:

build an image environment-belief testbed that lets us ask whether a small
amount of visual evidence can produce a stable, uncertainty-aware belief about
objects or concepts.

### Phase 1. Controlled few-shot concept environment

Keep the environment deliberately simple.

Candidate datasets:

- Omniglot
- miniImageNet
- CIFAR-FS
- controlled synthetic shapes or part-composition datasets

Hidden variables to infer:

- concept identity
- part composition
- transformation invariances
- nuisance style factors

Solver tasks:

- one-shot or few-shot classification
- retrieval
- support-set adaptation

Why this phase matters:

It isolates whether the belief can form good prototypes and invariances from
small evidence sets.

### Phase 2. Multi-view object belief

Move from single images to grouped evidence.

Design:

- local crop or view encoder produces `z_window`
- image or object aggregator produces `z_env`
- different views of the same object should agree on shared structure
- beliefs from different concepts should separate

This is the exact analogue of:

- local probe window
- env-level belief

on the physical RL side.

### Phase 3. Object-centric belief

Move from one pooled visual belief to slot-based or part-based beliefs.

Design:

- object slots or part slots become explicit latent substructures
- same object across views should map to stable slots
- nuisance backgrounds should be suppressed

Why this phase matters:

This is probably the closest vision-side analogue to "understanding the world"
rather than memorizing appearance.

### Phase 4. Active visual crawler

Stop treating the image side as static augmentations only.

Give the crawler choices like:

- which crop to inspect next
- which region to mask
- which support example to request
- which view or augmentation to compare
- which object slot to refine

The crawler reward should be something like:

- subset agreement improvement
- prototype stability improvement
- uncertainty reduction
- downstream few-shot performance after a fixed evidence budget

### Phase 5. Embodied or interactive vision

Once the passive setup is stable, add environments where visual actions change
what is observed.

Examples:

- moving-camera object views
- simple embodied navigation with recognition
- manipulation tasks where new views reveal hidden structure

At this point, the image side becomes very close to the RL side again.

### Phase 6. Shared cross-domain core

Only after the image setup is real should we push for shared modules.

The likely shared core is:

- evidence window data structure
- belief aggregator interface
- uncertainty interface
- prototype or memory interface
- solver-conditioning interface

The likely domain-specific adapters are:

- visual observation encoders
- crop or slot builders
- image-specific predictor heads

This is the boring, maintainable way to make the repo more general.

## Candidate Module Boundaries

If we eventually implement the image track, the code should probably stay flat
and explicit.

One plausible direction is:

- `teenyreason/domains/control/`
  Physical environment adapters and probe routines.
- `teenyreason/domains/language/`
  Text or dialogue environment adapters.
- `teenyreason/domains/vision/`
  Image or video environment adapters.
- `teenyreason/core/evidence.py`
  Domain-agnostic evidence window and grouping types.
- `teenyreason/core/belief.py`
  Domain-agnostic belief and uncertainty interfaces.
- `teenyreason/core/solver.py`
  Shared solver-conditioning interfaces.

Important note:

This is not a command to refactor the repo immediately. It is a future
direction if and when the current code earns it.

## Image-Side Metrics That Would Actually Matter

If we build an image belief testbed, the metrics should mirror the physical and
language ones but stay honest about vision.

### Belief metrics

- same-object subset agreement
- different-object or different-class separation
- prototype stability
- object-slot stability across views
- uncertainty calibration against classification or retrieval error

### Solver metrics

- one-shot or few-shot accuracy
- adaptation after limited support examples
- retrieval quality
- segmentation or object-centric transfer quality

### Crawler metrics

- uncertainty reduction per queried view
- prototype improvement per queried image
- value of active support selection versus random support selection

### Non-metrics

Things we should avoid overreading:

- one aggregate top-1 score with no belief analysis
- pretty embedding plots with weak support-set behavior
- benchmark wins that mostly come from heavy pretraining but say little about
  the belief machinery itself

## The One-Horse Thought Experiment

The "child sees a horse once" intuition is useful, but it needs to be made
precise or it becomes hand-wavy.

The clean version is:

- what priors did the child already have?
- what object categories were already known?
- what part-whole structure was already understood?
- was language already helping bind a name to a concept?
- how much earlier world knowledge was transferred in?

If the answer is "strong priors, existing concept hierarchies, object-centric
structure, and good uncertainty handling," then one new example can teach a
great deal.

If the answer is "blank learner with no structure," then one example is wildly
insufficient.

That is the main bridge between image learning and the RL question.

In both cases, fast learning comes from:

- good priors
- good latent structure
- good evidence selection
- good transfer

not from magic.

## What Not To Do

Do not let the image analogy degenerate into any of the following:

- "one embedding dimension reduction is enough"
- "classification accuracy equals concept understanding"
- "one-shot benchmark success proves open-world child-like learning"
- "more augmentations automatically solve data efficiency"
- "a pooled latent is always enough without object structure"

Those are exactly the shortcuts the image papers warn us away from.

## Practical Guidance For Future Agents

If you are writing future code after reading this file, keep these rules in
mind.

### Rule 1

Do not build image support by pretending a class embedding is already a world
belief.

### Rule 2

Do build image support by preserving the same abstract phases:

- gather evidence
- infer hidden visual structure
- quantify uncertainty
- solve downstream tasks

### Rule 3

When evaluating an image belief, separate:

- object structure
- class identity
- nuisance invariances
- prototype geometry
- true uncertainty

Do not treat one score as covering all of them.

### Rule 4

If an image-side benchmark improves, ask whether the improvement came from:

- a better belief over objects or concepts
- stronger generic pretraining
- easier optimization
- a solver that compensated for a weak belief

That is the same discipline we already need on the RL side.

## Bottom Line

The vision papers do not say that unrestricted one-shot natural-image learning
is solved.

What they do say, taken together, is more useful:

- strong priors matter
- support-set geometry matters
- self-supervised latent learning matters
- object-centric structure matters
- uncertainty matters
- language can help shape concepts

That is exactly the shape of the broader repo thesis.

We are trying to build a general crawler-belief mechanism that can enter a new
domain, discover its hidden constraints quickly, and give a downstream solver a
head start.
