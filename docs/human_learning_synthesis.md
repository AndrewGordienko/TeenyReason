# Human Learning Synthesis

This file is the developmental-science and neuroscience bridge for the repo.

The purpose is not to turn the project into cognitive science fan fiction.
The purpose is to make the human inspiration explicit enough that we can import
the right ideas into code without over-claiming that current ML systems are
"learning like children" just because they used fewer examples once.

The question here is:

What do developmental psychology, cognitive science, and neuroscience suggest
about how a child can build reusable latent world structure from small amounts
of experience?

The short answer is:

- children do not learn from a blank slate
- they bring structured priors about objects, agents, causes, and space
- they use surprise to decide what to investigate
- they learn statistical regularities from passive streams
- they use active interventions to disambiguate causal hypotheses
- they build predictive maps that support reuse, not just memorization

That is extremely close to the repo's intended `crawler -> belief -> solver`
pattern.

## The Human Story We Actually Want

The most useful human-learning picture for this repo is:

1. A learner comes in with partial built-in structure.
2. The learner quickly extracts regularities from observation.
3. Surprise highlights which parts of the environment are unresolved.
4. The learner runs targeted experiments to reduce uncertainty.
5. The learner stores a compact, reusable world belief.
6. A downstream skill can then be learned quickly because the learner already
   understands the world better.

That is the same high-level story we want in:

- physical RL
- language learning
- image understanding
- eventually any interactive environment with hidden rules

## 1. Core Knowledge

Reference:

- `papers/spelke_kinzler_2007_core_knowledge_view.pdf`

Main idea:

- children are not starting from zero
- early cognition appears to contain structured expectations about:
  - objects
  - agents
  - number
  - geometry / space

What matters for this repo:

- we should not ask the crawler-belief system to discover every useful factor
  from scratch
- we should encode structural priors in the architecture

What to import into code:

- domain-specific factorization when justified
  - physical envs: mass, force, damping, contact, controllability
  - language: syntax, lexicon, discourse state, intent
  - images: objects, parts, relations, viewpoint, style
- object or entity grouping when possible
- belief heads that separate hidden mechanics from surface style

What not to over-claim:

- "core knowledge" does not mean hand-author the answer for every benchmark
- the lesson is architectural bias, not brittle hardcoding of solutions

Repo consequence:

- latent factors should be organized around causal structure, not generic
  compression
- if a domain naturally supports slots, objects, entities, or mechanics
  variables, that structure should be made available to the model

## 2. Infant Statistical Learning

References:

- `papers/kuhl_2011_social_mechanisms_early_language_acquisition.pdf`
- `../papers/README.md` for canonical child statistical-learning papers that are
  important but not trivially mirrorable via automated download

Main idea:

- infants rapidly absorb distributional regularities from relatively small
  streams of experience
- this learning is not only supervised category labeling
- temporal contingency, transition probability, and repetition structure matter

What matters for this repo:

- the crawler does not need to act constantly to learn
- passive observation can already sharpen beliefs if the model is structured
  correctly

What to import into code:

- keep passive evidence as a first-class mode of learning, not just an
  embarrassing baseline
- make sequence statistics matter in belief updates
- treat evidence windows as structured time series, not unordered bags

Cross-domain meaning:

- RL: short passive decay / free-response windows are informative
- language: token transitions, phrase regularities, and discourse recurrence
  provide fast structure
- images/video: view co-occurrence, motion continuity, and part persistence
  build reusable visual priors

What not to over-claim:

- passive statistical learning is not enough by itself for causal
  understanding
- it gives constraints, not a full explanatory model

## 3. Social Gating And Rich Teaching Signals

Reference:

- `papers/kuhl_2011_social_mechanisms_early_language_acquisition.pdf`

Main idea:

- in children, learning is often amplified by social interaction, joint
  attention, and structured demonstrations
- the learner is not only reading raw data; it also benefits from cues about
  what matters

What matters for this repo:

- not all useful evidence must come from blind trial and error
- demonstrations, hints, or curricula are legitimate evidence channels

What to import into code:

- allow environments to provide multiple evidence modes:
  - self-generated probes
  - passive observation
  - demonstrations
  - textual descriptions / metadata
- make the belief model capable of integrating these modes instead of assuming
  every domain is solitary motor interaction

Cross-domain meaning:

- RL: start states, demonstrations, or teacher probes can shrink uncertainty
- language: explanations, dialogue, and grounded feedback matter
- images: captions or class descriptions can act like concept-level teaching

## 4. Causal Learning In Children

Reference:

- `papers/begus_bonawitz_2024_informativeness_of_evidence_and_predictive_looking.pdf`
- `../papers/README.md` for canonical children-as-scientists and causal-model
  papers that remain part of the repo's conceptual background

Main idea:

- children do not only memorize correlations
- they appear to form structured causal hypotheses and update them from
  interventions and observations

What matters for this repo:

- the latent should behave more like a causal hypothesis than a generic style
  code
- interventions should be chosen for disambiguation value

What to import into code:

- explicit hidden-world hypotheses
- intervention-aware belief updates
- losses that reward invariance across different experiments on the same world
- metrics that ask whether the belief supports causal prediction and not just
  classification

What not to over-claim:

- we are not building symbolic Bayes nets for every domain tomorrow
- the lesson is structural hypothesis testing, not dogmatic symbolic-only
  modeling

## 5. Surprise Drives Learning

References:

- `papers/begus_bonawitz_2024_informativeness_of_evidence_and_predictive_looking.pdf`
- `papers/berger_tzur_german_daum_dehaene_2015_neural_dynamics_prediction_surprise_infants.pdf`

Main idea:

- violation of expectation does not just produce a curiosity-like signal
- it changes what infants examine, manipulate, and learn next
- neuroscience work also suggests that infant brains carry explicit predictive
  and surprise-like signatures during these updates

What matters for this repo:

- surprise should be part of crawler control, not only a dashboard curiosity
- unresolved mismatches between prediction and outcome should redirect probes

What to import into code:

- novelty and disagreement are not enough by themselves
- the crawler should track expectation violations tied to mechanics hypotheses
- high surprise should trigger targeted follow-up experiments, not generic
  repetition

Cross-domain meaning:

- RL: follow up on actions whose outcomes sharply contradict the current world
  belief
- language: focus on constructions or tokens that violate the current grammar
  or discourse model
- images: focus on views or crops that violate object permanence or part
  consistency assumptions

## 6. Predictive Maps And Reusable Spatial / Sequential Structure

References:

- `papers/stachenfeld_botvinick_gershman_2017_hippocampus_predictive_map.pdf`
- `papers/garvert_dolan_behrens_2023_hippocampal_spatiopredictive_cognitive_maps.pdf`

Main idea:

- biological memory and navigation systems seem to encode predictive structure
- a useful representation stores what future states or outcomes are likely from
  here, not only what has already been seen

What matters for this repo:

- the belief should support counterfactual and successor-style prediction
- good world representations are predictive maps, not merely compressed
  summaries

What to import into code:

- successor-like or predictive-map losses where useful
- local geometry checks that ask whether nearby beliefs imply nearby futures
- retrieval metrics that reward world-consistent neighborhood structure

What not to over-claim:

- predictive maps are not a complete theory of intelligence
- the lesson is to preserve future-relevant structure, not to fetishize one
  neuroscience formalism

## 7. Active Science, Not Just Reactive Compression

Across these papers, the human lesson is not:

- "children have tiny models and therefore one-shot everything"

It is:

- children combine prior structure, passive statistical learning, surprise,
  intervention, memory, and predictive reuse

That is exactly the repo's intended algorithmic shape.

## What This Means For The Repo

The cross-domain target is now clearer:

- `crawler`
  gathers informative evidence using passive observation, active intervention,
  and eventually demonstrations or language cues
- `belief`
  stores reusable causal / predictive structure about the world
- `solver`
  learns the downstream task quickly because the world model is already partly
  identified

The human papers especially argue for five design commitments:

1. Keep strong priors.
   Do not make every domain rediscover objectness, causality, or sequence
   structure from scratch.

2. Let surprise steer exploration.
   Unexpected outcomes should change what the crawler does next.

3. Separate evidence from policy.
   World-identification is a different problem from task solving.

4. Prefer predictive beliefs over static embeddings.
   The latent should preserve future-relevant structure.

5. Treat uncertainty as operational.
   Uncertainty should tell the crawler where another experiment would help and
   tell the solver how much to trust the belief.

## What Success Would Look Like

If we were really importing the human lesson well, we would eventually see:

- small, diverse support sets that still recover useful mechanics or grammar
- uncertainty that rises on ambiguous or contradictory evidence
- surprise-driven follow-up probes instead of repeated comfortable probes
- beliefs whose local geometry preserves causal or predictive similarity
- faster downstream learning because the solver no longer has to infer the
  world from scratch

That is the developmental-science version of the repo thesis.
