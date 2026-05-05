# Paper Synthesis

This file is the paper-to-algorithm bridge for the repo.

The goal is not to summarize every paper in a generic way. The goal is to
extract what each paper suggests about the exact thing we are trying to build:

an agent that actively probes a new environment, forms a compact belief about
its hidden mechanics, and then uses that belief to solve downstream control
tasks quickly.

For each paper, this file answers:

- what the paper contributes
- what idea matters most for this repo
- what we should import into code
- what we should not over-claim from it

## The Common Thread

Across all of these papers, the common message is:

- the representation should be a belief, not just a compressed history
- the crawler should ask informative questions, not just act randomly
- the latent should preserve task-relevant hidden structure, not every detail
- uncertainty should mean something operational
- downstream control should benefit because inference and control are separated

That is the main reason this repo should be thought of as an active
environment-belief project rather than as "probe-conditioned PPO".

For the developmental-science and neuroscience bridge, read:

- [human_learning_synthesis.md](./human_learning_synthesis.md)
- [human_learning_design.md](./human_learning_design.md)

For the current world-model and solver-handoff review, including V-JEPA 2, read:

- [research/world_model_handoff_review.md](./research/world_model_handoff_review.md)

## 1. PEARL

Paper:

- Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context
  Variables

Core contribution:

- infers a latent context variable from experience
- conditions the policy on that inferred context
- uses posterior sampling to adapt across tasks

Most relevant lesson for this repo:

- the latent should be a probabilistic belief over hidden environment or task
  variables
- inference should be treated as a first-class object, not just an auxiliary
  encoder
- the controller should get belief state, not raw history

What to import into code:

- latent posterior over hidden environment mechanics
- separation between evidence collection and task execution
- explicit uncertainty-aware belief input to the controller

What not to over-import:

- PEARL is not itself a full active probing system
- it does not solve the whole "child learns physics" story by itself
- it is still more about adaptation to task variables than causal intuitive
  physics in the richer sense we care about

Repo consequence:

- use env-level belief objects
- maintain mean + uncertainty
- evaluate downstream adaptation as a function of probe evidence

## 2. VariBAD

Paper:

- VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-RL

Core contribution:

- learns a recurrent belief over hidden task or environment variables
- uses full interaction history as the source of inference
- emphasizes Bayes-adaptive behavior, where acting and learning are coupled

Most relevant lesson for this repo:

- the right object is a belief over hidden dynamics inferred from sequence
- uncertainty is not optional
- early actions should sometimes be chosen to improve future decision quality,
  not just immediate reward

What to import into code:

- recurrent or sequence-based belief updates
- belief as an online object, not just an offline window summary
- belief evaluation in terms of adaptation quality

What not to over-import:

- VariBAD does not mean every recurrent latent is automatically a good env
  representation
- a recurrent encoder can still learn probe style or trajectory texture unless
  the losses constrain it

Repo consequence:

- the env belief should be refined from accumulated evidence
- online belief updates during early control are part of the design
- uncertainty should be interpretable as unresolved mechanics knowledge

## 3. ContraBAR

Paper:

- ContraBAR: Contrastive Bayes-Adaptive RL

Core contribution:

- avoids relying only on full reconstruction losses
- uses contrastive objectives to preserve task-relevant information for
  Bayes-adaptive behavior

Most relevant lesson for this repo:

- we do not need a latent that stores every detail of a probe window
- we need a latent that retains the distinctions that matter for hidden-world
  inference and later control

What to import into code:

- multi-view contrastive losses around same-world evidence
- representation pressure that says "same env, different probe -> similar
  belief"
- less emphasis on generic compression and more emphasis on sufficiency
- anti-collapse geometry pressure so the belief cloud keeps usable global scale
  instead of turning into a tiny but decodable codebook

What not to over-import:

- contrastive objectives are not magic by themselves
- if the positive pairs are badly chosen, the model can still learn the wrong
  invariances

Repo consequence:

- use same-env positives
- use probe-mode-invariant multi-view losses
- if the belief cloud collapses globally, add explicit repulsion or uniformity
  pressure instead of only tuning PCA plots or uncertainty heads
- do not mistake a low reconstruction loss for a good env belief

## 4. DreamerV3

Paper:

- DreamerV3

Core contribution:

- robust world-model learning that supports strong control across many domains
- shows that a latent can be highly useful when tied directly to predictive
  modeling and policy learning

Most relevant lesson for this repo:

- representations matter most when they support useful prediction and
  decision-making, not just clustering
- the latent should support counterfactuals and imagined consequences

What to import into code:

- predictive consistency pressures
- belief usefulness judged partly by rollout or outcome prediction
- tighter coupling between representation and downstream use

What not to over-import:

- we are not trying to rewrite the repo into Dreamer
- the main lesson is not "replace PPO with imagination tomorrow"
- the main lesson is "a belief should earn its keep by supporting useful
  prediction"

Repo consequence:

- include predictive losses that tie belief to controllable futures
- judge the belief partly by whether it makes task learning easier

## 5. TD-MPC2

Paper:

- TD-MPC2

Core contribution:

- strong latent world models can scale to broad continuous-control settings
- representation quality matters for downstream control and planning

Most relevant lesson for this repo:

- the latent should be organized well enough that downstream decision-making
  can trust it
- geometry and predictive structure both matter

What to import into code:

- stronger concern for latent geometry
- latent should support control-relevant comparisons and counterfactuals
- world-model quality is part of the controller story, not separate from it

What not to over-import:

- TD-MPC2 is not evidence that any low-dimensional latent is fine
- a decodable latent is not automatically a good metric space

Repo consequence:

- we should explicitly care about local geometry
- nearest-neighbor latent behavior should mean something

## 6. Building Minimal and Reusable Causal State Abstractions

Paper:

- Building Minimal and Reusable Causal State Abstractions for Reinforcement
  Learning

Core contribution:

- argues for minimal causal abstractions instead of large entangled state
  summaries
- reusable abstractions should preserve the causes that matter for decisions

Most relevant lesson for this repo:

- the target latent is not "everything compressed"
- it is "just enough of the hidden world to support prediction and control"

What to import into code:

- pressure toward minimal mechanics belief
- skepticism toward rich probe-shaped embeddings
- evaluation based on causal usefulness, not only predictive correlation

What not to over-import:

- "causal" does not mean the repo will suddenly discover formal causal graphs
- we are using the idea operationally: hidden mechanics that govern outcomes

Repo consequence:

- separate local probe evidence from env-level belief
- penalize nuisance structure like probe identity when it leaks into `z_env`

## 7. ASID

Paper:

- Active Exploration for System Identification

Core contribution:

- the best exploration for system ID is not arbitrary exploration
- it is action selection explicitly optimized to improve model or parameter
  inference

Most relevant lesson for this repo:

- the crawler should be an experiment designer
- probe policy quality should be measured by how much it improves env belief

What to import into code:

- active probe scoring based on uncertainty reduction
- disagreement or identification-value objectives
- stopping conditions based on belief sharpness

What not to over-import:

- not every disagreement metric is good system ID
- random novelty bonuses alone are not enough

Repo consequence:

- treat crawler behavior as its own research object
- compare scripted probes against active ones in terms of belief quality

## 8. Dynamics as Prompts

Paper:

- Dynamics as Prompts

Core contribution:

- past interaction history can act as prompt-like context for fast adaptation
- the agent can use history as an adaptation substrate rather than relearning
  from scratch

Most relevant lesson for this repo:

- the controller should use the belief as contextual conditioning
- history-derived env belief should be refined online when needed

What to import into code:

- contextual downstream control
- online belief refinement
- support for fast transition from probing to task execution

What not to over-import:

- prompt-like context is not itself a guarantee of good structure
- the history still needs the right inductive biases and losses

Repo consequence:

- `state + z_env + u_env` should be the main interface to the controller

## 9. Building Machines That Learn and Think Like People

Paper:

- Building Machines That Learn and Think Like People

Core contribution:

- human-like sample efficiency comes from stronger structure:
  compositionality, causality, intuitive theories, and learning-to-learn

Most relevant lesson for this repo:

- our problem is not mainly "how do we use less data with the same objectives"
- it is "what priors and structure let a small amount of interaction teach the
  right things"

What to import into code:

- explicit separation of exploration and control
- stronger emphasis on causal, abstract, reusable world knowledge
- less tolerance for representation drift toward superficial rollout features

What not to over-import:

- this is not an algorithm paper
- it provides principles, not a drop-in implementation recipe

Repo consequence:

- the repo should optimize for structured understanding, not just compressed
  experience

## 10. PLATO / Intuitive Physics Learning

Paper:

- Intuitive physics learning in a deep-learning model inspired by developmental
  psychology

Core contribution:

- strong inductive biases and object-structured reasoning can lead to more
  human-like learning of physics

Most relevant lesson for this repo:

- if we want child-like rapid physics learning, the belief must reflect hidden
  mechanics, not only policy-relevant correlations
- representation quality should be judged against interpretable mechanics

What to import into code:

- per-parameter or per-mechanic diagnostics
- belief representations that can be inspected for physics content
- healthy skepticism about purely black-box latent spaces

What not to over-import:

- our current environments do not have rich object-centric structure yet
- this paper is more inspiration for design principles than a direct template

Repo consequence:

- always ask whether the latent really tracks mechanics
- keep the dashboard grounded in mechanics readouts, not only geometry plots

## 11. TurboQuant

Paper:

- TurboQuant

Source:

- https://arxiv.org/abs/2504.19874

Core contribution:

- treats vector compression as a rate-distortion problem rather than an ad hoc
  storage trick
- emphasizes online, data-oblivious compression
- uses random rotation plus simple scalar quantization to preserve useful
  geometry efficiently
- for similarity-heavy use cases, uses a coarse code plus residual sketch
  rather than assuming mean-squared error is the only quantity that matters

Most relevant lesson for this repo:

- the env belief should be treated as a communication object between crawler
  and solver
- we should ask how many bits are needed to preserve useful world knowledge
- we should not assume that "good latent" and "good compressed latent" are the
  same evaluation problem

What to import into code:

- explicit rate-limited bottlenecks for `z_env`
- rate-distortion evaluation curves for belief quality
- separate tests for:
  - mechanics decode under compression
  - geometry preservation under compression
  - downstream control under compression
- possible coarse belief plus residual belief sketch designs

What not to over-import:

- TurboQuant does not tell us how to learn the right semantics in the first
  place
- compressing a bad latent only preserves the wrong thing efficiently
- quantization quality is not a substitute for mechanics alignment or
  uncertainty calibration

Repo consequence:

- evaluate `z_env` as a minimal message from crawler to solver
- add belief communication experiments, not just bigger latent experiments
- use compression as a diagnostic for sufficiency, not just as an efficiency
  trick

## What The Papers Say Together

When combined, the eleven papers point toward this design:

- an active crawler
- a local evidence encoder
- an env-level belief over hidden mechanics
- calibrated uncertainty
- a rate-limited belief message that preserves what the solver actually needs
- downstream control conditioned on that belief
- evaluation based on mechanics capture, stability, and adaptation speed

They do not point toward:

- one giant opaque latent
- probe-script memorization
- optimizing whichever dashboard scalar improved this week
- letting PPO hide representation failure
- assuming a large latent is automatically better than a compressed sufficient
  one

## The Most Important Practical Translation

The most important translation from paper to code is this:

The env belief is the real research object.

Everything else is upstream or downstream of that.

- the crawler exists to make the env belief better
- the window encoder exists to produce good evidence for the env belief
- the controller exists to exploit the env belief
- the dashboard exists to inspect the env belief

If a code change improves return but makes the env belief less mechanical, less
stable, or less interpretable, then it is moving away from the core idea.

## Current Implementation Translation

The latest CartPole artifacts point to a few concrete paper-backed rules:

- ASID: active identification is not just "more probes"; saved support windows
  should be intervention-balanced. A stable center probe can be useful, but it
  should not dominate the evidence used to infer mechanics.
- ContraBAR: same-world positives are only meaningful when the two views come
  from genuinely different probe families. Track split retrieval alongside
  effective support-family count before increasing contrastive pressure.
- Causal state abstraction: punish probe identity only after measuring where it
  leaks. `window_mode_leakage` and `env_mode_leakage` answer different
  questions and should not be collapsed into one score.
- Dynamics-as-prompts and PEARL: a context message that is unused or harmful is
  not yet a useful belief. Downstream wins should be paired with matched
  zero/shuffled/stale/no-expression ablations.
- TurboQuant: communication diagnostics are meaningful only after the message
  clears the utility gate. Rate-distortion curves should not distract from a
  negative env-expression delta.

## The Import Rules

When future agents use these papers, they should follow these rules:

- import principles, not paper cargo-cult
- tie every new loss to one concrete failure mode in the current belief
- do not add a metric unless it changes a real research decision
- do not let downstream control gains substitute for belief quality
- always ask what the new change says about hidden mechanics inference

That is the standard these papers set for this repo.
