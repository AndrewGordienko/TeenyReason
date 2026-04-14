# General Crawler Belief

This file extends the repo thesis beyond the current physical-control testbeds.

For the concrete staged plan, metrics, and cross-domain operating rules, read
[general_program_plan.md](./general_program_plan.md) after this file.

The current codebase is still mainly an RL research repo. However, the deeper
idea is broader:

we want a general mechanism that can enter an unfamiliar environment, run a
small number of informative experiments, infer the hidden rules of that
environment, quantify what is still uncertain, and pass that belief to a
downstream solver so the solver does not need to rediscover the rules from
scratch.

That broader environment may be:

- a physical control system
- a text corpus
- an interactive dialogue setting
- a codebase
- a game world
- any other structured process that generates observations under hidden rules

This document defines that more general program.

## The General Problem

The problem is not:

- "learn a better embedding"
- "pretrain a bigger model"
- "add context to PPO"

The problem is:

How do we build a small scientist-like module that can infer the hidden
structure of a new environment from limited interaction, and then hand that
structured belief to a task solver?

That means the same overall machine should work in at least two very different
regimes:

1. Physical environments
   Example: a pendulum with unknown mass, friction, delay, or torque limits.
2. Linguistic environments
   Example: a new corpus, dialogue partner, or symbolic language with unknown
   grammar, lexicon, discourse regularities, and world assumptions.

The hidden object is different in each domain, but the architecture should be
the same in spirit.

## The Canonical Abstraction

The generic environment is:

- hidden rules `h`
- current local state `s_t`
- observation `o_t`
- intervention `a_t`
- next observation `o_{t+1}`
- optional reward or external objective `r_t`

The agent has two roles:

1. `crawler`
   Gathers informative evidence about `h`.
2. `solver`
   Uses a belief about `h` to solve a downstream task efficiently.

The canonical data flow is:

1. observe the current environment
2. choose an intervention
3. collect a short evidence window
4. encode that window into local evidence
5. update a global belief over the environment
6. stop crawling when the belief is sharp enough
7. hand the belief to the solver

This is the central abstraction for the repo.

## Domain Mapping

The table below is the intended mental model.

| Concept | Physical Control | Language / Text | Interactive Language |
| --- | --- | --- | --- |
| hidden rules `h` | gravity, mass, friction, delay, contacts, actuation limits | grammar, lexicon, discourse regularities, stylistic conventions, latent world assumptions | same as text plus partner behavior, grounding rules, tool affordances |
| observation `o_t` | state vector, pixels, contacts | token stream, sentence, paragraph, document context | utterance, response, external feedback |
| action `a_t` | torque, force, control action | read next span, mask/reconstruct, reorder, query, continue, compress, paraphrase | ask question, clarify, propose action, use tool |
| evidence window | short rollout | short text span plus local transformations or predictions | short interaction segment |
| downstream solver | PPO, MPC, Dreamer actor | generator, parser, QA system, reasoning policy | dialogue / tool-using policy |

The important point is that the environment is not defined by being physical.
It is defined by having hidden regularities that can be discovered through
structured interaction.

## The Latent We Actually Want

The repo has been calling the main object an env belief. Generalizing that
across domains, the latent should be factored conceptually into:

- `z_rules`
  The hidden regularities or laws governing transitions and constraints.
- `z_entities`
  The latent objects, symbols, roles, or participants that persist across
  local windows.
- `z_affordances`
  What kinds of interventions are effective, safe, or meaningful here.
- `z_state`
  The current local situation inside the environment.
- `z_surface`
  Superficial presentation details that should not dominate the belief.
- `u`
  What remains unresolved and how uncertain the system is.

In the current physical codebase, `z_env` is mostly carrying pieces of
`z_rules` and `z_affordances`.

In a language setting, the analogous belief should carry:

- syntax constraints
- lexicon structure
- discourse state
- latent world assumptions
- style as a nuisance or auxiliary factor rather than the main belief target

The central design rule is:

the belief should represent what is shared across many pieces of evidence from
the same environment, not what is idiosyncratic to one probe window.

## The Crawler

The crawler is not just a prelude. It is a hypothesis-testing mechanism.

Its job is to choose interventions that are informative about hidden rules.

That means the crawler should optimize things like:

- posterior shrinkage
- disagreement reduction
- same-environment subset consistency
- mechanics or rule prediction improvement
- counterfactual prediction improvement

It should not be optimized for:

- aesthetic coverage
- arbitrary diversity for its own sake
- benchmark reward unless the benchmark reward directly measures information

### Physical Crawler

In physical control, the crawler should:

- perturb the system
- test control directions
- examine recovery behavior
- observe passive decay
- probe controllability boundaries

The pendulum example is the right intuition:

push left, push right, let it swing, see how quickly it damps out, test whether
small torques matter, test whether large torques overshoot, and use those
outcomes to infer the mechanics.

### Language Crawler

In language, the crawler looks different but the role is the same.

Possible crawler actions include:

- choose which passage to read next
- mask spans and predict them
- test reorderings
- compare paraphrases
- track entities across chapters
- ask cloze questions
- test coreference hypotheses
- estimate whether a transformation preserves grammaticality or meaning
- compress and reconstruct a passage

In an interactive linguistic environment, the crawler can also:

- ask clarifying questions
- make tentative statements and observe correction
- query tools or external memory
- test the consequences of alternate word choices

The general principle is:

the crawler should gather evidence that reduces uncertainty over the hidden
language-generating process, not merely consume tokens passively in one fixed
order.

## Passive, Active, and Hybrid Environments

Not every environment allows the same kind of interaction.

There are three broad cases.

### 1. Passive Environments

Example:

- a fixed book with no external feedback

The crawler cannot change the world. It can still choose:

- reading order
- masking strategy
- which hypotheses to test
- which spans to compare
- which predictions to score

In passive environments, intervention means active inference over fixed data.

### 2. Interactive Environments

Example:

- a pendulum
- a robot
- a dialogue partner
- a game

The crawler can act and observe consequences directly.

This is the cleanest setting for active system identification.

### 3. Hybrid Environments

Example:

- codebases
- tool-using assistants
- multimodal language agents

These allow some direct interventions and some passive observation.

The repo should be designed to accommodate all three.

## What The Solver Should Receive

The solver should not receive raw windows and try to rediscover the world.

The solver interface should conceptually be:

- local task state
- env-level belief
- env-level uncertainty

That is:

`solver_input = local_state + z_env + u_env`

The solver's job is then:

- choose actions for the task
- optionally request additional evidence if uncertainty is too high
- adapt online without starting from zero

This matters because it forces a clean separation:

- the crawler learns the world
- the solver solves the task

The solver is the consumer of the belief, not the producer of its meaning.

## The Losses That Matter

Across domains, the important losses are conceptually the same.

### Same-environment consistency

Different evidence subsets from the same world should produce similar beliefs.

This is the single most important invariance in the system.

### Between-environment separation

Different worlds should be separated, ideally in proportion to how different
their hidden rules are.

### Predictive sufficiency

Given belief and intervention, the model should predict relevant consequences:

- next states
- future tokens
- discourse updates
- reward-relevant outcomes
- error boundaries

### Nuisance invariance

The belief should not be dominated by superficial presentation details:

- probe mode
- writing style, if style is not the target
- rollout phase
- formatting artifacts

### Uncertainty calibration

When the belief says it is uncertain, actual rule prediction error should be
high.

When the belief says it is certain, it should usually be right.

### Compression or minimality

The belief should be compact enough that it keeps what matters and discards
irrelevant texture.

This does not mean "smallest possible vector." It means minimal sufficient
structure.

For the concrete artifact contract, staged implementation path, and metrics,
read [general_program_plan.md](./general_program_plan.md).
