# Language Belief Design

This file is the design companion to
[language_domain_synthesis.md](./language_domain_synthesis.md).

That file explains what the language papers imply.

This file answers the follow-up question:

if we took those implications seriously, what would the language-side version
of the crawler-belief-solver architecture actually look like?

## The Language Version Of The Latent

The language analogue of the repo's env belief should not be one undifferentiated
vector called "language understanding."

It should be thought of as a structured belief with parts like:

- `z_syntax`
  Belief over grammatical and compositional rules.
- `z_lexicon`
  Belief over how local word forms map to latent meanings and roles.
- `z_discourse`
  Belief over entities, coreference, topic flow, and narrative state.
- `z_world`
  Belief over the local world model implied by the text.
- `z_style`
  Belief over authorial or corpus-specific stylistic regularities.
- `u_lang`
  What is still ambiguous or underdetermined.

Mapped back to the repo's general notation:

- `z_rules`
  roughly contains syntax and stable discourse constraints
- `z_entities`
  roughly contains lexicon plus discourse entity state
- `z_affordances`
  means what kinds of continuations, questions, or transformations are valid
- `u`
  tracks unresolved hypotheses

The key idea is:

language, like physics, has hidden structure that should be inferred from
multiple local pieces of evidence.

## The Language Crawler

The language version of the crawler is the most important design question.

It should not just ingest tokens left-to-right and call that exploration.

The crawler in a language environment should choose informative tests.

### Passive text crawler

For a fixed book or corpus, the crawler can choose:

- which chapters to sample first
- which spans to mask
- which paraphrases or reorderings to test
- which entity links to verify
- which continuations are most informative
- which compression hypotheses best explain the text

This is active inference over passive data.

### Interactive text crawler

For dialogue or tool use, the crawler can also:

- ask clarification questions
- propose alternate phrasings
- test assumptions and watch corrections
- query tools to ground ambiguous references
- explore which linguistic actions produce stable effects

This is much closer to active system identification in RL.

## Concrete Language Probes

If we ever build the language side in code, these are the kinds of probes that
best match the repo thesis.

### Syntax probes

- masked function-word prediction
- clause reordering checks
- agreement checks
- attachment ambiguity tests

### Lexicon probes

- word substitution tests
- synonym / antonym discrimination
- definition induction from local context
- role consistency across mentions

### Discourse probes

- coreference tracking
- entity state change prediction
- next-event prediction
- missing-bridge sentence selection

### Compression probes

- choose the shortest explanatory representation for a span
- compare alternate segmentations
- predict which summary or parse gives best future compression

### Counterfactual probes

- change one entity or relation and predict downstream consequences
- replace one phrase with another and test whether discourse remains coherent

These are language analogues of physical interventions.

## The Solver In Language

The language-side solver does not have to be PPO specifically.

The broader solver could be:

- a generator
- a parser
- a QA policy
- a reasoning policy
- a tool-using agent

The shared idea is that the solver receives a structured belief and solves a
task more efficiently because it does not have to infer the rules from scratch.

That is the direct analogue of:

first understand the pendulum, then balance it.

For language the analogue is:

first infer the local grammar, lexicon, discourse, and world assumptions, then
generate, answer, reason, or act.

## The General Cross-Domain Thesis

This repo should therefore treat language and physical control as two instances
of the same broader pattern.

The latent mechanism should be able to support:

- physical dynamics discovery
- language-structure discovery
- other symbolic or interactive domains

That does not mean the same observation encoder should be reused blindly.

It means the same architecture pattern should survive:

1. local evidence windows
2. belief updates over hidden rules
3. calibrated uncertainty
4. downstream solver conditioned on belief

## What We Should Import Into Code

If future agents extend the repo toward a more domain-general system, these are
the ideas worth importing.

### 1. Domain-agnostic interfaces

We should preserve clean abstractions like:

- evidence window
- belief update
- uncertainty estimate
- predictor heads
- solver interface

These abstractions should not be tied to continuous control tensors only.

### 2. Active evidence selection

The crawler should be able to score candidate interventions in any domain by:

- expected uncertainty reduction
- expected improvement in rule prediction
- expected improvement in subset agreement

### 3. Structured latent factorization

Avoid one monolithic latent.

At minimum, preserve a conceptual split between:

- rules
- local state
- nuisance surface features
- uncertainty

### 4. Compression-aware objectives

A good belief is a compact explanation of observations.

Compression and predictive sufficiency should be part of the design, not just
side metaphors.

### 5. Explicit grounding boundary

The repo should clearly distinguish:

- text-only belief learning
- grounded interactive belief learning

This is necessary to stay honest about what the system can and cannot infer.

## A Concrete Language Track For This Repo

If we actually wanted to prototype the language side in this repo, the first
goal should not be "build a better text generator."

The first goal should be:

build a language environment-belief testbed that lets us ask whether a small
amount of text can produce a stable, uncertainty-aware belief about the hidden
structure of a corpus or language-like environment.

### Phase 1. Passive one-book or few-book environment

Keep the environment deliberately simple.

Environment examples:

- one book
- one author across a few short books
- one synthetic corpus with known grammar variations
- one TinyStories-like controlled corpus

Hidden variables to infer:

- grammar family
- lexical conventions
- discourse regularities
- style parameters

Solver tasks:

- continuation
- cloze
- next-paragraph ranking
- entity tracking

Why this phase matters:

It isolates whether the belief can learn local language structure in a
small-data setting without dragging in full web-scale semantics.

### Phase 2. Multi-view corpus belief

Move from text windows to corpus-level beliefs.

Design:

- local span encoder produces `z_window`
- corpus or episode aggregator produces `z_env`
- different spans from the same book should agree on the shared structure
- beliefs from different books or grammars should separate

This is the exact analogue of:

- local probe window
- env-level belief

on the physical RL side.

### Phase 3. Active reading crawler

Stop consuming text in a fixed order only.

Give the crawler choices like:

- which chapter to read next
- which spans to mask
- which candidate continuations to evaluate
- which coreference ambiguities to resolve
- which compressions best explain the corpus

The crawler reward should be something like:

- subset agreement improvement
- grammar prediction improvement
- uncertainty reduction
- downstream task improvement after a fixed evidence budget

### Phase 4. Interactive language environment

Once the passive setup is stable, add an environment where actions affect what
comes back.

Examples:

- a dialogue partner with hidden conventions
- a text game with hidden world rules
- tool-use tasks with hidden schemas or APIs

At this point, the language side becomes a very close analogue of active system
identification in control.

### Phase 5. Shared cross-domain core

Only after the language setup is real should we push for shared modules.

The likely shared core is:

- evidence window data structure
- belief aggregator interface
- uncertainty interface
- predictor head interface
- solver-conditioning interface

The likely domain-specific adapters are:

- physical probe data collection
- text span or dialogue data collection
- observation encoders
- domain-specific predictor heads

This is the boring, maintainable way to make the repo more general.

## Candidate Module Boundaries

If we eventually implement the language track, the code should probably stay
mostly flat and explicit.

One plausible direction is:

- `teenyreason/domains/control/`
  Existing physical environment adapters and probe routines.
- `teenyreason/domains/language/`
  Text or dialogue environment adapters.
- `teenyreason/core/evidence.py`
  Domain-agnostic evidence window and grouping types.
- `teenyreason/core/belief.py`
  Domain-agnostic belief and uncertainty interfaces.
- `teenyreason/core/solver.py`
  Shared solver-conditioning interfaces.

Important note:

This is not a command to refactor the repo immediately. It is a direction for
future generalization if and when the current code earns it.

## Language-Side Metrics That Would Actually Matter

If we build a language belief testbed, the metrics should mirror the physical
ones but stay honest about text.

### Belief metrics

- same-book subset agreement
- different-book separation
- grammar decode quality
- discourse-state prediction quality
- uncertainty calibration against held-out structure prediction error

### Solver metrics

- few-shot continuation quality
- few-shot QA or cloze accuracy
- adaptation after limited reading budget
- robustness under style shift

### Crawler metrics

- uncertainty reduction per queried span
- grammar or discourse decode improvement per read
- value of active read order versus passive read order

### Non-metrics

Things we should avoid overreading:

- single perplexity values with no belief analysis
- generic generation quality that may be mostly style imitation
- benchmark wins that do not isolate what the belief contributed

## The One-Book Thought Experiment

The "give aliens one Harry Potter book" intuition is useful, but it needs to be
made precise or it becomes hand-wavy.

The clean version is:

- what priors did the aliens already have?
- do they already understand that symbols compose?
- do they already model agents, objects, and events?
- are they grounded in a world at all?
- are they allowed to ask questions or only read passively?

If the answer is "strong priors, good abstraction machinery, and active
hypothesis-testing," then one book might teach them a great deal about:

- local grammar
- recurring symbols
- discourse organization
- style

If the answer is "blank learner with no grounding and no structural prior,"
then one book is radically underdetermined.

That is the main bridge between the language question and the RL question.

In both cases, fast learning comes from:

- good priors
- good interventions
- good abstractions

not from magic.

## What Not To Do

Do not let the language analogy degenerate into any of the following:

- "one vector that contains all of English"
- "token prediction equals understanding"
- "one book is enough for fully grounded meaning"
- "style imitation proves semantic competence"
- "bigger model equals better latent"

Those are exactly the kinds of shortcuts the paper set should warn us away
from.

## Practical Guidance For Future Agents

If you are writing future code after reading this file, keep these rules in
mind.

### Rule 1

Do not build language support by jamming text into the existing physical probe
stack with only naming changes.

### Rule 2

Do build language support by preserving the same abstract phases:

- gather evidence
- infer hidden rules
- quantify uncertainty
- solve downstream tasks

### Rule 3

When evaluating a language belief, separate:

- grammar acquisition
- discourse tracking
- style modeling
- grounded meaning

Do not treat one score as covering all of them.

### Rule 4

If a language-side benchmark improves, ask whether the improvement came from:

- a better belief over hidden structure
- easier data
- better optimization
- a solver that simply learned to compensate

That is the same discipline we already need on the RL side.

## Bottom Line

The language papers do not say that one small corpus is enough for complete
general intelligence.

What they do say, taken together, is more useful:

- strong priors matter
- curated evidence matters
- small-data regimes are worth studying
- compression is a powerful lens
- prediction alone is not meaning
- a general world-model style architecture can span domains

That is exactly the shape of the broader repo thesis.

We are trying to build a general crawler-belief mechanism that can enter a new
domain, discover its hidden constraints quickly, and give a downstream solver a
head start.
