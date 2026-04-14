# Language Domain Synthesis

This file explains how the language-side papers connect to the repo's broader
environment-belief thesis.

For the concrete language-side architecture, module boundaries, metrics, and
prototype plan, read [language_belief_design.md](./language_belief_design.md)
after this file.

The point is not to turn the repo into a language model project tomorrow.

The point is to make one deeper idea precise:

the latent mechanism we are building for physical control should, in principle,
generalize to language and other structured environments.

This file therefore answers four questions:

1. What do the language papers actually imply?
2. What can a small amount of text teach, and what can it not teach?
3. What would the language version of the crawler-belief-solver architecture
   look like?
4. How should this influence future code and research decisions in this repo?

## The Core Language Question

The motivating intuition is:

if a child can learn a lot about language, grammar, and local reasoning from
far less data than a modern LLM sees, then maybe we are missing the right
latent structure and the right interaction pattern.

The strong version of the idea is not:

"one Harry Potter book should make a blank system understand all of English."

The stronger and more precise version is:

"with the right priors, abstractions, and active hypothesis-testing, a system
should be able to infer a great deal about the local generative rules of a new
linguistic environment from surprisingly little data."

That is directly analogous to the pendulum story:

- first infer the rules
- then solve the task

## What The Papers Say

This section uses the six user-provided papers and extracts the relevant lesson
for the repo.

### 1. Bender and Koller 2020

Paper:

- Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data

Source:

- https://aclanthology.org/2020.acl-main.463/

Key point:

- a system trained only on linguistic form has no a priori route to full
  meaning

Repo implication:

- we must not over-claim what a text-only crawler can infer
- a language belief learned only from text can plausibly capture:
  - syntax
  - local semantic regularities
  - discourse structure
  - style
  - latent world assumptions reflected in the corpus
- but it cannot be equated with fully grounded semantic understanding

Why this matters:

This paper is the guardrail that stops the repo from making empty claims like
"the latent understands language" just because it predicts tokens well.

### 2. BabyLM 2023

Paper:

- Findings of the BabyLM Challenge: Sample-Efficient Pretraining on
  Developmentally Plausible Corpora

Sources:

- https://aclanthology.org/2023.conll-babylm.1/
- https://www.research-collection.ethz.ch/entities/publication/97a3a491-c831-4ae6-bdb9-104da76d65a7

Key points:

- humans acquire language from less than 100M words
- the challenge sets small-data regimes like 10M and 100M words
- strong submissions used architecture, training objective, and data choices
  carefully rather than just scaling
- shorter input sequences and teacher-student setups helped
- curriculum learning was not a magic fix

Repo implication:

- small-data language learning is a valid research regime
- data efficiency should be treated as a first-class target
- curated evidence and better training signals matter as much as raw volume
- not every "cognitively inspired" trick pays off

Why this matters:

BabyLM supports the core intuition that brute-force scale is not the only game
in town. It does not prove one-book language mastery, but it does show that
there is meaningful headroom in small-data learning.

### 3. TinyStories 2023

Paper:

- TinyStories: How Small Can Language Models Be and Still Speak Coherent
  English?

Source:

- https://arxiv.org/abs/2305.07759

Key points:

- a highly simplified synthetic corpus can train very small models to produce
  coherent, grammatical, multi-paragraph text
- models below 10M parameters or even with only one transformer block can
  learn surprisingly strong fluency on that curated domain
- grammar and basic reasoning emerge in a narrow, controlled regime

Repo implication:

- language competence is not all-or-nothing
- syntax, coherence, and local reasoning can be learned in small models when
  the environment is simplified and the data is curated
- a latent system should separate:
  - general grammar acquisition
  - domain-specific world knowledge
  - broad grounded semantics

Why this matters:

TinyStories is evidence that the barrier to language learning is partly about
objective and data regime, not just parameter count. It suggests that a
well-designed language crawler could learn some constraints from a small corpus
far faster than the usual web-scale setup implies.

### 4. Textbooks Are All You Need 2023

Paper:

- Textbooks Are All You Need

Source:

- https://arxiv.org/abs/2306.11644

Key points:

- phi-1 achieved strong code performance from a relatively small model and a
  relatively small corpus by using textbook-quality web data plus synthetic
  textbooks and exercises
- the paper is about code, not general natural language
- the main lesson is that didactic, structured, explanatory data can be more
  valuable than huge noisy corpora

Repo implication:

- the crawler should prefer evidence that teaches rules cleanly
- high-quality explanatory sequences may be disproportionately valuable for
  latent belief formation
- synthetic exercises and targeted interventions are legitimate tools

Why this matters:

For the repo's broader thesis, this paper says:

good evidence is not just more evidence.

That is exactly the crawler intuition in another form.

### 5. Language Modeling Is Compression 2023

Paper:

- Language Modeling Is Compression

Source:

- https://arxiv.org/abs/2309.10668

Key points:

- predictive modeling and lossless compression are deeply linked
- large models can act as strong general-purpose compressors
- the compression view gives insight into scaling laws, tokenization, and
  in-context learning

Repo implication:

- the latent should be understood as a compact explanation of observations
- a belief is good when it makes future observations cheaper to describe
- compression is a useful lens for deciding what structure matters

Why this matters:

This gives the repo a principled way to think about the latent:

the belief should be the compact code that explains the environment's
constraints, not just another hidden vector optimized to make loss go down.

### 6. DreamerV3 2023

Paper:

- Mastering Diverse Domains through World Models

Source:

- https://arxiv.org/abs/2301.04104

Key points:

- a single world-model-based algorithm can work across many very different
  domains
- robust prediction and imagined futures make downstream control more general
- the model succeeds because the latent is tightly coupled to action and future
  consequence

Repo implication:

- the belief must be useful for prediction and downstream choice
- the crawler-belief-solver split is not just an RL trick; it is a candidate
  general AI architecture
- the language-side belief should also support counterfactual or alternative
  continuation reasoning, not just passive encoding

Why this matters:

DreamerV3 is the paper in this list that most strongly supports the idea that a
single latent-belief interface can span many domains if it is tied to
prediction and decision-making.

## What A Small Corpus Can And Cannot Teach

This section is the practical answer to the "one Harry Potter book" question.

### What a small corpus plausibly can teach

With the right priors and training setup, a small corpus can plausibly teach:

- local grammar
- common syntax patterns
- discourse transitions
- recurring entity structure
- style and tone
- local lexical regularities
- some bounded world assumptions visible in the text

This is the domain where the latent can become sharp quickly.

### What a small corpus probably cannot fully teach from form alone

A small text-only corpus is unlikely to fully teach:

- grounded reference
- broad open-world semantics
- sensorimotor meaning
- full pragmatic competence
- robust causal reasoning beyond what is implicit in the text

This is where Bender and Koller matters most.

The system can build a belief about the language-generating process without
necessarily achieving grounded understanding.

For the concrete language-side architecture, implementation phases, metrics, and
agent-writing rules, read [language_belief_design.md](./language_belief_design.md).
