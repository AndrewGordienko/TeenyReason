# Human Learning Design

This file converts the developmental-science and neuroscience synthesis into
design guidance for the repo.

The question is not:

- "How do we imitate babies literally?"

The question is:

- "Which computational commitments from human learning are strong enough and
  concrete enough to guide architecture and training choices here?"

## The Design Translation

The human literature suggests five algorithmic ingredients:

1. structured priors
2. passive regularity extraction
3. surprise-sensitive exploration
4. causal intervention and informative-evidence testing
5. predictive memory for reuse

In repo language, that becomes:

- `domain priors`
- `evidence windows`
- `surprise-aware crawler policy`
- `env belief update`
- `solver conditioned on predictive belief`

## 1. Structured Priors In Code

We should represent the world with factorized latent structure whenever the
domain supports it.

Examples:

- physical control:
  - hidden mechanics
  - controllability / affordances
  - safety / brittleness
- language:
  - syntax
  - lexicon
  - discourse state
  - communicative intent
- images:
  - objects
  - parts
  - scene relations
  - nuisance appearance

Implementation rule:

- prefer latent factorization that matches domain structure over one monolithic
  vector when that structure is concrete enough to justify

## 2. Passive Evidence Is Part Of The Crawler

Human learners get a lot from watching before acting.

So the crawler should support:

- passive windows
- low-intervention decay windows
- action-rich intervention windows
- later: social or demonstration windows

Implementation rule:

- do not benchmark only the most aggressive interventions
- evaluate what the belief can infer from a mix of passive and active evidence

## 3. Surprise Should Change What Happens Next

Surprise is not just a metric; it is a control signal.

The crawler policy should:

- predict what should happen under the current belief
- compare that against what actually happened
- allocate the next probe budget toward unresolved hypotheses
- value evidence that is especially diagnostic, not only evidence that is
  surprising in the abstract

Implementation rule:

- add explicit surprise or prediction-error bookkeeping at the experiment level
- make later probes conditional on which mechanics hypotheses remain confused

CartPole example:

- if passive decay already identifies length well but not force scale, choose
  probe families that stress actuation and recoverability

Language example:

- if the current belief is uncertain about word order but confident about local
  morphology, seek longer syntactic contexts instead of more token frequency

Image example:

- if object identity is stable but 3D shape or part relations remain uncertain,
  prefer additional views or part-sensitive crops

## 4. Causal Testing Beats Repetition

Children do not only gather more data. They gather more *useful* data.

Implementation rule:

- the crawler should track which hypothesis family each experiment is testing
- support sets should contain distinct experiment families where possible
- evaluation should punish probe collapse even if downstream PPO still wins

Current CartPole meaning:

- `passive_decay`, `impulse_left`, `impulse_right`, `chirp`, `boundary_push`,
  and `cart_brake` are not merely labels
- they are candidate causal tests for different mechanics variables

What to add over time:

- per-goal information gain estimates
- explicit "which hidden variable did this probe disambiguate?" diagnostics
- probe scheduling that uses expected belief improvement, not just current
  disagreement

Current translation in the repo:

- family-conditioned future-probe predictors estimate what each probe family is
  likely to reveal next
- the env belief now updates sequentially with family-conditioned evidence
  rather than only averaging a set of window embeddings
- the crawler scores each family by predicted mechanics gain, predictive gain,
  split-mismatch reduction, posterior entropy reduction, explicit
  hypothesis-separation value, and expected value per probe step
- fair mode stops early only after the belief is sharp enough and the predicted
  next-family gain is low enough relative to its expected interaction cost
- benchmark stop-reason summaries should count one final crawler stop decision
  per run, not every intermediate episode-level stop event

## 5. Predictive Maps, Not Static Codes

The belief should help answer:

- what futures are reachable here?
- which actions will reveal the world fastest?
- what transitions should be expected under this hypothesis?

Implementation rule:

- preserve local geometry only when it reflects predictive similarity
- if one latent cannot simultaneously support forecasting and good geometry,
  split it into a predictive belief plus a metric projector rather than forcing
  one vector to do every job
- use retrieval and neighbor metrics that are tied to hidden mechanics or
  successor-like prediction
- do not let the world-belief collapse into a tiny codebook if the agent is
  supposed to reuse it as a predictive map
- inspect the raw pre-normalization belief as the main predictive-map object;
  normalized copies are useful for comparison losses but should not be confused
  with the whole belief

This is why the repo should care about:

- split retrieval
- neighbor alignment
- world-consistent local geometry
- non-collapsed belief scale

and not only:

- PCA neatness
- raw latent variance coverage

## 6. Honest Uncertainty

Human learners are not merely low-data; they are selectively uncertain.

The repo should therefore treat uncertainty as a learned operational variable
with three jobs:

1. tell the crawler where another experiment would help
2. tell the dashboard where the belief is fragile
3. tell the solver how much trust to place in the belief

Implementation rule:

- uncertainty should be calibrated against actual mechanics prediction error
- disjoint-support disagreement matters more than overlapping-subset agreement
- leave-one-goal-out degradation matters more than decorative posterior spread
- probe budgets should be allowed to stop early when uncertainty is already low
  enough for the downstream solver
- uncertainty should lean more heavily on predictive failure than on pretty
  geometry numbers when the two disagree

Failure mode to avoid:

- an overconfident env belief that looks stable only because support sets are
  overlapping or too homogeneous

## 7. Domain-General Schema

The same outer algorithm should work across domains:

1. gather evidence
2. update latent world belief
3. estimate uncertainty
4. choose more evidence or hand off to solver
5. refine belief online if needed

What changes by domain is not the outer loop but:

- the evidence format
- the priors
- the probe families
- the downstream task head

That is the core reason this repo should not shrink back into "CartPole probe
tricks." CartPole is only the current easiest laboratory for the broader idea.

## 8. Concrete Repo Implications

Near-term code decisions should follow these rules:

- prefer few-shot diverse support over many repeated comfortable probes
- prefer belief metrics tied to hidden-world retrieval and prediction over
  aesthetic latent plots
- prefer surprise-driven follow-up probes over static probe questionnaires
- prefer calibrated uncertainty over deterministic confidence theater
- prefer domain-specific factorization when the domain genuinely supports it
- prefer solver-facing messages that can survive compression over large opaque
  belief vectors that only work because the controller sees everything

## 9. What To Ask Before Adding New Code

Every substantial edit should be checked against the human-learning frame:

1. Does this improve structured priors?
2. Does this help the crawler learn from both passive and active evidence?
3. Does this make surprise operational?
4. Does this improve causal disambiguation instead of repeated probes?
5. Does this improve predictive reuse?
6. Does this make uncertainty more honest?

If the answer is "no" to most of these, the change is probably benchmark-local
or metric-local rather than thesis-aligned.

## 10. The Long-Term Target

The long-term target is not one special latent for CartPole.

It is a domain-general world-belief system that:

- enters a new environment
- uses a small amount of structured evidence to infer the hidden rules
- knows what it still does not know
- and lets a downstream learner solve tasks quickly because the environment
  itself is already partly understood

That is the strongest child-inspired interpretation of what this repo is trying
to become.
