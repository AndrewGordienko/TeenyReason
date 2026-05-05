# Value-Aware Crawler Strategy

This file states the current research bet and the next engineering plan.

The short version:

the novel object should be the crawler, but not merely as a probe collector.
The crawler should spend interaction budget like an experimental scientist,
build a compact causal belief, and hand that belief to arbitrary solvers only
when the belief has positive measured value.

The project should not optimize for attractive latents. It should optimize for:

```text
value gained from belief > cost to acquire belief
```

That is the performance thesis.

## Current Empirical State

The latest four-domain suite says the crawler can often discover hidden
structure, but the solvers usually fail to use it efficiently.

| Domain | Discovery | Solver Use | Current Loss |
| --- | ---: | ---: | --- |
| CartPole | causal decode 1.00, counterfactual 1.00 | transfer 0.00 | probe/sample economics |
| Language | causal score 0.921 | transfer 0.0024, solver gain -0.0365 BPC | weak LM handoff |
| Image | causal decode 1.00 | transfer 0.0040, solver gain -0.0342 | belief too wide |
| Board | world score 0.838 | transfer 1.00 | only end-to-end pass |

Important concrete numbers from
`artifacts/four_domain_belief_suite_20260503_160719.json`:

- CartPole baseline PPO peaks and solves at 12,536 env steps.
- CartPole crawler/probe PPO peaks and solves at 148,136 env steps.
- CartPole crawler/probe loss versus baseline: -135,600 env steps.
- CartPole baseline and crawler both reach peak return 500.
- Language baseline and gated belief both peak at 200k chars.
- Language best local belief gain: +0.0347 BPC at 50k chars.
- Image baseline and gated belief both reach 0.90 accuracy at 4,096 labels.
- Image best local belief gain: +0.0466 accuracy at 256 labels.
- Board learned best-move accuracy: 1.0 versus 0.5556 on the strongest
  hidden-rule row.

Interpretation:

the crawler is not mainly blocked on seeing structure. It is blocked on turning
that structure into cheap solver value.

The V-JEPA 2 review in
[`research/world_model_handoff_review.md`](./research/world_model_handoff_review.md)
sharpens that interpretation: the current handoff is too much like "give PPO a
latent side channel" and not enough like "let the solver query an
action-conditioned world model."

## The Research Bet

Most nearby systems do one of these:

- train a policy directly from reward
- learn a latent from passive data
- learn a world model and plan inside it
- condition a solver on history or context

This repo's sharper claim is:

```text
active causal experiments -> compact world belief -> measured solver utility
```

The crawler's central question should be:

```text
What one experiment most improves the next solver decision per sample?
```

not:

```text
What probe gives the representation more evidence?
```

That is the eureka candidate. The crawler should learn which latent facts are
worth knowing, not just which facts are true.

## Shared Handoff Contract

Every domain should expose the same solver-facing belief contract:

- `belief.vector`
- `belief.confidence`
- `belief.uncertainty`
- `belief.cost`
- `belief.hidden_target`
- `belief.intervention_coverage`
- `belief.counterfactual_score`
- `belief.compression_bits`
- `belief.expected_solver_utility`

Every solver arm should support the same evaluation modes:

- correct belief
- zero belief
- shuffled belief
- stale belief
- compressed belief
- cheap belief
- cheap belief plus expensive fallback

The crawler remains general. Domain adapters define observation format,
intervention format, counterfactual scoring, and solver consumption.

Success gate:

no domain can claim that the latent helps unless correct belief beats
zero/shuffled/stale and has positive gain per sample.

## Next Implementation Plan

1. Add a shared belief handoff contract.

   This is the source-of-truth structure for the dashboard and artifacts. It
   records belief fields, ablation modes, economics, fallback accounting, and
   failure reasons.

2. Fix solver handoff before adding more crawler complexity.

   The current data says discovery is ahead of solver use. CartPole,
   language, and image need better belief consumers before more probes can pay
   off.

3. Add rate-distortion / compression curves.

   For each domain, measure useful belief at:

   - 8 bits
   - 16 bits
   - 32 bits
   - 64 bits
   - 128 bits
   - 256 bits
   - 512 bits

   Track causal lift per bit, solver gain per bit, sample cost per useful bit,
   transfer retained after compression, and ablation gaps at each bitrate.

   Success gate:

   compressed belief should keep at least 80% of causal utility while improving
   gain per bit.

4. Add cheap-first, expensive-fallback crawler policy.

   Expensive probes should wake only when cheap belief is uncertain or
   decision-sensitive.

   Wake-up conditions:

   - cheap confidence low
   - predicted action/value delta high
   - correct-vs-shuffled solver gap high
   - near instability or high uncertainty
   - bridge-to-real gap high
   - expected fallback ROI positive

   Success gate:

   expensive fallback must report positive net sample savings versus baseline,
   not just positive diagnostic ROI.

5. Repair domains in this order.

   Image compression curve first. It is the fastest way to test whether the
   visual belief contains compact useful factors or just excess features.

   Language adapter-vs-prefix second. Prefix conditioning is currently too
   weak; adapter, FiLM, or gated residual conditioning should be tested against
   prefix conditioning with zero/shuffled/stale controls.

   CartPole cheap-context PPO third. Use cheap mechanics context as the main
   input arm, with expensive fallback only when uncertainty or action
   sensitivity says it is worth paying.

   Board larger rule-space fourth. Board already works; scale it so the success
   is not only a tiny solved-game artifact.

## Dashboard Requirements

The dashboard should answer these questions directly:

- Did belief decode hidden factors?
- Did belief predict counterfactuals?
- Did belief beat zero/shuffled/stale?
- Did belief transfer into solver performance?
- How many samples to peak?
- How many samples to solve?
- What was probe cost?
- What was net sample savings?
- What bitrate was needed?
- Was expensive fallback worth it?
- What is the next bottleneck label?

No claim should depend on intuition. Every decision should have a row in the
artifact and dashboard.

## Target Table

The target is to flip the current table.

| Domain | Current State | Target |
| --- | --- | --- |
| CartPole | understands but slower than PPO | solves faster than PPO after probe cost |
| Language | causal signal but worse BPC | belief-conditioned LM beats baseline and ablations |
| Image | huge belief, tiny lift | compressed belief improves accuracy per bit |
| Board | works | still works in larger rule space |

That is the shortest path from the current repo to a general crawler library:
a crawler that builds useful world belief and hands it to different algorithms
without hardcoding the core around one environment.
