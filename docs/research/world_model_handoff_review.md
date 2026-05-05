# World Model Handoff Review

This note connects the local paper library to the repo's current blocker:

the crawler can often build a meaningful latent, but the solver does not yet
use that latent with human-like sample efficiency.

New local paper:

- `papers/2506.09985v1.pdf`
- V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and
  Planning
- arXiv: https://arxiv.org/abs/2506.09985

Latest empirical artifact used for this read:

- `artifacts/four_domain_belief_suite_20260503_160719.json`

## Current Empirical State

| Domain | Peak/Solve Sample Read | Best Local Belief Read | Current Blocker |
| --- | --- | --- | --- |
| CartPole | baseline PPO peaks and solves at 12,536 env steps; crawler PPO peaks and solves at 148,136 env steps | same peak return 500, but -135,600 env-step loss | solver handoff cost dominates |
| Language | baseline and gated belief both peak at 200k chars | best local gain +0.0347 BPC at 50k chars | early gain does not survive scale |
| Image | baseline and gated belief both hit 0.90 acc at 4,096 labels | best local gain +0.0466 acc at 256 labels | low-budget lift is not converted into peak/solve savings |
| Board | baseline and belief solve at 2 queries | best belief gain +0.4444 move accuracy | works, but rule space is small |

The data says discovery is not the whole problem. We need a handoff where the
solver can ask the belief, "what happens if I do this?", instead of receiving a
side-channel vector and having to learn its meaning through PPO.

## How The Papers Construct World Models

| Paper Family | Construction Pattern | What It Learns |
| --- | --- | --- |
| PEARL / VariBAD | infer latent context or recurrent belief from interaction history | task or environment variable belief |
| ContraBAR | contrastive Bayes-adaptive representation from same-task evidence | task-relevant distinctions, not full reconstruction |
| DreamerV3 | recurrent latent dynamics with prediction, reward, continuation, and value learning | an imagination model for policy optimization |
| TD-MPC2 | latent dynamics and value model with control-aware geometry | a compact latent state useful for MPC-style control |
| ASID | choose actions that improve identification | a dynamics or parameter belief from active probes |
| Causal State Abstraction | compress state to decision-relevant causes | minimal reusable causal state |
| TurboQuant | rate-distortion compression of vectors | communication-efficient latent messages |
| V-JEPA 2 | self-supervised video encoder predicts masked future/target regions in representation space; action-conditioned model predicts future embeddings | an abstract video world model that ignores pixel-level nuisance detail |

The common construction theme is:

- do not model raw sensory detail unless the solver needs it
- predict in latent space
- keep the latent tied to future consequences
- separate the representation from the policy when possible
- add action-conditioning when the solver needs planning

V-JEPA 2 is especially relevant because it avoids pixel reconstruction as the
main target. It learns a strong video representation first, then freezes that
encoder and trains a smaller action-conditioned world model for robot planning.

## How They Plug Into Algorithms

| Paper Family | Solver Handoff | Why It Helps Performance |
| --- | --- | --- |
| PEARL | policy consumes latent context | fast adaptation because the policy sees inferred task identity |
| VariBAD | policy acts on recurrent belief | exploration and exploitation are Bayes-adaptive |
| DreamerV3 | actor/critic train inside imagined latent rollouts | many policy updates per real sample |
| TD-MPC2 | planner optimizes action sequences in latent model | action search uses learned dynamics directly |
| Dynamics as Prompts | policy conditions on recent dynamics history | history acts as fast adaptation context |
| V-JEPA 2 | model-predictive control searches action sequences using predicted future embeddings and a goal embedding distance | planning uses the world model directly, without training a new policy from reward |

This is the important difference from the current repo.

Our failing CartPole path mostly does:

```text
crawler probes -> belief vector -> PPO input
```

The stronger world-model pattern does:

```text
crawler probes -> belief/world model -> predict consequences of action candidates -> choose action
```

That second form is much closer to the pendulum intuition. A person does not
need thousands of reward-labeled trials because they can internally ask: "if I
push this way, what happens next?"

## What V-JEPA 2 Adds To The Repo Thesis

V-JEPA 2 supports three concrete changes to our mental model.

1. The solver-facing latent should be predictive, not just descriptive.

   A belief that decodes hidden factors can still be hard for PPO to use. The
   belief should expose a predictive interface: current latent plus candidate
   intervention gives next latent or outcome distribution.

2. The handoff can be a planner, not only a policy input.

   For control, the next fair comparison should include a small MPC/CEM arm:
   baseline PPO versus crawler-belief PPO versus crawler-belief latent MPC.
   If the crawler really understands mechanics, MPC should be the first place
   it pays off.

3. Action-conditioning is the missing bridge.

   Our current `belief.vector` says something about the world. It does not
   necessarily answer what action should happen next. V-JEPA 2 makes the bridge
   explicit by training an action-conditioned predictor on top of the frozen
   representation.

## Research Gaps In Existing Approaches

These papers do not already solve our exact problem.

V-JEPA 2 gaps:

- it uses massive passive video pretraining, so it is not proof of few-shot
  from-scratch environment learning
- the robot planner still depends on an action-conditioned model trained from
  robot trajectories
- long-horizon planning can suffer from compounding prediction error and action
  search explosion
- it does not make active experiment design the central object
- uncertainty and value-of-information are not the core interface

Dreamer / TD-MPC2 gaps:

- they can be sample efficient relative to model-free RL, but they still learn
  large internal world models from many environment interactions
- they do not isolate a small crawler belief message as the research object
- the model and solver are tightly coupled, which makes cross-domain handoff
  less clean

PEARL / VariBAD / ContraBAR gaps:

- they show latent adaptation, but not necessarily child-like mechanics
  understanding
- recurrent context can become a history shortcut unless constrained by
  counterfactual and causal tests
- the policy still has to learn how to exploit the latent

Our opportunity:

```text
active experiments -> compact predictive belief -> action-conditioned solver handoff
```

That is narrower than "build a huge world model" and stronger than "add a
latent side channel."

## What Should Be Generic Versus Domain-Specific

The crawler should be general at the level of contracts, not at the level of
raw observations.

Generic:

- evidence window schema
- belief update interface
- uncertainty and confidence fields
- query value accounting
- counterfactual prediction API
- rate-distortion accounting
- solver handoff modes and ablations
- sample-to-peak and sample-to-solve reporting

Domain-specific adapters:

- how to observe state or pixels or tokens
- which interventions are legal
- how to execute an intervention
- how to encode a local evidence window
- what counts as a hidden target
- how a solver consumes the belief
- what success threshold means

So CartPole-specific code is not wrong if it is an adapter. It is wrong only if
CartPole logic leaks into the crawler core. The current repo still feels a bit
off because the benchmark compatibility path has more CartPole-specific handoff
logic than the clean public API should eventually expose.

The clean shape should be:

```text
WorldAdapter        domain-specific
EvidenceEncoder     domain-specific or pretrained
BeliefBackend       generic interface, possibly shared implementation
QueryPolicy         generic value-of-information policy over adapter queries
LatentDynamicsModel generic interface
SolverConsumer      algorithm-specific, not crawler-core
```

## Next Implementation Strategy

1. Add a generic action-conditioned latent dynamics contract.

   Minimal interface:

   ```text
   encode_observation(o_t) -> z_t
   encode_goal(goal) -> z_goal
   predict(z_t, action_sequence, belief) -> z_future
   score(z_future, z_goal_or_task) -> scalar
   uncertainty(z_t, action_sequence, belief) -> scalar
   ```

2. Build a CartPole latent-MPC arm before more PPO handoff tuning.

   The crawler already has mechanics probes. Use them to train or calibrate a
   tiny action-conditioned transition model, then run MPC/CEM over action
   candidates. Compare:

   - baseline PPO
   - current crawler PPO
   - crawler belief plus latent MPC
   - no-belief latent MPC
   - oracle-mechanics MPC

3. Change crawler probe value to planner value.

   A probe should be worth buying if it improves future action choice:

   ```text
   expected reduction in planner action regret per probe step
   ```

   This is sharper than generic belief confidence.

4. Extend the same handoff shape to other domains.

   Board:
   belief plus action-value planner is already close to minimax.

   Image:
   action means crop, mask, view, transform, or label query. The latent model
   predicts how evidence changes class or object belief.

   Language:
   action means read span, mask, query, paraphrase, or ask. The latent model
   predicts how the belief changes next-token, entity, or rule decisions.

5. Add metrics that test the actual missing bridge.

   New required rows:

   - one-step latent prediction error
   - k-step latent prediction error
   - action-ranking accuracy versus oracle or rollout
   - planner regret with correct, zero, shuffled, and stale belief
   - samples to solve for latent-MPC versus PPO
   - probe value measured as action-regret reduction per sample

## Bottom Line

The next breakthrough is probably not a bigger latent.

The missing low-level idea is:

```text
make the crawler belief action-conditioned and planner-readable
```

In the current repo, the latent often knows something but the solver has to
learn how to use it through the normal training loop. V-JEPA 2, Dreamer, and
TD-MPC2 all suggest the same correction: the world model should be queried by
the solver for predicted consequences. That is the path that could plausibly
turn "I played with the pendulum for a few seconds" into "I can balance it in a
few tries."

## Implemented V0: Predictive CartPole MPC

Added a generic action-conditioned planning contract and a CartPole
belief-MPC arm:

```text
crawler probes -> mechanics belief -> action-conditioned rollout model
-> random shooting planner -> action
```

Smoke artifact:

- `artifacts/four_domain_belief_suite_20260503_164439.json`

64-seed check with `horizon=4`, `candidate_count=32`:

| Metric | Value |
| --- | ---: |
| decode accuracy | 1.0000 |
| no-belief MPC return | -7.6987 |
| belief MPC return | -6.4293 |
| oracle MPC return | -6.4293 |
| solver gain vs no belief | +1.2694 |
| content lift vs no/shuffled/stale | +0.2347 |
| oracle gap | 0.0000 |
| belief action match vs oracle | 1.0000 |
| no-belief action match vs oracle | 0.7584 |
| belief k-step prediction MSE | 0.0000 |
| no-belief k-step prediction MSE | 0.7690 |
| belief solve rate | 0.9844 |
| no-belief solve rate | 1.0000 |
| belief samples to peak return | 151.8594 |
| no-belief samples to peak return | 80.0000 |
| belief samples to solve | 152.0000 |
| no-belief samples to solve | 80.0000 |
| net samples-to-solve savings | -72.0000 |

Interpretation:

- The handoff now proves that a correct crawler belief can be planner-readable:
  it exactly matches the oracle mechanics planner and beats the no-belief model
  on return and action ranking.
- It still does not prove sample-efficiency victory. The probe costs 72
  environment steps, and the no-belief MPC already solves this controlled
  CartPole slice.
- The next performance blocker is not representation decoding. It is
  value-of-information: only buy probes when they reduce planner action regret
  enough to pay back their environment-sample cost, and test on harder
  mechanics settings where wrong dynamics cause failure rather than only worse
  local cost.
