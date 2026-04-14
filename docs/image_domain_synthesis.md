# Image Domain Synthesis

This file explains how the vision-side papers connect to the repo's broader
environment-belief thesis.

For the concrete image-side architecture, crawler behavior, metrics, and
prototype plan, read [image_belief_design.md](./image_belief_design.md) after
this file.

The point is not to turn the repo into an image benchmark zoo.

The point is to make one deeper idea precise:

the latent mechanism we are building for physical control should, in principle,
generalize to images and visual concepts too.

This file therefore answers four questions:

1. What do the image papers actually imply?
2. Why are children more sample-efficient than standard classifiers?
3. What kind of latent should a cross-domain crawler build for images?
4. How should this change future code and research decisions in this repo?

## The Core Image Question

The motivating intuition is:

if a child can see a horse once, or a few times, and learn the concept far
faster than a standard supervised classifier trained from scratch, then the
main missing ingredient is probably not "more labels."

It is probably:

- stronger priors
- better latent structure
- better object decomposition
- better transfer from earlier experience
- better uncertainty handling

The strong version of the idea is not:

"one image is always enough to solve open-world visual recognition."

The stronger and more precise version is:

"with the right priors, abstractions, and support-set reasoning, a system
should be able to infer a lot about a new visual concept from very little
supervision."

That is directly analogous to the pendulum story:

- first infer the structure of the world or concept
- then solve the downstream task quickly

## What The Papers Say

This section uses the user-provided papers and extracts the relevant lesson for
the repo.

## 1. Lake et al. 2015

Paper:

- Human-level concept learning through probabilistic program induction

Source:

- https://pubmed.ncbi.nlm.nih.gov/26659050/

Key points:

- one-shot visual concept learning is much easier when concepts are represented
  as structured programs rather than flat pixel templates
- human-level one-shot classification in Omniglot depends on compositional and
  causal structure
- prior knowledge is part of the solution, not something to be avoided

Repo implication:

- the image latent should not just be an embedding for nearest-neighbor lookup
- the latent should reflect reusable structure such as parts, composition, and
  transformation rules
- the crawler should learn concepts by building structured hypotheses, not only
  by separating classes in feature space

Why this matters:

This is the strongest vision paper in the set for the child-like learning
intuition. It says the right latent is a structured explanation of a concept,
not merely a numerical descriptor.

## 2. Matching Networks 2016

Paper:

- Matching Networks for One Shot Learning

Source:

- https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning

Key points:

- one-shot classification can be framed as comparison to a small support set
- a good embedding plus episodic training can support fast adaptation to new
  classes

Repo implication:

- the downstream solver should be able to consume a support-set belief
- a latent is valuable when a tiny amount of new evidence can create a usable
  comparison structure

Why this matters:

This paper is not the full answer, but it is a very clean reminder that the
solver should be designed for small-data adaptation, not only for fixed-class
supervised training.

## 3. Prototypical Networks 2017

Paper:

- Prototypical Networks for Few-shot Learning

Source:

- https://papers.nips.cc/paper/6996-prototypical-networks-for-fe

Key points:

- classes can often be represented as prototypes in latent space
- few-shot classification improves when the latent geometry makes those
  prototypes meaningful

Repo implication:

- the env belief should support prototype-like reasoning
- the image-side belief should make it easy to form class or concept summaries
  from a handful of examples

Why this matters:

This paper is one of the clearest arguments that latent geometry is not a side
issue. The geometry must make few-example concepts easy to form and compare.

## 4. MAML 2017

Paper:

- Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

Source:

- https://proceedings.mlr.press/v70/finn17a.html

Key points:

- a model can be trained so that a small number of examples lead to rapid
  adaptation
- fast learning is a first-class training objective

Repo implication:

- the solver should be optimized for fast adaptation from the belief, not just
  for asymptotic accuracy
- the crawler-belief stack should be judged partly by how easily the solver
  adapts from small support evidence

Why this matters:

This is the direct analogue of what we want in RL: build the system so few-shot
adaptation is the point, not a side effect.

## 5. SimCLR 2020

Paper:

- A Simple Framework for Contrastive Learning of Visual Representations

Source:

- https://proceedings.mlr.press/v119/chen20j.html

Key points:

- strong visual representations can be learned without labels using view
  invariance
- good latents make downstream tasks much more label-efficient

Repo implication:

- the crawler should gather multiple views or transformations that reveal
  stable structure
- the latent should learn invariances before heavy supervision arrives

Why this matters:

This paper supports the idea that representation learning should happen before
or alongside classification, not be reduced to it.

## 6. BYOL 2020

Paper:

- Bootstrap Your Own Latent

Source:

- https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html

Key points:

- self-distillation across views can produce strong image latents without
  contrastive negatives
- representation stability can come from predicting one view from another

Repo implication:

- cross-view latent consistency is a useful principle for the crawler-belief
  system
- we should think beyond only negative-sampling contrastive objectives

Why this matters:

BYOL reinforces the general idea that a good belief should be consistent across
multiple observations of the same underlying structure.

## 7. FixMatch 2020

Paper:

- FixMatch: Simplifying Semi-Supervised Learning with Consistency and
  Confidence

Source:

- https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html

Key points:

- once the model is partly competent, unlabeled data becomes much more useful
  through confidence-gated consistency learning
- label efficiency depends heavily on uncertainty and confidence discipline

Repo implication:

- uncertainty is not just for logging; it should control when the crawler or
  solver trusts its own hypotheses
- pseudo-label-like updates should be gated by confidence

Why this matters:

This is a very practical paper for our broader thesis: once the belief is good
enough, the system should exploit lots of cheap unlabeled or weakly labeled
evidence without fooling itself.

## 8. Slot Attention 2020

Paper:

- Object-Centric Learning with Slot Attention

Source:

- https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html

Key points:

- scenes are easier to model when the latent is decomposed into object-like
  slots
- object-centric structure improves compositional generalization

Repo implication:

- the image-side latent should probably have object or part structure, not only
  one pooled embedding
- the crawler should be able to collect evidence about parts and objects

Why this matters:

If the child-like story is real, object-centricity is one of the biggest
missing ingredients in a flat latent pipeline.

## 9. DINO 2021

Paper:

- Emerging Properties in Self-Supervised Vision Transformers

Source:

- https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html

Key points:

- self-supervised ViTs can learn surprisingly semantic and segmentation-like
  features
- k-NN performance and object-aware properties can emerge without labels

Repo implication:

- the latent can acquire object and concept structure before explicit labels
- a good crawler should exploit multi-view self-supervision heavily

Why this matters:

DINO is one of the strongest practical signs that image latents can become much
more semantic than simple texture embeddings if the self-supervised objective
is good.

## 10. DeiT 2021

Paper:

- Training data-efficient image transformers and distillation through attention

Source:

- https://proceedings.mlr.press/v139/touvron21a.html

Key points:

- training procedure and distillation can matter enormously for data efficiency
- data-efficient performance is not only about architecture depth or scale

Repo implication:

- the solver should be trained with efficiency in mind
- knowledge transfer and teacher signals are legitimate tools

Why this matters:

This paper is a reminder that the adaptation head and training recipe matter,
not only the latent itself.

## 11. CLIP 2021

Paper:

- Learning Transferable Visual Models From Natural Language Supervision

Source:

- https://proceedings.mlr.press/v139/radford21a.html

Key points:

- language supervision can build broader, more transferable visual concepts
- open-vocabulary classification becomes possible when image and text concepts
  share a latent space

Repo implication:

- a child-like visual latent probably benefits from language
- a general crawler should be able to use textual names, relations, and
  descriptions as concept priors

Why this matters:

This is one of the strongest bridges between the image and language sides of
the repo. It suggests the cross-domain latent should not treat language and
vision as separate islands.

## 12. MAE 2022

Paper:

- Masked Autoencoders Are Scalable Vision Learners

Source:

- https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html

Key points:

- masked prediction can learn strong visual structure efficiently
- partial observation is enough to force the latent to model the whole image

Repo implication:

- missing-information probes are meaningful for image crawling
- the latent should be trained to complete structure from partial evidence

Why this matters:

This is the vision analogue of learning the hidden rules of a world from small
windows of evidence.

## What The Image Papers Say Together

Together, these papers say:

- fast image learning depends on reusable latent structure
- few-shot success depends on support-set-friendly geometry
- self-supervised learning is essential for label efficiency
- object-centricity matters
- confidence and uncertainty matter
- language can supply concept priors
- strong priors and structure are part of the solution, not cheating

They do not say:

- raw supervised classification from scratch is enough
- one-shot natural-image classification is solved in the wild
- a single pooled embedding is always the right representation
- more labels are the only route to competence

## The Child Analogy, Made Precise

The useful version of the child analogy is:

a child does not see a horse once as a blank learner.

The child already has:

- object priors
- part-whole priors
- motion and geometry priors
- concept hierarchies
- language
- uncertainty handling

Then one new example mostly updates an existing belief system.

That is exactly the kind of latent we should be aiming for across domains.

## What This Means For The Repo

The image-side lesson for this repo is not "add an image classifier."

The lesson is:

build a crawler-belief system that can infer reusable object and concept
structure from a small number of visual views, then let a downstream solver use
that belief for classification, retrieval, or control.

That is the image analogue of:

- infer the pendulum mechanics
- then balance it

The latent should be a belief over visual structure, not only a class logit
precursor.
