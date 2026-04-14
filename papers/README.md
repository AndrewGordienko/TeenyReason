# Papers Folder

This folder is the local paper library for the repo.

The point is not to hoard PDFs. The point is to keep the specific papers that
directly shape the crawler-belief-solver thesis close to the code and docs so
future work can stay anchored to the actual research idea.

## Main Buckets

### RL / World Belief

- `53_efficient_off_policy_meta_rein.pdf`
  PEARL / latent context for fast adaptation
- `1910.08348v2.pdf`
  VariBAD / belief from interaction history
- `2438_ContraBAR_Contrastive_Bay.pdf`
  contrastive Bayes-adaptive RL
- `2301.04104v2.pdf`
  DreamerV3 / predictive world models
- `2310.16828v2.pdf`
  TD-MPC2 / latent geometry for control
- `2401.12497v1.pdf`
  causal state abstraction
- `2404.12308v2.pdf`
  active system identification
- `2410.20357v2.pdf`
  dynamics as prompts
- `2504.19874v1.pdf`
  TurboQuant / rate-distortion and belief communication

### Language / Small-Data Structure Learning

- `2020.acl-main.463.pdf`
  form vs meaning caution
- `2023.conll-babylm.1.pdf`
  BabyLM / child-scale language budgets
- `2305.07759v2.pdf`
  TinyStories / coherent language from small simplified corpora
- `2306.11644v2.pdf`
  textbooks / curated data and efficient learning
- `2309.10668v2.pdf`
  language modeling as compression
- `P19-1228.pdf`
  structured latent grammar ideas

### Image / Few-Shot and Self-Supervised Vision

- `science.aab3050.pdf`
  probabilistic program induction / one-shot concepts
- `NIPS-2016-matching-networks-for-one-shot-learning-Paper.pdf`
  Matching Networks
- `NIPS-2017-prototypical-networks-for-few-shot-learning-Paper.pdf`
  Prototypical Networks
- `finn17a.pdf`
  MAML
- `chen20j.pdf`
  SimCLR
- `NeurIPS-2020-bootstrap-your-own-latent-a-new-approach-to-self-supervised-learning-Paper.pdf`
  BYOL
- `NeurIPS-2020-fixmatch-simplifying-semi-supervised-learning-with-consistency-and-confidence-Paper.pdf`
  FixMatch
- `NeurIPS-2020-object-centric-learning-with-slot-attention-Paper.pdf`
  Slot Attention
- `Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf`
  DINO
- `touvron21a.pdf`
  DeiT
- `He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf`
  MAE
- `radford21a.pdf`
  CLIP

### Human Learning / Developmental Science / Neuroscience

- `spelke_kinzler_2007_core_knowledge_view.pdf`
  core priors over objects, agents, number, and space
- `kuhl_2011_social_mechanisms_early_language_acquisition.pdf`
  social gating and rich teaching signals
- `begus_bonawitz_2024_informativeness_of_evidence_and_predictive_looking.pdf`
  infants track informativeness of evidence and update causal expectations
- `berger_tzur_german_daum_dehaene_2015_neural_dynamics_prediction_surprise_infants.pdf`
  neural signatures of prediction and surprise in infants
- `stachenfeld_botvinick_gershman_2017_hippocampus_predictive_map.pdf`
  predictive-map account of reusable latent structure
- `garvert_dolan_behrens_2023_hippocampal_spatiopredictive_cognitive_maps.pdf`
  predictive maps and reward generalization

### Canonical But Not Mirrored Locally Yet

These papers are still important to the repo story, but automated download ran
into publisher or anti-bot protection on this machine. Their canonical URLs
are listed here so future work can still find them quickly.

- Infant statistical learning
  https://www.annualreviews.org/doi/10.1146/annurev-psych-122216-011805
- Foreign-language experience in infancy: effects of short-term exposure and
  social interaction on phonetic learning
  https://www.pnas.org/doi/10.1073/pnas.1532872100
- A theory of causal learning in children: causal maps and Bayes nets
  https://escholarship.org/uc/item/4js8k8hc
- Observing the unexpected enhances infants' learning and exploration
  https://pubmed.ncbi.nlm.nih.gov/29507157/
- Stable individual differences in infants' responses to violations of
  intuitive physics
  https://www.pnas.org/doi/10.1073/pnas.2103805118

## How To Use This Folder

- treat the docs as the first reading path
- use the PDFs when a doc needs to be sharpened or challenged
- prefer updating the docs after reading rather than letting paper ideas live
  only in chat

The corresponding docs are:

- `docs/paper_synthesis.md`
- `docs/language_domain_synthesis.md`
- `docs/image_domain_synthesis.md`
- `docs/human_learning_synthesis.md`
