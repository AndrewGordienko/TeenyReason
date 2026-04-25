# Papers Folder

This folder is the local paper library for the repo.

The point is not to hoard PDFs. The point is to keep the small set of papers
that directly shape the crawler-belief-solver thesis close to the code and
docs so future work can stay anchored to the actual research idea.

## Core Architecture Papers

These are the papers that most directly shape the library architecture.

### Belief From Interaction

- `1910.08348v2.pdf`
  VariBAD / belief from interaction history
- `53_efficient_off_policy_meta_rein.pdf`
  PEARL / latent context for fast adaptation
- `2438_ContraBAR_Contrastive_Bay.pdf`
  contrastive Bayes-adaptive RL

### Active Probing And System Identification

- `2404.12308v2.pdf`
  active system identification
- `2410.20357v2.pdf`
  dynamics as prompts

### Uncertainty-Aware Adaptation

- `2301.04104v2.pdf`
  DreamerV3 / predictive world models
- `2310.16828v2.pdf`
  TD-MPC2 / latent geometry for control
- `2401.12497v1.pdf`
  causal state abstraction

### Communication And Rate-Distortion

- `2504.19874v1.pdf`
  TurboQuant / rate-distortion and belief communication
- `2309.10668v2.pdf`
  language modeling as compression

## Domain Reference Papers

These support recipe design for specific modalities. They should not be used to
justify turning every benchmark trick into a crawler-core abstraction.

### Language

- `2020.acl-main.463.pdf`
- `2023.conll-babylm.1.pdf`
- `2305.07759v2.pdf`
- `2306.11644v2.pdf`
- `P19-1228.pdf`

### Image

- `science.aab3050.pdf`
- `NIPS-2016-matching-networks-for-one-shot-learning-Paper.pdf`
- `NIPS-2017-prototypical-networks-for-few-shot-learning-Paper.pdf`
- `finn17a.pdf`
- `chen20j.pdf`
- `NeurIPS-2020-bootstrap-your-own-latent-a-new-approach-to-self-supervised-learning-Paper.pdf`
- `NeurIPS-2020-fixmatch-simplifying-semi-supervised-learning-with-consistency-and-confidence-Paper.pdf`
- `NeurIPS-2020-object-centric-learning-with-slot-attention-Paper.pdf`
- `Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf`
- `touvron21a.pdf`
- `He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf`
- `radford21a.pdf`

### Human Learning / Development

- `spelke_kinzler_2007_core_knowledge_view.pdf`
- `kuhl_2011_social_mechanisms_early_language_acquisition.pdf`
- `begus_bonawitz_2024_informativeness_of_evidence_and_predictive_looking.pdf`
- `berger_tzur_german_daum_dehaene_2015_neural_dynamics_prediction_surprise_infants.pdf`
- `stachenfeld_botvinick_gershman_2017_hippocampus_predictive_map.pdf`
- `garvert_dolan_behrens_2023_hippocampal_spatiopredictive_cognitive_maps.pdf`

## Archive / Extra

These may still be useful, but they are not core architecture drivers.

- `2603.19312v2.pdf`
- `building-machines-that-learn-and-think-like-people.pdf`
- `machines_that_think.pdf`
- `s41467-022-32012-w.pdf`
- `s41562-022-01394-8.pdf`

## Local Cleanup Rules

- keep one copy of each PDF
- keep core papers easy to spot
- do not let every interesting paper imply a new subsystem
- prefer updating docs after reading rather than letting ideas live only in the
  PDF folder

## Canonical But Not Mirrored Locally Yet

These papers are still important to the repo story, but automated download ran
into publisher or anti-bot protection on this machine.

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
