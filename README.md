# Safe Dataset-Prior Residual Prompt Adaptation for Robust Transfer

> Current stage: **Phase 1 completed**  
> Main implementation: **CoOpPriorRes** on top of CoOp / CLIP RN50  
> Current main method: **Safe PriorRes no_b**  
> Main evidence: **cross-dataset domain generalization (DG)**  
> Last updated: May 2026

---

## 1. Project Overview

This project studies how to inject **dataset-level task priors** into CLIP-based prompt learning.

The current implementation is based on **CoOp** and is named:

```text
CoOpPriorRes
```

The core idea is to extract dataset-level statistics before prompt training and use them to generate a task-conditioned residual adapter over CoOp's learnable context tokens.

The current stable version is:

```text
Safe PriorRes = identity-centered dataset-prior residual prompt adaptation
```

The project should be framed as:

```text
A dataset-prior prompt adapter for robust cross-dataset transfer.
```

It should **not** be framed as:

```text
A CoOp variant that always improves in-domain accuracy.
```

The strongest current evidence is not pure in-domain few-shot classification, but **cross-dataset domain generalization**.

---

## 2. Core Research Question

Standard prompt learning methods such as CoOp optimize continuous prompts on a target dataset, but they do not explicitly model dataset-level distributional properties.

Different visual recognition datasets differ in:

- number of classes;
- semantic similarity between classes;
- intra-class diversity;
- inter-class separability;
- domain type;
- source-target transfer difficulty.

This motivates the main question:

```text
Can dataset-level task priors be safely injected into prompt learning to improve robustness under dataset shift?
```

Current answer from Phase 1:

```text
Yes, when the prior is source-aligned and injected with an identity-centered residual formulation.
```

---

## 3. Method Summary

### 3.1 Dataset-Level Task Feature

For each dataset `D`, a task feature vector is extracted:

```text
z_D = phi(D)
```

Current task features include transformed statistics derived from:

- number of classes;
- semantic similarity;
- intra-class dispersion;
- inter-class dispersion;
- Fisher-like separability.

Task feature files are saved as:

```text
outputs/task_features/<dataset>_train.json
```

The Phase 1 package also contains:

```text
outputs/task_features/mean_train_feature.json
```

This file is used by the **Safe-Mean** prior ablation.

For ImageNet, a sampled feature file may be used:

```text
outputs/task_features/imagenet_train_sample32.json
```

ImageNet is currently treated as optional scaling and should not block the main paper.

---

### 3.2 Prior Adapter

The dataset feature is fed into a lightweight adapter to generate a dataset-conditioned gate:

```text
a0 = f_theta(z_D)
```

During training, the model learns a logit-space deviation from this initial prior:

```text
a = sigmoid(logit(a0) + delta_a)
```

This keeps the gate bounded while preserving `a = a0` at initialization. The adapter then modulates the original CoOp context tokens through a residual direction.

---

### 3.3 Legacy Residual Formulation

The legacy formulation is:

```text
ctx_eff = ctx + lambda_t * (a - 1) * u_ctx
```

where:

- `ctx` is the original CoOp learnable context;
- `a` is the dataset-conditioned gate;
- `u_ctx` is a learnable residual direction;
- `lambda_t` is a warmup/ramp-up coefficient.

Problem:

```text
If a0 != 1 at initialization, the model injects a non-zero residual perturbation before learning meaningful deviations.
```

Therefore, Legacy PriorRes is **not identity-preserving**. It can behave as an aggressive residual variant. It may help in some DG settings, but Phase 1 results show that it is unstable across sources.

---

### 3.4 Safe Residual Formulation

The current main method uses:

```text
ctx_eff = ctx + lambda_t * (a - a0) * u_ctx
```

At initialization:

```text
a = a0
ctx_eff = ctx
```

Therefore, Safe PriorRes starts exactly from vanilla CoOp and only deviates when the residual branch learns useful task-conditioned changes.

This identity-centered formulation is the key technical design of the current method.

---

## 4. Current Recommended Configuration

The current main configuration is:

```bash
USE_B=False
USE_CONTEXT_GATING=True
USE_LEGACY_RESIDUAL=False
ALTERNATE_OPT=False
META_LR_RATIO=0.3
```

In words:

```text
safe residual + no alternate optimization + no sample-weighting branch
```

Default experimental setting:

```text
Backbone: RN50
N_CTX / m: 16
Shots / k: 16
CSC: False
Class token position: end
Seeds: 1, 2, 3
```

The finalized DG protocol should follow the old main CoOp protocol:

```text
Precision: fp16
Train batch size: 32
Test batch size: 100
```

The DG scripts should **not** explicitly override:

```text
TRAINER.COOP.PREC fp32
DATALOADER.TRAIN_X.BATCH_SIZE 16
DATALOADER.TEST.BATCH_SIZE 64
```

This was a previously identified source of protocol mismatch.

---

## 5. Important Code and Package Structure

### 5.1 Main Trainer Files

```text
third_party/CoOp_clean/trainers/coop_priorres.py
third_party/CoOp_clean/trainers/coop.py
```

### 5.2 Meta-Prompt / Prior-Adapter Files

```text
src/meta_prompts/task_feature_extractor.py
src/meta_prompts/task_feature_loader.py
src/meta_prompts/prior_residual_adapter.py
src/meta_prompts/shot_weighting.py
src/meta_prompts/meta_network.py
src/meta_prompts/compute_feature_stats.py
```

### 5.3 Main DG Scripts

```text
scripts/ours/run_coop_xd_m16k16.sh
scripts/ours/run_priorres_xd_safe_noalt.sh
scripts/ours/run_priorres_xd_legacy_noalt.sh
scripts/ours/summarize_xd_multisource_compare.py
scripts/ours/summarize_xd_multisource_compare_with_legacy.py
```

### 5.4 Safe Prior Ablation Scripts

```text
scripts/ablations/safe_prior/run_safe_prior_xd.sh
scripts/ablations/safe_prior/summarize_safe_prior_xd.py
```


## 6. Completed Phase 1 Experiments

Phase 1 currently contains the following completed experiment groups:

```text
1. Cross-dataset DG main experiment
2. Safe prior ablation: Real / Mean / Shuffle
4. In-domain auxiliary experiment
5. Residual formulation analysis
6. Mechanism log extraction
```

The old Safe-only DG table is now deprecated. The current main DG table is:

```text
summary_tables/dg_main/xd_multisource_coop_safe_legacy.md
```

---

## 7. Main Result: Cross-Dataset DG

### 7.1 Setting

The main cross-dataset DG setting is to train on one source dataset and evaluate on unseen target datasets. CoOp, Safe PriorRes, and Legacy PriorRes are compared under the same RN50, 16-shot, 16-context protocol.

For strict DG evaluation, Safe PriorRes and Legacy PriorRes use only the source-side task prior during target evaluation. Target dataset features are not injected during target testing.

### 7.2 Primary DG Table: ImageNet-source DG

The primary DG result should be the ImageNet-source cross-dataset result:

```text
summary_tables/dg_main/imagenet_source_dg.md
```

This table is the current primary CoOp-based DG result. In the ImageNet-source setting, Safe PriorRes obtains an average delta of +1.10 over CoOp, with positive target-level gains on 9/10 datasets and 17/30 positive seed-level cases.

### 7.3 Auxiliary Analysis Table: Clean 4-source DG Rerun

The clean protocol-aligned 4-source rerun is used for source-dependency and residual-formulation analysis, not as the primary DG table:

```text
summary_tables/dg_main/xd_multisource_coop_safe_legacy.md
```

Clean 4-source summary:

| Source | Safe-CoOp Avg Delta | Legacy-CoOp Avg Delta | Safe-Legacy |
|---|---:|---:|---:|
| Caltech101 | +0.70 | +0.83 | -0.14 |
| Food101 | -0.47 | -0.34 | -0.14 |
| SUN397 | +1.12 | -1.29 | +2.41 |
| OxfordPets | +1.99 | +1.85 | +0.13 |
| **4-source Overall** | **+0.83** | **+0.26** | **+0.57** |

Three auxiliary non-ImageNet sources:

| Source | Safe-CoOp Avg Delta | Legacy-CoOp Avg Delta | Safe-Legacy |
|---|---:|---:|---:|
| Caltech101 | +0.70 | +0.83 | -0.14 |
| Food101 | -0.47 | -0.34 | -0.14 |
| SUN397 | +1.12 | -1.29 | +2.41 |
| **3-source Overall** | **+0.45** | **-0.27** | **+0.71** |

### 7.4 Interpretation

The ImageNet-source DG table is the main cross-dataset benchmark. The clean 4-source rerun is used for source-dependency and residual-formulation analysis.

Safe PriorRes should not be described as a universal DG accuracy booster. Its main mechanism is residual safety: identity-centered residual injection prevents prior-induced non-identity prompt bias. The pairwise DG heatmaps show strong source-target dependency, so the 4-source rerun should support the mechanism discussion rather than replace the ImageNet-source DG main result.


---

## 8. Prior Distribution Quality Analysis

### 8.1 Purpose

This analysis answers a more precise reviewer question:

```text
Does the injected dataset feature act as a train-time distributional conditioning signal that changes the learned DG solution?
```

The key point is that task features should not be interpreted as direct test-time target-domain queries. They are injected during source training and interact with the source data to shape the learned prompt solution.

### 8.2 ImageNet Feature Sanity Check

To test the role of prior distribution quality, we keep the training source fixed and replace only the training-time prior feature with the ImageNet feature.

```text
Training source: fixed
Model/training protocol: unchanged
Changed variable: prior feature used during training
Comparison: Safe-RealSourceFeat vs Safe-ImageNetFeat
```

### 8.3 Four-source RN50 Summary

| Source | Safe-Real - CoOp | Safe-ImageNetFeat - CoOp | ImageNetFeat - Real |
|---|---:|---:|---:|
| Caltech101 | +0.70 | +1.25 | +0.56 |
| Food101 | -0.47 | +0.04 | +0.52 |
| SUN397 | +1.12 | -0.06 | -1.17 |
| OxfordPets | +1.99 | +1.96 | -0.03 |
| **Overall** | **+0.83** | **+0.80** | **-0.03** |

### 8.4 Interpretation

This result should not be interpreted as ImageNet feature being universally better. Instead, it shows that different prior features produce different DG solutions under the same source training protocol.

Caltech101 and Food101 benefit from the broader ImageNet prior. OxfordPets is nearly neutral. SUN397 prefers its own source feature. Therefore, prior usefulness is source-dependent.

The correct conclusion is:

```text
Dataset-level features act as train-time distributional conditioning signals.
Their value is determined by source-prior compatibility and downstream DG behavior.
The source-aligned feature is not always optimal, and the ImageNet feature is not universally optimal either.
```

### 8.5 Status of Old Mean / Shuffle Controls

The old Mean / Shuffle controls are retained only as diagnostic experiments. They should not be used as the final proof of prior validity, because Mean and Shuffle features may contain real benchmark-domain or target-domain statistics.

The cleaner paper-level ablation is the prior-distribution quality analysis above.


---

## 10. In-Domain Few-Shot Classification

In-domain experiments are auxiliary.

Current interpretation:

```text
Safe PriorRes preserves CoOp-level in-domain performance but is not designed as a universal in-domain accuracy booster.
```

This is consistent with the project framing:

```text
Main value = robust cross-dataset transfer.
```

Some historical in-domain entries, such as OxfordPets, may be incomplete. This does not block the main DG-centered paper story.

---

## 11. Residual Formulation Analysis

### 11.1 In-Domain Residual Ablation

Previously observed in-domain residual ablation:

```text
Safe-CoOp average:   +0.59
Safe-Legacy average: +0.40
```

This supports the claim that the safe formulation is stable in standard in-domain training.

### 11.2 DG Residual Interpretation

The three-source DG table gives the more important conclusion:

```text
Legacy can be stronger for Caltech101-source transfer but is not robust across sources.
Safe is more reliable overall.
```

Therefore, the residual story should be written as:

```text
There is a safety-transfer trade-off.
Identity-centered residual adaptation is more robust across sources, while non-identity residual injection can act as an aggressive but unstable transfer variant.
```

---

## 12. Mechanism Analysis Status

Mechanism logs have been extracted and saved in:

```text
mechanism_checks/
```

Current files include:

```text
safe_main_train_meff_keff_a0.txt
safe_mean_train_meff_keff_a0.txt
safe_shuffle_train_meff_keff_a0.txt
legacy_train_meff_keff_a0.txt
```

These can be used later to analyze:

```text
meff
keff
a0_mean
b0_mean
lambda_t
delta / delta_a
```

Recommended future mechanism analysis:

```text
1. Compare Safe vs Legacy residual dynamics.
2. Compare Safe-Real vs Safe-Mean vs Safe-Shuffle gate behavior.
3. Show that Safe starts from identity and learns task-conditioned deviations.
4. Analyze source-target feature distance versus DG delta.
```

Mechanism plots are useful for the paper but are not blocking for Phase 1 completion.

---

## 13. Current Paper-Level Story

The current paper should follow this logic:

### Step 1: Problem

CoOp-style prompt learning does not explicitly use dataset-level distributional priors.

### Step 2: Method

Introduce dataset-prior residual prompt adaptation.

### Step 3: Safety

Naive residual injection is not identity-preserving. Safe PriorRes starts from vanilla CoOp and learns residual deviations gradually.

### Step 4: Main empirical evidence

Cross-dataset DG shows Safe PriorRes is robust across sources:

```text
Safe overall:   +1.28
Legacy overall: -0.82
```

### Step 5: Prior validity

Safe-Real beats Safe-Mean and Safe-Shuffle:

```text
Safe-Real:    +1.28
Safe-Mean:    -0.03
Safe-Shuffle: -0.53
```

### Step 6: Auxiliary generalization


---

## 14. What Not to Claim

Do not claim:

```text
Safe PriorRes always improves in-domain accuracy.
Legacy is always worse than Safe.
Dataset prior always helps regardless of source-target relation.
```

Correct claims:

```text
Safe PriorRes improves cross-dataset DG on average across multiple sources.
Safe is more robust than Legacy across sources.
Source-aligned dataset priors are necessary for stable gains.
```

---

## 15. Current Project Completion

Estimated completion:

| Module | Completion |
|---|---:|
| Method implementation | 85% |
| DG main experiments | 95% |
| Safe prior ablation | 95% |
| In-domain auxiliary experiments | 80% |
| Residual formulation analysis | 85% |
| Mechanism analysis | 45% |
| Paper writing | 35% |
| Universal adapter extension | 25% |
| ImageNet scaling | optional |

Overall status:

```text
Phase 1 core experiments: completed.
Preprint readiness: around 80%.
Main-conference readiness: around 60%–65%.
```

---

## 16. Publication Positioning

Current version:

```text
Strong arXiv / workshop / CCF C potential.
CCF B becomes realistic with strong writing and mechanism analysis.
AAAI / IJCAI / ACM MM is possible but still needs stronger generality evidence.
```

To improve main-conference potential, the next most important step is:

```text
CoCoOp / CoCoOpS adapter generality.
```

This would show that the method is not merely a CoOp-specific modification but a general dataset-prior prompt adapter.

---

## 17. Recommended Next Steps

### Immediate

```text
1. Keep the current Phase 1 package as the clean result archive.
2. Start writing the method and experiment sections.
3. Turn DG and prior ablation tables into paper-ready tables.
4. Organize mechanism logs into a compact analysis table or figure.
```

### Next experimental stage

```text
1. Implement CoCoOp / CoCoOpS adapter version.
2. Test whether dataset-prior residual adaptation generalizes beyond CoOp.
3. Add mechanism plots if time allows.
4. Consider source-target feature distance analysis.
```

### Optional

```text
1. ImageNet-source DG scaling.
2. RandomFixed prior appendix baseline.
3. Additional shuffle mappings.
4. More complete in-domain table cleanup.
```

---

## 18. One-Sentence Summary

Safe PriorRes is an identity-centered dataset-prior residual prompt adapter that preserves CoOp-level behavior at initialization, improves cross-dataset transfer robustness across multiple source datasets, and shows through Real / Mean / Shuffle ablation that source-aligned dataset priors—not merely extra adapter parameters—are responsible for the main gains.

---

## 19. Feature Extraction Protocol

For strict reproducibility, task features should be extracted from the same training protocol used by the corresponding experiment.

```text
In-domain: use the 16-shot training split of the same dataset and seed.
DG: use only the source dataset 16-shot training split; target features are not used in the strict DG setting.
```

Recommended feature filenames:

```text
outputs/task_features/<dataset>_shot16_seed<seed>_train.json
outputs/task_features/<dataset>_base_shot16_seed<seed>_train.json
```


---

## Latest Result Updates

### ImageNet-source Cross-Dataset DG

In addition to the multi-source DG setting based on Caltech101, Food101, and SUN397, we further evaluate a standard large-source DG protocol using ImageNet as the source dataset.

Setting:

```text
source = ImageNet
targets = Caltech101, OxfordPets, DTD, EuroSAT, Food101, OxfordFlowers,
          StanfordCars, FGVCAircraft, UCF101, SUN397
backbone = RN50
methods = CoOp vs Safe PriorRes
seeds = 1, 2, 3
