# Safe Dataset-Prior Residual Prompt Adaptation for Robust Transfer

## 1. Project Overview

This project studies how to inject dataset-level task priors into prompt learning for CLIP-based vision-language models.

The current implementation is based on **CoOp** and is named:

```text
CoOpPriorRes
```

The central idea is to use dataset-level statistics to generate a task-conditioned residual prompt adapter. Instead of manually treating prompt hyperparameters such as context length `m` and shots `k` as fixed static choices, the model learns a dataset-conditioned modulation over prompt tokens.

The current stable version is:

```text
Safe PriorRes = dataset-conditioned residual prompt adaptation with identity-centered initialization
```

The project has evolved from a simple CoOp modification into a broader direction:

```text
Safe Dataset-Prior Prompt Adapter
```

The long-term goal is to build a universal adapter interface that can be inserted into prompt learning models such as CoOp, CoCoOp, MaPLe, PromptSRC, and related methods.

---

## 2. Core Motivation

Existing prompt learning methods such as CoOp optimize continuous prompts on a given dataset but usually do not explicitly model dataset-level distributional properties.

However, different datasets have very different characteristics:

- number of classes;
- semantic similarity between classes;
- intra-class diversity;
- inter-class separability;
- domain type;
- source-target transfer difficulty.

The hypothesis of this project is:

```text
Dataset-level priors can help prompt learning become more robust under dataset/domain shift, as long as the prior is injected safely.
```

The project does **not** claim that dataset priors universally improve in-domain few-shot accuracy. Instead, the current evidence suggests:

```text
Safe dataset-prior residual prompting is more useful for cross-dataset transfer than for pure in-domain few-shot classification.
```

---

## 3. Method Summary

### 3.1 Dataset-Level Task Feature

For each dataset, we extract a task feature vector:

```text
z_D = phi(D)
```

The current task features include transformed statistics derived from:

- number of classes;
- semantic similarity;
- intra-class dispersion;
- inter-class dispersion;
- Fisher-like separability.

These features are saved as JSON files:

```text
outputs/task_features/<dataset>_train.json
```

For ImageNet, a sampled version is used due to scale:

```text
outputs/task_features/imagenet_train_sample32.json
```

---

### 3.2 Legacy Residual Formulation

The initial residual formulation was:

```text
ctx_eff = ctx + lambda_t * (a - 1) * u_ctx
```

where:

- `ctx` is the original CoOp learnable context;
- `a` is the dataset-conditioned gate;
- `u_ctx` is a learnable residual direction;
- `lambda_t` is a warmup/ramp-up coefficient.

This formulation has a major problem:

```text
If a0 is not exactly 1 at initialization, the model injects a non-zero random residual before learning.
```

This caused seed-dependent negative transfer, especially on EuroSAT.

Observed failure case:

```text
EuroSAT seed3
Legacy PriorRes: 71.3%
Safe PriorRes:   77.9%
```

---

### 3.3 Safe Residual Formulation

The current main method uses identity-centered residual adaptation:

```text
ctx_eff = ctx + lambda_t * (a - a0) * u_ctx
```

where `a0` is the initial dataset-conditioned prior gate.

At initialization:

```text
a = a0
ctx_eff = ctx
```

Therefore, the model starts exactly as vanilla CoOp and only deviates when the learned residual becomes useful during training.

This is the key technical design of the current project.

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

Default prompt setting:

```text
backbone = RN50
m = 16
k = 16
CSC = False
class token position = end
seeds = 1, 2, 3
```

The current main model should be referred to as:

```text
Safe PriorRes no_b
```

---

## 5. Code Structure

Important files:

```text
third_party/CoOp_clean/trainers/coop_priorres.py
third_party/CoOp_clean/trainers/coop.py
src/meta_prompts/task_feature_extractor.py
src/meta_prompts/task_feature_loader.py
src/meta_prompts/prior_residual_adapter.py
src/meta_prompts/shot_weighting.py
```

Important running scripts:

```text
third_party/CoOp_clean/scripts/ours/run_priorres_any_dataset.sh
scripts/ours/run_priorres_b2n_safe_noalt.sh
scripts/ours/run_coop_b2n_m16k16.sh
scripts/ours/run_priorres_xd_safe_noalt.sh
scripts/ours/run_coop_xd_m16k16.sh
scripts/ours/run_xd_multisource_pair_compare.sh
scripts/ours/summarize_xd_compare.py
scripts/ours/summarize_xd_multisource_compare.py
scripts/ours/summarize_b2n_compare.py
```

---

## 6. Completed Experiments

### 6.1 In-Domain Few-Shot Classification

Datasets:

```text
OxfordPets
EuroSAT
DTD
Food101
OxfordFlowers
Caltech101
StanfordCars
FGVCAircraft
UCF101
SUN397
```

Main conclusion:

```text
Safe PriorRes achieves competitive same-parameter CoOp performance on in-domain few-shot classification.
```

It does not consistently outperform CoOp on all datasets. Its role in the paper should be framed as:

```text
Safe PriorRes preserves CoOp-level in-domain performance while enabling more robust transfer under dataset shift.
```

Key observation:

```text
Safe residual fixes the instability of the legacy residual formulation.
```

---

### 6.2 Base-to-New Generalization

B2N setting:

```text
train on base classes
test on base classes
test on new classes
report base / new / HM
```

Current result summary:

```text
Average Delta:
Base: -0.17
New:  +0.05
HM:   +0.12
```

Strong positive case:

```text
DTD:
Delta New = +9.13
Delta HM  = +7.99
```

Other positive cases:

```text
FGVC-Aircraft:
Delta HM = +1.21

UCF101:
Delta HM = +1.30
```

Conclusion:

```text
B2N performance is comparable to CoOp on average, with dataset-dependent gains.
```

B2N should be used as auxiliary evidence rather than the main claim.

---

### 6.3 Cross-Dataset Domain Generalization

This is currently the strongest evidence for the project.

The main cross-dataset transfer setting is:

```text
train on one source dataset
evaluate on the remaining target datasets
compare Safe PriorRes against CoOp
```

Recommended main DG sources:

```text
Caltech101
Food101
SUN397
```

Source-level results:

| Source | Avg Delta |
|---|---:|
| Caltech101 | +1.43 |
| Food101 | +1.88 |
| SUN397 | +0.20 |

Estimated main-source average:

```text
(+1.43 + 1.88 + 0.20) / 3 = +1.17
```

Additional source-dependency analysis:

| Source | Avg Delta |
|---|---:|
| OxfordPets | -1.12 |

Interpretation:

```text
Broad and diverse source datasets such as Caltech101 and Food101 provide more transferable dataset-conditioned priors.
SUN397 is more difficult and semantically different from several targets, but still gives weak positive average transfer.
OxfordPets is a narrow fine-grained source and may induce source-specific bias, resulting in negative average transfer.
```

Main DG claim:

```text
Safe PriorRes improves cross-dataset transfer on average across multiple broad/diverse source datasets, while the benefit remains source-target dependent.
```

---

## 7. Current Interpretation

The current evidence supports the following refined research claim:

```text
Dataset-conditioned residual prompt adaptation is not primarily an in-domain accuracy booster.
Its main value lies in improving robustness and transferability under dataset shift.
```

The safe residual design is important because naive residual injection can hurt performance by introducing non-identity perturbations at initialization.

The project should be framed as:

```text
A safe dataset-prior prompt adapter for robust cross-dataset transfer.
```

rather than:

```text
A CoOp variant that always improves accuracy.
```

---

## 8. Recommended Main Experimental Story

The current paper should follow this logic:

### Step 1: Identify the problem

CoOp learns prompts in a dataset-specific way but does not explicitly model dataset-level distributional properties.

### Step 2: Propose dataset-prior residual adaptation

Use task features to generate prompt modulation.

### Step 3: Show naive residual is unsafe

Legacy residual can cause seed-dependent negative transfer.

### Step 4: Propose identity-centered safe residual

Safe residual preserves CoOp behavior at initialization.

### Step 5: Show in-domain preservation

Safe PriorRes remains close to same-parameter CoOp on base few-shot tasks.

### Step 6: Show DG advantage

Safe PriorRes improves cross-dataset transfer on average across broad/diverse source datasets.

### Step 7: Analyze source dependency

OxfordPets negative result shows that narrow fine-grained sources may produce less transferable priors.

---

## 9. Next Ablation Plan

The next experiments should not blindly add more datasets. They should directly support the core mechanism.

### 9.1 Residual Formulation Ablation

Compare:

| Variant | Formula | Purpose |
|---|---|---|
| CoOp | no residual | baseline |
| Legacy PriorRes | `ctx + lambda*(a-1)*u` | show unsafe residual |
| Safe PriorRes | `ctx + lambda*(a-a0)*u` | main method |

Important case study:

```text
EuroSAT seed3:
Legacy: 71.3%
Safe:   77.9%
```

Expected conclusion:

```text
Identity-centered residual injection is necessary to avoid seed-dependent negative transfer.
```

---

### 9.2 Dataset Prior Ablation

Compare:

| Variant | Purpose |
|---|---|
| CoOp | no prior |
| Safe residual with random feature | controls for extra parameters |
| Safe residual with constant feature | controls for task-independent adapter |
| Safe residual with dataset feature | main method |

Expected conclusion:

```text
Dataset-level features provide meaningful task-conditioned modulation beyond simply adding parameters.
```

---

### 9.3 a-Branch and b-Branch Ablation

Current main method uses:

```text
a branch only
USE_B=False
```

Ablation variants:

| Variant | Purpose |
|---|---|
| a-only | main stable context residual |
| b-only | sample/shot weighting only |
| a+b | combined prior |
| no_b | main recommended configuration |

Current observation:

```text
b branch with B_LOSS_WEIGHT=0.2 is unstable on base tasks.
```

Recommended conclusion:

```text
Context-side dataset prior is more stable than sample-weighting prior in the current setting.
```

---

### 9.4 Source Diversity Ablation

Use source datasets with different properties:

| Source | Type | Expected Behavior |
|---|---|---|
| Caltech101 | broad object source | strong positive |
| Food101 | broad natural source | strong positive |
| SUN397 | large scene source | weak positive |
| OxfordPets | narrow fine-grained source | negative or unstable |

Analysis goal:

```text
Study how source dataset diversity affects transferability of dataset-conditioned priors.
```

This is important because it turns OxfordPets from a bad result into a meaningful negative case.

---

### 9.5 Strength and Warmup Ablation

Ablate:

```text
lambda_max
warmup_epochs
ramp_epochs
META_LR_RATIO
INIT_GATE_BIAS
```

Recommended small grid:

| Hyperparameter | Values |
|---|---|
| `lambda_max` | 0.5, 1.0 |
| `META_LR_RATIO` | 0.1, 0.3, 1.0 |
| `INIT_GATE_BIAS` | 2.0, 4.0 |
| `warmup_epochs` | 0, 5 |
| `ramp_epochs` | 5, 10 |

Do not overdo this grid. The purpose is not to tune the best number, but to show robustness.

---

## 10. Mechanism and Theoretical Analysis Plan

### 10.1 Identity-Preserving Initialization

Key theoretical point:

```text
Legacy residual does not guarantee identity initialization.
Safe residual guarantees identity initialization.
```

Legacy:

```text
ctx_eff = ctx + lambda*(a-1)*u
```

At initialization:

```text
a = a0
ctx_eff = ctx + lambda*(a0-1)*u
```

If `a0 != 1`, the model is already perturbed.

Safe:

```text
ctx_eff = ctx + lambda*(a-a0)*u
```

At initialization:

```text
a = a0
ctx_eff = ctx
```

Therefore, safe residual starts exactly from CoOp.

This can be written as a stability guarantee:

```text
Safe residual bounds the initial perturbation norm to zero.
```

---

### 10.2 Residual Norm Analysis

Track:

```text
|| lambda_t * (a - a0) * u ||
```

over training.

Expected observation:

```text
Safe residual starts near zero and gradually increases.
Legacy residual can start with non-zero perturbation.
```

This supports the argument that safe residual avoids unsafe early-stage prompt perturbation.

---

### 10.3 Gate Deviation Analysis

Track:

```text
a0
a
delta_a = a - a0
meff
```

Potential table:

| Dataset | mean(a0) | mean(a) | mean(|a-a0|) | meff | DG Delta |
|---|---:|---:|---:|---:|---:|

Goal:

```text
Show that the adapter learns non-trivial task-conditioned deviations rather than remaining identical to CoOp.
```

---

### 10.4 Source-Target Feature Distance

Compute distance between source and target task features:

```text
d(source, target) = || z_source - z_target ||
```

Then compare it with:

```text
Delta = Acc(PriorRes) - Acc(CoOp)
```

The goal is not necessarily to prove strong linear correlation. The goal is to show:

```text
The effectiveness of dataset priors depends on the source-target distribution relationship.
```

This can support the interpretation of:

```text
Caltech101 / Food101 positive
SUN397 weak positive
OxfordPets negative
```

---

### 10.5 Case Studies

Recommended case studies:

| Case | Purpose |
|---|---|
| EuroSAT legacy failure | shows unsafe residual can cause negative transfer |
| DTD B2N positive | shows texture-oriented transfer may benefit |
| Caltech101/Food101 DG positive | shows broad source benefit |
| OxfordPets source negative | shows narrow fine-grained source bias |
| SUN397 weak positive | shows large but semantically distant source gives limited but stable gain |

---

## 11. ImageNet Status

ImageNet is currently treated as an optional scaling experiment.

Reason:

```text
ImageNet has 1000 classes, and CoOp-style prompt learning encodes all class prompts through the text encoder.
This creates a very large text-side computation graph and causes high memory usage on 24GB GPUs.
```

Possible strategies:

1. use A100 for ImageNet-source DG;
2. use text prompt chunking;
3. use smaller batch size;
4. use AMP/fp16;
5. use class-subset logits;
6. leave full ImageNet-source scaling as future work.

Current recommendation:

```text
Do not let ImageNet block the main paper.
Use non-ImageNet multi-source DG as the main result.
```

---

## 12. Universal Adapter Direction

The next-stage goal is to make the method model-agnostic.

The adapter should be abstracted as:

```python
class DatasetPriorAdapter(nn.Module):
    def forward(task_feature):
        return {
            "context_gate": a,
            "initial_gate": a0,
            "residual_direction": u,
            "sample_gate": b,
        }
```

The core interface is:

```python
ctx_eff = ctx + lambda_t * (a - a0) * u_ctx
```

Potential target models:

| Model | Adapter Location |
|---|---|
| CoOp | learnable context tokens |
| CoCoOp | generated conditional context |
| MaPLe | shallow/deep text and vision prompts |
| PromptSRC | prompt tokens and regularization branch |

Recommended first extension:

```text
CoCoOp + DatasetPriorAdapter
```

Reason:

```text
CoCoOp is structurally closer to CoOp and easier to adapt than MaPLe or PromptSRC.
```

---

## 13. Current Project Completion

Current completion estimate:

| Module | Completion |
|---|---:|
| Method implementation | 85% |
| Base few-shot experiments | 90% |
| B2N experiments | 90% |
| Multi-source DG experiments | 85% |
| b-branch validation | 70% |
| Mechanism analysis | 35% |
| Paper writing | 30% |
| Universal adapter abstraction | 25% |
| ImageNet scaling | optional |

Overall:

```text
Preprint readiness: ~80%
Main-conference readiness: ~60–65%
Undergraduate research value: 90%+
```

---

## 14. Publication Positioning

Current version:

```text
Strong arXiv / workshop / CCF C
CCF B competitive with better analysis
AAAI / CCF A possible but not stable
```

With complete ablation and mechanism analysis:

```text
CCF B competitive
AAAI / IJCAI / ACM MM can be seriously attempted
```

With universal adapter + CoCoOp proof-of-concept:

```text
Main-conference potential becomes significantly stronger
```

---

## 15. Recommended Next Steps

Immediate next steps:

1. Finalize DG main table:

```bash
python scripts/ours/summarize_xd_multisource_compare.py \
  caltech101 food101 sun397 \
  > outputs/xd_multisource_caltech_food_sun397_compare.md
```

2. Generate source-dependency table:

```bash
python scripts/ours/summarize_xd_multisource_compare.py \
  caltech101 food101 sun397 oxford_pets \
  > outputs/xd_multisource_with_pets_analysis.md
```

3. Prepare residual formulation ablation:

```text
CoOp vs Legacy PriorRes vs Safe PriorRes
```

4. Extract mechanism logs:

```text
a0, a, delta_a, meff, residual norm
```

5. Write paper draft.

6. Start universal adapter refactor.

---

## 16. One-Sentence Summary

Safe PriorRes is a dataset-conditioned residual prompt adaptation framework that preserves CoOp-level in-domain performance, avoids unsafe residual initialization, and improves cross-dataset transfer on average across broad and diverse source datasets, while revealing source-dependent behavior in prompt transfer.