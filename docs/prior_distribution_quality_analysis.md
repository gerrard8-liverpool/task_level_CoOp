# Prior Distribution Quality Analysis

This note summarizes the current interpretation of dataset-level task features after the clean DG rerun and prior-quality sanity checks.

## Core Interpretation

The dataset feature should not be interpreted as a direct test-time target-domain query. Instead, it is a train-time distributional conditioning signal.

During source-domain training, the injected task feature interacts with the source samples and changes the learned prompt solution. The final DG behavior therefore depends on the compatibility between the prior feature, the source training distribution, and the desired cross-dataset transfer behavior.

## Main Evidence

The primary ImageNet-source DG table shows that Safe PriorRes improves CoOp by +1.10 average points, with positive target-level gains on 9 out of 10 datasets and 17/30 positive seed-level cases.

The prior-quality sanity check keeps the training source fixed and replaces only the training-time prior feature with the ImageNet feature.

## Four-source Prior-quality Summary

| Source | Safe-Real - CoOp | Safe-ImageNetFeat - CoOp | ImageNetFeat - Real |
|---|---:|---:|---:|
| Caltech101 | +0.70 | +1.25 | +0.56 |
| Food101 | -0.47 | +0.04 | +0.52 |
| SUN397 | +1.12 | -0.06 | -1.17 |
| OxfordPets | +1.99 | +1.96 | -0.03 |
| Overall | +0.83 | +0.80 | -0.03 |

## Interpretation

The ImageNet feature is not universally better than the source feature. Instead, the effect is source-dependent.

Caltech101 and Food101 benefit from the broader ImageNet prior. OxfordPets is almost neutral. SUN397 prefers its own source feature, suggesting that scene-centric sources may require source-aligned priors.

This supports the claim that task features are effective train-time distributional conditioning signals, but the quality of a prior depends on source-prior compatibility rather than simple source alignment alone.

## How to Use in Paper Writing

Use this analysis as prior-distribution quality evidence. Do not use the old Mean / Shuffle ablation as the final proof of prior validity, because those controls can contain real benchmark-domain statistics and are not clean negative controls.

Recommended claim:

Dataset-level priors affect the learned DG solution through train-time conditioning. Their effectiveness is source-dependent and should be evaluated by downstream DG behavior rather than by assuming that the source-aligned feature is always optimal.
