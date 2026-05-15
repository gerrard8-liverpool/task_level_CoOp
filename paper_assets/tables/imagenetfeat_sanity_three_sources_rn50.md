# ImageNet Feature Sanity Check: Three-source Summary

Strict RN50-only summary over Caltech101, Food101, and SUN397. Training source is fixed; only the prior feature is replaced by ImageNet feature.

| Source | Safe-Real - CoOp | Safe-ImageNetFeat - CoOp | ImageNetFeat - Real |
|---|---:|---:|---:|
| caltech101 | +0.70 | +1.25 | +0.56 |
| food101 | -0.47 | +0.04 | +0.52 |
| sun397 | +1.12 | -0.06 | -1.17 |
| **Overall** | **+0.45** | **+0.41** | **-0.03** |

## Seed-level Positive Cases

| Comparison | Positive cases |
|---|---:|
| Safe-Real > CoOp | 44/81 |
| Safe-ImageNetFeat > CoOp | 44/81 |
| Safe-ImageNetFeat > Safe-Real | 39/81 |

## Interpretation

This sanity check tests whether the dataset feature acts as a train-time distributional conditioning signal. A positive ImageNetFeat - Real value suggests that a broader prior feature can provide a better DG training signal than the narrow source feature for some sources.
