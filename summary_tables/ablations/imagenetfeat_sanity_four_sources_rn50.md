# ImageNet Feature Sanity Check: Four-source Summary

Strict RN50-only summary. Training source is fixed; only the prior feature is replaced by ImageNet feature.

| Source | Safe-Real - CoOp | Safe-ImageNetFeat - CoOp | ImageNetFeat - Real |
|---|---:|---:|---:|
| caltech101 | +0.70 | +1.25 | +0.56 |
| food101 | -0.47 | +0.04 | +0.52 |
| sun397 | +1.12 | -0.06 | -1.17 |
| oxford_pets | +1.99 | +1.95 | -0.04 |
| **Overall** | **+0.83** | **+0.80** | **-0.03** |

## Seed-level Positive Cases

| Comparison | Positive cases |
|---|---:|
| Safe-Real > CoOp | 65/108 |
| Safe-ImageNetFeat > CoOp | 65/108 |
| Safe-ImageNetFeat > Safe-Real | 50/108 |

## Interpretation

This sanity check tests whether the dataset feature acts as a train-time distributional conditioning signal. The result should be interpreted as prior-quality analysis rather than a claim that ImageNet feature is universally better for every source.
