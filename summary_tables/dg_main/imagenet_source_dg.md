# ImageNet-source Cross-Dataset DG Summary

This is the primary CoOp-based cross-dataset DG table.

Source: `imagenet`

Protocol:

- Backbone: RN50
- Shots: 16
- Context length: 16
- CSC: False
- Class token position: end
- Seeds: 1, 2, 3
- Delta = PriorRes - CoOp
- PriorRes setting: safe noalt, `USE_B=False`, `USE_LEGACY_RESIDUAL=False`, `ALTERNATE_OPT=False`

| Source | Target | CoOp | PriorRes | Delta |
|---|---|---:|---:|---:|
| imagenet | oxford_pets | 78.53 | 80.77 | +2.23 |
| imagenet | eurosat | 26.97 | 27.77 | +0.80 |
| imagenet | dtd | 27.67 | 28.47 | +0.80 |
| imagenet | food101 | 64.40 | 66.50 | +2.10 |
| imagenet | oxford_flowers | 46.53 | 50.03 | +3.50 |
| imagenet | caltech101 | 81.57 | 80.30 | -1.27 |
| imagenet | stanford_cars | 41.07 | 41.57 | +0.50 |
| imagenet | fgvc_aircraft | 10.03 | 10.87 | +0.83 |
| imagenet | ucf101 | 50.20 | 50.60 | +0.40 |
| imagenet | sun397 | 50.63 | 51.70 | +1.07 |
| **imagenet** | **Average** |  |  | **+1.10** |

## Source-level Average

| Source | Avg Delta | Positive Seed Cases |
|---|---:|---:|
| imagenet | +1.10 | 17/30 |
| **Overall** | **+1.10** | **17/30** |

## Interpretation

The ImageNet-source DG result is the primary CoOp-based cross-dataset benchmark.

Safe PriorRes obtains a modest positive average gain over CoOp under the standard ImageNet-source DG protocol. The improvement is positive on 9 out of 10 target datasets, with an average delta of +1.10 points and 17/30 positive seed-level cases.

This result should be used as the main empirical DG evidence. The clean 4-source rerun should be used as auxiliary source-dependency and residual-formulation analysis.
