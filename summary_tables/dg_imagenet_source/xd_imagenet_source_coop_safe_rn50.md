# ImageNet-source Cross-dataset DG: CoOp vs Safe PriorRes

Source: `imagenet`  
Backbone: RN50  
Protocol: train on ImageNet and directly evaluate on ten target datasets.  
PriorRes setting: Safe noalt, `USE_B=False`, `USE_LEGACY_RESIDUAL=False`, `ALTERNATE_OPT=False`.

| Source | Target | CoOp | Safe PriorRes | Delta |
|---|---|---:|---:|---:|
| imagenet | oxford_pets | 78.53 | 80.77 | 2.23 |
| imagenet | eurosat | 26.97 | 27.77 | 0.80 |
| imagenet | dtd | 27.67 | 28.47 | 0.80 |
| imagenet | food101 | 64.40 | 66.50 | 2.10 |
| imagenet | oxford_flowers | 46.53 | 50.03 | 3.50 |
| imagenet | caltech101 | 81.57 | 80.30 | -1.27 |
| imagenet | stanford_cars | 41.07 | 41.57 | 0.50 |
| imagenet | fgvc_aircraft | 10.03 | 10.87 | 0.83 |
| imagenet | ucf101 | 50.20 | 50.60 | 0.40 |
| imagenet | sun397 | 50.63 | 51.70 | 1.07 |
| **imagenet** | **Average** |  |  | **1.10** |

## Source-level Average

| Source | Avg Delta | Positive Seed Cases |
|---|---:|---:|
| imagenet | 1.10 | 17/30 |
| **Overall** | **1.10** | **17/30** |

## Notes

- Delta = Safe PriorRes - CoOp.
- ImageNet-source DG is used as the standard large-source cross-dataset validation.
- The multi-source DG setting is still useful for source-dependency analysis.
