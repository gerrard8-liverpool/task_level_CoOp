# Food101 Source with ImageNet Feature Sanity Check

Strict RN50-only summary. Training source: food101. Prior feature for sanity variant: ImageNet feature. Seeds: 1, 2, 3.

| Target | CoOp RN50 | Safe-Real RN50 | Safe-ImageNetFeat RN50 | Real-CoOp | ImageNetFeat-CoOp | ImageNetFeat-Real |
|---|---:|---:|---:|---:|---:|---:|
| oxford_pets | 56.10 | 50.37 | 49.10 | -5.73 | -7.00 | -1.27 |
| eurosat | 23.90 | 22.03 | 25.77 | -1.87 | +1.87 | +3.73 |
| dtd | 24.97 | 21.70 | 22.33 | -3.27 | -2.63 | +0.63 |
| oxford_flowers | 37.37 | 38.83 | 39.23 | +1.47 | +1.87 | +0.40 |
| caltech101 | 63.43 | 65.93 | 66.53 | +2.50 | +3.10 | +0.60 |
| stanford_cars | 47.77 | 49.90 | 50.17 | +2.13 | +2.40 | +0.27 |
| fgvc_aircraft | 6.17 | 6.60 | 5.87 | +0.43 | -0.30 | -0.73 |
| ucf101 | 48.17 | 47.53 | 49.03 | -0.63 | +0.87 | +1.50 |
| sun397 | 38.67 | 39.37 | 38.90 | +0.70 | +0.23 | -0.47 |
| **Average** |  |  |  | **-0.47** | **+0.04** | **+0.52** |

## Seed-level Positive Cases

| Comparison | Positive cases |
|---|---:|
| Safe-Real > CoOp | 14/27 |
| Safe-ImageNetFeat > CoOp | 17/27 |
| Safe-ImageNetFeat > Safe-Real | 16/27 |

## Missing Entries

No missing entries.
