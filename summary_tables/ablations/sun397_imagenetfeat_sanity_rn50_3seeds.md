# SUN397 Source with ImageNet Feature Sanity Check

Strict RN50-only summary. Training source: sun397. Prior feature for sanity variant: ImageNet feature. Seeds: 1, 2, 3.

| Target | CoOp RN50 | Safe-Real RN50 | Safe-ImageNetFeat RN50 | Real-CoOp | ImageNetFeat-CoOp | ImageNetFeat-Real |
|---|---:|---:|---:|---:|---:|---:|
| oxford_pets | 57.63 | 61.23 | 57.53 | +3.60 | -0.10 | -3.70 |
| eurosat | 26.60 | 25.93 | 26.67 | -0.67 | +0.07 | +0.73 |
| dtd | 22.60 | 24.53 | 24.13 | +1.93 | +1.53 | -0.40 |
| food101 | 61.23 | 62.60 | 61.37 | +1.37 | +0.13 | -1.23 |
| oxford_flowers | 39.83 | 40.67 | 40.57 | +0.83 | +0.73 | -0.10 |
| caltech101 | 73.07 | 77.40 | 74.50 | +4.33 | +1.43 | -2.90 |
| stanford_cars | 48.73 | 46.90 | 48.03 | -1.83 | -0.70 | +1.13 |
| fgvc_aircraft | 9.37 | 8.33 | 6.63 | -1.03 | -2.73 | -1.70 |
| ucf101 | 48.27 | 49.80 | 47.40 | +1.53 | -0.87 | -2.40 |
| **Average** |  |  |  | **+1.12** | **-0.06** | **-1.17** |

## Seed-level Positive Cases

| Comparison | Positive cases |
|---|---:|
| Safe-Real > CoOp | 14/27 |
| Safe-ImageNetFeat > CoOp | 9/27 |
| Safe-ImageNetFeat > Safe-Real | 8/27 |

## Missing Entries

No missing entries.
