# Caltech101 Source with ImageNet Feature Sanity Check

Strict RN50-only summary. Training source: caltech101. Prior feature for sanity variant: ImageNet feature. Seeds: 1, 2, 3.

| Target | CoOp RN50 | Safe-Real RN50 | Safe-ImageNetFeat RN50 | Real-CoOp | ImageNetFeat-CoOp | ImageNetFeat-Real |
|---|---:|---:|---:|---:|---:|---:|
| oxford_pets | 72.50 | 70.80 | 73.53 | -1.70 | +1.03 | +2.73 |
| eurosat | 21.10 | 25.20 | 24.07 | +4.10 | +2.97 | -1.13 |
| dtd | 29.33 | 31.70 | 30.80 | +2.37 | +1.47 | -0.90 |
| food101 | 64.93 | 66.53 | 68.40 | +1.60 | +3.47 | +1.87 |
| oxford_flowers | 48.60 | 49.33 | 49.37 | +0.73 | +0.77 | +0.03 |
| stanford_cars | 50.33 | 47.90 | 49.37 | -2.43 | -0.97 | +1.47 |
| fgvc_aircraft | 11.50 | 11.20 | 11.73 | -0.30 | +0.23 | +0.53 |
| ucf101 | 47.77 | 49.10 | 48.40 | +1.33 | +0.63 | -0.70 |
| sun397 | 48.33 | 48.90 | 50.00 | +0.57 | +1.67 | +1.10 |
| **Average** |  |  |  | **+0.70** | **+1.25** | **+0.56** |

## Seed-level Positive Cases

| Comparison | Positive cases |
|---|---:|
| Safe-Real > CoOp | 16/27 |
| Safe-ImageNetFeat > CoOp | 18/27 |
| Safe-ImageNetFeat > Safe-Real | 15/27 |

## Missing Entries

No missing entries.
