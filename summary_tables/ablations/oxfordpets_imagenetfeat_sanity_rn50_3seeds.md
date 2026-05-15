# OxfordPets Source with ImageNet Feature Sanity Check

Strict RN50-only summary. Training source: oxford_pets. Prior feature for sanity variant: ImageNet feature. Seeds: 1, 2, 3.

| Target | CoOp RN50 | Safe-Real RN50 | Safe-ImageNetFeat RN50 | Real-CoOp | ImageNetFeat-CoOp | ImageNetFeat-Real |
|---|---:|---:|---:|---:|---:|---:|
| eurosat | 22.53 | 19.43 | 19.57 | -3.10 | -2.97 | +0.13 |
| dtd | 22.03 | 23.90 | 23.90 | +1.87 | +1.87 | +0.00 |
| food101 | 46.53 | 51.10 | 50.83 | +4.57 | +4.30 | -0.27 |
| oxford_flowers | 40.33 | 43.17 | 43.10 | +2.83 | +2.77 | -0.07 |
| caltech101 | 69.40 | 71.13 | 71.07 | +1.73 | +1.67 | -0.07 |
| stanford_cars | 44.37 | 47.03 | 47.13 | +2.67 | +2.77 | +0.10 |
| fgvc_aircraft | 5.70 | 7.17 | 7.33 | +1.47 | +1.63 | +0.17 |
| ucf101 | 38.83 | 42.47 | 42.20 | +3.63 | +3.37 | -0.27 |
| sun397 | 35.77 | 37.97 | 37.90 | +2.20 | +2.13 | -0.07 |
| **Average** |  |  |  | **+1.99** | **+1.95** | **-0.04** |

## Seed-level Positive Cases

| Comparison | Positive cases |
|---|---:|
| Safe-Real > CoOp | 21/27 |
| Safe-ImageNetFeat > CoOp | 21/27 |
| Safe-ImageNetFeat > Safe-Real | 11/27 |

## Missing Entries

No missing entries.
