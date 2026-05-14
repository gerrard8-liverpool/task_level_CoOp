# Clean Rerun Multi-source Cross-dataset DG Summary

> This table is regenerated from the clean protocol-aligned DG rerun.

## Protocol

- Backbone: RN50
- Shots: 16
- Context length: 16
- CSC: False
- Class token position: end
- Seeds: 1, 2, 3
- DG setting: train on one source dataset and evaluate on all remaining target datasets
- Safe / Legacy evaluation uses source task features under strict DG

Compared methods:

- CoOp
- Safe PriorRes: identity-centered residual, `ctx + lambda * (a - a0) * u_ctx`
- Legacy PriorRes: non-identity residual, `ctx + lambda * (a - 1) * u_ctx`

## Main 3-source Summary

| Source | n | CoOp | Safe | Legacy | Safe-CoOp | Legacy-CoOp | Safe-Legacy | Safe>CoOp | Legacy>CoOp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Caltech101 | 27 | 43.82 | 44.52 | 44.66 | +0.70 | +0.83 | -0.14 | 16/27 | 19/27 |
| Food101 | 27 | 38.50 | 38.03 | 38.17 | -0.47 | -0.34 | -0.14 | 14/27 | 12/27 |
| SUN397 | 27 | 43.04 | 44.16 | 41.74 | +1.12 | -1.29 | +2.41 | 14/27 | 7/27 |
| **Overall** | **81** | **41.79** | **42.23** | **41.52** | **+0.45** | **-0.27** | **+0.71** | **44/81** | **38/81** |

## 4-source Source-dependency Summary

| Source | n | CoOp | Safe | Legacy | Safe-CoOp | Legacy-CoOp | Safe-Legacy | Safe>CoOp | Legacy>CoOp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Caltech101 | 27 | 43.82 | 44.52 | 44.66 | +0.70 | +0.83 | -0.14 | 16/27 | 19/27 |
| Food101 | 27 | 38.50 | 38.03 | 38.17 | -0.47 | -0.34 | -0.14 | 14/27 | 12/27 |
| SUN397 | 27 | 43.04 | 44.16 | 41.74 | +1.12 | -1.29 | +2.41 | 14/27 | 7/27 |
| OxfordPets | 27 | 36.17 | 38.15 | 38.02 | +1.99 | +1.85 | +0.13 | 21/27 | 20/27 |
| **Overall** | **108** | **40.38** | **41.21** | **40.65** | **+0.83** | **+0.26** | **+0.57** | **65/108** | **58/108** |

## Interpretation

The clean rerun should be interpreted conservatively:

- Safe PriorRes is not a universal DG accuracy booster.
- Its main mechanism is residual safety: identity-centered residual injection prevents prior-induced non-identity prompt bias.
- Cross-dataset behavior is source-target dependent, so pairwise heatmaps are more informative than source-level averages alone.
- Legacy can be competitive or positive on some pairs, but it lacks the identity-preserving safety guarantee.

## Main 3-source Target-level Averages

| Source | Target | n | CoOp | Safe | Legacy | Safe-CoOp | Legacy-CoOp | Safe-Legacy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Caltech101 | OxfordPets | 3 | 72.50 | 70.80 | 72.70 | -1.70 | +0.20 | -1.90 |
| Caltech101 | EuroSAT | 3 | 21.10 | 25.20 | 23.07 | +4.10 | +1.97 | +2.13 |
| Caltech101 | DTD | 3 | 29.33 | 31.70 | 30.83 | +2.37 | +1.50 | +0.87 |
| Caltech101 | Food101 | 3 | 64.93 | 66.53 | 66.73 | +1.60 | +1.80 | -0.20 |
| Caltech101 | OxfordFlowers | 3 | 48.60 | 49.33 | 49.73 | +0.73 | +1.13 | -0.40 |
| Caltech101 | StanfordCars | 3 | 50.33 | 47.90 | 49.60 | -2.43 | -0.73 | -1.70 |
| Caltech101 | FGVCAircraft | 3 | 11.50 | 11.20 | 11.10 | -0.30 | -0.40 | +0.10 |
| Caltech101 | UCF101 | 3 | 47.77 | 49.10 | 48.57 | +1.33 | +0.80 | +0.53 |
| Caltech101 | SUN397 | 3 | 48.33 | 48.90 | 49.57 | +0.57 | +1.23 | -0.67 |
| Food101 | OxfordPets | 3 | 56.10 | 50.37 | 49.93 | -5.73 | -6.17 | +0.43 |
| Food101 | EuroSAT | 3 | 23.90 | 22.03 | 21.27 | -1.87 | -2.63 | +0.77 |
| Food101 | DTD | 3 | 24.97 | 21.70 | 21.93 | -3.27 | -3.03 | -0.23 |
| Food101 | OxfordFlowers | 3 | 37.37 | 38.83 | 41.30 | +1.47 | +3.93 | -2.47 |
| Food101 | Caltech101 | 3 | 63.43 | 65.93 | 66.50 | +2.50 | +3.07 | -0.57 |
| Food101 | StanfordCars | 3 | 47.77 | 49.90 | 49.70 | +2.13 | +1.93 | +0.20 |
| Food101 | FGVCAircraft | 3 | 6.17 | 6.60 | 5.63 | +0.43 | -0.53 | +0.97 |
| Food101 | UCF101 | 3 | 48.17 | 47.53 | 48.53 | -0.63 | +0.37 | -1.00 |
| Food101 | SUN397 | 3 | 38.67 | 39.37 | 38.70 | +0.70 | +0.03 | +0.67 |
| SUN397 | OxfordPets | 3 | 57.63 | 61.23 | 54.43 | +3.60 | -3.20 | +6.80 |
| SUN397 | EuroSAT | 3 | 26.60 | 25.93 | 24.37 | -0.67 | -2.23 | +1.57 |
| SUN397 | DTD | 3 | 22.60 | 24.53 | 23.00 | +1.93 | +0.40 | +1.53 |
| SUN397 | Food101 | 3 | 61.23 | 62.60 | 60.40 | +1.37 | -0.83 | +2.20 |
| SUN397 | OxfordFlowers | 3 | 39.83 | 40.67 | 38.40 | +0.83 | -1.43 | +2.27 |
| SUN397 | Caltech101 | 3 | 73.07 | 77.40 | 75.77 | +4.33 | +2.70 | +1.63 |
| SUN397 | StanfordCars | 3 | 48.73 | 46.90 | 46.80 | -1.83 | -1.93 | +0.10 |
| SUN397 | FGVCAircraft | 3 | 9.37 | 8.33 | 7.57 | -1.03 | -1.80 | +0.77 |
| SUN397 | UCF101 | 3 | 48.27 | 49.80 | 44.97 | +1.53 | -3.30 | +4.83 |

## 4-source Target-level Averages

| Source | Target | n | CoOp | Safe | Legacy | Safe-CoOp | Legacy-CoOp | Safe-Legacy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Caltech101 | OxfordPets | 3 | 72.50 | 70.80 | 72.70 | -1.70 | +0.20 | -1.90 |
| Caltech101 | EuroSAT | 3 | 21.10 | 25.20 | 23.07 | +4.10 | +1.97 | +2.13 |
| Caltech101 | DTD | 3 | 29.33 | 31.70 | 30.83 | +2.37 | +1.50 | +0.87 |
| Caltech101 | Food101 | 3 | 64.93 | 66.53 | 66.73 | +1.60 | +1.80 | -0.20 |
| Caltech101 | OxfordFlowers | 3 | 48.60 | 49.33 | 49.73 | +0.73 | +1.13 | -0.40 |
| Caltech101 | StanfordCars | 3 | 50.33 | 47.90 | 49.60 | -2.43 | -0.73 | -1.70 |
| Caltech101 | FGVCAircraft | 3 | 11.50 | 11.20 | 11.10 | -0.30 | -0.40 | +0.10 |
| Caltech101 | UCF101 | 3 | 47.77 | 49.10 | 48.57 | +1.33 | +0.80 | +0.53 |
| Caltech101 | SUN397 | 3 | 48.33 | 48.90 | 49.57 | +0.57 | +1.23 | -0.67 |
| Food101 | OxfordPets | 3 | 56.10 | 50.37 | 49.93 | -5.73 | -6.17 | +0.43 |
| Food101 | EuroSAT | 3 | 23.90 | 22.03 | 21.27 | -1.87 | -2.63 | +0.77 |
| Food101 | DTD | 3 | 24.97 | 21.70 | 21.93 | -3.27 | -3.03 | -0.23 |
| Food101 | OxfordFlowers | 3 | 37.37 | 38.83 | 41.30 | +1.47 | +3.93 | -2.47 |
| Food101 | Caltech101 | 3 | 63.43 | 65.93 | 66.50 | +2.50 | +3.07 | -0.57 |
| Food101 | StanfordCars | 3 | 47.77 | 49.90 | 49.70 | +2.13 | +1.93 | +0.20 |
| Food101 | FGVCAircraft | 3 | 6.17 | 6.60 | 5.63 | +0.43 | -0.53 | +0.97 |
| Food101 | UCF101 | 3 | 48.17 | 47.53 | 48.53 | -0.63 | +0.37 | -1.00 |
| Food101 | SUN397 | 3 | 38.67 | 39.37 | 38.70 | +0.70 | +0.03 | +0.67 |
| SUN397 | OxfordPets | 3 | 57.63 | 61.23 | 54.43 | +3.60 | -3.20 | +6.80 |
| SUN397 | EuroSAT | 3 | 26.60 | 25.93 | 24.37 | -0.67 | -2.23 | +1.57 |
| SUN397 | DTD | 3 | 22.60 | 24.53 | 23.00 | +1.93 | +0.40 | +1.53 |
| SUN397 | Food101 | 3 | 61.23 | 62.60 | 60.40 | +1.37 | -0.83 | +2.20 |
| SUN397 | OxfordFlowers | 3 | 39.83 | 40.67 | 38.40 | +0.83 | -1.43 | +2.27 |
| SUN397 | Caltech101 | 3 | 73.07 | 77.40 | 75.77 | +4.33 | +2.70 | +1.63 |
| SUN397 | StanfordCars | 3 | 48.73 | 46.90 | 46.80 | -1.83 | -1.93 | +0.10 |
| SUN397 | FGVCAircraft | 3 | 9.37 | 8.33 | 7.57 | -1.03 | -1.80 | +0.77 |
| SUN397 | UCF101 | 3 | 48.27 | 49.80 | 44.97 | +1.53 | -3.30 | +4.83 |
| OxfordPets | EuroSAT | 3 | 22.53 | 19.43 | 19.37 | -3.10 | -3.17 | +0.07 |
| OxfordPets | DTD | 3 | 22.03 | 23.90 | 23.87 | +1.87 | +1.83 | +0.03 |
| OxfordPets | Food101 | 3 | 46.53 | 51.10 | 50.60 | +4.57 | +4.07 | +0.50 |
| OxfordPets | OxfordFlowers | 3 | 40.33 | 43.17 | 42.70 | +2.83 | +2.37 | +0.47 |
| OxfordPets | Caltech101 | 3 | 69.40 | 71.13 | 71.07 | +1.73 | +1.67 | +0.07 |
| OxfordPets | StanfordCars | 3 | 44.37 | 47.03 | 47.07 | +2.67 | +2.70 | -0.03 |
| OxfordPets | FGVCAircraft | 3 | 5.70 | 7.17 | 7.30 | +1.47 | +1.60 | -0.13 |
| OxfordPets | UCF101 | 3 | 38.83 | 42.47 | 42.33 | +3.63 | +3.50 | +0.13 |
| OxfordPets | SUN397 | 3 | 35.77 | 37.97 | 37.87 | +2.20 | +2.10 | +0.10 |

## Missing Logs

No missing logs were found in the clean rerun summary.
