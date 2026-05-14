# Multi-source Cross-dataset DG Summary: CoOp vs Safe PriorRes vs Legacy PriorRes

Sources: `caltech101`, `food101`, `sun397`

| Source | Target | CoOp | Safe | Legacy | Safe-CoOp | Legacy-CoOp | Legacy-Safe |
|---|---|---:|---:|---:|---:|---:|---:|
| caltech101 | oxford_pets | 73.33 | 73.43 | 72.67 | 0.10 | -0.67 | -0.77 |
| caltech101 | eurosat | 16.17 | 24.50 | 26.13 | 8.33 | 9.97 | 1.63 |
| caltech101 | dtd | 29.33 | 30.63 | 31.30 | 1.30 | 1.97 | 0.67 |
| caltech101 | food101 | 65.43 | 67.00 | 68.13 | 1.57 | 2.70 | 1.13 |
| caltech101 | oxford_flowers | 49.10 | 49.47 | 49.83 | 0.37 | 0.73 | 0.37 |
| caltech101 | stanford_cars | 49.10 | 49.27 | 50.20 | 0.17 | 1.10 | 0.93 |
| caltech101 | fgvc_aircraft | 11.93 | 10.87 | 11.67 | -1.07 | -0.27 | 0.80 |
| caltech101 | ucf101 | 48.10 | 48.37 | 50.23 | 0.27 | 2.13 | 1.87 |
| caltech101 | sun397 | 47.77 | 49.60 | 50.27 | 1.83 | 2.50 | 0.67 |
| **caltech101** | **Average** |  |  |  | **1.43** | **2.24** | **0.81** |
| food101 | oxford_pets | 55.00 | 61.10 | 44.47 | 6.10 | -10.53 | -16.63 |
| food101 | eurosat | 17.40 | 26.67 | 22.90 | 9.27 | 5.50 | -3.77 |
| food101 | dtd | 20.63 | 22.57 | 21.07 | 1.93 | 0.43 | -1.50 |
| food101 | oxford_flowers | 40.87 | 40.97 | 41.63 | 0.10 | 0.77 | 0.67 |
| food101 | caltech101 | 65.80 | 65.77 | 66.60 | -0.03 | 0.80 | 0.83 |
| food101 | stanford_cars | 49.37 | 48.87 | 49.73 | -0.50 | 0.37 | 0.87 |
| food101 | fgvc_aircraft | 9.47 | 7.53 | 5.57 | -1.93 | -3.90 | -1.97 |
| food101 | ucf101 | 46.17 | 46.90 | 47.40 | 0.73 | 1.23 | 0.50 |
| food101 | sun397 | 34.13 | 38.33 | 39.13 | 4.20 | 5.00 | 0.80 |
| **food101** | **Average** |  |  |  | **2.21** | **-0.04** | **-2.24** |
| sun397 | oxford_pets | 65.60 | 65.17 | 54.87 | -0.43 | -10.73 | -10.30 |
| sun397 | eurosat | 25.33 | 20.30 | 24.33 | -5.03 | -1.00 | 4.03 |
| sun397 | dtd | 25.83 | 30.40 | 24.13 | 4.57 | -1.70 | -6.27 |
| sun397 | food101 | 63.00 | 66.40 | 61.23 | 3.40 | -1.77 | -5.17 |
| sun397 | oxford_flowers | 46.97 | 45.77 | 40.33 | -1.20 | -6.63 | -5.43 |
| sun397 | caltech101 | 77.73 | 78.80 | 74.70 | 1.07 | -3.03 | -4.10 |
| sun397 | stanford_cars | 52.07 | 52.70 | 46.07 | 0.63 | -6.00 | -6.63 |
| sun397 | fgvc_aircraft | 10.20 | 10.43 | 7.30 | 0.23 | -2.90 | -3.13 |
| sun397 | ucf101 | 53.50 | 52.07 | 45.40 | -1.43 | -8.10 | -6.67 |
| **sun397** | **Average** |  |  |  | **0.20** | **-4.65** | **-4.85** |

# Source-level Average

| Source | Safe-CoOp Avg Delta | Legacy-CoOp Avg Delta | Legacy-Safe |
|---|---:|---:|---:|
| caltech101 | 1.43 | 2.24 | 0.81 |
| food101 | 2.21 | -0.04 | -2.24 |
| sun397 | 0.20 | -4.65 | -4.85 |
| **Overall** | **1.28** | **-0.82** | **-2.10** |

Safe > CoOp seed-level cases: **52/81**
Legacy > CoOp seed-level cases: **39/81**

# Notes

- Safe = identity-centered residual, `nctx16_cscFalse_ctpend_safe_noalt`.
- Legacy = non-identity residual, `nctx16_cscFalse_ctpend_legacy_noalt`.
- All logs are read from `third_party/CoOp_clean/output/xd`.
