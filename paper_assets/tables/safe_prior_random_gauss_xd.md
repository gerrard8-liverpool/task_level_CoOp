# Safe Prior Random Gaussian Control: Cross-dataset DG

RandomGaussian is a target-independent control. It does not use target-domain or benchmark-level statistics.

| Source | Target | CoOp | Safe-Real | Safe-Random | Real-CoOp | Random-CoOp | Real-Random |
|---|---|---:|---:|---:|---:|---:|---:|
| caltech101 | oxford_pets | 72.50 | 70.80 | 73.90 | -1.70 | +1.40 | -3.10 |
| caltech101 | eurosat | 21.10 | 25.20 | 25.20 | +4.10 | +4.10 | +0.00 |
| caltech101 | dtd | 29.33 | 31.70 | 31.30 | +2.37 | +1.97 | +0.40 |
| caltech101 | food101 | 64.93 | 66.53 | 68.23 | +1.60 | +3.30 | -1.70 |
| caltech101 | oxford_flowers | 48.60 | 49.33 | 48.93 | +0.73 | +0.33 | +0.40 |
| caltech101 | stanford_cars | 50.33 | 47.90 | 48.80 | -2.43 | -1.53 | -0.90 |
| caltech101 | fgvc_aircraft | 11.50 | 11.20 | 11.17 | -0.30 | -0.33 | +0.03 |
| caltech101 | ucf101 | 47.77 | 49.10 | 49.67 | +1.33 | +1.90 | -0.57 |
| caltech101 | sun397 | 48.33 | 48.90 | 50.10 | +0.57 | +1.77 | -1.20 |
| **caltech101** | **Average** |  |  |  | **+0.70** | **+1.43** | **-0.74** |
| food101 | oxford_pets | 56.10 | 50.37 | 45.53 | -5.73 | -10.57 | +4.83 |
| food101 | eurosat | 23.90 | 22.03 | 23.90 | -1.87 | -0.00 | -1.87 |
| food101 | dtd | 24.97 | 21.70 | 22.47 | -3.27 | -2.50 | -0.77 |
| food101 | oxford_flowers | 37.37 | 38.83 | 40.13 | +1.47 | +2.77 | -1.30 |
| food101 | caltech101 | 63.43 | 65.93 | 66.30 | +2.50 | +2.87 | -0.37 |
| food101 | stanford_cars | 47.77 | 49.90 | 49.17 | +2.13 | +1.40 | +0.73 |
| food101 | fgvc_aircraft | 6.17 | 6.60 | 5.27 | +0.43 | -0.90 | +1.33 |
| food101 | ucf101 | 48.17 | 47.53 | 47.57 | -0.63 | -0.60 | -0.03 |
| food101 | sun397 | 38.67 | 39.37 | 38.67 | +0.70 | +0.00 | +0.70 |
| **food101** | **Average** |  |  |  | **-0.47** | **-0.84** | **+0.36** |
| sun397 | oxford_pets | 57.63 | 61.23 | 59.50 | +3.60 | +1.87 | +1.73 |
| sun397 | eurosat | 26.60 | 25.93 | 24.70 | -0.67 | -1.90 | +1.23 |
| sun397 | dtd | 22.60 | 24.53 | 24.80 | +1.93 | +2.20 | -0.27 |
| sun397 | food101 | 61.23 | 62.60 | 64.20 | +1.37 | +2.97 | -1.60 |
| sun397 | oxford_flowers | 39.83 | 40.67 | 42.20 | +0.83 | +2.37 | -1.53 |
| sun397 | caltech101 | 73.07 | 77.40 | 81.70 | +4.33 | +8.63 | -4.30 |
| sun397 | stanford_cars | 48.73 | 46.90 | 47.60 | -1.83 | -1.13 | -0.70 |
| sun397 | fgvc_aircraft | 9.37 | 8.33 | 10.90 | -1.03 | +1.53 | -2.57 |
| sun397 | ucf101 | 48.27 | 49.80 | 47.80 | +1.53 | -0.47 | +2.00 |
| **sun397** | **Average** |  |  |  | **+1.12** | **+1.79** | **-0.67** |

## Source-level Average

| Source | Real-CoOp | Random-CoOp | Real-Random |
|---|---:|---:|---:|
| caltech101 | +0.70 | +1.43 | -0.74 |
| food101 | -0.47 | -0.84 | +0.36 |
| sun397 | +1.12 | +1.79 | -0.67 |
| **Overall** | **+0.45** | **+0.79** | **-0.35** |

## Missing Entries

No missing entries.

## Notes

- Safe-Real uses the true source dataset feature.
- Safe-Random uses a Gaussian random vector with source-feature L2 norm matching.
- RandomGaussian is target-independent and avoids target-domain feature leakage.
- CoOp and Safe-Real are read from `third_party/CoOp_clean/output/xd/test`.
- Safe-Random is read from `outputs/ablations/safe_prior/runs/xd/test`.
