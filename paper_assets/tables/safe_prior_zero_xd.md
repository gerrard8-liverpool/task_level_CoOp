# Safe Prior Zero Control: Cross-dataset DG

Safe-Zero is a blank-information control. It uses an all-zero task feature and does not use any source, target, or benchmark-level statistics.

| Source | Target | CoOp | Safe-Real | Safe-Zero | Real-CoOp | Zero-CoOp | Real-Zero |
|---|---|---:|---:|---:|---:|---:|---:|
| caltech101 | oxford_pets | 72.50 | 70.80 | 70.70 | -1.70 | -1.80 | +0.10 |
| caltech101 | eurosat | 21.10 | 25.20 | 23.50 | +4.10 | +2.40 | +1.70 |
| caltech101 | dtd | 29.33 | 31.70 | 29.30 | +2.37 | -0.03 | +2.40 |
| caltech101 | food101 | 64.93 | 66.53 | 70.60 | +1.60 | +5.67 | -4.07 |
| caltech101 | oxford_flowers | 48.60 | 49.33 | 49.90 | +0.73 | +1.30 | -0.57 |
| caltech101 | stanford_cars | 50.33 | 47.90 | 52.90 | -2.43 | +2.57 | -5.00 |
| caltech101 | fgvc_aircraft | 11.50 | 11.20 | 10.90 | -0.30 | -0.60 | +0.30 |
| caltech101 | ucf101 | 47.77 | 49.10 | 46.20 | +1.33 | -1.57 | +2.90 |
| caltech101 | sun397 | 48.33 | 48.90 | 48.50 | +0.57 | +0.17 | +0.40 |
| **caltech101** | **Average** |  |  |  | **+0.70** | **+0.90** | **-0.20** |
| food101 | oxford_pets | 56.10 | 50.37 |  | -5.73 |  |  |
| food101 | eurosat | 23.90 | 22.03 |  | -1.87 |  |  |
| food101 | dtd | 24.97 | 21.70 |  | -3.27 |  |  |
| food101 | oxford_flowers | 37.37 | 38.83 |  | +1.47 |  |  |
| food101 | caltech101 | 63.43 | 65.93 |  | +2.50 |  |  |
| food101 | stanford_cars | 47.77 | 49.90 |  | +2.13 |  |  |
| food101 | fgvc_aircraft | 6.17 | 6.60 |  | +0.43 |  |  |
| food101 | ucf101 | 48.17 | 47.53 |  | -0.63 |  |  |
| food101 | sun397 | 38.67 | 39.37 |  | +0.70 |  |  |
| **food101** | **Average** |  |  |  | **-0.47** | **** | **** |
| sun397 | oxford_pets | 57.63 | 61.23 |  | +3.60 |  |  |
| sun397 | eurosat | 26.60 | 25.93 |  | -0.67 |  |  |
| sun397 | dtd | 22.60 | 24.53 |  | +1.93 |  |  |
| sun397 | food101 | 61.23 | 62.60 |  | +1.37 |  |  |
| sun397 | oxford_flowers | 39.83 | 40.67 |  | +0.83 |  |  |
| sun397 | caltech101 | 73.07 | 77.40 |  | +4.33 |  |  |
| sun397 | stanford_cars | 48.73 | 46.90 |  | -1.83 |  |  |
| sun397 | fgvc_aircraft | 9.37 | 8.33 |  | -1.03 |  |  |
| sun397 | ucf101 | 48.27 | 49.80 |  | +1.53 |  |  |
| **sun397** | **Average** |  |  |  | **+1.12** | **** | **** |

## Source-level Average

| Source | Real-CoOp | Zero-CoOp | Real-Zero |
|---|---:|---:|---:|
| caltech101 | +0.70 | +0.90 | -0.20 |
| food101 | -0.47 |  |  |
| sun397 | +1.12 |  |  |
| **Overall** | **+0.45** | **+0.90** | **-0.20** |

## Missing Entries

| Source | Target | Method |
|---|---|---|
| food101 | oxford_pets | Safe-Zero |
| food101 | eurosat | Safe-Zero |
| food101 | dtd | Safe-Zero |
| food101 | oxford_flowers | Safe-Zero |
| food101 | caltech101 | Safe-Zero |
| food101 | stanford_cars | Safe-Zero |
| food101 | fgvc_aircraft | Safe-Zero |
| food101 | ucf101 | Safe-Zero |
| food101 | sun397 | Safe-Zero |
| sun397 | oxford_pets | Safe-Zero |
| sun397 | eurosat | Safe-Zero |
| sun397 | dtd | Safe-Zero |
| sun397 | food101 | Safe-Zero |
| sun397 | oxford_flowers | Safe-Zero |
| sun397 | caltech101 | Safe-Zero |
| sun397 | stanford_cars | Safe-Zero |
| sun397 | fgvc_aircraft | Safe-Zero |
| sun397 | ucf101 | Safe-Zero |

## Notes

- Safe-Real uses the true source dataset feature.
- Safe-Zero uses an all-zero feature vector.
- Safe-Zero is target-independent and contains no dataset statistics.
- CoOp and Safe-Real are read from `third_party/CoOp_clean/output/xd/test`.
- Safe-Zero is read from `outputs/ablations/safe_prior/runs/xd/test`.
