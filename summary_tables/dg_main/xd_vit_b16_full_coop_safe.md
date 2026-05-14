# Cross-dataset DG Backbone Extension: vit_b16

| Source | Target | CoOp | Safe | Safe-CoOp |
|---|---|---:|---:|---:|
| caltech101 | oxford_pets | 80.17 | 78.30 | -1.87 |
| caltech101 | eurosat | 44.40 | 45.50 | 1.10 |
| caltech101 | dtd | 35.37 | 33.73 | -1.63 |
| caltech101 | food101 | 81.93 | 80.57 | -1.37 |
| caltech101 | oxford_flowers | 59.57 | 59.37 | -0.20 |
| caltech101 | stanford_cars | 60.53 | 61.60 | 1.07 |
| caltech101 | fgvc_aircraft | 17.33 | 14.83 | -2.50 |
| caltech101 | ucf101 | 57.50 | 57.77 | 0.27 |
| caltech101 | sun397 | 57.17 | 57.97 | 0.80 |
| **caltech101** | **Average** |  |  | **-0.48** |
| food101 | oxford_pets | 74.93 | 75.70 | 0.77 |
| food101 | eurosat | 36.63 | 47.37 | 10.73 |
| food101 | dtd | 29.50 | 27.43 | -2.07 |
| food101 | oxford_flowers | 51.87 | 53.20 | 1.33 |
| food101 | caltech101 | 77.80 | 77.50 | -0.30 |
| food101 | stanford_cars | 59.00 | 58.63 | -0.37 |
| food101 | fgvc_aircraft | 9.40 | 8.73 | -0.67 |
| food101 | ucf101 | 51.13 | 53.30 | 2.17 |
| food101 | sun397 | 41.70 | 44.93 | 3.23 |
| **food101** | **Average** |  |  | **1.65** |
| sun397 | oxford_pets | 64.87 | 67.23 | 2.37 |
| sun397 | eurosat | 38.50 | 44.37 | 5.87 |
| sun397 | dtd | 34.67 | 33.73 | -0.93 |
| sun397 | food101 | 76.47 | 77.20 | 0.73 |
| sun397 | oxford_flowers | 50.97 | 53.17 | 2.20 |
| sun397 | caltech101 | 86.90 | 88.67 | 1.77 |
| sun397 | stanford_cars | 59.93 | 58.20 | -1.73 |
| sun397 | fgvc_aircraft | 13.30 | 10.27 | -3.03 |
| sun397 | ucf101 | 60.53 | 61.00 | 0.47 |
| **sun397** | **Average** |  |  | **0.86** |

# Source-level Average

| Source | Safe-CoOp Avg Delta |
|---|---:|
| caltech101 | -0.48 |
| food101 | 1.65 |
| sun397 | 0.86 |
| **Overall** | **0.67** |

Safe > CoOp seed-level cases: **44/81**
