# Five-dataset experiment summary (strict dedup)

## Best setting per dataset / method / setting_tag

| dataset | method | setting_tag | shots | nctx | acc_mean | acc_std | f1_mean | f1_std | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dtd | CoOp | baseline | 14 | 16 | 64.5667 | 0.2625 | 64.2333 | 0.2867 | 1,2,3 |
| dtd | CoOpPriorRes | b_w0.2 | 16 | 16 | 65.4333 | 0.5185 | 65.1333 | 0.5907 | 1,2,3 |
| dtd | CoOpPriorRes | no_b | 16 | 16 | 64.6667 | 0.4028 | 64.2 | 0.432 | 1,2,3 |
| eurosat | CoOp | baseline | 16 | 14 | 81.0333 | 0.9672 | 80.2667 | 1.0209 | 1,2,3 |
| eurosat | CoOpPriorRes | b_w0.2 | 16 | 16 | 77.8667 | 4.5682 | 77.1667 | 4.619 | 1,2,3 |
| eurosat | CoOpPriorRes | no_b | 16 | 16 | 77.7333 | 4.6147 | 76.9667 | 4.619 | 1,2,3 |
| food101 | CoOp | baseline | 16 | 2 | 78.6667 | 0.0943 | 78.6 | 0.0816 | 1,2,3 |
| food101 | CoOpPriorRes | b_w0.2 | 16 | 16 | 77.3 | 0.2449 | 77.2333 | 0.2867 | 1,2,3 |
| food101 | CoOpPriorRes | no_b | 16 | 16 | 77.3 | 0.3 | 77.25 | 0.25 | 2,3 |
| oxford_flowers | CoOp | baseline | 16 | 12 | 93.6 | 0.1633 | 93.4 | 0.216 | 1,2,3 |
| oxford_flowers | CoOpPriorRes | b_w0.2 | 16 | 16 | 93.0667 | 0.411 | 92.6667 | 0.6128 | 1,2,3 |
| oxford_flowers | CoOpPriorRes | no_b | 16 | 16 | 93.4667 | 0.2055 | 93.3 | 0.1414 | 1,2,3 |
| oxford_pets | CoOpPriorRes | b_w0.2 | 16 | 16 | 88.8667 | 0.33 | 88.7667 | 0.3682 | 1,2,3 |
| oxford_pets | CoOpPriorRes | no_b | 16 | 16 | 88.9 | 0.0816 | 88.8 | 0.0816 | 1,2,3 |

## Duplicate review needed

| formal_key | issue | override | candidates |
| --- | --- | --- | --- |
| dtd|CoOp|baseline|k16|m16|seed1 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/dtd/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/dtd/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_cachebuild/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/dtd/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_smoke/log.txt |
| dtd|CoOp|baseline|k16|m16|seed2 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/dtd/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed2/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/dtd/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed2_cachebuild/log.txt |
| dtd|CoOp|baseline|k16|m16|seed3 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/dtd/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed3/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/dtd/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed3_cachebuild/log.txt |
| eurosat|CoOp|baseline|k16|m16|seed1 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/eurosat/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/eurosat/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_fresh/log.txt |
| eurosat|CoOp|baseline|k16|m16|seed2 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/eurosat/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed2/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/eurosat/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed2_fresh/log.txt |
| eurosat|CoOp|baseline|k16|m16|seed3 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/eurosat/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed3/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/eurosat/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed3_fresh/log.txt |
| food101|CoOp|baseline|k16|m16|seed1 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_cachebuild/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_smoke/log.txt |
| food101|CoOp|baseline|k16|m16|seed2 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed2/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed2_cachebuild/log.txt |
| food101|CoOp|baseline|k16|m16|seed3 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed3/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed3_cachebuild/log.txt |
| food101|CoOpPriorRes|no_b|k16|m16|seed1 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOpPriorRes/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_nob_fresh/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/food101/CoOpPriorRes/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_nob_smoke/log.txt |
| oxford_flowers|CoOp|baseline|k16|m16|seed1 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_flowers/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_flowers/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_cachebuild/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_flowers/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed1_smoke2/log.txt |
| oxford_flowers|CoOp|baseline|k16|m16|seed2 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_flowers/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed2/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_flowers/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed2_cachebuild/log.txt |
| oxford_flowers|CoOp|baseline|k16|m16|seed3 | duplicate_requires_manual_confirmation |  | /workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_flowers/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed3/log.txt || /workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_flowers/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend_seed3_cachebuild/log.txt |

## oxford_pets

| method | setting_tag | shots | nctx | num_runs | seeds | acc_mean | acc_std | f1_mean | f1_std | best_acc | worst_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoOpPriorRes | b_w0.2 | 16 | 16 | 3 | 1,2,3 | 88.8667 | 0.33 | 88.7667 | 0.3682 | 89.3 | 88.5 |
| CoOpPriorRes | no_b | 16 | 16 | 3 | 1,2,3 | 88.9 | 0.0816 | 88.8 | 0.0816 | 89.0 | 88.8 |

## eurosat

| method | setting_tag | shots | nctx | num_runs | seeds | acc_mean | acc_std | f1_mean | f1_std | best_acc | worst_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoOp | baseline | 2 | 2 | 3 | 1,2,3 | 46.9 | 10.7241 | 43.7 | 11.8696 | 56.8 | 32.0 |
| CoOp | baseline | 2 | 4 | 3 | 1,2,3 | 57.6 | 5.7868 | 56.4333 | 5.9668 | 65.5 | 51.8 |
| CoOp | baseline | 2 | 6 | 3 | 1,2,3 | 52.2333 | 1.5283 | 49.7667 | 2.0758 | 53.6 | 50.1 |
| CoOp | baseline | 2 | 8 | 3 | 1,2,3 | 52.2 | 6.7587 | 49.0 | 7.8128 | 61.6 | 46.0 |
| CoOp | baseline | 2 | 10 | 3 | 1,2,3 | 51.7667 | 6.8786 | 49.6667 | 7.1598 | 60.9 | 44.3 |
| CoOp | baseline | 2 | 12 | 3 | 1,2,3 | 56.0667 | 1.4974 | 55.2 | 1.1225 | 57.5 | 54.0 |
| CoOp | baseline | 2 | 14 | 3 | 1,2,3 | 54.9333 | 3.8056 | 54.1667 | 3.6003 | 59.1 | 49.9 |
| CoOp | baseline | 2 | 16 | 3 | 1,2,3 | 57.9333 | 0.8957 | 57.2667 | 1.4704 | 59.2 | 57.3 |
| CoOp | baseline | 4 | 2 | 3 | 1,2,3 | 58.6 | 0.432 | 55.9333 | 0.419 | 59.2 | 58.2 |
| CoOp | baseline | 4 | 4 | 3 | 1,2,3 | 60.8667 | 2.3414 | 58.8667 | 3.6554 | 63.9 | 58.2 |
| CoOp | baseline | 4 | 6 | 3 | 1,2,3 | 57.0667 | 2.5316 | 55.0333 | 2.1453 | 60.2 | 54.0 |
| CoOp | baseline | 4 | 8 | 3 | 1,2,3 | 58.1333 | 9.6043 | 55.7 | 10.2101 | 67.4 | 44.9 |
| CoOp | baseline | 4 | 10 | 3 | 1,2,3 | 53.7 | 5.6622 | 51.0 | 5.9755 | 61.7 | 49.4 |
| CoOp | baseline | 4 | 12 | 3 | 1,2,3 | 58.6 | 0.7874 | 56.3333 | 1.7632 | 59.7 | 57.9 |
| CoOp | baseline | 4 | 14 | 3 | 1,2,3 | 63.1667 | 2.0138 | 61.5333 | 1.8696 | 66.0 | 61.5 |
| CoOp | baseline | 4 | 16 | 3 | 1,2,3 | 63.3667 | 2.4567 | 61.4667 | 3.1415 | 66.5 | 60.5 |
| CoOp | baseline | 8 | 2 | 3 | 1,2,3 | 64.4333 | 5.5295 | 63.6 | 6.0437 | 71.5 | 58.0 |
| CoOp | baseline | 8 | 4 | 3 | 1,2,3 | 68.1667 | 1.8696 | 67.5333 | 1.6997 | 70.0 | 65.6 |
| CoOp | baseline | 8 | 6 | 3 | 1,2,3 | 67.5667 | 6.3163 | 66.0 | 6.7661 | 76.4 | 62.0 |
| CoOp | baseline | 8 | 8 | 2 | 2,3 | 68.85 | 3.75 | 68.35 | 3.85 | 72.6 | 65.1 |
| CoOp | baseline | 8 | 10 | 3 | 1,2,3 | 69.8333 | 6.5281 | 69.2333 | 6.5429 | 79.0 | 64.3 |
| CoOp | baseline | 8 | 12 | 3 | 1,2,3 | 70.5667 | 3.7259 | 69.3667 | 3.9381 | 75.6 | 66.7 |
| CoOp | baseline | 8 | 14 | 3 | 1,2,3 | 68.1667 | 1.819 | 67.1333 | 2.2216 | 69.6 | 65.6 |
| CoOp | baseline | 8 | 16 | 3 | 1,2,3 | 71.4 | 2.0461 | 70.8333 | 2.1669 | 73.8 | 68.8 |
| CoOp | baseline | 10 | 2 | 3 | 1,2,3 | 73.5667 | 0.33 | 72.9 | 0.432 | 74.0 | 73.2 |
| CoOp | baseline | 10 | 4 | 2 | 1,2 | 73.6 | 1.7 | 72.8 | 2.2 | 75.3 | 71.9 |
| CoOp | baseline | 10 | 6 | 3 | 1,2,3 | 70.3667 | 6.2098 | 69.4667 | 6.2899 | 77.8 | 62.6 |
| CoOp | baseline | 10 | 8 | 3 | 1,2,3 | 73.8 | 6.7186 | 73.0667 | 6.8295 | 79.7 | 64.4 |
| CoOp | baseline | 10 | 10 | 3 | 1,2,3 | 75.8333 | 2.0287 | 75.1667 | 2.0138 | 78.7 | 74.3 |
| CoOp | baseline | 10 | 12 | 2 | 2,3 | 76.7 | 2.2 | 76.15 | 2.05 | 78.9 | 74.5 |
| CoOp | baseline | 10 | 14 | 3 | 1,2,3 | 75.7 | 1.5578 | 75.0 | 1.2028 | 77.9 | 74.5 |
| CoOp | baseline | 10 | 16 | 3 | 1,2,3 | 74.1333 | 1.0781 | 73.4667 | 1.0499 | 75.3 | 72.7 |
| CoOp | baseline | 12 | 2 | 3 | 1,2,3 | 69.2 | 4.9538 | 68.1 | 5.4485 | 73.7 | 62.3 |
| CoOp | baseline | 12 | 4 | 3 | 1,2,3 | 73.9333 | 2.1761 | 73.2667 | 1.9602 | 75.9 | 70.9 |
| CoOp | baseline | 12 | 6 | 3 | 1,2,3 | 70.8667 | 5.5428 | 70.0 | 5.678 | 78.7 | 66.7 |
| CoOp | baseline | 12 | 8 | 3 | 1,2,3 | 71.4333 | 4.6162 | 70.6667 | 4.5065 | 77.8 | 67.0 |
| CoOp | baseline | 12 | 10 | 3 | 1,2,3 | 72.9 | 3.9825 | 72.1667 | 4.3208 | 78.2 | 68.6 |
| CoOp | baseline | 12 | 12 | 3 | 1,2,3 | 75.5333 | 2.4998 | 74.7 | 2.5923 | 79.0 | 73.2 |
| CoOp | baseline | 12 | 14 | 3 | 1,2,3 | 76.2333 | 1.9067 | 75.5333 | 1.7461 | 78.3 | 73.7 |
| CoOp | baseline | 12 | 16 | 3 | 1,2,3 | 76.8667 | 1.3225 | 76.0 | 1.5578 | 77.9 | 75.0 |
| CoOp | baseline | 14 | 2 | 3 | 1,2,3 | 74.8333 | 2.2455 | 73.9 | 2.3036 | 77.6 | 72.1 |
| CoOp | baseline | 14 | 4 | 3 | 1,2,3 | 76.0333 | 1.9258 | 75.1667 | 1.9189 | 78.5 | 73.8 |
| CoOp | baseline | 14 | 6 | 3 | 1,2,3 | 74.3 | 2.6882 | 73.2667 | 2.4527 | 76.3 | 70.5 |
| CoOp | baseline | 14 | 8 | 3 | 1,2,3 | 71.8 | 3.1496 | 70.7667 | 3.1138 | 76.2 | 69.0 |
| CoOp | baseline | 14 | 10 | 3 | 1,2,3 | 75.9333 | 1.0339 | 75.4333 | 0.9428 | 76.9 | 74.5 |
| CoOp | baseline | 14 | 12 | 3 | 1,2,3 | 77.1 | 1.4514 | 76.2667 | 1.5173 | 78.5 | 75.1 |
| CoOp | baseline | 14 | 14 | 3 | 1,2,3 | 77.5 | 1.1431 | 76.6 | 0.9933 | 79.1 | 76.5 |
| CoOp | baseline | 14 | 16 | 3 | 1,2,3 | 77.6 | 2.4536 | 76.8667 | 2.5773 | 79.9 | 74.2 |
| CoOp | baseline | 16 | 2 | 3 | 1,2,3 | 74.5333 | 0.419 | 73.8667 | 0.33 | 75.1 | 74.1 |
| CoOp | baseline | 16 | 4 | 3 | 1,2,3 | 78.2333 | 1.7211 | 77.5333 | 1.7932 | 79.5 | 75.8 |
| CoOp | baseline | 16 | 6 | 3 | 1,2,3 | 77.8667 | 0.826 | 77.1 | 0.9201 | 78.5 | 76.7 |
| CoOp | baseline | 16 | 8 | 3 | 1,2,3 | 79.0 | 1.0614 | 78.2333 | 1.0873 | 79.8 | 77.5 |
| CoOp | baseline | 16 | 10 | 3 | 1,2,3 | 78.2333 | 1.4704 | 77.6 | 1.49 | 80.3 | 77.0 |
| CoOp | baseline | 16 | 12 | 3 | 1,2,3 | 80.9 | 0.7874 | 80.2333 | 0.8055 | 81.6 | 79.8 |
| CoOp | baseline | 16 | 14 | 3 | 1,2,3 | 81.0333 | 0.9672 | 80.2667 | 1.0209 | 82.4 | 80.3 |
| CoOpPriorRes | b_w0.2 | 16 | 16 | 3 | 1,2,3 | 77.8667 | 4.5682 | 77.1667 | 4.619 | 82.0 | 71.5 |
| CoOpPriorRes | no_b | 16 | 16 | 3 | 1,2,3 | 77.7333 | 4.6147 | 76.9667 | 4.619 | 81.9 | 71.3 |

## dtd

| method | setting_tag | shots | nctx | num_runs | seeds | acc_mean | acc_std | f1_mean | f1_std | best_acc | worst_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoOp | baseline | 1 | 2 | 3 | 1,2,3 | 43.9333 | 1.4384 | 41.7 | 1.4166 | 45.0 | 41.9 |
| CoOp | baseline | 1 | 4 | 3 | 1,2,3 | 42.8333 | 2.0434 | 40.6 | 1.9026 | 45.4 | 40.4 |
| CoOp | baseline | 1 | 6 | 3 | 1,2,3 | 34.7333 | 8.448 | 30.2333 | 10.8644 | 43.2 | 23.2 |
| CoOp | baseline | 1 | 8 | 3 | 1,2,3 | 42.5333 | 2.0072 | 40.0333 | 2.2066 | 45.3 | 40.6 |
| CoOp | baseline | 1 | 10 | 3 | 1,2,3 | 43.0333 | 1.6214 | 41.2667 | 1.2658 | 44.6 | 40.8 |
| CoOp | baseline | 1 | 12 | 3 | 1,2,3 | 43.2333 | 0.0943 | 41.6667 | 0.4028 | 43.3 | 43.1 |
| CoOp | baseline | 1 | 14 | 3 | 1,2,3 | 43.7 | 1.5297 | 41.8333 | 1.7969 | 45.2 | 41.6 |
| CoOp | baseline | 1 | 16 | 3 | 1,2,3 | 43.6333 | 1.1025 | 41.4333 | 1.0371 | 45.0 | 42.3 |
| CoOp | baseline | 2 | 2 | 3 | 1,2,3 | 48.5333 | 1.5107 | 45.9333 | 1.271 | 50.4 | 46.7 |
| CoOp | baseline | 2 | 4 | 3 | 1,2,3 | 50.0667 | 0.838 | 48.2667 | 1.2284 | 51.2 | 49.2 |
| CoOp | baseline | 2 | 6 | 3 | 1,2,3 | 42.2 | 6.0017 | 38.7 | 7.6424 | 49.5 | 34.8 |
| CoOp | baseline | 2 | 8 | 3 | 1,2,3 | 47.5667 | 2.473 | 45.7333 | 3.4179 | 49.7 | 44.1 |
| CoOp | baseline | 2 | 10 | 3 | 1,2,3 | 47.6667 | 0.826 | 45.8667 | 1.4079 | 48.3 | 46.5 |
| CoOp | baseline | 2 | 12 | 3 | 1,2,3 | 48.4333 | 2.1761 | 47.5333 | 2.0039 | 50.4 | 45.4 |
| CoOp | baseline | 2 | 14 | 3 | 1,2,3 | 48.1667 | 1.8154 | 46.8667 | 2.1297 | 49.5 | 45.6 |
| CoOp | baseline | 2 | 16 | 3 | 1,2,3 | 49.4 | 0.432 | 48.1 | 0.7257 | 49.8 | 48.8 |
| CoOp | baseline | 4 | 2 | 3 | 1,2,3 | 52.0333 | 2.4851 | 50.2333 | 2.7717 | 55.5 | 49.8 |
| CoOp | baseline | 4 | 4 | 3 | 1,2,3 | 54.2667 | 1.4008 | 53.0667 | 1.676 | 56.1 | 52.7 |
| CoOp | baseline | 4 | 6 | 3 | 1,2,3 | 52.0333 | 1.3225 | 51.0333 | 1.674 | 53.9 | 51.0 |
| CoOp | baseline | 4 | 8 | 3 | 1,2,3 | 52.3333 | 2.6285 | 51.1667 | 2.917 | 55.1 | 48.8 |
| CoOp | baseline | 4 | 10 | 3 | 1,2,3 | 53.4 | 0.0816 | 52.4 | 0.2944 | 53.5 | 53.3 |
| CoOp | baseline | 4 | 12 | 3 | 1,2,3 | 54.4333 | 0.17 | 53.8333 | 0.3091 | 54.6 | 54.2 |
| CoOp | baseline | 4 | 14 | 3 | 1,2,3 | 54.3333 | 1.6997 | 53.4333 | 1.6438 | 56.0 | 52.0 |
| CoOp | baseline | 4 | 16 | 3 | 1,2,3 | 54.3 | 1.0231 | 53.4333 | 0.9672 | 55.6 | 53.1 |
| CoOp | baseline | 8 | 2 | 3 | 1,2,3 | 57.1667 | 1.7556 | 56.4 | 1.9596 | 59.3 | 55.0 |
| CoOp | baseline | 8 | 4 | 3 | 1,2,3 | 59.5333 | 0.9428 | 58.9667 | 1.0403 | 60.2 | 58.2 |
| CoOp | baseline | 8 | 6 | 3 | 1,2,3 | 58.3333 | 0.8957 | 57.6667 | 1.0209 | 59.6 | 57.7 |
| CoOp | baseline | 8 | 8 | 3 | 1,2,3 | 60.9333 | 1.4522 | 60.6667 | 1.5195 | 62.5 | 59.0 |
| CoOp | baseline | 8 | 10 | 3 | 1,2,3 | 60.7667 | 0.7318 | 60.5 | 0.6481 | 61.8 | 60.2 |
| CoOp | baseline | 8 | 12 | 3 | 1,2,3 | 60.7667 | 0.6182 | 60.3667 | 0.834 | 61.3 | 59.9 |
| CoOp | baseline | 8 | 14 | 3 | 1,2,3 | 59.6333 | 1.0403 | 59.4333 | 0.8994 | 61.1 | 58.8 |
| CoOp | baseline | 8 | 16 | 3 | 1,2,3 | 61.4 | 0.5099 | 61.2 | 0.4546 | 61.9 | 60.7 |
| CoOp | baseline | 10 | 2 | 3 | 1,2,3 | 58.4 | 1.8991 | 57.8333 | 2.1061 | 60.9 | 56.3 |
| CoOp | baseline | 10 | 4 | 3 | 1,2,3 | 60.0667 | 0.9568 | 59.5 | 1.0677 | 61.4 | 59.2 |
| CoOp | baseline | 10 | 6 | 3 | 1,2,3 | 60.5667 | 2.1669 | 60.0 | 2.2091 | 63.3 | 58.0 |
| CoOp | baseline | 10 | 8 | 3 | 1,2,3 | 61.2 | 0.7348 | 60.6333 | 0.8179 | 62.1 | 60.3 |
| CoOp | baseline | 10 | 10 | 3 | 1,2,3 | 61.4333 | 1.2037 | 61.1333 | 1.2815 | 63.1 | 60.3 |
| CoOp | baseline | 10 | 12 | 3 | 1,2,3 | 61.0333 | 0.9843 | 60.7667 | 0.9393 | 62.3 | 59.9 |
| CoOp | baseline | 10 | 14 | 3 | 1,2,3 | 62.1 | 0.7118 | 61.9 | 0.7789 | 62.7 | 61.1 |
| CoOp | baseline | 10 | 16 | 3 | 1,2,3 | 61.6667 | 1.8927 | 61.5667 | 1.8874 | 63.2 | 59.0 |
| CoOp | baseline | 12 | 2 | 3 | 1,2,3 | 59.4333 | 0.9393 | 58.5333 | 1.0339 | 60.6 | 58.3 |
| CoOp | baseline | 12 | 4 | 3 | 1,2,3 | 62.1 | 0.5657 | 61.5667 | 0.6182 | 62.5 | 61.3 |
| CoOp | baseline | 12 | 6 | 3 | 1,2,3 | 62.4 | 0.5099 | 61.8667 | 0.3771 | 63.1 | 61.9 |
| CoOp | baseline | 12 | 8 | 3 | 1,2,3 | 62.7333 | 0.1247 | 62.2667 | 0.1247 | 62.9 | 62.6 |
| CoOp | baseline | 12 | 10 | 3 | 1,2,3 | 62.0333 | 0.3091 | 61.6 | 0.216 | 62.3 | 61.6 |
| CoOp | baseline | 12 | 12 | 3 | 1,2,3 | 63.6333 | 0.411 | 63.1667 | 0.4784 | 64.1 | 63.1 |
| CoOp | baseline | 12 | 14 | 3 | 1,2,3 | 63.7333 | 0.8219 | 63.3333 | 0.9104 | 64.8 | 62.8 |
| CoOp | baseline | 12 | 16 | 3 | 1,2,3 | 63.3 | 0.6164 | 62.9333 | 0.6944 | 64.0 | 62.5 |
| CoOp | baseline | 14 | 2 | 3 | 1,2,3 | 60.8 | 0.9092 | 59.9 | 1.0708 | 61.8 | 59.6 |
| CoOp | baseline | 14 | 4 | 3 | 1,2,3 | 62.8667 | 0.3682 | 62.3333 | 0.3859 | 63.3 | 62.4 |
| CoOp | baseline | 14 | 6 | 3 | 1,2,3 | 62.8333 | 0.9534 | 62.3667 | 0.8654 | 64.1 | 61.8 |
| CoOp | baseline | 14 | 8 | 3 | 1,2,3 | 62.8 | 0.3266 | 62.3333 | 0.3771 | 63.2 | 62.4 |
| CoOp | baseline | 14 | 10 | 3 | 1,2,3 | 62.4667 | 1.2472 | 62.1667 | 1.3021 | 63.8 | 60.8 |
| CoOp | baseline | 14 | 12 | 3 | 1,2,3 | 63.9333 | 0.9393 | 63.5 | 1.0614 | 65.1 | 62.8 |
| CoOp | baseline | 14 | 14 | 3 | 1,2,3 | 63.6667 | 0.9393 | 63.4 | 0.9416 | 64.8 | 62.5 |
| CoOp | baseline | 14 | 16 | 3 | 1,2,3 | 64.5667 | 0.2625 | 64.2333 | 0.2867 | 64.8 | 64.2 |
| CoOp | baseline | 16 | 2 | 3 | 1,2,3 | 60.8333 | 0.411 | 60.2 | 0.3559 | 61.3 | 60.3 |
| CoOp | baseline | 16 | 4 | 3 | 1,2,3 | 64.1 | 0.5657 | 63.5333 | 0.5249 | 64.5 | 63.3 |
| CoOp | baseline | 16 | 6 | 3 | 1,2,3 | 64.4 | 0.5354 | 63.9333 | 0.5312 | 65.1 | 63.8 |
| CoOp | baseline | 16 | 8 | 3 | 1,2,3 | 64.2667 | 0.3399 | 63.7667 | 0.2625 | 64.6 | 63.8 |
| CoOp | baseline | 16 | 10 | 3 | 1,2,3 | 63.8667 | 1.8927 | 63.2667 | 1.8927 | 65.4 | 61.2 |
| CoOp | baseline | 16 | 12 | 3 | 1,2,3 | 64.5667 | 0.6944 | 64.3667 | 0.6549 | 65.4 | 63.7 |
| CoOp | baseline | 16 | 14 | 3 | 1,2,3 | 64.0 | 0.3742 | 63.5 | 0.4967 | 64.5 | 63.6 |
| CoOpPriorRes | b_w0.2 | 16 | 16 | 3 | 1,2,3 | 65.4333 | 0.5185 | 65.1333 | 0.5907 | 65.8 | 64.7 |
| CoOpPriorRes | no_b | 16 | 16 | 3 | 1,2,3 | 64.6667 | 0.4028 | 64.2 | 0.432 | 65.0 | 64.1 |

## food101

| method | setting_tag | shots | nctx | num_runs | seeds | acc_mean | acc_std | f1_mean | f1_std | best_acc | worst_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoOp | baseline | 1 | 2 | 3 | 1,2,3 | 72.2333 | 0.8576 | 71.6333 | 0.9463 | 73.3 | 71.2 |
| CoOp | baseline | 1 | 4 | 3 | 1,2,3 | 74.5333 | 1.5151 | 74.0 | 1.7378 | 76.3 | 72.6 |
| CoOp | baseline | 1 | 6 | 3 | 1,2,3 | 72.2333 | 0.6944 | 71.7 | 0.8832 | 73.2 | 71.6 |
| CoOp | baseline | 1 | 8 | 3 | 1,2,3 | 73.4333 | 1.9293 | 73.0 | 2.197 | 76.1 | 71.6 |
| CoOp | baseline | 1 | 10 | 3 | 1,2,3 | 74.6333 | 1.4659 | 74.3 | 1.6513 | 76.0 | 72.6 |
| CoOp | baseline | 1 | 12 | 3 | 1,2,3 | 74.4333 | 1.6007 | 74.0333 | 1.7016 | 76.5 | 72.6 |
| CoOp | baseline | 1 | 14 | 3 | 1,2,3 | 74.0667 | 1.212 | 73.6667 | 1.4817 | 75.7 | 72.8 |
| CoOp | baseline | 1 | 16 | 3 | 1,2,3 | 74.6333 | 0.2055 | 74.3 | 0.2449 | 74.9 | 74.4 |
| CoOp | baseline | 2 | 2 | 3 | 1,2,3 | 75.2333 | 1.7783 | 74.9667 | 1.8117 | 77.0 | 72.8 |
| CoOp | baseline | 2 | 4 | 3 | 1,2,3 | 74.3667 | 2.424 | 74.1 | 2.514 | 76.9 | 71.1 |
| CoOp | baseline | 2 | 6 | 3 | 1,2,3 | 74.3 | 2.3537 | 73.9667 | 2.4418 | 77.4 | 71.7 |
| CoOp | baseline | 2 | 8 | 3 | 1,2,3 | 76.1333 | 1.7327 | 75.9 | 1.8457 | 77.6 | 73.7 |
| CoOp | baseline | 2 | 10 | 3 | 1,2,3 | 76.5333 | 0.9534 | 76.3667 | 0.9978 | 77.8 | 75.5 |
| CoOp | baseline | 2 | 12 | 3 | 1,2,3 | 76.0 | 1.314 | 75.8 | 1.4445 | 77.5 | 74.3 |
| CoOp | baseline | 2 | 14 | 3 | 1,2,3 | 76.6667 | 0.8654 | 76.5 | 0.9092 | 77.8 | 75.7 |
| CoOp | baseline | 2 | 16 | 3 | 1,2,3 | 75.9 | 0.8524 | 75.8333 | 0.8994 | 77.1 | 75.2 |
| CoOp | baseline | 4 | 2 | 3 | 1,2,3 | 76.9333 | 0.5793 | 76.8333 | 0.4989 | 77.7 | 76.3 |
| CoOp | baseline | 4 | 4 | 3 | 1,2,3 | 75.7667 | 1.782 | 75.6 | 1.8184 | 77.7 | 73.4 |
| CoOp | baseline | 4 | 6 | 3 | 1,2,3 | 76.0 | 1.0801 | 75.8667 | 1.1898 | 77.0 | 74.5 |
| CoOp | baseline | 4 | 8 | 3 | 1,2,3 | 76.3 | 1.0424 | 76.1667 | 1.0781 | 77.7 | 75.2 |
| CoOp | baseline | 4 | 10 | 3 | 1,2,3 | 77.1 | 0.4243 | 77.0 | 0.4243 | 77.4 | 76.5 |
| CoOp | baseline | 4 | 12 | 3 | 1,2,3 | 76.2667 | 0.7587 | 76.1333 | 0.8014 | 76.9 | 75.2 |
| CoOp | baseline | 4 | 14 | 3 | 1,2,3 | 76.3333 | 0.5437 | 76.2333 | 0.6236 | 76.9 | 75.6 |
| CoOp | baseline | 4 | 16 | 3 | 1,2,3 | 76.5 | 1.4855 | 76.4 | 1.4855 | 77.6 | 74.4 |
| CoOp | baseline | 8 | 2 | 3 | 1,2,3 | 77.6333 | 0.3399 | 77.5333 | 0.3399 | 78.1 | 77.3 |
| CoOp | baseline | 8 | 4 | 3 | 1,2,3 | 76.6333 | 0.4989 | 76.5 | 0.5099 | 77.3 | 76.1 |
| CoOp | baseline | 8 | 6 | 3 | 1,2,3 | 76.7333 | 0.4028 | 76.6667 | 0.4497 | 77.3 | 76.4 |
| CoOp | baseline | 8 | 8 | 3 | 1,2,3 | 76.8333 | 0.0943 | 76.7333 | 0.17 | 76.9 | 76.7 |
| CoOp | baseline | 8 | 10 | 3 | 1,2,3 | 76.3333 | 0.33 | 76.2333 | 0.3399 | 76.8 | 76.1 |
| CoOp | baseline | 8 | 12 | 3 | 1,2,3 | 76.2667 | 0.3771 | 76.1667 | 0.3771 | 76.8 | 76.0 |
| CoOp | baseline | 8 | 14 | 3 | 1,2,3 | 76.0667 | 0.3859 | 75.9 | 0.432 | 76.6 | 75.7 |
| CoOp | baseline | 8 | 16 | 3 | 1,2,3 | 76.3333 | 0.4643 | 76.2 | 0.5099 | 76.8 | 75.7 |
| CoOp | baseline | 10 | 2 | 3 | 1,2,3 | 78.0333 | 0.3682 | 78.0 | 0.3742 | 78.5 | 77.6 |
| CoOp | baseline | 10 | 4 | 3 | 1,2,3 | 77.2333 | 0.5437 | 77.1333 | 0.6182 | 78.0 | 76.8 |
| CoOp | baseline | 10 | 6 | 3 | 1,2,3 | 76.2 | 0.5715 | 76.1 | 0.5715 | 76.7 | 75.4 |
| CoOp | baseline | 10 | 8 | 3 | 1,2,3 | 76.5 | 0.4546 | 76.4 | 0.5354 | 77.0 | 75.9 |
| CoOp | baseline | 10 | 10 | 3 | 1,2,3 | 76.5667 | 0.8576 | 76.4333 | 0.8994 | 77.6 | 75.5 |
| CoOp | baseline | 10 | 12 | 3 | 1,2,3 | 76.5 | 0.9092 | 76.4 | 0.9092 | 77.5 | 75.3 |
| CoOp | baseline | 10 | 14 | 3 | 1,2,3 | 75.7 | 0.7483 | 75.5667 | 0.8731 | 76.5 | 74.7 |
| CoOp | baseline | 10 | 16 | 3 | 1,2,3 | 76.0333 | 0.8576 | 75.9 | 0.9416 | 77.1 | 75.0 |
| CoOp | baseline | 12 | 2 | 3 | 1,2,3 | 78.0333 | 0.3682 | 77.9 | 0.3742 | 78.5 | 77.6 |
| CoOp | baseline | 12 | 4 | 3 | 1,2,3 | 77.1667 | 0.6799 | 77.0333 | 0.7134 | 78.1 | 76.5 |
| CoOp | baseline | 12 | 6 | 3 | 1,2,3 | 77.1 | 0.4243 | 76.9333 | 0.5185 | 77.4 | 76.5 |
| CoOp | baseline | 12 | 8 | 3 | 1,2,3 | 76.6333 | 0.4497 | 76.4667 | 0.4643 | 77.2 | 76.1 |
| CoOp | baseline | 12 | 10 | 3 | 1,2,3 | 76.4 | 0.6481 | 76.2667 | 0.665 | 77.3 | 75.8 |
| CoOp | baseline | 12 | 12 | 3 | 1,2,3 | 76.7333 | 0.3091 | 76.6667 | 0.2625 | 77.0 | 76.3 |
| CoOp | baseline | 12 | 14 | 3 | 1,2,3 | 76.4 | 0.6377 | 76.2 | 0.6481 | 77.3 | 75.9 |
| CoOp | baseline | 12 | 16 | 3 | 1,2,3 | 76.2 | 0.6377 | 76.0333 | 0.6182 | 77.1 | 75.7 |
| CoOp | baseline | 14 | 2 | 3 | 1,2,3 | 78.4 | 0.0816 | 78.3333 | 0.1247 | 78.5 | 78.3 |
| CoOp | baseline | 14 | 4 | 3 | 1,2,3 | 77.8333 | 0.2055 | 77.7667 | 0.2055 | 78.1 | 77.6 |
| CoOp | baseline | 14 | 6 | 3 | 1,2,3 | 77.7 | 0.2944 | 77.6333 | 0.2625 | 78.1 | 77.4 |
| CoOp | baseline | 14 | 8 | 3 | 1,2,3 | 77.6 | 0.1414 | 77.5 | 0.1414 | 77.7 | 77.4 |
| CoOp | baseline | 14 | 10 | 3 | 1,2,3 | 77.3 | 0.0 | 77.2 | 0.0 | 77.3 | 77.3 |
| CoOp | baseline | 14 | 12 | 3 | 1,2,3 | 77.1333 | 0.3771 | 77.0333 | 0.3771 | 77.4 | 76.6 |
| CoOp | baseline | 14 | 14 | 3 | 1,2,3 | 76.8 | 0.3742 | 76.7 | 0.3742 | 77.3 | 76.4 |
| CoOp | baseline | 14 | 16 | 3 | 1,2,3 | 76.8 | 0.2449 | 76.7 | 0.2449 | 77.1 | 76.5 |
| CoOp | baseline | 16 | 2 | 3 | 1,2,3 | 78.6667 | 0.0943 | 78.6 | 0.0816 | 78.8 | 78.6 |
| CoOp | baseline | 16 | 4 | 3 | 1,2,3 | 78.3667 | 0.17 | 78.3 | 0.216 | 78.6 | 78.2 |
| CoOp | baseline | 16 | 6 | 3 | 1,2,3 | 77.8333 | 0.2494 | 77.7667 | 0.2055 | 78.1 | 77.5 |
| CoOp | baseline | 16 | 8 | 3 | 1,2,3 | 77.6333 | 0.1886 | 77.5 | 0.1414 | 77.9 | 77.5 |
| CoOp | baseline | 16 | 10 | 3 | 1,2,3 | 77.5 | 0.1414 | 77.4333 | 0.1247 | 77.7 | 77.4 |
| CoOp | baseline | 16 | 12 | 3 | 1,2,3 | 77.4667 | 0.6018 | 77.4 | 0.5715 | 78.3 | 76.9 |
| CoOp | baseline | 16 | 14 | 3 | 1,2,3 | 77.2 | 0.2944 | 77.1333 | 0.2625 | 77.6 | 76.9 |
| CoOpPriorRes | b_w0.2 | 16 | 16 | 3 | 1,2,3 | 77.3 | 0.2449 | 77.2333 | 0.2867 | 77.6 | 77.0 |
| CoOpPriorRes | no_b | 16 | 16 | 2 | 2,3 | 77.3 | 0.3 | 77.25 | 0.25 | 77.6 | 77.0 |

## oxford_flowers

| method | setting_tag | shots | nctx | num_runs | seeds | acc_mean | acc_std | f1_mean | f1_std | best_acc | worst_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoOp | baseline | 1 | 2 | 3 | 1,2,3 | 66.5 | 2.3281 | 61.9333 | 3.337 | 69.4 | 63.7 |
| CoOp | baseline | 1 | 4 | 3 | 1,2,3 | 67.2333 | 2.9227 | 63.0667 | 4.0036 | 70.3 | 63.3 |
| CoOp | baseline | 1 | 6 | 3 | 1,2,3 | 65.5667 | 2.7475 | 61.6667 | 2.4226 | 69.4 | 63.1 |
| CoOp | baseline | 1 | 8 | 3 | 1,2,3 | 65.1667 | 0.7587 | 60.6333 | 0.6944 | 65.8 | 64.1 |
| CoOp | baseline | 1 | 10 | 3 | 1,2,3 | 65.5333 | 1.2365 | 60.1 | 0.9798 | 66.6 | 63.8 |
| CoOp | baseline | 1 | 12 | 3 | 1,2,3 | 67.4333 | 0.8576 | 64.2 | 0.7874 | 68.5 | 66.4 |
| CoOp | baseline | 1 | 14 | 3 | 1,2,3 | 67.6 | 0.2449 | 64.0 | 0.5715 | 67.9 | 67.3 |
| CoOp | baseline | 1 | 16 | 3 | 1,2,3 | 68.6333 | 2.4784 | 65.5667 | 3.351 | 71.9 | 65.9 |
| CoOp | baseline | 2 | 2 | 3 | 1,2,3 | 69.2 | 1.7282 | 64.5333 | 2.0171 | 70.8 | 66.8 |
| CoOp | baseline | 2 | 4 | 3 | 1,2,3 | 69.9667 | 3.6745 | 66.3333 | 4.8582 | 74.5 | 65.5 |
| CoOp | baseline | 2 | 6 | 3 | 1,2,3 | 68.7667 | 5.1719 | 65.3333 | 5.4469 | 75.9 | 63.8 |
| CoOp | baseline | 2 | 8 | 3 | 1,2,3 | 67.4 | 0.1633 | 62.7667 | 0.5437 | 67.6 | 67.2 |
| CoOp | baseline | 2 | 10 | 3 | 1,2,3 | 68.8667 | 1.5755 | 64.4 | 0.6481 | 70.4 | 66.7 |
| CoOp | baseline | 2 | 12 | 3 | 1,2,3 | 70.9 | 2.2226 | 67.0667 | 2.1061 | 73.4 | 68.0 |
| CoOp | baseline | 2 | 14 | 3 | 1,2,3 | 73.3 | 1.3367 | 69.7667 | 2.6712 | 74.7 | 71.5 |
| CoOp | baseline | 2 | 16 | 3 | 1,2,3 | 73.9333 | 2.5953 | 70.7333 | 4.0417 | 77.5 | 71.4 |
| CoOp | baseline | 4 | 2 | 3 | 1,2,3 | 72.6667 | 3.0269 | 69.7667 | 3.7348 | 75.6 | 68.5 |
| CoOp | baseline | 4 | 4 | 3 | 1,2,3 | 75.5667 | 5.0632 | 73.0667 | 5.9779 | 80.1 | 68.5 |
| CoOp | baseline | 4 | 6 | 3 | 1,2,3 | 76.5 | 4.0133 | 73.2333 | 5.2665 | 82.1 | 72.9 |
| CoOp | baseline | 4 | 8 | 3 | 1,2,3 | 74.6 | 3.7974 | 71.8333 | 4.485 | 78.0 | 69.3 |
| CoOp | baseline | 4 | 10 | 3 | 1,2,3 | 76.6 | 1.3367 | 73.8333 | 1.33 | 78.4 | 75.2 |
| CoOp | baseline | 4 | 12 | 3 | 1,2,3 | 79.7667 | 4.6764 | 78.3333 | 4.8002 | 84.2 | 73.3 |
| CoOp | baseline | 4 | 14 | 3 | 1,2,3 | 84.0667 | 0.3682 | 82.8333 | 0.704 | 84.5 | 83.6 |
| CoOp | baseline | 4 | 16 | 3 | 1,2,3 | 81.8667 | 3.5743 | 80.2 | 4.0571 | 86.5 | 77.8 |
| CoOp | baseline | 8 | 2 | 3 | 1,2,3 | 81.4667 | 2.0997 | 79.9667 | 2.5038 | 84.4 | 79.6 |
| CoOp | baseline | 8 | 4 | 3 | 1,2,3 | 86.5333 | 0.9031 | 85.7 | 1.0614 | 87.7 | 85.5 |
| CoOp | baseline | 8 | 6 | 3 | 1,2,3 | 86.6333 | 0.33 | 85.7667 | 0.3771 | 87.0 | 86.2 |
| CoOp | baseline | 8 | 8 | 3 | 1,2,3 | 88.0 | 0.6683 | 87.3 | 0.7483 | 88.7 | 87.1 |
| CoOp | baseline | 8 | 10 | 3 | 1,2,3 | 89.4333 | 0.7409 | 88.8 | 0.5888 | 90.1 | 88.4 |
| CoOp | baseline | 8 | 12 | 3 | 1,2,3 | 89.5 | 0.4546 | 89.3 | 0.5354 | 90.0 | 88.9 |
| CoOp | baseline | 8 | 14 | 3 | 1,2,3 | 88.6667 | 0.6944 | 88.1333 | 0.7409 | 89.3 | 87.7 |
| CoOp | baseline | 8 | 16 | 3 | 1,2,3 | 89.7667 | 0.838 | 89.2667 | 0.704 | 90.9 | 88.9 |
| CoOp | baseline | 10 | 2 | 3 | 1,2,3 | 84.1 | 0.2944 | 82.5 | 0.2449 | 84.5 | 83.8 |
| CoOp | baseline | 10 | 4 | 3 | 1,2,3 | 87.9333 | 0.4922 | 87.2333 | 0.6236 | 88.5 | 87.3 |
| CoOp | baseline | 10 | 6 | 3 | 1,2,3 | 89.8 | 0.5099 | 89.1333 | 0.3682 | 90.3 | 89.1 |
| CoOp | baseline | 10 | 8 | 3 | 1,2,3 | 90.7 | 0.6683 | 90.0667 | 0.5312 | 91.4 | 89.8 |
| CoOp | baseline | 10 | 10 | 3 | 1,2,3 | 90.4667 | 0.5312 | 90.0667 | 0.66 | 91.1 | 89.8 |
| CoOp | baseline | 10 | 12 | 3 | 1,2,3 | 90.3667 | 0.3859 | 89.8333 | 0.2625 | 90.9 | 90.0 |
| CoOp | baseline | 10 | 14 | 3 | 1,2,3 | 91.0 | 0.7874 | 90.4333 | 0.7409 | 91.7 | 89.9 |
| CoOp | baseline | 10 | 16 | 3 | 1,2,3 | 90.6333 | 1.3275 | 90.2 | 1.6673 | 91.9 | 88.8 |
| CoOp | baseline | 12 | 2 | 3 | 1,2,3 | 85.9667 | 0.6018 | 84.8667 | 0.6549 | 86.8 | 85.4 |
| CoOp | baseline | 12 | 4 | 3 | 1,2,3 | 89.3 | 0.2944 | 88.8667 | 0.3859 | 89.7 | 89.0 |
| CoOp | baseline | 12 | 6 | 3 | 1,2,3 | 90.9 | 0.432 | 90.4333 | 0.3682 | 91.5 | 90.5 |
| CoOp | baseline | 12 | 8 | 3 | 1,2,3 | 91.0 | 0.3742 | 90.3333 | 0.33 | 91.4 | 90.5 |
| CoOp | baseline | 12 | 10 | 3 | 1,2,3 | 91.5667 | 0.4497 | 91.2 | 0.4899 | 92.1 | 91.0 |
| CoOp | baseline | 12 | 12 | 3 | 1,2,3 | 92.1667 | 0.5312 | 91.8 | 0.4082 | 92.8 | 91.5 |
| CoOp | baseline | 12 | 14 | 3 | 1,2,3 | 92.0333 | 0.2625 | 91.8 | 0.3742 | 92.4 | 91.8 |
| CoOp | baseline | 12 | 16 | 3 | 1,2,3 | 91.8667 | 0.4922 | 91.5667 | 0.419 | 92.5 | 91.3 |
| CoOp | baseline | 14 | 2 | 3 | 1,2,3 | 86.4 | 0.9416 | 85.6 | 1.2028 | 87.7 | 85.5 |
| CoOp | baseline | 14 | 4 | 3 | 1,2,3 | 90.6 | 0.2944 | 89.9667 | 0.2055 | 90.9 | 90.2 |
| CoOp | baseline | 14 | 6 | 3 | 1,2,3 | 91.2667 | 0.419 | 90.9667 | 0.6342 | 91.7 | 90.7 |
| CoOp | baseline | 14 | 8 | 3 | 1,2,3 | 91.9333 | 0.4497 | 91.5333 | 0.5249 | 92.3 | 91.3 |
| CoOp | baseline | 14 | 10 | 3 | 1,2,3 | 92.1667 | 0.33 | 91.7 | 0.3742 | 92.6 | 91.8 |
| CoOp | baseline | 14 | 12 | 3 | 1,2,3 | 92.3667 | 0.6128 | 91.9667 | 0.6128 | 93.1 | 91.6 |
| CoOp | baseline | 14 | 14 | 3 | 1,2,3 | 92.6667 | 0.6342 | 92.3333 | 0.8055 | 93.3 | 91.8 |
| CoOp | baseline | 14 | 16 | 3 | 1,2,3 | 93.1333 | 0.1247 | 93.0 | 0.0816 | 93.3 | 93.0 |
| CoOp | baseline | 16 | 2 | 3 | 1,2,3 | 88.0 | 0.5715 | 87.2 | 0.5888 | 88.8 | 87.5 |
| CoOp | baseline | 16 | 4 | 3 | 1,2,3 | 91.2667 | 0.4497 | 90.8667 | 0.411 | 91.9 | 90.9 |
| CoOp | baseline | 16 | 6 | 3 | 1,2,3 | 91.8667 | 0.704 | 91.4333 | 0.8219 | 92.8 | 91.1 |
| CoOp | baseline | 16 | 8 | 3 | 1,2,3 | 92.7333 | 0.5437 | 92.5667 | 0.5185 | 93.5 | 92.3 |
| CoOp | baseline | 16 | 10 | 3 | 1,2,3 | 92.7 | 0.2449 | 92.5 | 0.3266 | 93.0 | 92.4 |
| CoOp | baseline | 16 | 12 | 3 | 1,2,3 | 93.6 | 0.1633 | 93.4 | 0.216 | 93.8 | 93.4 |
| CoOp | baseline | 16 | 14 | 3 | 1,2,3 | 93.4333 | 0.4028 | 93.2333 | 0.2867 | 94.0 | 93.1 |
| CoOpPriorRes | b_w0.2 | 16 | 16 | 3 | 1,2,3 | 93.0667 | 0.411 | 92.6667 | 0.6128 | 93.6 | 92.6 |
| CoOpPriorRes | no_b | 16 | 16 | 3 | 1,2,3 | 93.4667 | 0.2055 | 93.3 | 0.1414 | 93.7 | 93.2 |
