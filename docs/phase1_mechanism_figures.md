# Phase 1 机制图与干净 DG 重跑说明

## 1. 当前核心结论

Safe PriorRes 不应被包装成一个普遍提升 DG 准确率的模型。

当前更稳的结论是：Safe PriorRes 的主要贡献是 identity-centered residual interface。它可以防止 dataset-conditioned prior 被 legacy residual 公式转化成有害的 non-identity prompt bias。

因此，论文叙事应强调 residual safety、source-target dependency 和 conservative transfer behavior，而不是强调 universal improvement。

## 2. 推荐主图

### 图 1：EuroSAT seed3 residual dynamics

作用：展示 Legacy residual 会产生明显的 effective prompt perturbation，而 Safe residual 会将该扰动压到接近 0。

正确解释：Safe removes harmful prior-induced prompt shift。

不要解释成：Safe learns a large useful residual。

### 图 2：Aggregate residual safety

文件名建议：
- fig_residual_safety.pdf
- fig_residual_safety.png

作用：展示在多个 source-seed pair 上，Safe 的 effective prompt perturbation 都比 Legacy 小几个数量级。

这说明 Safe 的安全性不是单个 case 的偶然现象。

### 图 3：Safe - CoOp DG delta heatmap

文件名建议：
- fig_heatmap_safe_delta.pdf
- fig_heatmap_safe_delta.png

作用：展示 Safe 相对 CoOp 的跨数据集迁移表现。

这张图说明 Safe 的收益具有 source-target 依赖性，有些 pair 正，有些 pair 负。

### 图 4：Legacy - CoOp DG delta heatmap

文件名建议：
- fig_heatmap_legacy_delta.pdf
- fig_heatmap_legacy_delta.png

作用：展示 Legacy 不是完全无效，但它更激进、更不稳定。

正确解释：Legacy is an aggressive but less stable residual formulation。

### 图 5：Safe - Legacy accuracy gap heatmap

文件名建议：
- fig_heatmap_safe_minus_legacy.pdf
- fig_heatmap_safe_minus_legacy.png

作用：直接比较 identity-centered residual 和 legacy residual 的效果。

这张图可以作为 residual formulation ablation 的性能证据。

### 图 6：Per-source delta distribution

文件名建议：
- fig_source_delta_distribution.pdf
- fig_source_delta_distribution.png

作用：展示每个 source 内部 target-seed pair 的方差很大，说明单纯 source mean 不足以解释 DG 行为。

## 3. 不建议作为主图的图

Source mean bar chart 不建议作为主图。

原因：
1. 误差棒太大；
2. source 平均会掩盖 pair-level 结构；
3. 容易引发不必要的 source 排名解释；
4. 信息量低于 heatmap 和 distribution plot。

如果保留，建议放 appendix。

## 4. Clean rerun 当前 source-level 结果

| Source | Safe-CoOp mean | Legacy-CoOp mean | n |
|---|---:|---:|---:|
| Caltech101 | +0.70 | +0.83 | 27 |
| Food101 | -0.47 | -0.34 | 27 |
| SUN397 | +1.12 | -1.29 | 27 |
| OxfordPets | +1.99 | +1.85 | 27 |

这组结果来自 clean protocol rerun。后续论文和 Codex 写作应优先使用这组结果，而不是旧日志和新日志混合扫描出来的结果。

## 5. 推荐论文表述

Safe PriorRes does not improve transfer by applying large explicit prompt residuals. Instead, its main contribution is an identity-centered residual interface that prevents dataset-conditioned priors from being converted into harmful non-identity prompt shifts. Residual dynamics and aggregate safety analysis show that Safe consistently suppresses effective prompt perturbation compared with Legacy. The DG heatmaps further show that transfer behavior is source-target dependent rather than universally positive.

## 6. 推荐给 Codex 的文件顺序

1. README.md
2. docs/phase1_mechanism_figures.md
3. paper_assets/tables/final_dg_source_summary.csv
4. paper_assets/tables/final_dg_pair_deltas.csv
5. paper_assets/tables/final_residual_safety.csv
6. paper_assets/figures/

## 7. 一句话总结

Safe PriorRes 是一个 identity-centered dataset-prior residual prompt adapter。它的核心价值是 residual safety，而不是 universal accuracy improvement。跨数据集迁移结果应解释为 source-target dependent。
