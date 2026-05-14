# Residual Formula Ablation Scripts

Copy the `scripts/ablations/residual_formula` directory into `/workspace/meta_prompt_1/scripts/ablations/residual_formula`.

Create output namespace:

```bash
cd /workspace/meta_prompt_1
mkdir -p outputs/ablations/residual_formula/{runs,tables,logs}
chmod +x scripts/ablations/residual_formula/*.sh
```

Run in-domain first round:

```bash
cd /workspace/meta_prompt_1
GPU=0 bash scripts/ablations/residual_formula/run_first_round_indomain.sh
```

Run cross-dataset first round with Caltech101 source:

```bash
cd /workspace/meta_prompt_1
GPU=0 bash scripts/ablations/residual_formula/run_first_round_xd_caltech.sh
```

Summaries are written to:

```text
outputs/ablations/residual_formula/tables/indomain_residual_formula.md
outputs/ablations/residual_formula/tables/xd_caltech101_residual_formula.md
```
