import csv
import math
import re
import statistics as st
from pathlib import Path

BASE_STAGE2 = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets/CoOpPriorRes_stage2")
BASE_COOP = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets/CoOp")
OUT_CSV = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets/oxfordpets_stage2_summary.csv")

MODES = ["aonly", "bonly", "ab"]
SEEDS = [1, 2, 3, 4, 5]

PATS = {
    "acc": re.compile(r"\* accuracy:\s*([0-9.]+)%"),
    "f1": re.compile(r"\* macro_f1:\s*([0-9.]+)%"),
    "meff": re.compile(r"meff\s+([0-9.]+)\s+\("),
    "keff": re.compile(r"keff\s+([0-9.]+)\s+\("),
    "a0_mean": re.compile(r"a0_mean\s+([0-9.]+)\s+\("),
    "a_mean": re.compile(r"a_mean\s+([0-9.]+)\s+\("),
    "b0_mean": re.compile(r"b0_mean\s+([0-9.]+)\s+\("),
    "b_mean": re.compile(r"b_mean\s+([0-9.]+)\s+\("),
    "delta_a_norm": re.compile(r"delta_a_norm\s+([0-9.]+)\s+\("),
    "delta_b_norm": re.compile(r"delta_b_norm\s+([0-9.]+)\s+\("),
    "loss_b": re.compile(r"loss_b\s+([0-9.]+)\s+\("),
    "b_entropy": re.compile(r"b_entropy\s+([0-9.]+)\s+\("),
    "top1_weight": re.compile(r"top1_weight\s+([0-9.]+)\s+\("),
    "top4_weight_sum": re.compile(r"top4_weight_sum\s+([0-9.]+)\s+\("),
}

ACC_PAT_BASE = re.compile(r"\* accuracy:\s*([0-9.]+)%")
F1_PAT_BASE = re.compile(r"\* macro_f1:\s*([0-9.]+)%")

def last_float(pat, text):
    xs = pat.findall(text)
    return float(xs[-1]) if xs else None

def parse_stage2_log(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {k: last_float(v, text) for k, v in PATS.items()}
    out["log_path"] = str(path)
    return out

def parse_baseline_log(path: Path):
    if not path.exists():
        return {"baseline_acc": None, "baseline_f1": None}
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "baseline_acc": last_float(ACC_PAT_BASE, text),
        "baseline_f1": last_float(F1_PAT_BASE, text),
    }

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None, 0)
    if len(vals) == 1:
        return (vals[0], 0.0, 1)
    return (st.mean(vals), st.stdev(vals), len(vals))

rows = []

# per-seed rows
for mode in MODES:
    for seed in SEEDS:
        log = BASE_STAGE2 / mode / f"rn50_16shots_beta0.2_seed{seed}" / "log.txt"
        row = parse_stage2_log(log)
        if row is None:
            continue

        base_log = BASE_COOP / f"rn50_16shots_seed{seed}" / "log.txt"
        base = parse_baseline_log(base_log)

        row_out = {
            "row_type": "seed",
            "mode": mode,
            "seed": seed,
            **row,
            **base,
            "delta_vs_coop": None,
            "n_runs": None,
        }
        if row_out["acc"] is not None and row_out["baseline_acc"] is not None:
            row_out["delta_vs_coop"] = row_out["acc"] - row_out["baseline_acc"]

        rows.append(row_out)

# summary rows
for mode in MODES:
    mode_rows = [r for r in rows if r["row_type"] == "seed" and r["mode"] == mode]
    if not mode_rows:
        continue

    summary = {
        "row_type": "summary",
        "mode": mode,
        "seed": "",
        "log_path": "",
    }

    keys = [
        "acc", "f1", "meff", "keff", "a0_mean", "a_mean", "b0_mean", "b_mean",
        "delta_a_norm", "delta_b_norm", "loss_b", "b_entropy", "top1_weight",
        "top4_weight_sum", "baseline_acc", "baseline_f1", "delta_vs_coop"
    ]

    for k in keys:
        m, s, n = mean_std([r.get(k) for r in mode_rows])
        summary[k] = m
        summary[f"{k}_std"] = s
        summary["n_runs"] = n

    rows.append(summary)

fieldnames = [
    "row_type", "mode", "seed",
    "acc", "acc_std",
    "f1", "f1_std",
    "meff", "meff_std",
    "keff", "keff_std",
    "a0_mean", "a0_mean_std",
    "a_mean", "a_mean_std",
    "b0_mean", "b0_mean_std",
    "b_mean", "b_mean_std",
    "delta_a_norm", "delta_a_norm_std",
    "delta_b_norm", "delta_b_norm_std",
    "loss_b", "loss_b_std",
    "b_entropy", "b_entropy_std",
    "top1_weight", "top1_weight_std",
    "top4_weight_sum", "top4_weight_sum_std",
    "baseline_acc", "baseline_acc_std",
    "baseline_f1", "baseline_f1_std",
    "delta_vs_coop", "delta_vs_coop_std",
    "n_runs", "log_path",
]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"CSV saved to: {OUT_CSV}")
