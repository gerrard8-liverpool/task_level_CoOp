import os
import re
import statistics as st
from pathlib import Path

BASE = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets")
K_VALUES = [1, 2, 4, 8, 10, 12, 14, 16]
SEEDS = [1, 2, 3]

ACC_PAT = re.compile(r"\* accuracy:\s*([0-9.]+)%")
F1_PAT = re.compile(r"\* macro_f1:\s*([0-9.]+)%")
MEFF_PAT = re.compile(r"meff\s+([0-9.]+)\s+\(")
A0_PAT = re.compile(r"a0_mean\s+([0-9.]+)\s+\(")
A_PAT = re.compile(r"a_mean\s+([0-9.]+)\s+\(")
DA_PAT = re.compile(r"delta_a_norm\s+([0-9.]+)\s+\(")

def parse_log(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    accs = ACC_PAT.findall(text)
    f1s = F1_PAT.findall(text)
    meffs = MEFF_PAT.findall(text)
    a0s = A0_PAT.findall(text)
    a_s = A_PAT.findall(text)
    das = DA_PAT.findall(text)
    return {
        "acc": float(accs[-1]) if accs else None,
        "f1": float(f1s[-1]) if f1s else None,
        "meff": float(meffs[-1]) if meffs else None,
        "a0_mean": float(a0s[-1]) if a0s else None,
        "a_mean": float(a_s[-1]) if a_s else None,
        "delta_a_norm": float(das[-1]) if das else None,
    }

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.stdev(vals)

def fmt(ms):
    m, s = ms
    if m is None:
        return "N/A"
    return f"{m:.2f} ± {s:.2f}"

def collect(prefix_func):
    out = {}
    for k in K_VALUES:
        rows = []
        for seed in SEEDS:
            path = prefix_func(k, seed)
            rows.append(parse_log(path))
        out[k] = rows
    return out

coop = collect(lambda k, seed: BASE / "CoOp" / f"rn50_{k}shots_seed{seed}" / "log.txt")
prior = collect(lambda k, seed: BASE / "CoOpPriorRes_datasetonly" / f"rn50_{k}shots_b1.0_lr0.3_seed{seed}" / "log.txt")

print("=" * 120)
print("Detailed per-seed results")
print("=" * 120)
for k in K_VALUES:
    print(f"\n[K={k}]")
    for seed, row in zip(SEEDS, coop[k]):
        print(f"  CoOp      seed={seed}: {row}")
    for seed, row in zip(SEEDS, prior[k]):
        print(f"  PriorRes  seed={seed}: {row}")

print("\n" + "=" * 120)
print("Markdown table")
print("=" * 120)
print("| k | CoOp acc | PriorRes acc | Δacc | PriorRes macro-F1 | PriorRes meff | PriorRes a0_mean | PriorRes a_mean | delta_a_norm |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

for k in K_VALUES:
    coop_acc = mean_std([r["acc"] if r else None for r in coop[k]])
    prior_acc = mean_std([r["acc"] if r else None for r in prior[k]])
    prior_f1 = mean_std([r["f1"] if r else None for r in prior[k]])
    prior_meff = mean_std([r["meff"] if r else None for r in prior[k]])
    prior_a0 = mean_std([r["a0_mean"] if r else None for r in prior[k]])
    prior_a = mean_std([r["a_mean"] if r else None for r in prior[k]])
    prior_da = mean_std([r["delta_a_norm"] if r else None for r in prior[k]])

    delta = "N/A"
    if coop_acc[0] is not None and prior_acc[0] is not None:
        delta = f"{prior_acc[0] - coop_acc[0]:.2f}"

    print(
        f"| {k} | {fmt(coop_acc)} | {fmt(prior_acc)} | {delta} | "
        f"{fmt(prior_f1)} | {fmt(prior_meff)} | {fmt(prior_a0)} | {fmt(prior_a)} | {fmt(prior_da)} |"
    )

print("\n" + "=" * 120)
print("Plain summary")
print("=" * 120)
for k in K_VALUES:
    coop_acc = mean_std([r["acc"] if r else None for r in coop[k]])
    prior_acc = mean_std([r["acc"] if r else None for r in prior[k]])
    prior_meff = mean_std([r["meff"] if r else None for r in prior[k]])

    delta = None
    if coop_acc[0] is not None and prior_acc[0] is not None:
        delta = prior_acc[0] - coop_acc[0]

    print(
        f"k={k:>2} | CoOp={fmt(coop_acc)} | "
        f"PriorRes={fmt(prior_acc)} | "
        f"Δacc={delta:.2f}" if delta is not None else f"k={k:>2} | CoOp={fmt(coop_acc)} | PriorRes={fmt(prior_acc)} | Δacc=N/A",
        end=""
    )
    print(f" | meff={fmt(prior_meff)}")
