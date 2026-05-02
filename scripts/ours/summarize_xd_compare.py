#!/usr/bin/env python3
import re
import sys
from pathlib import Path

ROOT = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/xd")

SOURCE = sys.argv[1] if len(sys.argv) >= 2 else "caltech101"
TARGETS = sys.argv[2:] if len(sys.argv) >= 3 else ["dtd", "eurosat", "ucf101", "sun397"]
SEEDS = [1, 2, 3]

ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")

def read_acc(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    vals = ACC_RE.findall(text)
    return float(vals[-1]) if vals else None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def fmt(x):
    return "" if x is None else f"{x:.2f}"

def log_path(method, target, seed):
    if method == "CoOp":
        return ROOT / "test" / target / f"source_{SOURCE}/shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed{seed}/log.txt"
    if method == "PriorRes":
        return ROOT / "test" / target / f"source_{SOURCE}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed{seed}/log.txt"
    raise ValueError(method)

print(f"# Cross-dataset DG: source = {SOURCE}")
print()
print("| Target | CoOp | PriorRes | Delta |")
print("|---|---:|---:|---:|")

deltas = []
positive_seed = 0
total_seed = 0

for target in TARGETS:
    coop_vals = []
    ours_vals = []

    for seed in SEEDS:
        c = read_acc(log_path("CoOp", target, seed))
        o = read_acc(log_path("PriorRes", target, seed))
        coop_vals.append(c)
        ours_vals.append(o)

        if c is not None and o is not None:
            total_seed += 1
            if o - c > 0:
                positive_seed += 1

    coop_m = mean(coop_vals)
    ours_m = mean(ours_vals)
    delta = None if coop_m is None or ours_m is None else ours_m - coop_m

    if delta is not None:
        deltas.append(delta)

    print(f"| {target} | {fmt(coop_m)} | {fmt(ours_m)} | {fmt(delta)} |")

print(f"| **Average** |  |  | **{fmt(mean(deltas))}** |")
print()
print(f"Positive seed-level cases: **{positive_seed}/{total_seed}**")

print()
print("## Seed-level details")
print()
print("| Target | Seed | CoOp | PriorRes | Delta |")
print("|---|---:|---:|---:|---:|")

for target in TARGETS:
    for seed in SEEDS:
        c = read_acc(log_path("CoOp", target, seed))
        o = read_acc(log_path("PriorRes", target, seed))
        d = None if c is None or o is None else o - c
        print(f"| {target} | {seed} | {fmt(c)} | {fmt(o)} | {fmt(d)} |")
