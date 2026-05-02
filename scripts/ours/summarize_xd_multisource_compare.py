#!/usr/bin/env python3
import re
import sys
from pathlib import Path

ROOT = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/xd")

ALL_DATASETS = [
    "oxford_pets",
    "eurosat",
    "dtd",
    "food101",
    "oxford_flowers",
    "caltech101",
    "stanford_cars",
    "fgvc_aircraft",
    "ucf101",
    "sun397",
]

SOURCES = sys.argv[1:] if len(sys.argv) >= 2 else ["caltech101", "food101", "oxford_pets"]
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

def log_path(method, source, target, seed):
    if method == "CoOp":
        return ROOT / "test" / target / f"source_{source}/shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed{seed}/log.txt"
    if method == "PriorRes":
        return ROOT / "test" / target / f"source_{source}/shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed{seed}/log.txt"
    raise ValueError(method)

all_deltas = []
all_seed_positive = 0
all_seed_total = 0

print("# Multi-source Cross-dataset DG Summary")
print()
print(f"Sources: `{', '.join(SOURCES)}`")
print()
print("| Source | Target | CoOp | PriorRes | Delta |")
print("|---|---|---:|---:|---:|")

source_avg_rows = []

for source in SOURCES:
    source_deltas = []
    source_seed_positive = 0
    source_seed_total = 0

    targets = [d for d in ALL_DATASETS if d != source]

    for target in targets:
        coop_vals = []
        ours_vals = []

        for seed in SEEDS:
            c = read_acc(log_path("CoOp", source, target, seed))
            o = read_acc(log_path("PriorRes", source, target, seed))
            coop_vals.append(c)
            ours_vals.append(o)

            if c is not None and o is not None:
                source_seed_total += 1
                all_seed_total += 1
                if o - c > 0:
                    source_seed_positive += 1
                    all_seed_positive += 1

        c_mean = mean(coop_vals)
        o_mean = mean(ours_vals)
        delta = None if c_mean is None or o_mean is None else o_mean - c_mean

        if delta is not None:
            source_deltas.append(delta)
            all_deltas.append(delta)

        print(f"| {source} | {target} | {fmt(c_mean)} | {fmt(o_mean)} | {fmt(delta)} |")

    s_avg = mean(source_deltas)
    source_avg_rows.append((source, s_avg, source_seed_positive, source_seed_total))
    print(f"| **{source}** | **Average** |  |  | **{fmt(s_avg)}** |")

overall = mean(all_deltas)

print()
print("# Source-level Average")
print()
print("| Source | Avg Delta | Positive Seed Cases |")
print("|---|---:|---:|")
for source, avg, pos, total in source_avg_rows:
    print(f"| {source} | {fmt(avg)} | {pos}/{total} |")

print(f"| **Overall** | **{fmt(overall)}** | **{all_seed_positive}/{all_seed_total}** |")

print()
print("# Notes")
print()
print("- Delta = PriorRes - CoOp.")
print("- PriorRes setting: safe noalt, `USE_B=False`, `USE_LEGACY_RESIDUAL=False`, `ALTERNATE_OPT=False`.")
print("- Empty cells mean the corresponding logs were not found or did not contain final accuracy.")
