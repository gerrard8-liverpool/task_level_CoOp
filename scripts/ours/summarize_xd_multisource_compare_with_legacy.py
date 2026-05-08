#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from statistics import mean

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

SOURCES = sys.argv[1:] if len(sys.argv) >= 2 else ["caltech101", "food101", "sun397"]
SEEDS = [1, 2, 3]

ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")

def read_acc(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    vals = ACC_RE.findall(text)
    return float(vals[-1]) if vals else None

def avg(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None

def fmt(x):
    return "" if x is None else f"{x:.2f}"

def log_path(method, source, target, seed):
    base = ROOT / "test" / target / f"source_{source}" / "shots_16"
    if method == "CoOp":
        return base / f"CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed{seed}/log.txt"
    if method == "Safe":
        return base / f"CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed{seed}/log.txt"
    if method == "Legacy":
        return base / f"CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_legacy_noalt/seed{seed}/log.txt"
    raise ValueError(method)

all_safe_deltas = []
all_legacy_deltas = []
safe_seed_positive = 0
legacy_seed_positive = 0
seed_total = 0

print("# Multi-source Cross-dataset DG Summary: CoOp vs Safe PriorRes vs Legacy PriorRes")
print()
print("Sources:", ", ".join(f"`{s}`" for s in SOURCES))
print()
print("| Source | Target | CoOp | Safe | Legacy | Safe-CoOp | Legacy-CoOp | Legacy-Safe |")
print("|---|---|---:|---:|---:|---:|---:|---:|")

source_rows = []

for source in SOURCES:
    targets = [d for d in ALL_DATASETS if d != source]
    safe_deltas = []
    legacy_deltas = []

    for target in targets:
        coop_vals = []
        safe_vals = []
        legacy_vals = []

        for seed in SEEDS:
            c = read_acc(log_path("CoOp", source, target, seed))
            s = read_acc(log_path("Safe", source, target, seed))
            l = read_acc(log_path("Legacy", source, target, seed))

            coop_vals.append(c)
            safe_vals.append(s)
            legacy_vals.append(l)

            if c is not None and s is not None and l is not None:
                seed_total += 1
                if s > c:
                    safe_seed_positive += 1
                if l > c:
                    legacy_seed_positive += 1

        c_mean = avg(coop_vals)
        s_mean = avg(safe_vals)
        l_mean = avg(legacy_vals)

        ds = None if c_mean is None or s_mean is None else s_mean - c_mean
        dl = None if c_mean is None or l_mean is None else l_mean - c_mean
        dls = None if s_mean is None or l_mean is None else l_mean - s_mean

        if ds is not None:
            safe_deltas.append(ds)
            all_safe_deltas.append(ds)
        if dl is not None:
            legacy_deltas.append(dl)
            all_legacy_deltas.append(dl)

        print(f"| {source} | {target} | {fmt(c_mean)} | {fmt(s_mean)} | {fmt(l_mean)} | {fmt(ds)} | {fmt(dl)} | {fmt(dls)} |")

    s_avg = avg(safe_deltas)
    l_avg = avg(legacy_deltas)
    source_rows.append((source, s_avg, l_avg))
    print(f"| **{source}** | **Average** |  |  |  | **{fmt(s_avg)}** | **{fmt(l_avg)}** | **{fmt(None if s_avg is None or l_avg is None else l_avg - s_avg)}** |")

overall_safe = avg(all_safe_deltas)
overall_legacy = avg(all_legacy_deltas)

print()
print("# Source-level Average")
print()
print("| Source | Safe-CoOp Avg Delta | Legacy-CoOp Avg Delta | Legacy-Safe |")
print("|---|---:|---:|---:|")
for source, s_avg, l_avg in source_rows:
    print(f"| {source} | {fmt(s_avg)} | {fmt(l_avg)} | {fmt(None if s_avg is None or l_avg is None else l_avg - s_avg)} |")
print(f"| **Overall** | **{fmt(overall_safe)}** | **{fmt(overall_legacy)}** | **{fmt(None if overall_safe is None or overall_legacy is None else overall_legacy - overall_safe)}** |")

print()
print(f"Safe > CoOp seed-level cases: **{safe_seed_positive}/{seed_total}**")
print(f"Legacy > CoOp seed-level cases: **{legacy_seed_positive}/{seed_total}**")
print()
print("# Notes")
print()
print("- Safe = identity-centered residual, `nctx16_cscFalse_ctpend_safe_noalt`.")
print("- Legacy = non-identity residual, `nctx16_cscFalse_ctpend_legacy_noalt`.")
print("- All logs are read from `third_party/CoOp_clean/output/xd`.")
