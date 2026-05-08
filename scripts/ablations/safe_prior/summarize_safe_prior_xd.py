#!/usr/bin/env python3
import re
from pathlib import Path
from statistics import mean

PROJECT = Path("/workspace/meta_prompt_1")
MAIN_XD = PROJECT / "third_party/CoOp_clean/output/xd/test"
ABL_XD = PROJECT / "outputs/ablations/safe_prior/runs/xd/test"

SOURCES = ["caltech101", "food101", "sun397"]

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

SEEDS = [1, 2, 3]
ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")

SHUFFLE_TAG = {
    "caltech101": "safe_shuffle_foodfeat",
    "food101": "safe_shuffle_sunfeat",
    "sun397": "safe_shuffle_caltechfeat",
}

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

def main_log(method, source, target, seed):
    base = MAIN_XD / target / f"source_{source}" / "shots_16"
    if method == "coop":
        return base / f"CoOp/rn50_ep50/nctx16_cscFalse_ctpend/seed{seed}/log.txt"
    if method == "safe_real":
        return base / f"CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt/seed{seed}/log.txt"
    raise ValueError(method)

def abl_log(tag, source, target, seed):
    return ABL_XD / target / f"source_{source}" / "shots_16/CoOpPriorRes/rn50_ep50" / f"nctx16_cscFalse_ctpend_{tag}" / f"seed{seed}/log.txt"

print("# Safe Dataset-prior Ablation: Cross-dataset DG")
print()
print("| Source | Target | CoOp | Safe-Real | Safe-Mean | Safe-Shuffle | Real-CoOp | Mean-CoOp | Shuffle-CoOp | Real-Mean | Real-Shuffle |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

overall = {
    "real": [],
    "mean": [],
    "shuffle": [],
    "real_mean": [],
    "real_shuffle": [],
}

source_rows = []

for source in SOURCES:
    targets = [d for d in ALL_DATASETS if d != source]
    source_real = []
    source_mean = []
    source_shuffle = []
    source_real_mean = []
    source_real_shuffle = []

    shuffle_tag = SHUFFLE_TAG[source]

    for target in targets:
        coop_vals = [read_acc(main_log("coop", source, target, seed)) for seed in SEEDS]
        real_vals = [read_acc(main_log("safe_real", source, target, seed)) for seed in SEEDS]
        mean_vals = [read_acc(abl_log("safe_mean", source, target, seed)) for seed in SEEDS]
        shuffle_vals = [read_acc(abl_log(shuffle_tag, source, target, seed)) for seed in SEEDS]

        c = avg(coop_vals)
        r = avg(real_vals)
        m = avg(mean_vals)
        sh = avg(shuffle_vals)

        d_r = None if c is None or r is None else r - c
        d_m = None if c is None or m is None else m - c
        d_sh = None if c is None or sh is None else sh - c
        d_rm = None if r is None or m is None else r - m
        d_rsh = None if r is None or sh is None else r - sh

        for arr, val in [
            (source_real, d_r),
            (source_mean, d_m),
            (source_shuffle, d_sh),
            (source_real_mean, d_rm),
            (source_real_shuffle, d_rsh),
        ]:
            if val is not None:
                arr.append(val)

        if d_r is not None:
            overall["real"].append(d_r)
        if d_m is not None:
            overall["mean"].append(d_m)
        if d_sh is not None:
            overall["shuffle"].append(d_sh)
        if d_rm is not None:
            overall["real_mean"].append(d_rm)
        if d_rsh is not None:
            overall["real_shuffle"].append(d_rsh)

        print(f"| {source} | {target} | {fmt(c)} | {fmt(r)} | {fmt(m)} | {fmt(sh)} | {fmt(d_r)} | {fmt(d_m)} | {fmt(d_sh)} | {fmt(d_rm)} | {fmt(d_rsh)} |")

    sr = avg(source_real)
    sm = avg(source_mean)
    ss = avg(source_shuffle)
    srm = avg(source_real_mean)
    srsh = avg(source_real_shuffle)

    source_rows.append((source, sr, sm, ss, srm, srsh))

    print(f"| **{source}** | **Average** |  |  |  |  | **{fmt(sr)}** | **{fmt(sm)}** | **{fmt(ss)}** | **{fmt(srm)}** | **{fmt(srsh)}** |")

print()
print("# Source-level Average")
print()
print("| Source | Real-CoOp | Mean-CoOp | Shuffle-CoOp | Real-Mean | Real-Shuffle |")
print("|---|---:|---:|---:|---:|---:|")

for row in source_rows:
    source, sr, sm, ss, srm, srsh = row
    print(f"| {source} | {fmt(sr)} | {fmt(sm)} | {fmt(ss)} | {fmt(srm)} | {fmt(srsh)} |")

print(f"| **Overall** | **{fmt(avg(overall['real']))}** | **{fmt(avg(overall['mean']))}** | **{fmt(avg(overall['shuffle']))}** | **{fmt(avg(overall['real_mean']))}** | **{fmt(avg(overall['real_shuffle']))}** |")

print()
print("# Notes")
print()
print("- Safe-Real uses the true source dataset feature.")
print("- Safe-Mean uses `outputs/task_features/mean_train_feature.json`.")
print("- Safe-Shuffle uses a fixed wrong feature: caltech101->food101, food101->sun397, sun397->caltech101.")
print("- CoOp and Safe-Real are read from `third_party/CoOp_clean/output/xd/test`.")
print("- Safe-Mean and Safe-Shuffle are read from `outputs/ablations/safe_prior/runs/xd/test`.")
