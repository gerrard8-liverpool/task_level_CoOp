#!/usr/bin/env python3
from pathlib import Path
from statistics import mean, pstdev
import re
import argparse

ROOT = Path("/workspace/meta_prompt_1")
OLD_BASE = ROOT / "third_party/CoOp_clean/output/xd/test"
LEGACY_BASE = ROOT / "outputs/ablations/residual_formula/runs/xd/test"
ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")
SEEDS = [1, 2, 3]

def read_acc(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    vals = ACC_RE.findall(text)
    return float(vals[-1]) if vals else None

def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0), len(vals)

def fmt(x):
    return "" if x is None else f"{x:.2f}"

def fmt_ms(m, s, n):
    return "" if m is None else f"{m:.2f}±{s:.2f} ({n}/3)"

def coop_log(source, target, seed):
    return OLD_BASE / target / f"source_{source}" / "shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend" / f"seed{seed}/log.txt"

def safe_log(source, target, seed):
    return OLD_BASE / target / f"source_{source}" / "shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt" / f"seed{seed}/log.txt"

def legacy_log(source, target, seed):
    return LEGACY_BASE / target / f"source_{source}" / "shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_legacy" / f"seed{seed}/log.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="caltech101")
    parser.add_argument("--targets", nargs="+", default=[
        "oxford_pets", "eurosat", "dtd", "food101", "oxford_flowers",
        "stanford_cars", "fgvc_aircraft", "ucf101", "sun397"
    ])
    args = parser.parse_args()

    print(f"# Residual Formula Ablation: Cross-dataset DG aligned with old protocol, source = {args.source}")
    print()
    print("Protocol: old main XD setting; CoOp/Safe reused from `third_party/CoOp_clean/output/xd`, Legacy newly run under the same default protocol.")
    print()
    print("| Target | CoOp(old) | Legacy(aligned) | Safe(old) | Safe-CoOp | Safe-Legacy |")
    print("|---|---:|---:|---:|---:|---:|")

    ds_sc = []
    ds_sl = []
    pos_sc = 0
    pos_sl = 0
    total_sc = 0
    total_sl = 0

    seed_rows = []

    for target in args.targets:
        coop = [read_acc(coop_log(args.source, target, seed)) for seed in SEEDS]
        legacy = [read_acc(legacy_log(args.source, target, seed)) for seed in SEEDS]
        safe = [read_acc(safe_log(args.source, target, seed)) for seed in SEEDS]

        cm, cs, cn = stat(coop)
        lm, ls, ln = stat(legacy)
        sm, ss, sn = stat(safe)

        d_sc = None if sm is None or cm is None else sm - cm
        d_sl = None if sm is None or lm is None else sm - lm

        if d_sc is not None:
            ds_sc.append(d_sc)
        if d_sl is not None:
            ds_sl.append(d_sl)

        for seed, c, l, s in zip(SEEDS, coop, legacy, safe):
            if c is not None and s is not None:
                total_sc += 1
                if s > c:
                    pos_sc += 1
            if l is not None and s is not None:
                total_sl += 1
                if s > l:
                    pos_sl += 1
            seed_rows.append((target, seed, c, l, s))

        print(f"| {target} | {fmt_ms(cm, cs, cn)} | {fmt_ms(lm, ls, ln)} | {fmt_ms(sm, ss, sn)} | {fmt(d_sc)} | {fmt(d_sl)} |")

    avg_sc = mean(ds_sc) if ds_sc else None
    avg_sl = mean(ds_sl) if ds_sl else None
    print(f"| **Average Delta** |  |  |  | **{fmt(avg_sc)}** | **{fmt(avg_sl)}** |")
    print()
    print(f"Safe > CoOp seed-level cases: **{pos_sc}/{total_sc}**")
    print(f"Safe > Legacy seed-level cases: **{pos_sl}/{total_sl}**")
    print()
    print("## Seed-level details")
    print()
    print("| Target | Seed | CoOp(old) | Legacy(aligned) | Safe(old) | Safe-CoOp | Safe-Legacy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for target, seed, c, l, s in seed_rows:
        print(f"| {target} | {seed} | {fmt(c)} | {fmt(l)} | {fmt(s)} | {fmt(None if c is None or s is None else s-c)} | {fmt(None if l is None or s is None else s-l)} |")

if __name__ == "__main__":
    main()
