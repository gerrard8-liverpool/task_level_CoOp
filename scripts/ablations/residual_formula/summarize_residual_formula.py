#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Optional

ROOT = Path("/workspace/meta_prompt_1")
BASE = ROOT / "outputs" / "ablations" / "residual_formula" / "runs"
SEEDS = [1, 2, 3]
ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")


def read_acc(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    vals = ACC_RE.findall(text)
    return float(vals[-1]) if vals else None


def stat(xs: Iterable[Optional[float]]):
    vals = [x for x in xs if x is not None]
    if not vals:
        return None, None, 0
    return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.2f}"


def fmt_ms(m: Optional[float], s: Optional[float], n: int) -> str:
    if m is None:
        return ""
    return f"{m:.2f}±{s:.2f} ({n}/3)"


def indomain_log(dataset: str, variant: str, seed: int) -> Path:
    if variant == "coop":
        return BASE / "indomain" / dataset / "coop" / "shots_16" / "CoOp" / "rn50_ep50" / "nctx16_cscFalse_ctpend" / f"seed{seed}" / "log.txt"
    return BASE / "indomain" / dataset / variant / "shots_16" / "CoOpPriorRes" / "rn50_ep50" / "nctx16_cscFalse_ctpend" / f"seed{seed}" / "log.txt"


def xd_log(source: str, target: str, variant: str, seed: int) -> Path:
    if variant == "coop":
        return BASE / "xd" / "test" / target / f"source_{source}" / "shots_16" / "CoOp" / "rn50_ep50" / "nctx16_cscFalse_ctpend" / f"seed{seed}" / "log.txt"
    return BASE / "xd" / "test" / target / f"source_{source}" / "shots_16" / "CoOpPriorRes" / "rn50_ep50" / f"nctx16_cscFalse_ctpend_{variant}" / f"seed{seed}" / "log.txt"


def summarize_indomain(datasets: list[str]):
    print("# Residual Formula Ablation: In-domain")
    print()
    print("| Dataset | CoOp | Legacy | Safe | Safe-CoOp | Safe-Legacy |")
    print("|---|---:|---:|---:|---:|---:|")

    all_safe_minus_coop = []
    all_safe_minus_legacy = []

    for dataset in datasets:
        vals = {}
        for variant in ["coop", "legacy", "safe"]:
            accs = [read_acc(indomain_log(dataset, variant, seed)) for seed in SEEDS]
            vals[variant] = (*stat(accs), accs)

        coop_m, coop_s, coop_n, _ = vals["coop"]
        legacy_m, legacy_s, legacy_n, _ = vals["legacy"]
        safe_m, safe_s, safe_n, _ = vals["safe"]
        d_coop = None if safe_m is None or coop_m is None else safe_m - coop_m
        d_legacy = None if safe_m is None or legacy_m is None else safe_m - legacy_m
        if d_coop is not None:
            all_safe_minus_coop.append(d_coop)
        if d_legacy is not None:
            all_safe_minus_legacy.append(d_legacy)

        print(
            f"| {dataset} | {fmt_ms(coop_m, coop_s, coop_n)} | {fmt_ms(legacy_m, legacy_s, legacy_n)} | "
            f"{fmt_ms(safe_m, safe_s, safe_n)} | {fmt(d_coop)} | {fmt(d_legacy)} |"
        )

    avg_sc = mean(all_safe_minus_coop) if all_safe_minus_coop else None
    avg_sl = mean(all_safe_minus_legacy) if all_safe_minus_legacy else None
    print(f"| **Average Delta** |  |  |  | **{fmt(avg_sc)}** | **{fmt(avg_sl)}** |")
    print()
    print("## Seed-level details")
    print()
    print("| Dataset | Seed | CoOp | Legacy | Safe | Safe-CoOp | Safe-Legacy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for seed in SEEDS:
            c = read_acc(indomain_log(dataset, "coop", seed))
            l = read_acc(indomain_log(dataset, "legacy", seed))
            s = read_acc(indomain_log(dataset, "safe", seed))
            print(f"| {dataset} | {seed} | {fmt(c)} | {fmt(l)} | {fmt(s)} | {fmt(None if s is None or c is None else s-c)} | {fmt(None if s is None or l is None else s-l)} |")


def summarize_xd(source: str, targets: list[str]):
    print(f"# Residual Formula Ablation: Cross-dataset DG, source = {source}")
    print()
    print("| Target | CoOp | Legacy | Safe | Safe-CoOp | Safe-Legacy |")
    print("|---|---:|---:|---:|---:|---:|")

    deltas_coop = []
    deltas_legacy = []
    positive_safe_vs_coop = 0
    positive_safe_vs_legacy = 0
    total_pairs = 0

    for target in targets:
        vals = {}
        for variant in ["coop", "legacy", "safe"]:
            accs = [read_acc(xd_log(source, target, variant, seed)) for seed in SEEDS]
            vals[variant] = (*stat(accs), accs)

        coop_m, coop_s, coop_n, coop_accs = vals["coop"]
        legacy_m, legacy_s, legacy_n, legacy_accs = vals["legacy"]
        safe_m, safe_s, safe_n, safe_accs = vals["safe"]
        d_coop = None if safe_m is None or coop_m is None else safe_m - coop_m
        d_legacy = None if safe_m is None or legacy_m is None else safe_m - legacy_m
        if d_coop is not None:
            deltas_coop.append(d_coop)
        if d_legacy is not None:
            deltas_legacy.append(d_legacy)

        for c, l, s in zip(coop_accs, legacy_accs, safe_accs):
            if s is not None and c is not None:
                total_pairs += 1
                if s > c:
                    positive_safe_vs_coop += 1
            if s is not None and l is not None:
                if s > l:
                    positive_safe_vs_legacy += 1

        print(
            f"| {target} | {fmt_ms(coop_m, coop_s, coop_n)} | {fmt_ms(legacy_m, legacy_s, legacy_n)} | "
            f"{fmt_ms(safe_m, safe_s, safe_n)} | {fmt(d_coop)} | {fmt(d_legacy)} |"
        )

    avg_sc = mean(deltas_coop) if deltas_coop else None
    avg_sl = mean(deltas_legacy) if deltas_legacy else None
    print(f"| **Average Delta** |  |  |  | **{fmt(avg_sc)}** | **{fmt(avg_sl)}** |")
    print()
    print(f"Safe > CoOp seed-level cases: **{positive_safe_vs_coop}/{total_pairs}**")
    print(f"Safe > Legacy seed-level cases: **{positive_safe_vs_legacy}/{total_pairs}**")
    print()
    print("## Seed-level details")
    print()
    print("| Target | Seed | CoOp | Legacy | Safe | Safe-CoOp | Safe-Legacy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for target in targets:
        for seed in SEEDS:
            c = read_acc(xd_log(source, target, "coop", seed))
            l = read_acc(xd_log(source, target, "legacy", seed))
            s = read_acc(xd_log(source, target, "safe", seed))
            print(f"| {target} | {seed} | {fmt(c)} | {fmt(l)} | {fmt(s)} | {fmt(None if s is None or c is None else s-c)} | {fmt(None if s is None or l is None else s-l)} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["indomain", "xd"], required=True)
    parser.add_argument("--datasets", nargs="*", default=["eurosat", "dtd", "caltech101"])
    parser.add_argument("--source", default="caltech101")
    parser.add_argument("--targets", nargs="*", default=["oxford_pets", "eurosat", "dtd", "food101", "oxford_flowers", "stanford_cars", "fgvc_aircraft", "ucf101", "sun397"])
    args = parser.parse_args()

    if args.mode == "indomain":
        summarize_indomain(args.datasets)
    else:
        summarize_xd(args.source, args.targets)


if __name__ == "__main__":
    main()
