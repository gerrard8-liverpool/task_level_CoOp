#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from statistics import mean

ROOT = Path('/workspace/meta_prompt_1/third_party/CoOp_clean/output/xd/test')
SEEDS = [1, 2, 3]
ALL_DATASETS = [
    'oxford_pets', 'eurosat', 'dtd', 'food101', 'oxford_flowers',
    'caltech101', 'stanford_cars', 'fgvc_aircraft', 'ucf101', 'sun397'
]
ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")
CFG_TAG = {
    'rn101': 'rn101_ep50',
    'vit_b16': 'vit_b16_ep50',
    'vit_b32': 'vit_b32_ep50',
}

def read_acc(p: Path):
    if not p.exists():
        return None
    text = p.read_text(errors='ignore')
    vals = ACC_RE.findall(text)
    return float(vals[-1]) if vals else None

def avg(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None

def fmt(x):
    return '' if x is None else f'{x:.2f}'

def path(method, cfg_tag, source, target, seed):
    base = ROOT / target / f'source_{source}' / 'shots_16'
    if method == 'coop':
        return base / f'CoOp/{cfg_tag}/nctx16_cscFalse_ctpend/seed{seed}/log.txt'
    if method == 'safe':
        return base / f'CoOpPriorRes/{cfg_tag}/nctx16_cscFalse_ctpend_safe_noalt/seed{seed}/log.txt'
    raise ValueError(method)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('backbone_tag', choices=sorted(CFG_TAG.keys()))
    ap.add_argument('--sources', nargs='+', default=['caltech101', 'food101', 'sun397'])
    ap.add_argument('--targets', nargs='*', default=None, help='Optional fixed target subset for all sources')
    args = ap.parse_args()

    cfg_tag = CFG_TAG[args.backbone_tag]
    all_deltas = []
    seed_pos = 0
    seed_total = 0
    source_rows = []

    print(f'# Cross-dataset DG Backbone Extension: {args.backbone_tag}')
    print()
    print('| Source | Target | CoOp | Safe | Safe-CoOp |')
    print('|---|---|---:|---:|---:|')

    for source in args.sources:
        targets = args.targets if args.targets else [d for d in ALL_DATASETS if d != source]
        source_deltas = []
        for target in targets:
            if target == source:
                continue
            coop_vals = [read_acc(path('coop', cfg_tag, source, target, s)) for s in SEEDS]
            safe_vals = [read_acc(path('safe', cfg_tag, source, target, s)) for s in SEEDS]
            c = avg(coop_vals)
            sf = avg(safe_vals)
            delta = None if c is None or sf is None else sf - c
            if delta is not None:
                source_deltas.append(delta)
                all_deltas.append(delta)
            for cv, sv in zip(coop_vals, safe_vals):
                if cv is not None and sv is not None:
                    seed_total += 1
                    if sv > cv:
                        seed_pos += 1
            print(f'| {source} | {target} | {fmt(c)} | {fmt(sf)} | {fmt(delta)} |')
        s_avg = avg(source_deltas)
        source_rows.append((source, s_avg))
        print(f'| **{source}** | **Average** |  |  | **{fmt(s_avg)}** |')

    print()
    print('# Source-level Average')
    print()
    print('| Source | Safe-CoOp Avg Delta |')
    print('|---|---:|')
    for source, val in source_rows:
        print(f'| {source} | {fmt(val)} |')
    print(f'| **Overall** | **{fmt(avg(all_deltas))}** |')
    print()
    print(f'Safe > CoOp seed-level cases: **{seed_pos}/{seed_total}**')

if __name__ == '__main__':
    main()
