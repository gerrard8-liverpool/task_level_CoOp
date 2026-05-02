#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/base2new")
DATASETS = ["caltech101", "oxford_pets", "dtd", "eurosat"]
SEEDS = [1, 2, 3]

ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")

def read_acc(path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    vals = ACC_RE.findall(text)
    return float(vals[-1]) if vals else None

def hm(base, new):
    if base is None or new is None or base + new == 0:
        return None
    return 2 * base * new / (base + new)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

print("| Dataset | Seed | Base | New | HM |")
print("|---|---:|---:|---:|---:|")

for dataset in DATASETS:
    rows = []

    for seed in SEEDS:
        base_log = ROOT / "test_base" / dataset / "shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend" / f"seed{seed}" / "log.txt"
        new_log = ROOT / "test_new" / dataset / "shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend" / f"seed{seed}" / "log.txt"

        base = read_acc(base_log)
        new = read_acc(new_log)
        h = hm(base, new)
        rows.append((base, new, h))

        base_s = "" if base is None else f"{base:.2f}"
        new_s = "" if new is None else f"{new:.2f}"
        h_s = "" if h is None else f"{h:.2f}"
        print(f"| {dataset} | {seed} | {base_s} | {new_s} | {h_s} |")

    mb = mean([r[0] for r in rows])
    mn = mean([r[1] for r in rows])
    mh = mean([r[2] for r in rows])

    mb_s = "" if mb is None else f"{mb:.2f}"
    mn_s = "" if mn is None else f"{mn:.2f}"
    mh_s = "" if mh is None else f"{mh:.2f}"

    print(f"| **{dataset}** | **mean** | **{mb_s}** | **{mn_s}** | **{mh_s}** |")
