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
    m = ACC_RE.findall(text)
    return float(m[-1]) if m else None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def hm(a, b):
    if a is None or b is None or a + b == 0:
        return None
    return 2 * a * b / (a + b)

print("| Dataset | Seed | Base | New | HM |")
print("|---|---:|---:|---:|---:|")

all_rows = []
for dataset in DATASETS:
    rows = []
    for seed in SEEDS:
        base_log = ROOT / "test_base" / dataset / "shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt" / f"seed{seed}" / "log.txt"
        new_log = ROOT / "test_new" / dataset / "shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt" / f"seed{seed}" / "log.txt"
        b = read_acc(base_log)
        n = read_acc(new_log)
        h = hm(b, n)
        rows.append((b, n, h))
        print(f"| {dataset} | {seed} | {b if b is not None else ''} | {n if n is not None else ''} | {h:.2f} |" if h is not None else f"| {dataset} | {seed} | {b or ''} | {n or ''} |  |")

    mb = mean([x[0] for x in rows])
    mn = mean([x[1] for x in rows])
    mh = mean([x[2] for x in rows])
    print(f"| **{dataset}** | **mean** | **{mb:.2f}** | **{mn:.2f}** | **{mh:.2f}** |" if mh is not None else f"| **{dataset}** | **mean** |  |  |  |")
