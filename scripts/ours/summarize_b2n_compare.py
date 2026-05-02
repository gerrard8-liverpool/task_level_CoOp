#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/base2new")

DATASETS = [
    "caltech101",
    "oxford_pets",
    "dtd",
    "eurosat",
    "food101",
    "oxford_flowers",
    "stanford_cars",
    "fgvc_aircraft",
    "ucf101",
    "sun397",
]

SEEDS = [1, 2, 3]
ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")

def read_acc(path: Path):
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

def fmt(x):
    return "" if x is None else f"{x:.2f}"

def get_logs(method, dataset, seed):
    if method == "CoOp":
        base_log = ROOT / "test_base" / dataset / "shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend" / f"seed{seed}" / "log.txt"
        new_log = ROOT / "test_new" / dataset / "shots_16/CoOp/rn50_ep50/nctx16_cscFalse_ctpend" / f"seed{seed}" / "log.txt"
    elif method == "PriorRes":
        base_log = ROOT / "test_base" / dataset / "shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt" / f"seed{seed}" / "log.txt"
        new_log = ROOT / "test_new" / dataset / "shots_16/CoOpPriorRes/rn50_ep50/nctx16_cscFalse_ctpend_safe_noalt" / f"seed{seed}" / "log.txt"
    else:
        raise ValueError(method)
    return base_log, new_log

print("# B2N CoOp vs Safe PriorRes")
print()
print("| Dataset | Method | Base | New | HM |")
print("|---|---|---:|---:|---:|")

compare_rows = []

for dataset in DATASETS:
    method_means = {}

    for method in ["CoOp", "PriorRes"]:
        rows = []
        for seed in SEEDS:
            base_log, new_log = get_logs(method, dataset, seed)
            base = read_acc(base_log)
            new = read_acc(new_log)
            h = hm(base, new)
            rows.append((base, new, h))

        mb = mean([r[0] for r in rows])
        mn = mean([r[1] for r in rows])
        mh = mean([r[2] for r in rows])

        method_means[method] = (mb, mn, mh)
        print(f"| {dataset} | {method} | {fmt(mb)} | {fmt(mn)} | {fmt(mh)} |")

    cb, cn, ch = method_means["CoOp"]
    ob, on, oh = method_means["PriorRes"]

    db = None if cb is None or ob is None else ob - cb
    dn = None if cn is None or on is None else on - cn
    dh = None if ch is None or oh is None else oh - ch

    compare_rows.append((dataset, db, dn, dh))
    print(f"| {dataset} | Δ PriorRes-CoOp | {fmt(db)} | {fmt(dn)} | {fmt(dh)} |")

print()
print("# Delta Summary")
print()
print("| Dataset | ΔBase | ΔNew | ΔHM |")
print("|---|---:|---:|---:|")

for dataset, db, dn, dh in compare_rows:
    print(f"| {dataset} | {fmt(db)} | {fmt(dn)} | {fmt(dh)} |")

avg_db = mean([r[1] for r in compare_rows])
avg_dn = mean([r[2] for r in compare_rows])
avg_dh = mean([r[3] for r in compare_rows])

print(f"| **Average** | **{fmt(avg_db)}** | **{fmt(avg_dn)}** | **{fmt(avg_dh)}** |")
