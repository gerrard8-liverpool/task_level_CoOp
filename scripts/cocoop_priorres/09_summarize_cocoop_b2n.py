from pathlib import Path
import re
import statistics as stats

ROOT = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output_cocoop_priorres/base2new")
OUT = Path("/workspace/meta_prompt_1/outputs/cocoop_priorres_b2n_summary.md")
ACC_PAT = re.compile(r"\* accuracy:\s*([0-9.]+)%|accuracy:\s*([0-9.]+)")

DATASETS = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
SEEDS = [1, 2, 3]
METHODS = {
    "CoCoOp": "CoCoOp/rn50_c4_ep10_batch1/nctx4_ctxinit_a_photo_of_a",
    "CoCoOpPriorRes": "CoCoOpPriorRes/rn50_c4_ep10_batch1/nctx4_ctxinit_a_photo_of_a_safe_noalt_baseprior",
}

def read_acc(path: Path):
    if not path.is_file():
        return None
    txt = path.read_text(errors="ignore")
    vals = []
    for m in ACC_PAT.finditer(txt):
        v = m.group(1) or m.group(2)
        vals.append(float(v))
    return vals[-1] if vals else None

def hm(b, n):
    if b is None or n is None or b + n == 0:
        return None
    return 2 * b * n / (b + n)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def fmt(x):
    return "" if x is None else f"{x:.2f}"

rows = []
for d in DATASETS:
    for s in SEEDS:
        rec = {"dataset": d, "seed": s}
        for method, subdir in METHODS.items():
            base_log = ROOT / "test_base" / d / "shots_16" / subdir / f"seed{s}" / "log.txt"
            new_log = ROOT / "test_new" / d / "shots_16" / subdir / f"seed{s}" / "log.txt"
            b = read_acc(base_log)
            n = read_acc(new_log)
            rec[f"{method}_base"] = b
            rec[f"{method}_new"] = n
            rec[f"{method}_hm"] = hm(b, n)
        rows.append(rec)

lines = []
lines.append("# CoCoOpPriorRes B2N Summary\n")
lines.append("| Dataset | CoCoOp Base | CoCoOp New | CoCoOp HM | PriorRes Base | PriorRes New | PriorRes HM | ΔHM |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

for d in DATASETS:
    ds = [r for r in rows if r["dataset"] == d]
    cb = mean([r["CoCoOp_base"] for r in ds])
    cn = mean([r["CoCoOp_new"] for r in ds])
    ch = mean([r["CoCoOp_hm"] for r in ds])
    pb = mean([r["CoCoOpPriorRes_base"] for r in ds])
    pn = mean([r["CoCoOpPriorRes_new"] for r in ds])
    ph = mean([r["CoCoOpPriorRes_hm"] for r in ds])
    delta = ph - ch if ph is not None and ch is not None else None
    lines.append(f"| {d} | {fmt(cb)} | {fmt(cn)} | {fmt(ch)} | {fmt(pb)} | {fmt(pn)} | {fmt(ph)} | {fmt(delta)} |")

cb = mean([r["CoCoOp_base"] for r in rows])
cn = mean([r["CoCoOp_new"] for r in rows])
ch = mean([r["CoCoOp_hm"] for r in rows])
pb = mean([r["CoCoOpPriorRes_base"] for r in rows])
pn = mean([r["CoCoOpPriorRes_new"] for r in rows])
ph = mean([r["CoCoOpPriorRes_hm"] for r in rows])
delta = ph - ch if ph is not None and ch is not None else None
lines.append(f"| **Average** | **{fmt(cb)}** | **{fmt(cn)}** | **{fmt(ch)}** | **{fmt(pb)}** | **{fmt(pn)}** | **{fmt(ph)}** | **{fmt(delta)}** |")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\n[WROTE] {OUT}")
