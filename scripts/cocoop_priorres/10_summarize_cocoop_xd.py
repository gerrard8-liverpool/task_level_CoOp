from pathlib import Path
import re

ROOT = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output_cocoop_priorres/xd")
OUT = Path("/workspace/meta_prompt_1/outputs/cocoop_priorres_xd_summary.md")
ACC_PAT = re.compile(r"\* accuracy:\s*([0-9.]+)%|accuracy:\s*([0-9.]+)")

ALL = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
SOURCES = ["caltech101", "food101", "sun397"]
SEEDS = [1, 2, 3]

METHODS = {
    "CoCoOp": "CoCoOp/rn50_c4_ep10_batch1/nctx4_ctxinit_a_photo_of_a",
    "CoCoOpPriorRes": "CoCoOpPriorRes/rn50_c4_ep10_batch1/nctx4_ctxinit_a_photo_of_a_safe_noalt_sourceprior",
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

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def fmt(x):
    return "" if x is None else f"{x:.2f}"

records = []
for src in SOURCES:
    targets = [d for d in ALL if d != src]
    for tgt in targets:
        for seed in SEEDS:
            rec = {"source": src, "target": tgt, "seed": seed}
            for method, subdir in METHODS.items():
                log = ROOT / "test" / tgt / f"source_{src}" / "shots_16" / subdir / f"seed{seed}" / "log.txt"
                rec[method] = read_acc(log)
            records.append(rec)

lines = []
lines.append("# CoCoOpPriorRes Cross-Dataset DG Summary\n")
lines.append("| Source | Target | CoCoOp | PriorRes | Δ |")
lines.append("|---|---|---:|---:|---:|")

for src in SOURCES:
    targets = [d for d in ALL if d != src]
    for tgt in targets:
        rs = [r for r in records if r["source"] == src and r["target"] == tgt]
        c = mean([r["CoCoOp"] for r in rs])
        p = mean([r["CoCoOpPriorRes"] for r in rs])
        delta = p - c if p is not None and c is not None else None
        lines.append(f"| {src} | {tgt} | {fmt(c)} | {fmt(p)} | {fmt(delta)} |")
    rs = [r for r in records if r["source"] == src]
    c = mean([r["CoCoOp"] for r in rs])
    p = mean([r["CoCoOpPriorRes"] for r in rs])
    delta = p - c if p is not None and c is not None else None
    lines.append(f"| **{src}** | **Average** | **{fmt(c)}** | **{fmt(p)}** | **{fmt(delta)}** |")

c = mean([r["CoCoOp"] for r in records])
p = mean([r["CoCoOpPriorRes"] for r in records])
delta = p - c if p is not None and c is not None else None
lines.append(f"| **Overall** | **Average** | **{fmt(c)}** | **{fmt(p)}** | **{fmt(delta)}** |")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\n[WROTE] {OUT}")
