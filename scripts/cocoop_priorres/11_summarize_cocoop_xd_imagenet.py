from pathlib import Path
import re
import statistics

PROJECT_ROOT = Path("/workspace/meta_prompt_1")
ROOT = PROJECT_ROOT / "third_party/CoOp_clean/output_cocoop_priorres/xd/test"
OUT = PROJECT_ROOT / "outputs/cocoop_priorres/cocoop_priorres_xd_imagenet_source_summary.md"

SOURCE = "imagenet"
CFG_TAG = "rn50_c4_ep10_batch4_a100"

TARGETS = [
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

METHODS = {
    "CoCoOp": "CoCoOp",
    "PriorRes": "CoCoOpPriorRes",
}

ACC_PATTERNS = [
    re.compile(r"\* accuracy:\s*([0-9.]+)%?"),
    re.compile(r"accuracy:\s*([0-9.]+)%?"),
]


def read_acc(log_path: Path):
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="ignore")
    vals = []
    for pat in ACC_PATTERNS:
        vals.extend(float(m.group(1)) for m in pat.finditer(text))
    return vals[-1] if vals else None


def find_log(target: str, method_dir: str, seed: int):
    base = ROOT / target / f"source_{SOURCE}" / "shots_16" / method_dir
    if not base.exists():
        return None

    candidates = list(base.glob(f"{CFG_TAG}/**/seed{seed}/log.txt"))
    if not candidates:
        candidates = list(base.glob(f"**/seed{seed}/log.txt"))

    if not candidates:
        return None

    # Prefer logs containing the expected cfg tag.
    candidates = sorted(candidates, key=lambda p: (CFG_TAG not in str(p), len(str(p))))
    return candidates[0]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def fmt(x):
    return "" if x is None else f"{x:.2f}"


records = []
for target in TARGETS:
    for seed in SEEDS:
        row = {"target": target, "seed": seed}
        for name, method_dir in METHODS.items():
            log = find_log(target, method_dir, seed)
            row[name] = read_acc(log) if log else None
            row[f"{name}_log"] = str(log) if log else ""
        records.append(row)

lines = []
lines.append("# CoCoOpPriorRes ImageNet-source Cross-Dataset DG Summary\n")
lines.append(f"Config tag: `{CFG_TAG}`\n")
lines.append("| Source | Target | CoCoOp | PriorRes | Δ | Done Seeds |")
lines.append("|---|---|---:|---:|---:|---:|")

pos_seed = 0
valid_seed = 0

for target in TARGETS:
    rs = [r for r in records if r["target"] == target]
    c_vals = [r["CoCoOp"] for r in rs]
    p_vals = [r["PriorRes"] for r in rs]

    c = mean(c_vals)
    p = mean(p_vals)
    d = p - c if c is not None and p is not None else None

    done = sum(1 for r in rs if r["CoCoOp"] is not None or r["PriorRes"] is not None)

    lines.append(f"| {SOURCE} | {target} | {fmt(c)} | {fmt(p)} | {fmt(d)} | {done}/3 |")

    for r in rs:
        if r["CoCoOp"] is not None and r["PriorRes"] is not None:
            valid_seed += 1
            if r["PriorRes"] > r["CoCoOp"]:
                pos_seed += 1

all_c = [r["CoCoOp"] for r in records]
all_p = [r["PriorRes"] for r in records]
avg_c = mean(all_c)
avg_p = mean(all_p)
avg_d = avg_p - avg_c if avg_c is not None and avg_p is not None else None

lines.append(f"| **{SOURCE}** | **Average** | **{fmt(avg_c)}** | **{fmt(avg_p)}** | **{fmt(avg_d)}** |  |")
lines.append("")
lines.append("# Source-level Average\n")
lines.append("| Source | Avg Delta | Positive Seed Cases |")
lines.append("|---|---:|---:|")
lines.append(f"| {SOURCE} | {fmt(avg_d)} | {pos_seed}/{valid_seed} |")
lines.append(f"| **Overall** | **{fmt(avg_d)}** | **{pos_seed}/{valid_seed}** |")

lines.append("\n# Seed-level Details\n")
lines.append("| Target | Seed | CoCoOp | PriorRes | Δ |")
lines.append("|---|---:|---:|---:|---:|")
for r in records:
    c = r["CoCoOp"]
    p = r["PriorRes"]
    d = p - c if c is not None and p is not None else None
    lines.append(f"| {r['target']} | {r['seed']} | {fmt(c)} | {fmt(p)} | {fmt(d)} |")

lines.append("\n# Missing / Incomplete Logs\n")
for r in records:
    for name in METHODS:
        if r[name] is None:
            lines.append(f"- Missing or no accuracy: target={r['target']}, seed={r['seed']}, method={name}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\n[WROTE] {OUT}")
