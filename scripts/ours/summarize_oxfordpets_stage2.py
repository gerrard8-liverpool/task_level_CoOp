import re
import statistics as st
from pathlib import Path

BASE = Path("/workspace/meta_prompt_1/third_party/CoOp_clean/output/oxford_pets/CoOpPriorRes_stage2")
MODES = ["aonly", "bonly", "ab"]
SEEDS = [1, 2, 3]

ACC_PAT = re.compile(r"\* accuracy:\s*([0-9.]+)%")
F1_PAT = re.compile(r"\* macro_f1:\s*([0-9.]+)%")

MEFF_PAT = re.compile(r"meff\s+([0-9.]+)\s+\(")
KEFF_PAT = re.compile(r"keff\s+([0-9.]+)\s+\(")
A0_PAT = re.compile(r"a0_mean\s+([0-9.]+)\s+\(")
A_PAT = re.compile(r"a_mean\s+([0-9.]+)\s+\(")
B0_PAT = re.compile(r"b0_mean\s+([0-9.]+)\s+\(")
B_PAT = re.compile(r"b_mean\s+([0-9.]+)\s+\(")
DA_PAT = re.compile(r"delta_a_norm\s+([0-9.]+)\s+\(")
DB_PAT = re.compile(r"delta_b_norm\s+([0-9.]+)\s+\(")
LB_PAT = re.compile(r"loss_b\s+([0-9.]+)\s+\(")
BE_PAT = re.compile(r"b_entropy\s+([0-9.]+)\s+\(")
T1_PAT = re.compile(r"top1_weight\s+([0-9.]+)\s+\(")
T4_PAT = re.compile(r"top4_weight_sum\s+([0-9.]+)\s+\(")

def last_float(pat, text):
    xs = pat.findall(text)
    return float(xs[-1]) if xs else None

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (st.mean(vals), st.stdev(vals))

def fmt(ms):
    if ms is None:
        return "N/A"
    m, s = ms
    return f"{m:.2f} ± {s:.2f}"

def parse_log(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "acc": last_float(ACC_PAT, text),
        "f1": last_float(F1_PAT, text),
        "meff": last_float(MEFF_PAT, text),
        "keff": last_float(KEFF_PAT, text),
        "a0": last_float(A0_PAT, text),
        "a": last_float(A_PAT, text),
        "b0": last_float(B0_PAT, text),
        "b": last_float(B_PAT, text),
        "da": last_float(DA_PAT, text),
        "db": last_float(DB_PAT, text),
        "loss_b": last_float(LB_PAT, text),
        "b_entropy": last_float(BE_PAT, text),
        "top1": last_float(T1_PAT, text),
        "top4": last_float(T4_PAT, text),
        "path": str(path),
    }

all_rows = {}
for mode in MODES:
    rows = []
    for seed in SEEDS:
        p = BASE / mode / f"rn50_16shots_beta0.2_seed{seed}" / "log.txt"
        row = parse_log(p)
        rows.append(row)
        print(f"[{mode}] seed={seed}: {'OK' if row else 'MISSING'} | {p}")
    all_rows[mode] = rows

print("\n=== Per-seed details ===")
for mode in MODES:
    for seed, row in zip(SEEDS, all_rows[mode]):
        print(f"\n[{mode}] seed={seed}")
        if row is None:
            print("  MISSING")
            continue
        for k in ["acc","f1","meff","keff","a0","a","b0","b","da","db","loss_b","b_entropy","top1","top4"]:
            print(f"  {k}: {row[k]}")

print("\n=== Summary Table ===")
print("| mode | acc | macro_f1 | meff | keff | a0_mean | a_mean | b0_mean | b_mean | delta_a_norm | delta_b_norm | loss_b | b_entropy | top1_weight | top4_weight_sum |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for mode in MODES:
    rows = [r for r in all_rows[mode] if r is not None]
    if not rows:
        print(f"| {mode} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        continue

    print(
        f"| {mode} | "
        f"{fmt(mean_std([r['acc'] for r in rows]))} | "
        f"{fmt(mean_std([r['f1'] for r in rows]))} | "
        f"{fmt(mean_std([r['meff'] for r in rows]))} | "
        f"{fmt(mean_std([r['keff'] for r in rows]))} | "
        f"{fmt(mean_std([r['a0'] for r in rows]))} | "
        f"{fmt(mean_std([r['a'] for r in rows]))} | "
        f"{fmt(mean_std([r['b0'] for r in rows]))} | "
        f"{fmt(mean_std([r['b'] for r in rows]))} | "
        f"{fmt(mean_std([r['da'] for r in rows]))} | "
        f"{fmt(mean_std([r['db'] for r in rows]))} | "
        f"{fmt(mean_std([r['loss_b'] for r in rows]))} | "
        f"{fmt(mean_std([r['b_entropy'] for r in rows]))} | "
        f"{fmt(mean_std([r['top1'] for r in rows]))} | "
        f"{fmt(mean_std([r['top4'] for r in rows]))} |"
    )
