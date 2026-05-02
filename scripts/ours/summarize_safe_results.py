#!/usr/bin/env python3
import re
import csv
import math
import statistics as stats
from pathlib import Path
from datetime import datetime

ROOT = Path("/workspace/meta_prompt_1")
OUT_ROOT = ROOT / "third_party/CoOp_clean/output"
SAVE_DIR = ROOT / "outputs/safe_summary_20260428"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "oxford_pets",
    "eurosat",
    "dtd",
    "food101",
    "oxford_flowers",
    "caltech101",
    "stanford_cars",
    "fgvc_aircraft",
    "ucf101",
    "sun397",
]

SEEDS = [1, 2, 3]

METHODS = {
    "coop_m16k16": {
        "trainer": "CoOp",
        "patterns": [
            "rn50_ep50_16shots/**/log.txt",
        ],
        "must_contain": ["nctx16", "cscFalse", "ctpend"],
        "exclude": ["smoke", "diag", "cachebuild"],
    },
    "safe_noalt": {
        "trainer": "CoOpPriorRes",
        "patterns": [
            "rn50_ep50_16shots/*safe_noalt_fresh*/log.txt",
        ],
        "must_contain": [],
        "exclude": ["smoke", "diag", "cachebuild"],
    },
    "safe_b0p2": {
        "trainer": "CoOpPriorRes",
        "patterns": [
            "rn50_ep50_16shots/*safe_b0p2_fresh*/log.txt",
        ],
        "must_contain": [],
        "exclude": ["smoke", "diag", "cachebuild"],
    },
    "legacy_nob": {
        "trainer": "CoOpPriorRes",
        "patterns": [
            "rn50_ep50_16shots/*nob_fresh*/log.txt",
        ],
        "must_contain": [],
        "exclude": ["safe", "smoke", "diag", "cachebuild"],
    },
}

ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")
F1_RE = re.compile(r"\*\s*macro_f1:\s*([0-9.]+)%")

def parse_metric(log_path: Path):
    text = log_path.read_text(errors="ignore")
    accs = ACC_RE.findall(text)
    f1s = F1_RE.findall(text)
    acc = float(accs[-1]) if accs else None
    f1 = float(f1s[-1]) if f1s else None
    return acc, f1

def path_ok(path: Path, seed: int, must_contain, exclude):
    s = str(path)
    if f"seed{seed}" not in s and f"seed_{seed}" not in s:
        return False
    for x in must_contain:
        if x not in s:
            return False
    for x in exclude:
        if x in s:
            return False
    return True

def find_log(dataset: str, method: str, seed: int):
    spec = METHODS[method]
    base = OUT_ROOT / dataset / spec["trainer"]
    if not base.exists():
        return None

    candidates = []
    for pat in spec["patterns"]:
        for p in base.glob(pat):
            if path_ok(p, seed, spec["must_contain"], spec["exclude"]):
                candidates.append(p)

    if not candidates:
        return None

    # Prefer paths with "fresh"; if multiple, use newest file.
    candidates = sorted(
        candidates,
        key=lambda p: (("fresh" in str(p)), p.stat().st_mtime),
        reverse=True,
    )
    return candidates[0]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) <= 1:
        return None
    return stats.stdev(xs)

def fmt(x, nd=2):
    if x is None:
        return ""
    return f"{x:.{nd}f}"

seed_rows = []

for dataset in DATASETS:
    for method in METHODS:
        for seed in SEEDS:
            log_path = find_log(dataset, method, seed)
            if log_path is None:
                seed_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "accuracy": "",
                    "macro_f1": "",
                    "status": "MISSING",
                    "log_path": "",
                })
                continue

            acc, f1 = parse_metric(log_path)
            seed_rows.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "accuracy": acc if acc is not None else "",
                "macro_f1": f1 if f1 is not None else "",
                "status": "OK" if acc is not None else "NO_ACC",
                "log_path": str(log_path),
            })

summary_rows = []
for dataset in DATASETS:
    for method in METHODS:
        vals = [
            r["accuracy"]
            for r in seed_rows
            if r["dataset"] == dataset and r["method"] == method and isinstance(r["accuracy"], float)
        ]
        f1s = [
            r["macro_f1"]
            for r in seed_rows
            if r["dataset"] == dataset and r["method"] == method and isinstance(r["macro_f1"], float)
        ]
        summary_rows.append({
            "dataset": dataset,
            "method": method,
            "n": len(vals),
            "acc_mean": mean(vals),
            "acc_std": std(vals),
            "macro_f1_mean": mean(f1s),
            "macro_f1_std": std(f1s),
            "status": "OK" if len(vals) == 3 else f"PARTIAL_{len(vals)}/3",
        })

def get_summary(dataset, method, key):
    for r in summary_rows:
        if r["dataset"] == dataset and r["method"] == method:
            return r[key]
    return None

main_rows = []
for dataset in DATASETS:
    coop = get_summary(dataset, "coop_m16k16", "acc_mean")
    safe = get_summary(dataset, "safe_noalt", "acc_mean")
    legacy = get_summary(dataset, "legacy_nob", "acc_mean")
    b = get_summary(dataset, "safe_b0p2", "acc_mean")

    main_rows.append({
        "dataset": dataset,
        "coop_m16k16": coop,
        "legacy_nob": legacy,
        "safe_noalt": safe,
        "safe_b0p2": b,
        "delta_safe_vs_coop": (safe - coop) if safe is not None and coop is not None else None,
        "delta_safe_vs_legacy": (safe - legacy) if safe is not None and legacy is not None else None,
        "delta_b_vs_safe": (b - safe) if b is not None and safe is not None else None,
    })

def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, float):
                    out[k] = f"{v:.4f}"
                else:
                    out[k] = v
            w.writerow(out)

write_csv(
    SAVE_DIR / "seed_results.csv",
    seed_rows,
    ["dataset", "method", "seed", "accuracy", "macro_f1", "status", "log_path"],
)

write_csv(
    SAVE_DIR / "summary_by_dataset.csv",
    summary_rows,
    ["dataset", "method", "n", "acc_mean", "acc_std", "macro_f1_mean", "macro_f1_std", "status"],
)

write_csv(
    SAVE_DIR / "main_compare_safe_vs_coop.csv",
    main_rows,
    [
        "dataset",
        "coop_m16k16",
        "legacy_nob",
        "safe_noalt",
        "safe_b0p2",
        "delta_safe_vs_coop",
        "delta_safe_vs_legacy",
        "delta_b_vs_safe",
    ],
)

# Markdown report
md = []
md.append("# Safe Residual Summary\n")
md.append(f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n")
md.append("\n## Main comparison\n")
md.append("| Dataset | CoOp m16/k16 | Legacy no_b | Safe noalt | Safe b0p2 | Safe-CoOp | Safe-Legacy | b-Safe |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in main_rows:
    md.append(
        f"| {r['dataset']} | {fmt(r['coop_m16k16'])} | {fmt(r['legacy_nob'])} | "
        f"{fmt(r['safe_noalt'])} | {fmt(r['safe_b0p2'])} | "
        f"{fmt(r['delta_safe_vs_coop'])} | {fmt(r['delta_safe_vs_legacy'])} | {fmt(r['delta_b_vs_safe'])} |"
    )

safe_vals = [r["safe_noalt"] for r in main_rows if r["safe_noalt"] is not None]
coop_vals = [r["coop_m16k16"] for r in main_rows if r["coop_m16k16"] is not None]
common_deltas = [r["delta_safe_vs_coop"] for r in main_rows if r["delta_safe_vs_coop"] is not None]

md.append("\n## Aggregate\n")
md.append(f"- Safe noalt average over available datasets: **{fmt(mean(safe_vals))}**")
md.append(f"- CoOp m16/k16 average over available datasets: **{fmt(mean(coop_vals))}**")
md.append(f"- Average delta safe vs CoOp over matched datasets: **{fmt(mean(common_deltas))}**")

md.append("\n## Notes\n")
md.append("- `safe_noalt` corresponds to `USE_B=False`, `USE_LEGACY_RESIDUAL=False`, `ALTERNATE_OPT=False`.")
md.append("- `safe_b0p2` corresponds to `USE_B=True`, `B_LOSS_WEIGHT=0.2`, `USE_LEGACY_RESIDUAL=False`, `ALTERNATE_OPT=False`.")
md.append("- Missing CoOp or legacy entries mean the script did not find matching logs under the expected output directories.")
md.append("- For the current base-task paper table, use `safe_noalt` as the main method. Treat `safe_b0p2` as an auxiliary ablation only.")

(SAVE_DIR / "safe_summary.md").write_text("\n".join(md))

print(f"[OK] Wrote summary files to: {SAVE_DIR}")
print(f"  - {SAVE_DIR / 'seed_results.csv'}")
print(f"  - {SAVE_DIR / 'summary_by_dataset.csv'}")
print(f"  - {SAVE_DIR / 'main_compare_safe_vs_coop.csv'}")
print(f"  - {SAVE_DIR / 'safe_summary.md'}")
