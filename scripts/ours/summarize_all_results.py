#!/usr/bin/env python3
"""
Summarize CoOp / CoOpPriorRes experiment logs into CSV tables.

Designed for meta_prompt_1 / CoOp_clean output layout, including mixed cases:
- Some datasets have full CoOp k-m grids.
- Some datasets only have k=16 CoOp grids.
- PriorRes no_b / b0p2 are fixed at k=16, m=16 by default.
- ImageNet can be added later without changing the script; if logs appear under output/imagenet, they are parsed automatically.

Default output folder:
    /workspace/meta_prompt_1/outputs/summary_csv

Generated CSVs:
    run_level.csv
    summary_by_setting.csv
    main_table_k16.csv
    coop_best_available.csv
    missing_expected.csv
    coverage_by_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics as stats
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_DATASETS = [
    "oxford_pets",
    "eurosat",
    "dtd",
    "food101",
    "oxford_flowers",
    "caltech101",
    "fgvc_aircraft",
    "stanford_cars",
    "ucf101",
    "sun397",
    "imagenet",
]

DATASET_ALIASES = {
    "caltech-101": "caltech101",
    "caltech101": "caltech101",
}

EXCLUDE_KEYWORDS_DEFAULT = ["smoke", "cachebuild", "debug"]

FLOAT_KEYS = [
    "accuracy",
    "macro_f1",
    "meff",
    "keff",
    "a0_mean",
    "b0_mean",
    "loss_b",
    "lambda_t",
    "best_val_acc",
]


@dataclass
class RunRow:
    dataset: str
    dataset_dir: str
    trainer: str
    variant: str
    cfg: str
    shots: Optional[int]
    n_ctx: Optional[int]
    csc: Optional[str]
    ctp: Optional[str]
    seed: Optional[int]
    status: str
    accuracy: Optional[float]
    macro_f1: Optional[float]
    meff: Optional[float]
    keff: Optional[float]
    a0_mean: Optional[float]
    b0_mean: Optional[float]
    loss_b: Optional[float]
    lambda_t: Optional[float]
    best_val_acc: Optional[float]
    log_mtime: str
    log_path: str
    run_dir: str


def norm_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def maybe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def last_float(patterns: Iterable[str], text: str, flags: int = re.IGNORECASE | re.MULTILINE) -> Optional[float]:
    vals: List[str] = []
    for p in patterns:
        vals = re.findall(p, text, flags=flags)
        if vals:
            # If groups produce tuples, use first group.
            if isinstance(vals[-1], tuple):
                return maybe_float(vals[-1][0])
            return maybe_float(vals[-1])
    return None


def parse_accuracy(text: str) -> Optional[float]:
    # Prefer final Dassl test line: "* accuracy: 89.50%".
    acc = last_float([
        r"\*\s*accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"\btest\s+accuracy\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"\baccuracy\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"\bAccuracy\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    ], text)
    return acc


def parse_macro_f1(text: str) -> Optional[float]:
    return last_float([
        r"\*\s*macro[_\s-]?f1\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"\bmacro[_\s-]?f1\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"\bmacro_f1\s*,\s*([0-9]+(?:\.[0-9]+)?)",
    ], text)


def parse_named_metric(text: str, key: str) -> Optional[float]:
    # Handles: "meff: 11.42", "meff=11.42", "meff 11.42", CSV-like "meff,11.42".
    return last_float([
        rf"(?:^|[\s,;|]){re.escape(key)}\s*[:=]\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        rf"(?:^|[\s,;|]){re.escape(key)}\s+([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        rf"(?:^|[\s,;|]){re.escape(key)}\s*,\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    ], text)


def parse_best_val_acc(text: str) -> Optional[float]:
    return last_float([
        r"\bbest[_\s-]?val[_\s-]?acc\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"\bbest[_\s-]?accuracy\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
    ], text)


def parse_cfg_shots(parts: List[str]) -> Tuple[str, Optional[int]]:
    # e.g. rn50_ep50_16shots -> cfg=rn50_ep50, shots=16
    for part in parts:
        m = re.search(r"(.+)_([0-9]+)shots$", part)
        if m:
            return m.group(1), int(m.group(2))
    return "", None


def parse_setting_from_path(path_str: str) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[int]]:
    # nctx16_cscFalse_ctpend, optionally with seed suffix.
    n_ctx = None
    csc = None
    ctp = None
    seed = None

    m = re.search(r"nctx([0-9]+)", path_str)
    if m:
        n_ctx = int(m.group(1))

    m = re.search(r"csc(True|False)", path_str)
    if m:
        csc = m.group(1)

    m = re.search(r"ctp([A-Za-z0-9_\-]+)", path_str)
    if m:
        ctp = m.group(1)
        # Trim likely suffixes in flat PriorRes dirs.
        for marker in ["_seed", "_nob", "_b0p2", "_b", "_fresh", "_smoke"]:
            if marker in ctp:
                ctp = ctp.split(marker)[0]

    m = re.search(r"seed[_\-]?([0-9]+)", path_str)
    if m:
        seed = int(m.group(1))

    return n_ctx, csc, ctp, seed


def infer_variant(trainer: str, path_str: str) -> str:
    p = path_str.lower()
    if trainer.lower() == "coop":
        return "baseline"
    if "nob" in p or "no_b" in p:
        return "nob"
    if "b0p2" in p or "b0.2" in p:
        return "b0p2"
    if re.search(r"(^|[_/\-])b($|[_/\-])", p):
        return "b"
    if "aonly" in p or "a_only" in p:
        return "aonly"
    if "bonly" in p or "b_only" in p:
        return "bonly"
    if "datasetonly" in p or "dataset_only" in p:
        return "datasetonly"
    if "priorres" in p:
        return "priorres"
    return "unknown"


def infer_status(text: str, accuracy: Optional[float]) -> str:
    lower = text.lower()
    if accuracy is not None:
        return "complete"
    if "traceback" in lower or "error" in lower or "segmentation fault" in lower:
        return "failed"
    return "incomplete"


def parse_log(log_path: Path, output_root: Path) -> Optional[RunRow]:
    try:
        rel = log_path.relative_to(output_root)
    except ValueError:
        rel = log_path

    parts = list(rel.parts)
    if len(parts) < 3:
        return None

    dataset_dir = parts[0]
    dataset = norm_dataset_name(dataset_dir)

    trainer = "unknown"
    for part in parts:
        if part in {"CoOp", "CoOpPriorRes"}:
            trainer = part
            break

    cfg, shots = parse_cfg_shots(parts)
    path_str = str(rel)
    n_ctx, csc, ctp, seed = parse_setting_from_path(path_str)
    variant = infer_variant(trainer, path_str)

    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        text = ""

    accuracy = parse_accuracy(text)
    macro_f1 = parse_macro_f1(text)
    metrics = {k: parse_named_metric(text, k) for k in ["meff", "keff", "a0_mean", "b0_mean", "loss_b", "lambda_t"]}
    best_val_acc = parse_best_val_acc(text)
    status = infer_status(text, accuracy)
    mtime = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    return RunRow(
        dataset=dataset,
        dataset_dir=dataset_dir,
        trainer=trainer,
        variant=variant,
        cfg=cfg,
        shots=shots,
        n_ctx=n_ctx,
        csc=csc,
        ctp=ctp,
        seed=seed,
        status=status,
        accuracy=accuracy,
        macro_f1=macro_f1,
        meff=metrics["meff"],
        keff=metrics["keff"],
        a0_mean=metrics["a0_mean"],
        b0_mean=metrics["b0_mean"],
        loss_b=metrics["loss_b"],
        lambda_t=metrics["lambda_t"],
        best_val_acc=best_val_acc,
        log_mtime=mtime,
        log_path=str(log_path),
        run_dir=str(log_path.parent),
    )


def should_exclude(path: Path, include_smoke: bool, exclude_keywords: List[str]) -> bool:
    if include_smoke:
        return False
    p = str(path).lower()
    return any(k.lower() in p for k in exclude_keywords)


def scan_logs(output_root: Path, include_smoke: bool, exclude_keywords: List[str]) -> List[RunRow]:
    rows: List[RunRow] = []
    for log_path in output_root.rglob("log.txt"):
        if should_exclude(log_path, include_smoke, exclude_keywords):
            continue
        row = parse_log(log_path, output_root)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r.dataset, r.trainer, r.variant, r.shots or -1, r.n_ctx or -1, r.seed or -1, r.log_path))
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float], int, Optional[float], Optional[float]]:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None, None, 0, None, None
    mean = stats.mean(vals)
    std = stats.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std, len(vals), min(vals), max(vals)


def fmt_float(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def summarize_by_setting(rows: List[RunRow], expected_seeds: List[int]) -> List[Dict[str, object]]:
    groups: Dict[Tuple, List[RunRow]] = {}
    for r in rows:
        key = (r.dataset, r.trainer, r.variant, r.cfg, r.shots, r.n_ctx, r.csc, r.ctp)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple("" if x is None else str(x) for x in kv[0])):
        dataset, trainer, variant, cfg, shots, n_ctx, csc, ctp = key
        completed = [r for r in group if r.status == "complete" and r.accuracy is not None]
        acc_mean, acc_std, n, acc_min, acc_max = mean_std([r.accuracy for r in completed])
        f1_mean, f1_std, _, _, _ = mean_std([r.macro_f1 for r in completed])
        meff_mean, meff_std, _, _, _ = mean_std([r.meff for r in completed])
        keff_mean, keff_std, _, _, _ = mean_std([r.keff for r in completed])
        seeds = sorted({r.seed for r in completed if r.seed is not None})
        missing = [s for s in expected_seeds if s not in seeds]
        out.append({
            "dataset": dataset,
            "trainer": trainer,
            "variant": variant,
            "cfg": cfg,
            "shots": shots if shots is not None else "",
            "n_ctx": n_ctx if n_ctx is not None else "",
            "csc": csc or "",
            "ctp": ctp or "",
            "n_complete": n,
            "expected_n": len(expected_seeds),
            "completed_seeds": " ".join(map(str, seeds)),
            "missing_seeds": " ".join(map(str, missing)),
            "accuracy_mean": fmt_float(acc_mean),
            "accuracy_std": fmt_float(acc_std),
            "accuracy_min": fmt_float(acc_min),
            "accuracy_max": fmt_float(acc_max),
            "macro_f1_mean": fmt_float(f1_mean),
            "macro_f1_std": fmt_float(f1_std),
            "meff_mean": fmt_float(meff_mean),
            "meff_std": fmt_float(meff_std),
            "keff_mean": fmt_float(keff_mean),
            "keff_std": fmt_float(keff_std),
        })
    return out


def get_setting_summary(summary_rows: List[Dict[str, object]], dataset: str, trainer: str, variant: str,
                        shots: Optional[int], n_ctx: Optional[int]) -> Optional[Dict[str, object]]:
    candidates = []
    for r in summary_rows:
        if r["dataset"] != dataset or r["trainer"] != trainer or r["variant"] != variant:
            continue
        if shots is not None and str(r["shots"]) != str(shots):
            continue
        if n_ctx is not None and str(r["n_ctx"]) != str(n_ctx):
            continue
        if r.get("accuracy_mean", "") == "":
            continue
        candidates.append(r)
    if not candidates:
        return None
    # Prefer larger n_complete.
    return sorted(candidates, key=lambda x: (int(x["n_complete"]), float(x["accuracy_mean"])), reverse=True)[0]


def coop_best_rows(summary_rows: List[Dict[str, object]], datasets: List[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for d in datasets:
        for scope, only_k16 in [("best_k16", True), ("best_all_available", False)]:
            candidates = []
            for r in summary_rows:
                if r["dataset"] != d or r["trainer"] != "CoOp" or r["variant"] != "baseline":
                    continue
                if r.get("accuracy_mean", "") == "":
                    continue
                if only_k16 and str(r["shots"]) != "16":
                    continue
                candidates.append(r)
            if not candidates:
                out.append({
                    "dataset": d,
                    "scope": scope,
                    "shots": "",
                    "n_ctx": "",
                    "accuracy_mean": "",
                    "accuracy_std": "",
                    "n_complete": 0,
                    "completed_seeds": "",
                    "missing_seeds": "",
                })
                continue
            best = max(candidates, key=lambda x: float(x["accuracy_mean"]))
            out.append({
                "dataset": d,
                "scope": scope,
                "shots": best["shots"],
                "n_ctx": best["n_ctx"],
                "accuracy_mean": best["accuracy_mean"],
                "accuracy_std": best["accuracy_std"],
                "n_complete": best["n_complete"],
                "completed_seeds": best["completed_seeds"],
                "missing_seeds": best["missing_seeds"],
            })
    return out


def as_float_cell(row: Optional[Dict[str, object]], key: str = "accuracy_mean") -> Optional[float]:
    if row is None:
        return None
    val = row.get(key, "")
    if val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def main_table(summary_rows: List[Dict[str, object]], best_rows: List[Dict[str, object]], datasets: List[str]) -> List[Dict[str, object]]:
    best_lookup = {(r["dataset"], r["scope"]): r for r in best_rows}
    out: List[Dict[str, object]] = []
    for d in datasets:
        coop_m16 = get_setting_summary(summary_rows, d, "CoOp", "baseline", 16, 16)
        nob = get_setting_summary(summary_rows, d, "CoOpPriorRes", "nob", 16, 16)
        b0p2 = get_setting_summary(summary_rows, d, "CoOpPriorRes", "b0p2", 16, 16)
        coop_best_k16 = best_lookup.get((d, "best_k16"))
        coop_best_all = best_lookup.get((d, "best_all_available"))

        coop_m16_acc = as_float_cell(coop_m16)
        coop_best_k16_acc = as_float_cell(coop_best_k16)
        coop_best_all_acc = as_float_cell(coop_best_all)
        nob_acc = as_float_cell(nob)
        b_acc = as_float_cell(b0p2)

        out.append({
            "dataset": d,
            "CoOp_m16_k16_mean": fmt_float(coop_m16_acc),
            "CoOp_m16_k16_std": "" if coop_m16 is None else coop_m16.get("accuracy_std", ""),
            "CoOp_m16_k16_n": "" if coop_m16 is None else coop_m16.get("n_complete", ""),
            "CoOp_best_k16_m": "" if coop_best_k16 is None else coop_best_k16.get("n_ctx", ""),
            "CoOp_best_k16_mean": fmt_float(coop_best_k16_acc),
            "CoOp_best_k16_std": "" if coop_best_k16 is None else coop_best_k16.get("accuracy_std", ""),
            "CoOp_best_all_available_k": "" if coop_best_all is None else coop_best_all.get("shots", ""),
            "CoOp_best_all_available_m": "" if coop_best_all is None else coop_best_all.get("n_ctx", ""),
            "CoOp_best_all_available_mean": fmt_float(coop_best_all_acc),
            "CoOp_best_all_available_std": "" if coop_best_all is None else coop_best_all.get("accuracy_std", ""),
            "PriorRes_no_b_m16_k16_mean": fmt_float(nob_acc),
            "PriorRes_no_b_m16_k16_std": "" if nob is None else nob.get("accuracy_std", ""),
            "PriorRes_no_b_n": "" if nob is None else nob.get("n_complete", ""),
            "PriorRes_b0p2_m16_k16_mean": fmt_float(b_acc),
            "PriorRes_b0p2_m16_k16_std": "" if b0p2 is None else b0p2.get("accuracy_std", ""),
            "PriorRes_b0p2_n": "" if b0p2 is None else b0p2.get("n_complete", ""),
            "delta_no_b_vs_CoOp_m16": fmt_float(None if nob_acc is None or coop_m16_acc is None else nob_acc - coop_m16_acc),
            "delta_no_b_vs_CoOp_best_k16": fmt_float(None if nob_acc is None or coop_best_k16_acc is None else nob_acc - coop_best_k16_acc),
            "delta_b0p2_vs_no_b": fmt_float(None if b_acc is None or nob_acc is None else b_acc - nob_acc),
            "status": "ready" if nob_acc is not None and coop_best_k16_acc is not None else "partial_or_pending",
        })
    return out


def missing_expected(rows: List[RunRow], datasets: List[str], expected_seeds: List[int], expected_m: List[int]) -> List[Dict[str, object]]:
    seen = set()
    for r in rows:
        if r.status == "complete" and r.seed is not None:
            seen.add((r.dataset, r.trainer, r.variant, r.shots, r.n_ctx, r.seed))

    out: List[Dict[str, object]] = []
    for d in datasets:
        # Expected PriorRes mainline and b-branch at k=m=16.
        for variant in ["nob", "b0p2"]:
            for seed in expected_seeds:
                key = (d, "CoOpPriorRes", variant, 16, 16, seed)
                if key not in seen:
                    out.append({
                        "dataset": d,
                        "trainer": "CoOpPriorRes",
                        "variant": variant,
                        "shots": 16,
                        "n_ctx": 16,
                        "seed": seed,
                        "expected_reason": "main PriorRes k=16,m=16 seed coverage",
                    })
        # Expected CoOp k=16 m-grid.
        for m in expected_m:
            for seed in expected_seeds:
                key = (d, "CoOp", "baseline", 16, m, seed)
                if key not in seen:
                    out.append({
                        "dataset": d,
                        "trainer": "CoOp",
                        "variant": "baseline",
                        "shots": 16,
                        "n_ctx": m,
                        "seed": seed,
                        "expected_reason": "CoOp k=16 m-grid seed coverage",
                    })
    return out


def coverage_by_dataset(rows: List[RunRow], datasets: List[str]) -> List[Dict[str, object]]:
    out = []
    for d in datasets:
        dr = [r for r in rows if r.dataset == d]
        coop = [r for r in dr if r.trainer == "CoOp" and r.status == "complete"]
        nob = [r for r in dr if r.trainer == "CoOpPriorRes" and r.variant == "nob" and r.status == "complete"]
        b = [r for r in dr if r.trainer == "CoOpPriorRes" and r.variant == "b0p2" and r.status == "complete"]
        coop_k_values = sorted({r.shots for r in coop if r.shots is not None})
        coop_m_values_k16 = sorted({r.n_ctx for r in coop if r.shots == 16 and r.n_ctx is not None})
        out.append({
            "dataset": d,
            "total_logs": len(dr),
            "complete_logs": len([r for r in dr if r.status == "complete"]),
            "coop_complete_runs": len(coop),
            "coop_available_k_values": " ".join(map(str, coop_k_values)),
            "coop_available_m_values_at_k16": " ".join(map(str, coop_m_values_k16)),
            "priorres_nob_complete_runs": len(nob),
            "priorres_b0p2_complete_runs": len(b),
        })
    return out


def write_readme(csv_dir: Path, args: argparse.Namespace, n_logs: int) -> None:
    text = f"""# Experiment CSV summary

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Output root scanned:

```text
{args.output_root}
```

Datasets expected:

```text
{', '.join(args.datasets)}
```

Number of parsed log files: {n_logs}

Files:

- `run_level.csv`: one row per parsed `log.txt`.
- `summary_by_setting.csv`: mean/std grouped by dataset/trainer/variant/k/m.
- `main_table_k16.csv`: main paper-style table for k=16 comparisons.
- `coop_best_available.csv`: CoOp best setting using available logs, separated into best at k=16 and best over all available k.
- `missing_expected.csv`: missing expected seeds for CoOp k=16 m-grid and PriorRes k=m=16.
- `coverage_by_dataset.csv`: compact coverage report.

Notes:

- `smoke`, `cachebuild`, and `debug` runs are excluded by default.
- The script combines `caltech-101` and `caltech101` under normalized dataset name `caltech101`.
- If ImageNet results are added later under `output/imagenet`, rerun this script; no code changes are needed.
"""
    (csv_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coop-root", default="/workspace/meta_prompt_1/third_party/CoOp_clean")
    parser.add_argument("--output-root", default=None, help="Default: <coop-root>/output")
    parser.add_argument("--csv-dir", default="/workspace/meta_prompt_1/outputs/summary_csv")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--expected-m-grid", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12, 14, 16])
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--exclude-keywords", nargs="+", default=EXCLUDE_KEYWORDS_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coop_root = Path(args.coop_root)
    output_root = Path(args.output_root) if args.output_root else coop_root / "output"
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    args.output_root = str(output_root)
    args.datasets = [norm_dataset_name(d) for d in args.datasets]
    # De-duplicate while preserving order.
    seen_ds = set()
    datasets = []
    for d in args.datasets:
        if d not in seen_ds:
            datasets.append(d)
            seen_ds.add(d)
    args.datasets = datasets

    rows = scan_logs(output_root, include_smoke=args.include_smoke, exclude_keywords=args.exclude_keywords)
    # Keep all scanned rows, even datasets not in expected list, but expected tables use args.datasets.

    run_dicts = [asdict(r) for r in rows]
    run_fieldnames = list(asdict(RunRow("", "", "", "", "", None, None, None, None, None, "", None, None, None, None, None, None, None, None, None, "", "", "")).keys())
    write_csv(csv_dir / "run_level.csv", run_dicts, run_fieldnames)

    summary = summarize_by_setting(rows, args.expected_seeds)
    write_csv(csv_dir / "summary_by_setting.csv", summary)

    best = coop_best_rows(summary, datasets)
    write_csv(csv_dir / "coop_best_available.csv", best)

    main = main_table(summary, best, datasets)
    write_csv(csv_dir / "main_table_k16.csv", main)

    missing = missing_expected(rows, datasets, args.expected_seeds, args.expected_m_grid)
    write_csv(csv_dir / "missing_expected.csv", missing)

    coverage = coverage_by_dataset(rows, datasets)
    write_csv(csv_dir / "coverage_by_dataset.csv", coverage)

    write_readme(csv_dir, args, len(rows))

    print(f"[OK] Parsed {len(rows)} log files")
    print(f"[OK] CSV folder: {csv_dir}")
    print("Generated:")
    for name in [
        "run_level.csv",
        "summary_by_setting.csv",
        "main_table_k16.csv",
        "coop_best_available.csv",
        "missing_expected.csv",
        "coverage_by_dataset.csv",
        "README.md",
    ]:
        print(f"  - {csv_dir / name}")


if __name__ == "__main__":
    main()
