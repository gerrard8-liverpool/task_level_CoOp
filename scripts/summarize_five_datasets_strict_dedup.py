#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

ACC_RE = re.compile(r"\* accuracy:\s*([0-9.]+)%")
F1_RE = re.compile(r"\* macro_f1:\s*([0-9.]+)%")
USE_B_RE = re.compile(r"^\s*USE_B:\s*(True|False)\s*$", re.MULTILINE)
B_WEIGHT_RE = re.compile(r"^\s*B_LOSS_WEIGHT:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
META_LR_RE = re.compile(r"^\s*META_LR_RATIO:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)

PATH_RE = re.compile(
    r"""
    output/
    (?P<dataset>[^/]+)/
    (?P<method>CoOp|CoOpPriorRes)/
    rn50_ep50_(?P<shots>\d+)shots/
    nctx(?P<nctx>\d+)_cscFalse_ctpend
    (?:
        /seed(?P<seed_dir>\d+)
      |
        _seed(?P<seed_inline>\d+)(?P<suffix_inline>[^/]*)
    )
    /log\.txt$
    """,
    re.VERBOSE,
)

SEED_FALLBACK_RE = re.compile(r"seed(\d+)")
B_TAG_RE = re.compile(r"(?:^|_)b([0-9]+p[0-9]+)(?:_|$)")
NOB_TAG_RE = re.compile(r"(?:^|_)nob(?:_|$)")
FRESH_TAG_RE = re.compile(r"(?:^|_)fresh(?:_|$)")
SMOKE_TAG_RE = re.compile(r"(?:^|_)smoke\d*(?:_|$)")
CACHEBUILD_TAG_RE = re.compile(r"(?:^|_)cachebuild(?:_|$)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize experiment results with strict dedup rules.")
    p.add_argument("--root", type=str, default="/workspace/meta_prompt_1")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["oxford_pets", "eurosat", "dtd", "food101", "oxford_flowers"],
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--override-json",
        type=str,
        default=None,
        help="JSON mapping from formal key to selected log_path.",
    )
    return p.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def infer_seed_from_path(path: Path) -> Optional[int]:
    m = SEED_FALLBACK_RE.search(str(path))
    return int(m.group(1)) if m else None


def infer_setting_tag(path: Path, text: str, method: str) -> str:
    if method == "CoOp":
        return "baseline"

    m = USE_B_RE.search(text)
    if m:
        use_b = m.group(1) == "True"
        if not use_b:
            return "no_b"
        bw = B_WEIGHT_RE.search(text)
        if bw:
            try:
                return f"b_w{float(bw.group(1))}"
            except Exception:
                return f"b_w{bw.group(1)}"

    s = str(path)
    if NOB_TAG_RE.search(s):
        return "no_b"
    pm = B_TAG_RE.search(s)
    if pm:
        return f"b_w{pm.group(1).replace('p', '.')}"
    return "unknown"


def parse_flags_from_path(path: Path) -> Dict[str, bool]:
    s = str(path)
    return {
        "is_fresh": bool(FRESH_TAG_RE.search(s)),
        "is_smoke": bool(SMOKE_TAG_RE.search(s)),
        "is_cachebuild": bool(CACHEBUILD_TAG_RE.search(s)),
    }


def formal_key(row: Dict) -> str:
    return f'{row["dataset"]}|{row["method"]}|{row["setting_tag"]}|k{row["shots"]}|m{row["nctx"]}|seed{row["seed"]}'


def parse_log(log_path: Path) -> Optional[Dict]:
    text = read_text(log_path)
    accs = ACC_RE.findall(text)
    if not accs:
        return None

    f1s = F1_RE.findall(text)
    sp = str(log_path).replace("\\", "/")
    pm = PATH_RE.search(sp)
    if not pm:
        return None

    seed = pm.group("seed_dir") or pm.group("seed_inline")
    seed_val = int(seed) if seed is not None else infer_seed_from_path(log_path)
    if seed_val is None:
        return None

    bw = B_WEIGHT_RE.search(text)
    mlr = META_LR_RE.search(text)
    flags = parse_flags_from_path(log_path)

    row = {
        "dataset": pm.group("dataset"),
        "method": pm.group("method"),
        "shots": int(pm.group("shots")),
        "nctx": int(pm.group("nctx")),
        "seed": seed_val,
        "accuracy": float(accs[-1]),
        "macro_f1": float(f1s[-1]) if f1s else None,
        "use_b": USE_B_RE.search(text).group(1) if USE_B_RE.search(text) else "",
        "b_loss_weight": safe_float(bw.group(1)) if bw else None,
        "meta_lr_ratio": safe_float(mlr.group(1)) if mlr else None,
        "setting_tag": infer_setting_tag(log_path, text, pm.group("method")),
        "log_path": str(log_path),
        "is_fresh": flags["is_fresh"],
        "is_smoke": flags["is_smoke"],
        "is_cachebuild": flags["is_cachebuild"],
        "status": "",
        "exclude_reason": "",
    }
    row["formal_key"] = formal_key(row)
    return row


def collect_runs(root: Path, datasets: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    out_root = root / "third_party" / "CoOp_clean" / "output"
    for dataset in datasets:
        ds_root = out_root / dataset
        if not ds_root.exists():
            continue
        for log_path in ds_root.rglob("log.txt"):
            row = parse_log(log_path)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["method"], r["setting_tag"], r["shots"], r["nctx"], r["seed"], r["log_path"]))
    return rows


def load_overrides(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"override json not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("override json must be a dict")
    return {str(k): str(v) for k, v in obj.items()}


def dedup_rows(rows: List[Dict], overrides: Dict[str, str]) -> Tuple[List[Dict], List[Dict]]:
    kept: List[Dict] = []
    review: List[Dict] = []
    buckets: Dict[str, List[Dict]] = defaultdict(list)

    for r in rows:
        if r["is_smoke"]:
            r["status"] = "excluded"
            r["exclude_reason"] = "smoke"
            continue
        if r["is_cachebuild"]:
            r["status"] = "excluded"
            r["exclude_reason"] = "cachebuild"
            continue
        buckets[r["formal_key"]].append(r)

    for key, vals in buckets.items():
        vals = sorted(vals, key=lambda x: (x["is_fresh"], x["log_path"]))

        if key in overrides:
            chosen = overrides[key]
            matched = [v for v in vals if v["log_path"] == chosen]
            if len(matched) != 1:
                for v in vals:
                    v["status"] = "excluded"
                    v["exclude_reason"] = f"override_mismatch:{key}"
                review.append({
                    "formal_key": key,
                    "issue": "override_mismatch",
                    "override": chosen,
                    "candidates": " || ".join(v["log_path"] for v in vals),
                })
                continue

            keep = matched[0]
            keep["status"] = "kept"
            kept.append(keep)
            for v in vals:
                if v is keep:
                    continue
                v["status"] = "excluded"
                v["exclude_reason"] = f"duplicate_removed_by_override:{key}"
            continue

        if len(vals) == 1:
            vals[0]["status"] = "kept"
            kept.append(vals[0])
            continue

        nonfresh = [v for v in vals if not v["is_fresh"]]

        if len(nonfresh) == 1:
            keep = nonfresh[0]
            keep["status"] = "kept"
            kept.append(keep)
            for v in vals:
                if v is keep:
                    continue
                v["status"] = "excluded"
                if v["is_fresh"]:
                    v["exclude_reason"] = f"fresh_dropped_because_formal_exists:{key}"
                else:
                    v["exclude_reason"] = f"duplicate_formal_removed:{key}"
            continue

        issue = "duplicate_requires_manual_confirmation"
        for v in vals:
            v["status"] = "excluded"
            v["exclude_reason"] = issue
        review.append({
            "formal_key": key,
            "issue": issue,
            "override": "",
            "candidates": " || ".join(v["log_path"] for v in vals),
        })

    kept.sort(key=lambda r: (r["dataset"], r["method"], r["setting_tag"], r["shots"], r["nctx"], r["seed"], r["log_path"]))
    return kept, review


def group_rows(rows: List[Dict]) -> List[Dict]:
    buckets: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["method"], r["setting_tag"], r["shots"], r["nctx"])
        buckets[key].append(r)

    grouped = []
    for key, vals in sorted(buckets.items()):
        accs = [v["accuracy"] for v in vals if v["accuracy"] is not None]
        f1s = [v["macro_f1"] for v in vals if v["macro_f1"] is not None]
        grouped.append({
            "dataset": key[0],
            "method": key[1],
            "setting_tag": key[2],
            "shots": key[3],
            "nctx": key[4],
            "num_runs": len(vals),
            "seeds": ",".join(str(v["seed"]) for v in sorted(vals, key=lambda x: x["seed"])),
            "acc_mean": round(mean(accs), 4) if accs else None,
            "acc_std": round(pstdev(accs), 4) if len(accs) > 1 else (0.0 if accs else None),
            "f1_mean": round(mean(f1s), 4) if f1s else None,
            "f1_std": round(pstdev(f1s), 4) if len(f1s) > 1 else (0.0 if f1s else None),
            "best_acc": round(max(accs), 4) if accs else None,
            "worst_acc": round(min(accs), 4) if accs else None,
        })
    return grouped


def best_by_method(grouped: List[Dict]) -> List[Dict]:
    best = {}
    for row in grouped:
        key = (row["dataset"], row["method"], row["setting_tag"])
        if row["acc_mean"] is None:
            continue
        if key not in best or row["acc_mean"] > best[key]["acc_mean"]:
            best[key] = row
    return [best[k] for k in sorted(best.keys())]


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: List[Dict], cols: List[str]) -> str:
    if not rows:
        return "_No rows found._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "summary_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(root, args.datasets)
    overrides = load_overrides(args.override_json)
    kept, review = dedup_rows(rows, overrides)
    grouped = group_rows(kept)
    best = best_by_method(grouped)

    raw_csv = out_dir / "five_datasets_raw_runs_with_status.csv"
    kept_csv = out_dir / "five_datasets_deduped_formal_runs.csv"
    review_csv = out_dir / "five_datasets_duplicate_review.csv"
    grouped_csv = out_dir / "five_datasets_grouped_summary.csv"
    best_csv = out_dir / "five_datasets_best_by_method.csv"
    md_path = out_dir / "five_datasets_summary.md"

    write_csv(raw_csv, rows)
    write_csv(kept_csv, kept)
    write_csv(review_csv, review)
    write_csv(grouped_csv, grouped)
    write_csv(best_csv, best)

    by_dataset = defaultdict(list)
    for row in grouped:
        by_dataset[row["dataset"]].append(row)

    md = ["# Five-dataset experiment summary (strict dedup)\n"]
    md.append("## Best setting per dataset / method / setting_tag\n")
    md.append(md_table(best, ["dataset", "method", "setting_tag", "shots", "nctx", "acc_mean", "acc_std", "f1_mean", "f1_std", "seeds"]))
    if review:
        md.append("## Duplicate review needed\n")
        md.append(md_table(review, ["formal_key", "issue", "override", "candidates"]))
    for ds in args.datasets:
        md.append(f"## {ds}\n")
        md.append(md_table(by_dataset.get(ds, []), ["method", "setting_tag", "shots", "nctx", "num_runs", "seeds", "acc_mean", "acc_std", "f1_mean", "f1_std", "best_acc", "worst_acc"]))
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] raw runs with status -> {raw_csv}")
    print(f"[OK] deduped formal runs -> {kept_csv}")
    print(f"[OK] duplicate review    -> {review_csv}")
    print(f"[OK] grouped summary     -> {grouped_csv}")
    print(f"[OK] best by method      -> {best_csv}")
    print(f"[OK] markdown summary    -> {md_path}")


if __name__ == "__main__":
    main()
