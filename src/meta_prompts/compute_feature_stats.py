#!/usr/bin/env python3
"""Aggregate mean/std over transformed task features from multiple JSON files.

Input JSON files are expected to be outputs produced by task_feature_extractor.py.
This is useful once you move from the single-dataset first-stage validation to the
multi-dataset meta-training stage, because your document defines:
    f_hat = (f_tilde - mu) / (sigma + eps)
where mu and sigma are estimated from training tasks.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List

import numpy as np


FEATURE_KEYS = ["log_C", "Ssemantic", "log_Dintra", "log_Dinter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSON files or glob patterns produced by task_feature_extractor.py",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def resolve_inputs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if matched:
            files.extend(matched)
        elif Path(pattern).is_file():
            files.append(pattern)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No input feature JSON files found")
    return files


if __name__ == "__main__":
    args = parse_args()
    files = resolve_inputs(args.inputs)

    rows = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        transformed = payload["transformed_features"]
        rows.append([float(transformed[k]) for k in FEATURE_KEYS])

    arr = np.asarray(rows, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)

    output = {
        "feature_order": FEATURE_KEYS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_tasks": int(arr.shape[0]),
        "eps": float(args.eps),
        "source_files": files,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))
