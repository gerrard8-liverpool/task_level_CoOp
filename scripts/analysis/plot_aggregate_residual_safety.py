import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def parse_source_seed(path: Path):
    parts = path.parts
    source = None
    seed = None

    for p in parts:
        if p.startswith("source_"):
            source = p.replace("source_", "")
        if p.startswith("seed"):
            try:
                seed = int(p.replace("seed", ""))
            except Exception:
                seed = None

    return source, seed


def summarize_csv(path: Path, mode: str):
    df = pd.read_csv(path)

    def get(col, fn="max"):
        if col not in df.columns:
            return None
        if fn == "max":
            return float(df[col].max())
        if fn == "last":
            return float(df[col].iloc[-1])
        if fn == "mean":
            return float(df[col].mean())
        return None

    source, seed = parse_source_seed(path)

    return {
        "source": source,
        "seed": seed,
        "mode": mode,
        "csv": str(path),
        "max_relative_residual_norm": get("relative_residual_norm", "max"),
        "last_relative_residual_norm": get("relative_residual_norm", "last"),
        "max_safe_relative_residual_norm": get("safe_relative_residual_norm", "max"),
        "last_safe_relative_residual_norm": get("safe_relative_residual_norm", "last"),
        "max_legacy_relative_residual_norm": get("legacy_relative_residual_norm", "max"),
        "last_legacy_relative_residual_norm": get("legacy_relative_residual_norm", "last"),
        "max_delta_a_norm": get("delta_a_norm", "max"),
        "last_delta_a_norm": get("delta_a_norm", "last"),
        "last_a0_mean": get("a0_mean", "last"),
        "last_a_mean": get("a_mean", "last"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="third_party/CoOp_clean/output/xd/train")
    parser.add_argument("--cfg", default="rn50_ep50")
    parser.add_argument("--safe-tag", default="nctx16_cscFalse_ctpend_safe_noalt")
    parser.add_argument("--legacy-tag", default="nctx16_cscFalse_ctpend_legacy_noalt")
    parser.add_argument("--out-csv", default="outputs/analysis/aggregate_residual_safety.csv")
    parser.add_argument("--out-fig", default="outputs/figures/fig3_aggregate_residual_safety.pdf")
    args = parser.parse_args()

    root = Path(args.train_root)

    safe_paths = sorted(root.glob(f"source_*/shots_16/CoOpPriorRes/{args.cfg}/{args.safe_tag}/seed*/analysis_stats.csv"))
    legacy_paths = sorted(root.glob(f"source_*/shots_16/CoOpPriorRes/{args.cfg}/{args.legacy_tag}/seed*/analysis_stats.csv"))

    rows = []
    for p in safe_paths:
        rows.append(summarize_csv(p, "safe"))
    for p in legacy_paths:
        rows.append(summarize_csv(p, "legacy"))

    df = pd.DataFrame(rows)

    if df.empty:
        print("[ERROR] No analysis_stats.csv found.")
        return

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    safe_df = df[df["mode"] == "safe"].copy()
    legacy_df = df[df["mode"] == "legacy"].copy()

    merged = pd.merge(
        safe_df,
        legacy_df,
        on=["source", "seed"],
        suffixes=("_safe_run", "_legacy_run"),
        how="inner",
    )

    if merged.empty:
        print("[WARN] No matched safe/legacy pairs found.")
        print("Saved raw CSV:", out_csv)
        return

    # Use actual effective residual for each run.
    merged["safe_max_effective"] = merged["max_relative_residual_norm_safe_run"]
    merged["legacy_max_effective"] = merged["max_relative_residual_norm_legacy_run"]
    merged["legacy_over_safe"] = merged["legacy_max_effective"] / merged["safe_max_effective"].clip(lower=1e-12)

    merged = merged.sort_values(["source", "seed"]).reset_index(drop=True)

    pair_csv = out_csv.with_name(out_csv.stem + "_paired.csv")
    merged.to_csv(pair_csv, index=False)

    print("Saved raw CSV:", out_csv)
    print("Saved paired CSV:", pair_csv)
    print()
    print("Matched cases:")
    cols = ["source", "seed", "safe_max_effective", "legacy_max_effective", "legacy_over_safe"]
    print(merged[cols].to_string(index=False))

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"{s}\nseed{int(seed)}" for s, seed in zip(merged["source"], merged["seed"])]
    x = range(len(merged))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 0.9), 4.2))

    ax = axes[0]
    width = 0.38
    ax.bar([i - width/2 for i in x], merged["safe_max_effective"], width=width, label="Safe")
    ax.bar([i + width/2 for i in x], merged["legacy_max_effective"], width=width, label="Legacy")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Max relative prompt perturbation")
    ax.set_title("(a) Effective residual magnitude")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    ax.bar(list(x), merged["legacy_over_safe"])
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Legacy / Safe ratio")
    ax.set_title("(b) Suppression ratio")
    ax.grid(True, axis="y", alpha=0.3, which="both")

    fig.suptitle("Aggregate Residual Safety: Safe Suppresses Legacy Prompt Bias", y=1.03)
    fig.tight_layout()
    fig.savefig(out_fig)
    fig.savefig(out_fig.with_suffix(".png"), dpi=300)

    print()
    print("Saved figure:", out_fig)
    print("Saved figure:", out_fig.with_suffix(".png"))


if __name__ == "__main__":
    main()
