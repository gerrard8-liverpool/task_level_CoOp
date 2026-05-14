import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ACC_RE = re.compile(r"accuracy:\s*([0-9.]+)", re.IGNORECASE)


def read_acc(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    vals = [float(x) for x in ACC_RE.findall(text)]
    if not vals:
        return None
    return vals[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", default="third_party/CoOp_clean/output/xd/test")
    parser.add_argument("--cfg", default="rn50_ep50")
    parser.add_argument("--coop-tag", default="nctx16_cscFalse_ctpend")
    parser.add_argument("--safe-tag", default="nctx16_cscFalse_ctpend_safe_noalt")
    parser.add_argument("--out-csv", default="outputs/analysis/source_dependency_delta_pairs.csv")
    parser.add_argument("--out-summary", default="outputs/analysis/source_dependency_delta_summary.csv")
    parser.add_argument("--out-fig", default="outputs/figures/fig4_source_dependency_delta.pdf")
    args = parser.parse_args()

    root = Path(args.test_root)

    safe_logs = sorted(
        root.glob(
            f"*/source_*/shots_16/CoOpPriorRes/{args.cfg}/{args.safe_tag}/seed*/log.txt"
        )
    )

    rows = []
    for safe_log in safe_logs:
        parts = safe_log.parts

        # .../test/<target>/source_<source>/shots_16/CoOpPriorRes/...
        try:
            idx = parts.index("test")
            target = parts[idx + 1]
        except Exception:
            continue

        source = None
        seed = None
        for p in parts:
            if p.startswith("source_"):
                source = p.replace("source_", "")
            if p.startswith("seed"):
                try:
                    seed = int(p.replace("seed", ""))
                except Exception:
                    pass

        if source is None or seed is None:
            continue

        coop_log = (
            root
            / target
            / f"source_{source}"
            / "shots_16"
            / "CoOp"
            / args.cfg
            / args.coop_tag
            / f"seed{seed}"
            / "log.txt"
        )

        safe_acc = read_acc(safe_log)
        coop_acc = read_acc(coop_log)

        if safe_acc is None or coop_acc is None:
            continue

        rows.append(
            {
                "source": source,
                "target": target,
                "seed": seed,
                "coop_acc": coop_acc,
                "safe_acc": safe_acc,
                "delta": safe_acc - coop_acc,
                "safe_log": str(safe_log),
                "coop_log": str(coop_log),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("[ERROR] No matched Safe/CoOp test logs found.")
        return

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("source")
        .agg(
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            num_pairs=("delta", "count"),
            mean_safe=("safe_acc", "mean"),
            mean_coop=("coop_acc", "mean"),
        )
        .reset_index()
        .sort_values("mean_delta", ascending=False)
    )

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    print("Saved pair CSV:", out_csv)
    print("Saved summary CSV:", out_summary)
    print()
    print("Source summary:")
    print(summary.to_string(index=False))

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    sources = list(summary["source"])
    means = list(summary["mean_delta"])
    stds = [0.0 if pd.isna(x) else float(x) for x in summary["std_delta"]]
    counts = list(summary["num_pairs"])

    x = range(len(sources))

    fig, ax = plt.subplots(figsize=(max(8, len(sources) * 1.2), 4.5))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(sources, counts)], rotation=30, ha="right")
    ax.set_ylabel("Mean DG delta: Safe - CoOp")
    ax.set_title("Source-dependent Cross-dataset Transfer Gain")
    ax.grid(True, axis="y", alpha=0.3)

    for i, m in enumerate(means):
        va = "bottom" if m >= 0 else "top"
        offset = 0.3 if m >= 0 else -0.3
        ax.text(i, m + offset, f"{m:+.2f}", ha="center", va=va, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_fig)
    fig.savefig(out_fig.with_suffix(".png"), dpi=300)

    print()
    print("Saved figure:", out_fig)
    print("Saved figure:", out_fig.with_suffix(".png"))


if __name__ == "__main__":
    main()
