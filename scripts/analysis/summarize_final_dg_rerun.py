import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACC_RE = re.compile(r"accuracy:\s*([0-9.]+)", re.IGNORECASE)


METHODS = {
    "coop": {
        "trainer": "CoOp",
        "tag": "nctx16_cscFalse_ctpend",
    },
    "safe": {
        "trainer": "CoOpPriorRes",
        "tag": "nctx16_cscFalse_ctpend_safe_noalt",
    },
    "legacy": {
        "trainer": "CoOpPriorRes",
        "tag": "nctx16_cscFalse_ctpend_legacy_noalt",
    },
}


def read_acc(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    vals = [float(x) for x in ACC_RE.findall(text)]
    if not vals:
        return None
    return vals[-1]


def log_path(test_root: Path, source: str, target: str, seed: int, method: str, cfg: str):
    spec = METHODS[method]
    return (
        test_root
        / target
        / f"source_{source}"
        / "shots_16"
        / spec["trainer"]
        / cfg
        / spec["tag"]
        / f"seed{seed}"
        / "log.txt"
    )


def analysis_csv_path(train_root: Path, source: str, seed: int, method: str, cfg: str):
    if method == "coop":
        return None
    spec = METHODS[method]
    return (
        train_root
        / f"source_{source}"
        / "shots_16"
        / spec["trainer"]
        / cfg
        / spec["tag"]
        / f"seed{seed}"
        / "analysis_stats.csv"
    )


def save_bar_source(summary: pd.DataFrame, out_path: Path):
    sources = list(summary["source"])
    x = np.arange(len(sources))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(sources) * 1.3), 4.5))

    ax.bar(x - width / 2, summary["safe_delta_mean"], width, yerr=summary["safe_delta_std"], capsize=4, label="Safe - CoOp")
    ax.bar(x + width / 2, summary["legacy_delta_mean"], width, yerr=summary["legacy_delta_std"], capsize=4, label="Legacy - CoOp")

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={int(n)})" for s, n in zip(summary["source"], summary["num_pairs"])], rotation=25, ha="right")
    ax.set_ylabel("Mean DG delta")
    ax.set_title("Source-dependent Cross-dataset Transfer")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, v in enumerate(summary["safe_delta_mean"]):
        ax.text(i - width / 2, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    for i, v in enumerate(summary["legacy_delta_mean"]):
        ax.text(i + width / 2, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def save_heatmap(matrix: pd.DataFrame, title: str, out_path: Path):
    data = matrix.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.8), max(4, len(matrix.index) * 0.8)))

    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 1.0)

    im = ax.imshow(data, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def summarize_residual_safety(train_root: Path, sources, seeds, cfg: str, out_dir: Path):
    rows = []

    for source in sources:
        for seed in seeds:
            safe_csv = analysis_csv_path(train_root, source, seed, "safe", cfg)
            legacy_csv = analysis_csv_path(train_root, source, seed, "legacy", cfg)

            if safe_csv is None or legacy_csv is None:
                continue
            if not safe_csv.exists() or not legacy_csv.exists():
                rows.append({
                    "source": source,
                    "seed": seed,
                    "safe_csv": str(safe_csv),
                    "legacy_csv": str(legacy_csv),
                    "status": "missing",
                })
                continue

            safe_df = pd.read_csv(safe_csv)
            legacy_df = pd.read_csv(legacy_csv)

            safe_max = float(safe_df["relative_residual_norm"].max()) if "relative_residual_norm" in safe_df.columns else np.nan
            legacy_max = float(legacy_df["relative_residual_norm"].max()) if "relative_residual_norm" in legacy_df.columns else np.nan

            rows.append({
                "source": source,
                "seed": seed,
                "safe_max_effective": safe_max,
                "legacy_max_effective": legacy_max,
                "legacy_over_safe": legacy_max / max(safe_max, 1e-12),
                "safe_csv": str(safe_csv),
                "legacy_csv": str(legacy_csv),
                "status": "ok",
            })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "final_residual_safety.csv"
    df.to_csv(out_csv, index=False)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return df

    ok = ok.sort_values(["source", "seed"])
    labels = [f"{s}\nseed{int(seed)}" for s, seed in zip(ok["source"], ok["seed"])]
    x = np.arange(len(ok))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(ok) * 0.9), 4.3))

    ax = axes[0]
    ax.bar(x - width / 2, ok["safe_max_effective"], width, label="Safe")
    ax.bar(x + width / 2, ok["legacy_max_effective"], width, label="Legacy")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Max relative prompt perturbation")
    ax.set_title("(a) Effective residual magnitude")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    ax.bar(x, ok["legacy_over_safe"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Legacy / Safe ratio")
    ax.set_title("(b) Suppression ratio")
    ax.grid(True, axis="y", alpha=0.3, which="both")

    fig.suptitle("Aggregate Residual Safety", y=1.03)
    fig.tight_layout()

    out_fig = out_dir / "fig_residual_safety.pdf"
    fig.savefig(out_fig)
    fig.savefig(out_fig.with_suffix(".png"), dpi=300)
    plt.close(fig)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cfg", default="rn50_ep50")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    test_root = Path(args.test_root)
    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    for source in args.sources:
        for target in args.targets:
            if target == source:
                continue
            for seed in args.seeds:
                row = {
                    "source": source,
                    "target": target,
                    "seed": seed,
                }

                complete = True
                for method in METHODS:
                    p = log_path(test_root, source, target, seed, method, args.cfg)
                    acc = read_acc(p)
                    row[f"{method}_acc"] = acc
                    row[f"{method}_log"] = str(p)
                    if acc is None:
                        complete = False
                        missing.append({
                            "source": source,
                            "target": target,
                            "seed": seed,
                            "method": method,
                            "log": str(p),
                        })

                if row["coop_acc"] is not None and row["safe_acc"] is not None:
                    row["safe_delta"] = row["safe_acc"] - row["coop_acc"]
                else:
                    row["safe_delta"] = np.nan

                if row["coop_acc"] is not None and row["legacy_acc"] is not None:
                    row["legacy_delta"] = row["legacy_acc"] - row["coop_acc"]
                else:
                    row["legacy_delta"] = np.nan

                if row["safe_acc"] is not None and row["legacy_acc"] is not None:
                    row["safe_minus_legacy"] = row["safe_acc"] - row["legacy_acc"]
                else:
                    row["safe_minus_legacy"] = np.nan

                row["complete"] = complete
                rows.append(row)

    df = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing)

    df.to_csv(out_dir / "final_dg_pair_deltas.csv", index=False)
    missing_df.to_csv(out_dir / "final_dg_missing_logs.csv", index=False)

    complete_df = df[df["complete"]].copy()

    source_summary = (
        complete_df.groupby("source")
        .agg(
            num_pairs=("safe_delta", "count"),
            coop_mean=("coop_acc", "mean"),
            safe_mean=("safe_acc", "mean"),
            legacy_mean=("legacy_acc", "mean"),
            safe_delta_mean=("safe_delta", "mean"),
            safe_delta_std=("safe_delta", "std"),
            legacy_delta_mean=("legacy_delta", "mean"),
            legacy_delta_std=("legacy_delta", "std"),
            safe_minus_legacy_mean=("safe_minus_legacy", "mean"),
            safe_minus_legacy_std=("safe_minus_legacy", "std"),
        )
        .reset_index()
        .sort_values("safe_delta_mean", ascending=False)
    )

    source_summary.to_csv(out_dir / "final_dg_source_summary.csv", index=False)

    target_summary = (
        complete_df.groupby("target")
        .agg(
            num_pairs=("safe_delta", "count"),
            coop_mean=("coop_acc", "mean"),
            safe_mean=("safe_acc", "mean"),
            legacy_mean=("legacy_acc", "mean"),
            safe_delta_mean=("safe_delta", "mean"),
            safe_delta_std=("safe_delta", "std"),
            legacy_delta_mean=("legacy_delta", "mean"),
            legacy_delta_std=("legacy_delta", "std"),
        )
        .reset_index()
        .sort_values("safe_delta_mean", ascending=False)
    )

    target_summary.to_csv(out_dir / "final_dg_target_summary.csv", index=False)

    print("\n===== Missing logs =====")
    if missing_df.empty:
        print("None")
    else:
        print(missing_df.to_string(index=False))

    print("\n===== Source summary =====")
    print(source_summary.to_string(index=False))

    print("\n===== Target summary =====")
    print(target_summary.to_string(index=False))

    if not source_summary.empty:
        save_bar_source(source_summary, out_dir / "fig_source_dependency_delta.pdf")

    safe_matrix = (
        complete_df.groupby(["source", "target"])["safe_delta"]
        .mean()
        .unstack("target")
        .reindex(index=args.sources, columns=[t for t in args.targets if t in complete_df["target"].unique()])
    )
    legacy_matrix = (
        complete_df.groupby(["source", "target"])["legacy_delta"]
        .mean()
        .unstack("target")
        .reindex(index=args.sources, columns=[t for t in args.targets if t in complete_df["target"].unique()])
    )
    safe_minus_legacy_matrix = (
        complete_df.groupby(["source", "target"])["safe_minus_legacy"]
        .mean()
        .unstack("target")
        .reindex(index=args.sources, columns=[t for t in args.targets if t in complete_df["target"].unique()])
    )

    safe_matrix.to_csv(out_dir / "final_dg_safe_delta_heatmap_values.csv")
    legacy_matrix.to_csv(out_dir / "final_dg_legacy_delta_heatmap_values.csv")
    safe_minus_legacy_matrix.to_csv(out_dir / "final_dg_safe_minus_legacy_heatmap_values.csv")

    save_heatmap(safe_matrix, "Safe - CoOp DG Delta", out_dir / "fig_heatmap_safe_delta.pdf")
    save_heatmap(legacy_matrix, "Legacy - CoOp DG Delta", out_dir / "fig_heatmap_legacy_delta.pdf")
    save_heatmap(safe_minus_legacy_matrix, "Safe - Legacy Accuracy Gap", out_dir / "fig_heatmap_safe_minus_legacy.pdf")

    residual_df = summarize_residual_safety(train_root, args.sources, args.seeds, args.cfg, out_dir)

    print("\n===== Residual safety =====")
    if residual_df.empty:
        print("No residual stats found.")
    else:
        cols = [c for c in ["source", "seed", "safe_max_effective", "legacy_max_effective", "legacy_over_safe", "status"] if c in residual_df.columns]
        print(residual_df[cols].to_string(index=False))

    print("\nSaved all outputs to:", out_dir)


if __name__ == "__main__":
    main()
