import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NAME_MAP = {
    "oxford_pets": "OxfordPets",
    "eurosat": "EuroSAT",
    "dtd": "DTD",
    "food101": "Food101",
    "oxford_flowers": "OxfordFlowers",
    "caltech101": "Caltech101",
    "stanford_cars": "StanfordCars",
    "fgvc_aircraft": "FGVCAircraft",
    "ucf101": "UCF101",
    "sun397": "SUN397",
}


def pretty_name(x):
    return NAME_MAP.get(x, x)


def save_source_bar(summary: pd.DataFrame, out_dir: Path):
    df = summary.copy()

    # 让 source 顺序更自然
    preferred_order = ["caltech101", "food101", "sun397", "oxford_pets"]
    order_map = {k: i for i, k in enumerate(preferred_order)}
    df["__order"] = df["source"].map(lambda x: order_map.get(x, 999))
    df = df.sort_values("__order").drop(columns="__order")

    x = np.arange(len(df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11, 6.5))

    safe_bar = ax.bar(
        x - width / 2,
        df["safe_delta_mean"],
        width,
        yerr=df["safe_delta_std"],
        capsize=6,
        label="Safe - CoOp",
    )
    legacy_bar = ax.bar(
        x + width / 2,
        df["legacy_delta_mean"],
        width,
        yerr=df["legacy_delta_std"],
        capsize=6,
        label="Legacy - CoOp",
    )

    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Mean DG delta", fontsize=15)
    ax.set_title("Source-dependent Cross-dataset Transfer", fontsize=20, pad=14)
    ax.grid(True, axis="y", alpha=0.3)

    xticklabels = [
        f"{pretty_name(s)}\n(n={int(n)})"
        for s, n in zip(df["source"], df["num_pairs"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)

    # 数值标注：放到误差线外面，避免遮挡
    for i, row in df.iterrows():
        safe_y = row["safe_delta_mean"]
        safe_std = 0.0 if pd.isna(row["safe_delta_std"]) else row["safe_delta_std"]
        legacy_y = row["legacy_delta_mean"]
        legacy_std = 0.0 if pd.isna(row["legacy_delta_std"]) else row["legacy_delta_std"]

        ax.text(
            x[i] - width / 2,
            safe_y + (safe_std + 0.25 if safe_y >= 0 else -(safe_std + 0.35)),
            f"{safe_y:+.2f}",
            ha="center",
            va="bottom" if safe_y >= 0 else "top",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            x[i] + width / 2,
            legacy_y + (legacy_std + 0.25 if legacy_y >= 0 else -(legacy_std + 0.35)),
            f"{legacy_y:+.2f}",
            ha="center",
            va="bottom" if legacy_y >= 0 else "top",
            fontsize=12,
            fontweight="bold",
        )

    ax.legend(fontsize=13, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_source_dependency_delta_pretty.pdf")
    fig.savefig(out_dir / "fig_source_dependency_delta_pretty.png", dpi=300)
    plt.close(fig)


def save_heatmap(matrix: pd.DataFrame, title: str, out_pdf: Path):
    data = matrix.values.astype(float)

    row_labels = [pretty_name(x) for x in matrix.index]
    col_labels = [pretty_name(x) for x in matrix.columns]

    fig_w = max(11, 1.15 * len(col_labels))
    fig_h = max(4.8, 1.2 * len(row_labels) + 1.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 1.0)

    im = ax.imshow(data, vmin=-vmax, vmax=vmax)

    ax.set_title(title, fontsize=20, pad=12)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=13)
    ax.set_yticklabels(row_labels, fontsize=15)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="black",
                    fontweight="bold",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)

    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.24, top=0.86)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


def save_residual_safety(df: pd.DataFrame, out_dir: Path):
    data = df.copy()
    if "status" in data.columns:
        data = data[data["status"] == "ok"].copy()

    data = data.sort_values(["source", "seed"]).reset_index(drop=True)

    labels = [
        f"{pretty_name(src)}\nseed{int(seed)}"
        for src, seed in zip(data["source"], data["seed"])
    ]

    x = np.arange(len(data))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.3))

    # (a)
    ax = axes[0]
    ax.bar(x - width / 2, data["safe_max_effective"], width, label="Safe")
    ax.bar(x + width / 2, data["legacy_max_effective"], width, label="Legacy")
    ax.set_yscale("log")
    ax.set_ylabel("Max relative prompt perturbation", fontsize=15)
    ax.set_title("(a) Effective residual magnitude", fontsize=18, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend(fontsize=13, loc="upper right")

    # (b)
    ax = axes[1]
    ax.bar(x, data["legacy_over_safe"])
    ax.set_yscale("log")
    ax.set_ylabel("Legacy / Safe ratio", fontsize=15)
    ax.set_title("(b) Suppression ratio", fontsize=18, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, axis="y", alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_residual_safety_pretty.pdf")
    fig.savefig(out_dir / "fig_residual_safety_pretty.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)

    source_summary = pd.read_csv(summary_dir / "final_dg_source_summary.csv")
    safe_heat = pd.read_csv(summary_dir / "final_dg_safe_delta_heatmap_values.csv", index_col=0)
    legacy_heat = pd.read_csv(summary_dir / "final_dg_legacy_delta_heatmap_values.csv", index_col=0)
    gap_heat = pd.read_csv(summary_dir / "final_dg_safe_minus_legacy_heatmap_values.csv", index_col=0)
    residual = pd.read_csv(summary_dir / "final_residual_safety.csv")

    save_source_bar(source_summary, summary_dir)
    save_heatmap(safe_heat, "Safe - CoOp DG Delta", summary_dir / "fig_heatmap_safe_delta_pretty.pdf")
    save_heatmap(legacy_heat, "Legacy - CoOp DG Delta", summary_dir / "fig_heatmap_legacy_delta_pretty.pdf")
    save_heatmap(gap_heat, "Safe - Legacy Accuracy Gap", summary_dir / "fig_heatmap_safe_minus_legacy_pretty.pdf")
    save_residual_safety(residual, summary_dir)

    print("Saved pretty figures to:", summary_dir)


if __name__ == "__main__":
    main()
