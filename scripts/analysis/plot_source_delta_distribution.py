import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME_MAP = {
    "caltech101": "Caltech101",
    "food101": "Food101",
    "sun397": "SUN397",
    "oxford_pets": "OxfordPets",
}

def pretty(x):
    return NAME_MAP.get(x, x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    args = parser.parse_args()

    sdir = Path(args.summary_dir)
    df = pd.read_csv(sdir / "final_dg_pair_deltas.csv")
    df = df[df["complete"] == True].copy()

    sources = ["caltech101", "food101", "sun397", "oxford_pets"]
    sources = [s for s in sources if s in set(df["source"])]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)

    for ax, metric, title in [
        (axes[0], "safe_delta", "Safe - CoOp"),
        (axes[1], "legacy_delta", "Legacy - CoOp"),
    ]:
        data = [df[df["source"] == s][metric].dropna().values for s in sources]
        positions = np.arange(1, len(sources) + 1)

        ax.boxplot(data, positions=positions, widths=0.55, showmeans=True)

        # jittered points
        rng = np.random.default_rng(0)
        for pos, vals in zip(positions, data):
            jitter = rng.normal(0, 0.045, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=18, alpha=0.55)

        ax.axhline(0, linewidth=1)
        ax.set_title(title, fontsize=17)
        ax.set_xticks(positions)
        ax.set_xticklabels([pretty(s) for s in sources], rotation=25, ha="right", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=12)

    axes[0].set_ylabel("DG delta", fontsize=14)
    fig.suptitle("Per-source Distribution of Cross-dataset DG Delta", fontsize=20, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = sdir / "fig_source_delta_distribution.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("Saved:", out)
    print("Saved:", out.with_suffix(".png"))

if __name__ == "__main__":
    main()
