import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path):
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--ramp-epochs", type=int, default=10)
    args = parser.parse_args()

    safe = load_csv(args.safe)
    legacy = load_csv(args.legacy)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    x_safe = safe["epoch"]
    x_legacy = legacy["epoch"]

    safe_y = safe["safe_relative_residual_norm"]
    legacy_y = legacy["legacy_relative_residual_norm"]

    safe_max = float(safe_y.max())
    legacy_max = float(legacy_y.max())
    ratio = legacy_max / max(safe_max, 1e-12)

    warmup_end = args.warmup_epochs
    ramp_end = args.warmup_epochs + args.ramp_epochs

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Panel A: normal scale
    ax = axes[0]
    ax.plot(x_safe, safe_y, marker="o", linewidth=2, label="Safe PriorRes")
    ax.plot(x_legacy, legacy_y, marker="o", linewidth=2, label="Legacy PriorRes")

    if "legacy_init_bias_relative_norm" in legacy.columns:
        ax.plot(
            x_legacy,
            legacy["legacy_init_bias_relative_norm"],
            linestyle="--",
            linewidth=2,
            label="Legacy init bias, no lambda",
        )

    ax.axvline(warmup_end, linestyle=":", linewidth=1.8, label="warmup end")
    ax.axvline(ramp_end, linestyle="--", linewidth=1.8, label="ramp end")
    ax.set_title("(a) Relative prompt perturbation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\|C_{\mathrm{eff}} - C\|_2 / \|C\|_2$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    text = (
        f"max Legacy / max Safe\n"
        f"$\\approx$ {ratio:.1e}x"
    )
    ax.text(
        0.04,
        0.82,
        text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel B: log scale
    ax = axes[1]
    eps = 1e-10
    ax.plot(x_safe, safe_y.clip(lower=eps), marker="o", linewidth=2, label="Safe PriorRes")
    ax.plot(x_legacy, legacy_y.clip(lower=eps), marker="o", linewidth=2, label="Legacy PriorRes")

    ax.axvline(warmup_end, linestyle=":", linewidth=1.8, label="warmup end")
    ax.axvline(ramp_end, linestyle="--", linewidth=1.8, label="ramp end")
    ax.set_yscale("log")
    ax.set_title("(b) Same curve in log scale")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"relative perturbation, log scale")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle("Identity-centered Residual Injection Suppresses Legacy Prompt Bias", y=1.03)
    fig.tight_layout()

    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)

    print("Saved:", out)
    print("Saved:", out.with_suffix(".png"))
    print("safe max:", safe_max)
    print("legacy max:", legacy_max)
    print("legacy/safe ratio:", ratio)
    print("safe final:", float(safe_y.iloc[-1]))
    print("legacy final:", float(legacy_y.iloc[-1]))


if __name__ == "__main__":
    main()
