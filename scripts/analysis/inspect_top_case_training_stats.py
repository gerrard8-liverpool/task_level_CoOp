import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="outputs/analysis/top_dg_delta_cases.csv")
    parser.add_argument("--rank", type=int, default=1, help="1 means top positive case")
    parser.add_argument("--train-root", default="third_party/CoOp_clean/output/xd/train")
    parser.add_argument("--cfg", default="rn50_ep50")
    parser.add_argument("--safe-tag", default="nctx16_cscFalse_ctpend_safe_noalt")
    parser.add_argument("--legacy-tag", default="nctx16_cscFalse_ctpend_legacy_noalt")
    args = parser.parse_args()

    df = pd.read_csv(args.cases).sort_values("delta", ascending=False).reset_index(drop=True)
    row = df.iloc[args.rank - 1]

    source = row["source"]
    target = row["target"]
    seed = int(row["seed"])

    safe_csv = Path(args.train_root) / f"source_{source}" / "shots_16" / "CoOpPriorRes" / args.cfg / args.safe_tag / f"seed{seed}" / "analysis_stats.csv"
    legacy_csv = Path(args.train_root) / f"source_{source}" / "shots_16" / "CoOpPriorRes" / args.cfg / args.legacy_tag / f"seed{seed}" / "analysis_stats.csv"

    print("Selected case:")
    print(row.to_string())
    print()
    print("Safe training csv:")
    print(safe_csv)
    print("exists:", safe_csv.exists())
    print()
    print("Legacy training csv:")
    print(legacy_csv)
    print("exists:", legacy_csv.exists())

    if safe_csv.exists():
        s = pd.read_csv(safe_csv)
        print("\n==== SAFE TRAINING STATS ====")
        for col in [
            "relative_residual_norm",
            "safe_relative_residual_norm",
            "legacy_relative_residual_norm",
            "safe_residual_norm",
            "delta_a_norm",
            "a0_mean",
            "a_mean",
            "lambda_t",
            "meff",
            "keff",
            "loss",
            "acc",
        ]:
            if col in s.columns:
                print(col, "min=", s[col].min(), "max=", s[col].max(), "last=", s[col].iloc[-1])

    if legacy_csv.exists():
        l = pd.read_csv(legacy_csv)
        print("\n==== LEGACY TRAINING STATS ====")
        for col in [
            "relative_residual_norm",
            "safe_relative_residual_norm",
            "legacy_relative_residual_norm",
            "legacy_init_bias_relative_norm",
            "delta_a_norm",
            "a0_mean",
            "a_mean",
            "lambda_t",
            "loss",
            "acc",
        ]:
            if col in l.columns:
                print(col, "min=", l[col].min(), "max=", l[col].max(), "last=", l[col].iloc[-1])


if __name__ == "__main__":
    main()
