import argparse
from pathlib import Path
import pandas as pd


PRETTY = {
    "caltech101": "Caltech101",
    "food101": "Food101",
    "sun397": "SUN397",
    "oxford_pets": "OxfordPets",
    "eurosat": "EuroSAT",
    "dtd": "DTD",
    "oxford_flowers": "OxfordFlowers",
    "stanford_cars": "StanfordCars",
    "fgvc_aircraft": "FGVCAircraft",
    "ucf101": "UCF101",
}

MAIN_SOURCES = ["caltech101", "food101", "sun397"]
SOURCE_ORDER = ["caltech101", "food101", "sun397", "oxford_pets"]

TARGET_ORDER = [
    "oxford_pets",
    "eurosat",
    "dtd",
    "food101",
    "oxford_flowers",
    "caltech101",
    "stanford_cars",
    "fgvc_aircraft",
    "ucf101",
    "sun397",
]


def pretty(x):
    return PRETTY.get(x, x)


def fmt(x):
    if pd.isna(x):
        return ""
    return f"{x:.2f}"


def signed(x):
    if pd.isna(x):
        return ""
    return f"{x:+.2f}"


def source_summary(df, sources):
    sub = df[df["source"].isin(sources)].copy()

    rows = []
    for src in sources:
        g = sub[sub["source"] == src]
        if g.empty:
            continue
        rows.append({
            "source": src,
            "num_pairs": len(g),
            "coop_mean": g["coop_acc"].mean(),
            "safe_mean": g["safe_acc"].mean(),
            "legacy_mean": g["legacy_acc"].mean(),
            "safe_delta_mean": g["safe_delta"].mean(),
            "legacy_delta_mean": g["legacy_delta"].mean(),
            "safe_minus_legacy_mean": g["safe_minus_legacy"].mean(),
            "safe_positive": int((g["safe_delta"] > 0).sum()),
            "legacy_positive": int((g["legacy_delta"] > 0).sum()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    overall = {
        "source": "Overall",
        "num_pairs": int(out["num_pairs"].sum()),
        "coop_mean": sub["coop_acc"].mean(),
        "safe_mean": sub["safe_acc"].mean(),
        "legacy_mean": sub["legacy_acc"].mean(),
        "safe_delta_mean": sub["safe_delta"].mean(),
        "legacy_delta_mean": sub["legacy_delta"].mean(),
        "safe_minus_legacy_mean": sub["safe_minus_legacy"].mean(),
        "safe_positive": int((sub["safe_delta"] > 0).sum()),
        "legacy_positive": int((sub["legacy_delta"] > 0).sum()),
    }

    return pd.concat([out, pd.DataFrame([overall])], ignore_index=True)


def per_target_table(df, sources):
    rows = []
    for src in sources:
        targets = [t for t in TARGET_ORDER if t != src]
        for tgt in targets:
            g = df[(df["source"] == src) & (df["target"] == tgt)]
            if g.empty:
                continue
            rows.append({
                "source": src,
                "target": tgt,
                "n": len(g),
                "coop": g["coop_acc"].mean(),
                "safe": g["safe_acc"].mean(),
                "legacy": g["legacy_acc"].mean(),
                "safe_delta": g["safe_delta"].mean(),
                "legacy_delta": g["legacy_delta"].mean(),
                "safe_minus_legacy": g["safe_minus_legacy"].mean(),
            })
    return pd.DataFrame(rows)


def write_source_summary_md(lines, title, summary):
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Source | n | CoOp | Safe | Legacy | Safe-CoOp | Legacy-CoOp | Safe-Legacy | Safe>CoOp | Legacy>CoOp |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in summary.iterrows():
        src = "**Overall**" if r["source"] == "Overall" else pretty(r["source"])
        bold = r["source"] == "Overall"

        vals = [
            src,
            str(int(r["num_pairs"])),
            fmt(r["coop_mean"]),
            fmt(r["safe_mean"]),
            fmt(r["legacy_mean"]),
            signed(r["safe_delta_mean"]),
            signed(r["legacy_delta_mean"]),
            signed(r["safe_minus_legacy_mean"]),
            f'{int(r["safe_positive"])}/{int(r["num_pairs"])}',
            f'{int(r["legacy_positive"])}/{int(r["num_pairs"])}',
        ]

        if bold:
            vals = [f"**{v}**" if i not in [0] else v for i, v in enumerate(vals)]

        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")


def write_per_target_md(lines, title, table):
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Source | Target | n | CoOp | Safe | Legacy | Safe-CoOp | Legacy-CoOp | Safe-Legacy |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in table.iterrows():
        lines.append(
            "| "
            + " | ".join([
                pretty(r["source"]),
                pretty(r["target"]),
                str(int(r["n"])),
                fmt(r["coop"]),
                fmt(r["safe"]),
                fmt(r["legacy"]),
                signed(r["safe_delta"]),
                signed(r["legacy_delta"]),
                signed(r["safe_minus_legacy"]),
            ])
            + " |"
        )

    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", required=True)
    ap.add_argument("--out", default="summary_tables/dg_main/xd_multisource_coop_safe_legacy.md")
    args = ap.parse_args()

    sdir = Path(args.summary_dir)
    pair_csv = sdir / "final_dg_pair_deltas.csv"
    missing_csv = sdir / "final_dg_missing_logs.csv"

    df = pd.read_csv(pair_csv)
    if "complete" in df.columns:
        df = df[df["complete"] == True].copy()

    for col in ["coop_acc", "safe_acc", "legacy_acc", "safe_delta", "legacy_delta", "safe_minus_legacy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if missing_csv.exists() and missing_csv.stat().st_size > 0:
        try:
            missing = pd.read_csv(missing_csv)
        except pd.errors.EmptyDataError:
            missing = pd.DataFrame()
    else:
        missing = pd.DataFrame()

    main_summary = source_summary(df, MAIN_SOURCES)
    all_summary = source_summary(df, SOURCE_ORDER)

    main_target = per_target_table(df, MAIN_SOURCES)
    all_target = per_target_table(df, SOURCE_ORDER)

    lines = []
    lines.append("# Clean Rerun Multi-source Cross-dataset DG Summary")
    lines.append("")
    lines.append("> This table is regenerated from the clean protocol-aligned DG rerun.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Backbone: RN50")
    lines.append("- Shots: 16")
    lines.append("- Context length: 16")
    lines.append("- CSC: False")
    lines.append("- Class token position: end")
    lines.append("- Seeds: 1, 2, 3")
    lines.append("- DG setting: train on one source dataset and evaluate on all remaining target datasets")
    lines.append("- Safe / Legacy evaluation uses source task features under strict DG")
    lines.append("")
    lines.append("Compared methods:")
    lines.append("")
    lines.append("- CoOp")
    lines.append("- Safe PriorRes: identity-centered residual, `ctx + lambda * (a - a0) * u_ctx`")
    lines.append("- Legacy PriorRes: non-identity residual, `ctx + lambda * (a - 1) * u_ctx`")
    lines.append("")

    write_source_summary_md(lines, "Main 3-source Summary", main_summary)
    write_source_summary_md(lines, "4-source Source-dependency Summary", all_summary)

    lines.append("## Interpretation")
    lines.append("")
    lines.append("The clean rerun should be interpreted conservatively:")
    lines.append("")
    lines.append("- Safe PriorRes is not a universal DG accuracy booster.")
    lines.append("- Its main mechanism is residual safety: identity-centered residual injection prevents prior-induced non-identity prompt bias.")
    lines.append("- Cross-dataset behavior is source-target dependent, so pairwise heatmaps are more informative than source-level averages alone.")
    lines.append("- Legacy can be competitive or positive on some pairs, but it lacks the identity-preserving safety guarantee.")
    lines.append("")

    write_per_target_md(lines, "Main 3-source Target-level Averages", main_target)
    write_per_target_md(lines, "4-source Target-level Averages", all_target)

    lines.append("## Missing Logs")
    lines.append("")
    if missing.empty:
        lines.append("No missing logs were found in the clean rerun summary.")
    else:
        lines.append("| Source | Target | Seed | Method | Log |")
        lines.append("|---|---|---:|---|---|")
        for _, r in missing.iterrows():
            lines.append(f"| {r.get('source','')} | {r.get('target','')} | {r.get('seed','')} | {r.get('method','')} | `{r.get('log','')}` |")

    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", out)
    print("")
    print("Main 3-source summary:")
    print(main_summary.to_string(index=False))
    print("")
    print("4-source summary:")
    print(all_summary.to_string(index=False))


if __name__ == "__main__":
    main()
