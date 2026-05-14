import argparse
import re
from pathlib import Path
import pandas as pd


ACC_RE = re.compile(r"accuracy:\s*([0-9.]+)", re.IGNORECASE)


def read_acc(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    vals = [float(x) for x in ACC_RE.findall(text)]
    if not vals:
        return None
    return vals[-1]


def find_seed_dirs(root: Path, method_name: str, tag_contains: str):
    pattern = f"**/{method_name}/**/{tag_contains}/seed*/log.txt"
    return list(root.glob(pattern))


def parse_case_from_path(path: Path):
    parts = path.parts

    source = None
    target = None
    seed = None
    tag = None
    trainer = None
    cfg = None

    for p in parts:
        if p.startswith("source_"):
            source = p.replace("source_", "")
        if p.startswith("seed"):
            seed = p.replace("seed", "")

    # output/xd/test/<target>/source_<source>/...
    if "test" in parts:
        idx = parts.index("test")
        if idx + 1 < len(parts):
            target = parts[idx + 1]

    if "CoOpPriorRes" in parts:
        trainer = "CoOpPriorRes"
        idx = parts.index("CoOpPriorRes")
        if idx + 2 < len(parts):
            cfg = parts[idx + 1]
            tag = parts[idx + 2]
    elif "CoOp" in parts:
        trainer = "CoOp"
        idx = parts.index("CoOp")
        if idx + 2 < len(parts):
            cfg = parts[idx + 1]
            tag = parts[idx + 2]

    return {
        "source": source,
        "target": target,
        "seed": seed,
        "trainer": trainer,
        "cfg": cfg,
        "tag": tag,
        "path": str(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-root",
        default="third_party/CoOp_clean/output/xd/test",
    )
    parser.add_argument("--coop-tag", default="nctx16_cscFalse_ctpend")
    parser.add_argument("--safe-tag", default="nctx16_cscFalse_ctpend_safe_noalt")
    parser.add_argument("--cfg", default="rn50_ep50")
    parser.add_argument("--out", default="outputs/analysis/top_dg_delta_cases.csv")
    args = parser.parse_args()

    root = Path(args.test_root)

    safe_logs = list(root.glob(f"*/source_*/shots_16/CoOpPriorRes/{args.cfg}/{args.safe_tag}/seed*/log.txt"))

    rows = []
    for safe_log in safe_logs:
        meta = parse_case_from_path(safe_log)
        source = meta["source"]
        target = meta["target"]
        seed = meta["seed"]

        coop_log = root / target / f"source_{source}" / "shots_16" / "CoOp" / args.cfg / args.coop_tag / f"seed{seed}" / "log.txt"

        safe_acc = read_acc(safe_log)
        coop_acc = read_acc(coop_log)

        if safe_acc is None or coop_acc is None:
            continue

        rows.append({
            "source": source,
            "target": target,
            "seed": int(seed),
            "coop_acc": coop_acc,
            "safe_acc": safe_acc,
            "delta": safe_acc - coop_acc,
            "safe_log": str(safe_log),
            "coop_log": str(coop_log),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No matched cases found.")
        return

    df = df.sort_values("delta", ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Saved:", out)
    print()
    print("Top positive cases:")
    print(df.head(20).to_string(index=False))
    print()
    print("Top negative cases:")
    print(df.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
