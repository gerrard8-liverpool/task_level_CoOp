#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_int_list(text: str) -> list[int]:
    values = []
    for x in text.split(","):
        x = x.strip()
        if x:
            values.append(int(x))
    return values


def build_m_values(m_values_text: str | None, m_start: int, m_max: int, m_step: int) -> list[int]:
    if m_values_text:
        values = parse_int_list(m_values_text)
    else:
        values = list(range(m_start, m_max + 1, m_step))

    values = sorted(set(values))
    if not values:
        raise ValueError("没有可用的 m 值。")
    return values


def run_command(cmd: list[str], cwd: Path) -> str:
    print("\n" + "=" * 100, flush=True)
    print(f"[CWD] {cwd}", flush=True)
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    print("=" * 100, flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines = []
    assert proc.stdout is not None

    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)

    proc.wait()
    output = "".join(lines)

    if proc.returncode != 0:
        raise RuntimeError(f"命令执行失败，返回码={proc.returncode}")

    return output


def infer_output_dir(
    coop_root: Path,
    dataset: str,
    config: str,
    shots: int,
    nctx: int,
    csc: bool,
    ctp: str,
) -> Path:
    # CoOp 原脚本输出目录模式：
    # output/{dataset}/CoOp/{config}_{shots}shots/nctx{nctx}_csc{csc}_ctp{ctp}
    return (
        coop_root
        / "output"
        / dataset
        / "CoOp"
        / f"{config}_{shots}shots"
        / f"nctx{nctx}_csc{csc}_ctp{ctp}"
    )


def parse_mean_std(parse_output: str) -> tuple[float, float]:
    """
    尽量兼容 parse_test_res.py 的常见输出格式。
    优先匹配：
    1) accuracy: 85.73% +- 1.41%
    2) accuracy ... std ...
    3) fallback: 最后一行里最后两个百分数
    """

    patterns = [
        r"accuracy[^0-9]*([0-9]+(?:\.[0-9]+)?)%\s*(?:\+/-|\+-|±)\s*([0-9]+(?:\.[0-9]+)?)%",
        r"accuracy[^0-9]*([0-9]+(?:\.[0-9]+)?)%.*?std[^0-9]*([0-9]+(?:\.[0-9]+)?)%",
        r"average[^0-9]*([0-9]+(?:\.[0-9]+)?)%.*?std[^0-9]*([0-9]+(?:\.[0-9]+)?)%",
        r"avg[^0-9]*([0-9]+(?:\.[0-9]+)?)%.*?std[^0-9]*([0-9]+(?:\.[0-9]+)?)%",
    ]

    for pattern in patterns:
        match = re.search(pattern, parse_output, flags=re.IGNORECASE | re.DOTALL)
        if match:
            mean = float(match.group(1))
            std = float(match.group(2))
            return mean, std

    # fallback：从后往前找，取某一行里的前两个百分数
    lines = [line.strip() for line in parse_output.splitlines() if "%" in line]
    for line in reversed(lines):
        nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)%", line)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])

    raise ValueError("无法从 parse_test_res.py 的输出中解析 mean/std，请手动检查输出格式。")


def write_csv(rows: list[dict], csv_path: Path) -> None:
    headers = [
        "k",
        "m",
        "csc",
        "Dataset",
        "Backbone/Config",
        "token位置",
        "mean",
        "std",
        "status",
        "output_dir",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict], md_path: Path) -> None:
    headers = [
        "k",
        "m",
        "csc",
        "Dataset",
        "Backbone/Config",
        "token位置",
        "mean",
        "std",
        "status",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for row in rows:
        line = "| " + " | ".join(str(row[h]) for h in headers) + " |"
        lines.append(line)

    md_path.write_text("\n".join(lines), encoding="utf-8")


def save_all_tables(result_dir: Path, rows: list[dict], dataset: str, config: str, ctp: str) -> None:
    # 总表
    all_csv = result_dir / f"{dataset}_{config}_{ctp}_all.csv"
    all_md = result_dir / f"{dataset}_{config}_{ctp}_all.md"

    sorted_rows = sorted(rows, key=lambda x: (int(x["k"]), int(x["m"])))
    write_csv(sorted_rows, all_csv)
    write_markdown(sorted_rows, all_md)

    # 每个 k 一个表
    grouped = defaultdict(list)
    for row in sorted_rows:
        grouped[int(row["k"])].append(row)

    for k, k_rows in grouped.items():
        k_csv = result_dir / f"{dataset}_{config}_{ctp}_k{k}.csv"
        k_md = result_dir / f"{dataset}_{config}_{ctp}_k{k}.md"
        write_csv(k_rows, k_csv)
        write_markdown(k_rows, k_md)


def main():
    parser = argparse.ArgumentParser(description="自动运行 CoOp 的 (k, m) 网格实验，并汇总结果。")

    parser.add_argument("--dataset", type=str, required=True, help="例如：oxford_pets")
    parser.add_argument("--config", type=str, default="rn50_ep50", help="例如：rn50_ep50")
    parser.add_argument("--ctp", type=str, default="end", help="class token position，例如：end")
    parser.add_argument("--csc", type=str, default="False", help="是否使用 CSC，固定填 False 或 True")

    parser.add_argument(
        "--k-values",
        type=str,
        default="2,4,8,10,12,14,16",
        help="逗号分隔，例如：2,4,8,10,12,14,16",
    )

    parser.add_argument("--m-values", type=str, default=None, help="直接指定 m 列表，例如：2,4,6,8,10,12,14,16")
    parser.add_argument("--m-start", type=int, default=2, help="m 起始值")
    parser.add_argument("--m-max", type=int, default=16, help="m 最大值")
    parser.add_argument("--m-step", type=int, default=2, help="m 步长")

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若输出目录已存在，则跳过训练，只做 parse_test_res.py 汇总",
    )

    args = parser.parse_args()

    # 这里假设脚本就放在 third_party/CoOp 目录下
    coop_root = Path(__file__).resolve().parent

    main_sh = coop_root / "scripts" / "coop" / "main.sh"
    parse_script = coop_root / "parse_test_res.py"

    if not main_sh.exists():
        raise FileNotFoundError(f"找不到 main.sh: {main_sh}")

    if not parse_script.exists():
        raise FileNotFoundError(f"找不到 parse_test_res.py: {parse_script}")

    csc_bool = args.csc.strip().lower() == "true"
    csc_text = "True" if csc_bool else "False"

    k_values = parse_int_list(args.k_values)
    m_values = build_m_values(args.m_values, args.m_start, args.m_max, args.m_step)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = coop_root / "grid_results" / f"{args.dataset}_{args.config}_{args.ctp}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_txt = result_dir / "run_summary.txt"
    summary_txt.write_text(
        "\n".join([
            f"dataset={args.dataset}",
            f"config={args.config}",
            f"ctp={args.ctp}",
            f"csc={csc_text}",
            f"k_values={k_values}",
            f"m_values={m_values}",
            f"coop_root={coop_root}",
            f"result_dir={result_dir}",
        ]),
        encoding="utf-8",
    )

    rows: list[dict] = []

    print("\n开始实验：")
    print(f"dataset   = {args.dataset}")
    print(f"config    = {args.config}")
    print(f"ctp       = {args.ctp}")
    print(f"csc       = {csc_text}")
    print(f"k_values  = {k_values}")
    print(f"m_values  = {m_values}")
    print(f"result_dir= {result_dir}\n")

    total_jobs = len(k_values) * len(m_values)
    job_id = 0

    for k in k_values:
        for m in m_values:
            job_id += 1

            row = {
                "k": k,
                "m": m,
                "csc": csc_text,
                "Dataset": args.dataset.replace("_", "").title() if args.dataset != "oxford_pets" else "OxfordPets",
                "Backbone/Config": args.config,
                "token位置": args.ctp,
                "mean": "",
                "std": "",
                "status": "",
                "output_dir": str(infer_output_dir(coop_root, args.dataset, args.config, k, m, csc_bool, args.ctp)),
            }

            print(f"\n[{job_id}/{total_jobs}] 开始运行: dataset={args.dataset}, k={k}, m={m}")

            try:
                output_dir = infer_output_dir(
                    coop_root=coop_root,
                    dataset=args.dataset,
                    config=args.config,
                    shots=k,
                    nctx=m,
                    csc=csc_bool,
                    ctp=args.ctp,
                )

                # 1) 训练
                if args.skip_existing and output_dir.exists():
                    print(f"检测到输出目录已存在，跳过训练：{output_dir}")
                else:
                    train_cmd = [
                        "bash",
                        "scripts/coop/main.sh",
                        args.dataset,
                        args.config,
                        args.ctp,
                        str(m),
                        str(k),
                        csc_text,
                    ]
                    run_command(train_cmd, cwd=coop_root)

                # 2) 汇总 3 个 seed
                parse_cmd = [
                    sys.executable,
                    "parse_test_res.py",
                    str(output_dir),
                ]
                parse_output = run_command(parse_cmd, cwd=coop_root)

                mean, std = parse_mean_std(parse_output)

                row["mean"] = f"{mean:.2f}%"
                row["std"] = f"{std:.2f}%"
                row["status"] = "OK"

            except Exception as e:
                row["status"] = f"FAIL: {e}"
                print(f"出错：{e}")

            rows.append(row)
            save_all_tables(result_dir, rows, args.dataset, args.config, args.ctp)

    print("\n全部任务结束。")
    print(f"结果目录：{result_dir}")
    print(f"总表 CSV：{result_dir / f'{args.dataset}_{args.config}_{args.ctp}_all.csv'}")
    print(f"总表 MD ：{result_dir / f'{args.dataset}_{args.config}_{args.ctp}_all.md'}")


if __name__ == "__main__":
    main()