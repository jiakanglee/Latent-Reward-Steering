#!/usr/bin/env python3
"""
从 run_basic_overwrite（非 baseline_only）的 log_worker_*.txt 里解析每题 Base/Steer 对错，
输出列联表：steer 修好 / 修坏 / 双对 / 双错。

用法:
  python summarize_math500_steer_logs.py log2/math500_steer_overwrite_124069
"""
from __future__ import annotations

import argparse
import glob
import os
import re

BASE_LINE = re.compile(r"^Base\s+\[(✅|❌)\]\s+Len:\s*(\d+)\s*$")
STEER_LINE = re.compile(r"^Steer\s+\[(✅|❌)\]\s+Len:\s*(\d+)\s*$")
REPORT_BASE = re.compile(r"^\s*Base\s*:\s*(\d+)/(\d+)\s*$")
REPORT_STEER = re.compile(r"^\s*Steer:\s*(\d+)/(\d+)\s*$")


def parse_pairs_scan(path: str) -> list[tuple[bool, bool]]:
    pairs: list[tuple[bool, bool]] = []
    pending_base: bool | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            bm = BASE_LINE.match(s)
            if bm:
                pending_base = bm.group(1) == "✅"
                continue
            sm = STEER_LINE.match(s)
            if sm and pending_base is not None:
                pairs.append((pending_base, sm.group(1) == "✅"))
                pending_base = None
    return pairs


def summarize_reports(paths: list[str]) -> None:
    print("--- Final Report 行（各分片自报）---")
    for p in sorted(paths):
        base = steer = None
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                mb = REPORT_BASE.match(line)
                ms = REPORT_STEER.match(line)
                if mb:
                    base = (int(mb.group(1)), int(mb.group(2)))
                if ms:
                    steer = (int(ms.group(1)), int(ms.group(2)))
        print(f"  {os.path.basename(p)}: Base {base}, Steer {steer}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="LOG_DIR 或单个 log_worker_*.txt")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, "log_worker_*.txt")))
    else:
        paths = [args.path]

    if not paths:
        print("未找到 log_worker_*.txt")
        return

    all_pairs: list[tuple[bool, bool]] = []
    for p in paths:
        all_pairs.extend(parse_pairs_scan(p))

    n = len(all_pairs)
    if n == 0:
        print("未解析到任何 Base/Steer 行对；确认是 steering 模式日志。")
        return

    both_ok = sum(1 for b, s in all_pairs if b and s)
    both_bad = sum(1 for b, s in all_pairs if not b and not s)
    steer_fixes = sum(1 for b, s in all_pairs if not b and s)
    steer_breaks = sum(1 for b, s in all_pairs if b and not s)

    print(f"解析文件数: {len(paths)}，题块数: {n}")
    print()
    print("列联表（math_verify 口径，与日志对错一致）:")
    print(f"  双对 (Base ok & Steer ok):     {both_ok:4d}  ({100 * both_ok / n:.1f}%)")
    print(f"  仅 Base 对 (Steer 改错):       {steer_breaks:4d}")
    print(f"  仅 Steer 对 (Steer 修好):    {steer_fixes:4d}")
    print(f"  双错:                          {both_bad:4d}")
    print()
    print("汇总:")
    print(f"  Base 对:  {sum(1 for b, _ in all_pairs if b):4d} / {n}")
    print(f"  Steer 对: {sum(1 for _, s in all_pairs if s):4d} / {n}")
    print(f"  Steer 净变化: {steer_fixes - steer_breaks:+d} （修好 - 改坏）")
    print()
    summarize_reports(paths)
    print()
    print("若要拆分 rule_math:empty_pred_parse / math_verify_false：")
    print("当前 steering 日志不打印每题 judge；请用 --save_judge_reason 重跑或只看 baseline_only 日志里的 judge 摘要。")


if __name__ == "__main__":
    main()
