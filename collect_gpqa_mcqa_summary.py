#!/usr/bin/env python3
"""
汇总 GPQA-Diamond（rule MCQ 抽取）的 judge JSON，写出 CSV / 简短统计。
用法: python collect_gpqa_mcqa_summary.py LOG_DIR
输出: LOG_DIR/gpqa_mcqa_summary.csv 与终端打印 Base/Steer 准确率。
"""
import csv
import glob
import json
import os
import re
import sys


def main():
    log_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    jdir = os.path.join(log_dir, "judge_reasons")
    if not os.path.isdir(jdir):
        print(f"FATAL: no judge_reasons under {log_dir}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(jdir, "question_*_shard_*.json")))
    rows = []
    for fp in paths:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        m = re.search(r"question_(\d+)_shard_(\d+)", os.path.basename(fp))
        qidx = int(m.group(1)) if m else -1
        shard = int(m.group(2)) if m else -1
        rows.append(
            {
                "question_idx": d.get("question_idx", qidx),
                "shard": shard,
                "gold_letter": d.get("gold_letter", ""),
                "extracted_letter_base": d.get("extracted_letter_base", ""),
                "extracted_letter_steer": d.get("extracted_letter_steer", ""),
                "base_correct": d.get("base_correct", ""),
                "steer_correct": d.get("steer_correct", ""),
                "judge_mode": d.get("judge_mode", ""),
                "judge_base_reason": (d.get("judge_base_reason") or "")[:80],
                "judge_steer_reason": (d.get("judge_steer_reason") or "")[:80],
            }
        )

    rows.sort(key=lambda r: (int(r["question_idx"]) if str(r["question_idx"]).isdigit() else 0, r["shard"]))

    out_csv = os.path.join(log_dir, "gpqa_mcqa_summary.csv")
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n = len(rows)
    if n == 0:
        print("No JSON rows")
        sys.exit(0)

    def rate(key):
        ok = sum(1 for r in rows if r.get(key) is True or r.get(key) == "True")
        return ok, n, ok / n if n else 0.0

    b_ok, _, br = rate("base_correct")
    s_ok, _, sr = rate("steer_correct")
    print(f"Wrote {out_csv}  (n={n})")
    print(f"Base correct:  {b_ok}/{n}  ({br*100:.2f}%)")
    print(f"Steer correct: {s_ok}/{n}  ({sr*100:.2f}%)")


if __name__ == "__main__":
    main()
