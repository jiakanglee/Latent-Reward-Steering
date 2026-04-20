#!/usr/bin/env python3
"""
汇总 run_math500_steer_sweep_120.slurm 产出：每个 task 目录下 judge/*.json 的 steer_correct 计数。

用法:
  python summarize_math500_steer_sweep.py log2/math500_steer_sweep120_<ARRAY_JOB_ID>
  python summarize_math500_steer_sweep.py log2/math500_steer_sweep120_<ARRAY_JOB_ID> --csv sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_root", help="含 task_*_ns*_ss*/ 的根目录")
    ap.add_argument("--csv", metavar="PATH", help="写出 CSV（num_steps,step_size,steer_ok,n_json,...）")
    args = ap.parse_args()
    root = os.path.abspath(args.sweep_root)
    if not os.path.isdir(root):
        raise SystemExit(f"不是目录: {root}")

    task_dirs = sorted(
        d for d in glob(os.path.join(root, "task_*")) if os.path.isdir(d)
    )
    rows = []
    for td in task_dirs:
        meta_path = os.path.join(td, "sweep_task_meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        judges = glob(os.path.join(td, "judge", "question_*_shard_*.json"))
        steer_ok = 0
        base_ok = 0
        n = 0
        for p in judges:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            n += 1
            if d.get("steer_correct"):
                steer_ok += 1
            if d.get("base_correct"):
                base_ok += 1
        rows.append(
            {
                "task_dir": os.path.basename(td),
                "num_steps": meta.get("num_steps", ""),
                "step_size": meta.get("step_size", ""),
                "reward_threshold": meta.get("reward_threshold", ""),
                "confidence_threshold": meta.get("confidence_threshold", ""),
                "n_judge_json": n,
                "steer_correct": steer_ok,
                "steer_acc": round(steer_ok / n, 4) if n else "",
                "base_correct_meta": base_ok,
            }
        )

    if not rows:
        print("未找到 task_* 子目录:", root)
        return
    rows.sort(key=lambda r: (r["num_steps"] or 0, float(r["step_size"] or 0)))
    for r in rows:
        print(
            f"ns={r['num_steps']} ss={r['step_size']}\tsteer {r['steer_correct']}/{r['n_judge_json']}\t{r['task_dir']}"
        )

    if args.csv:
        out = os.path.abspath(args.csv)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        print("Wrote", out)


if __name__ == "__main__":
    main()
