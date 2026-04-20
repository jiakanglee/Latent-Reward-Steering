#!/usr/bin/env python3
"""
汇总 baseline 4 次重复实验：根目录下 rep_*/judge/*.json 的 base_correct 计数。

用法:
  python summarize_math500_baseline_reps.py log2/math500_baseline_4rep_<ARRAY_JOB_ID>          # rlab2 / 默认根目录名
  python summarize_math500_baseline_reps.py log2/math500_baseline_4rep_rlab1_<ARRAY_JOB_ID>   # rlab1 2×A100
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rep_root", help="含 rep_0, rep_1, ... 子目录的根路径")
    args = ap.parse_args()
    root = os.path.abspath(args.rep_root)
    if not os.path.isdir(root):
        raise SystemExit(f"不是目录: {root}")

    rep_dirs = sorted(d for d in glob(os.path.join(root, "rep_*")) if os.path.isdir(d))
    if not rep_dirs:
        raise SystemExit(f"未找到 rep_* 子目录: {root}")

    rows = []
    accs = []
    for rd in rep_dirs:
        judges = glob(os.path.join(rd, "judge", "question_*_shard_*.json"))
        ok = 0
        for p in judges:
            with open(p, encoding="utf-8") as f:
                if json.load(f).get("base_correct"):
                    ok += 1
        n = len(judges)
        acc = ok / n if n else 0.0
        accs.append(acc)
        rows.append((os.path.basename(rd), ok, n, acc))

    print(f"root={root}")
    print(f"n_reps={len(rows)}")
    for name, ok, n, acc in rows:
        print(f"  {name}: {ok}/{n} = {100*acc:.4f}%")
    if accs:
        mean = sum(accs) / len(accs)
        tot_ok = sum(r[1] for r in rows)
        tot_n = sum(r[2] for r in rows)
        pooled = tot_ok / tot_n if tot_n else 0.0
        print(f"mean_acc (各次平均): {100*mean:.4f}%")
        print(f"pooled ({len(rows)} 次合计): {tot_ok}/{tot_n} = {100*pooled:.4f}%")


if __name__ == "__main__":
    main()
