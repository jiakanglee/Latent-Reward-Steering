#!/usr/bin/env python3
"""
比较同一目录下多个 task_*/judge 里每题的 base_correct 是否一致（用于不同 num_steps/step_size 但应同分布的 base）。

用法:
  python compare_math500_base_across_tasks.py log2/math500_steer_quad500_125844

全部 task 均满 500 题时：若两两差异为 0，则四次实验 base 逐题完全一致。
未满 500 时仍打印进度与共现子集上的不一致数。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob


def load_base_by_qidx(judge_dir: str) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for p in glob(os.path.join(judge_dir, "question_*_shard_*.json")):
        m = re.search(r"question_(\d+)_shard_", os.path.basename(p))
        if not m:
            continue
        q = int(m.group(1))
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        out[q] = bool(d.get("base_correct"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_root", help="含 task_*/judge/ 的根目录")
    ap.add_argument("--expect", type=int, default=500, metavar="N", help="期望每 task 题数（默认 500）")
    args = ap.parse_args()
    root = os.path.abspath(args.sweep_root)
    if not os.path.isdir(root):
        print(f"不是目录: {root}", file=sys.stderr)
        sys.exit(1)

    task_dirs = sorted(
        d for d in glob(os.path.join(root, "task_*")) if os.path.isdir(os.path.join(d, "judge"))
    )
    if not task_dirs:
        print(f"未找到 task_*/judge: {root}", file=sys.stderr)
        sys.exit(1)

    names: list[str] = []
    bases: dict[str, dict[int, bool]] = {}
    for td in task_dirs:
        name = os.path.basename(td)
        names.append(name)
        bases[name] = load_base_by_qidx(os.path.join(td, "judge"))

    print(f"根目录: {root}\n")
    print("=== 各 task judge 覆盖 ===")
    all_full = True
    for n in names:
        k = len(bases[n])
        ok = k >= args.expect
        all_full = all_full and ok
        print(f"  {n}: {k}/{args.expect} {'✓' if ok else '(未满)'}")
    print()

    print("=== 两两 base_correct 不一致（共现题号）===")
    any_pair_diff = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ca, cb = bases[a], bases[b]
            common = sorted(set(ca) & set(cb))
            diff = [q for q in common if ca[q] != cb[q]]
            if diff:
                any_pair_diff = True
            print(f"  {a} vs {b}: 共现 {len(common)} 题, 不一致 {len(diff)}")
            if diff[:25]:
                print(f"    题号示例: {diff[:25]}{' ...' if len(diff) > 25 else ''}")

    if all_full and len(names) >= 2:
        print("\n=== 全量 0..499 两两不一致题数 ===")
        full_diff_any = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ca, cb = bases[a], bases[b]
                diff = [q for q in range(args.expect) if ca.get(q) != cb.get(q)]
                if diff:
                    full_diff_any = True
                print(f"  {a} vs {b}: {len(diff)}")
        if not full_diff_any:
            print("\n结论: 所有 task 在 0..499 上 base_correct 两两完全一致。")
        else:
            print("\n结论: 存在跨 task 的 base 不一致，不宜共用同一套「抽题」逻辑；需按单次 run 抽题或先查因。")
            sys.exit(2)
    elif not all_full:
        print("\n尚有 task 未满 {}；请作业跑完后重新运行本脚本做最终结论。".format(args.expect))
        sys.exit(1)

    if any_pair_diff:
        sys.exit(2)


if __name__ == "__main__":
    main()
