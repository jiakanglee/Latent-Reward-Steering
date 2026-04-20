#!/usr/bin/env python3
"""
在 math500_steer_quad500_* 等多 task 目录上，等各 task judge 满 500 后：

1) 分析「Base 对 & Steer 错」(反损 / harm) 的跨 run 重合与统计共性（判题理由、len_steer 等）
2) 导出三类题号的并集（任一 run 落入该格即计入），供后续 sweep 的 --example_idx_file 使用：
   - harm:   base 对 & steer 错
   - save:   base 错 & steer 对
   - both_bad: 双错

用法:
  python aggregate_quad_steer_cells_for_sweep.py log2/math500_steer_quad500_125844
  python aggregate_quad_steer_cells_for_sweep.py log2/math500_steer_quad500_125844 -o log2/math500_steer_quad500_125844/sweep_pools

未满 500 时会打印进度并以退出码 1 退出（可用 --allow-partial 仅看当前能读的题）。

写出（默认前缀为 sweep_root/sweep_cell_pools）:
  <prefix>_meta.json          # 全集 meta + 每 task 列表
  <prefix>_harm.json          # {"question_indices": [...]}  供 --example_idx_file
  <prefix>_save.json
  <prefix>_both_bad.json
  <prefix>_union_three.json   # 三类并集（去重）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from glob import glob


def load_judges(judge_dir: str) -> dict[int, dict]:
    by_q: dict[int, dict] = {}
    for p in glob(os.path.join(judge_dir, "question_*_shard_*.json")):
        m = re.search(r"question_(\d+)_shard_", os.path.basename(p))
        if not m:
            continue
        q = int(m.group(1))
        with open(p, encoding="utf-8") as f:
            by_q[q] = json.load(f)
    return by_q


def cell(q: int, d: dict) -> str | None:
    if "base_correct" not in d or "steer_correct" not in d:
        return None
    b, s = bool(d.get("base_correct")), bool(d.get("steer_correct"))
    if b and s:
        return "both_ok"
    if (not b) and s:
        return "save"
    if b and (not s):
        return "harm"
    return "both_bad"


def bucket_reason(reason: str) -> str:
    r = (reason or "").strip()
    if "empty_pred_parse" in r:
        return "empty_pred_parse"
    if "math_verify_false" in r:
        return "math_verify_false"
    if "math_verify_missing" in r:
        return "math_verify_missing"
    if "no_gold" in r:
        return "no_gold"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_root", help="含多个 task_*/judge/ 的根目录")
    ap.add_argument(
        "-o",
        "--output-prefix",
        default=None,
        metavar="PREFIX",
        help="输出文件前缀（不含扩展名）；默认 <sweep_root>/sweep_cell_pools",
    )
    ap.add_argument("--expect", type=int, default=500, metavar="N", help="每 task 至少 N 题（默认 500）")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="有 task 未满仍导出并分析（仅基于已有题；并集可能不全）",
    )
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

    task_names = [os.path.basename(d) for d in task_dirs]
    data: dict[str, dict[int, dict]] = {}
    n_per: dict[str, int] = {}
    for td, name in zip(task_dirs, task_names):
        j = load_judges(os.path.join(td, "judge"))
        data[name] = j
        n_per[name] = len(j)

    incomplete = [n for n, k in n_per.items() if k < args.expect]
    if incomplete and not args.allow_partial:
        print(f"以下 task 未满 {args.expect} 题，请跑完后再执行本脚本（或加 --allow-partial）：")
        for n in incomplete:
            print(f"  {n}: {n_per[n]}/{args.expect}")
        sys.exit(1)

    # per-task cell sets
    cells: dict[str, dict[str, set[int]]] = {}
    for name, by_q in data.items():
        c = {"harm": set(), "save": set(), "both_bad": set(), "both_ok": set()}
        for q, d in by_q.items():
            cl = cell(q, d)
            if cl and cl in c:
                c[cl].add(q)
        cells[name] = c

    harm_union: set[int] = set()
    save_union: set[int] = set()
    both_bad_union: set[int] = set()
    for c in cells.values():
        harm_union |= c["harm"]
        save_union |= c["save"]
        both_bad_union |= c["both_bad"]

    # harm occurrence count per question (how many runs)
    harm_count: Counter[int] = Counter()
    for c in cells.values():
        for q in c["harm"]:
            harm_count[q] += 1

    n_tasks = len(task_names)
    # 每个 run 的反损集合的交集（题号在所有 task 上都是 harm）
    harm_inter_all = {q for q in harm_count if harm_count[q] == n_tasks}

    # Aggregate harm rows for "共性": all (task, q) harm pairs
    harm_lens: list[int] = []
    harm_reasons: Counter[str] = Counter()
    harm_base_lens: list[int] = []
    for name, by_q in data.items():
        for q in cells[name]["harm"]:
            d = by_q[q]
            harm_reasons[bucket_reason(d.get("judge_steer_reason") or "")] += 1
            ls = d.get("len_steer")
            if isinstance(ls, (int, float)):
                harm_lens.append(int(ls))
            lb = d.get("len_base")
            if isinstance(lb, (int, float)):
                harm_base_lens.append(int(lb))

    def pct_ge(xs: list[int], cap: int) -> float:
        if not xs:
            return 0.0
        return 100.0 * sum(1 for x in xs if x >= cap) / len(xs)

    harm_analysis = {
        "n_tasks": n_tasks,
        "task_names": task_names,
        "harm_per_task_counts": {n: len(cells[n]["harm"]) for n in task_names},
        "harm_union_size": len(harm_union),
        "harm_intersection_all_runs_size": len(harm_inter_all),
        "harm_intersection_all_runs_indices": sorted(harm_inter_all),
        "harm_runs_histogram": dict(
            sorted(Counter(harm_count.values()).items())
        ),  # k runs -> how many questions
        "harm_judge_steer_reason_bucket": dict(harm_reasons),
        "harm_len_steer": {
            "n": len(harm_lens),
            "min": min(harm_lens) if harm_lens else None,
            "median": int(statistics.median(harm_lens)) if harm_lens else None,
            "mean": round(statistics.mean(harm_lens), 1) if harm_lens else None,
            "max": max(harm_lens) if harm_lens else None,
            "pct_ge_4000": round(pct_ge(harm_lens, 4000), 2),
        },
        "harm_len_base": {
            "n": len(harm_base_lens),
            "median": int(statistics.median(harm_base_lens)) if harm_base_lens else None,
            "mean": round(statistics.mean(harm_base_lens), 1) if harm_base_lens else None,
            "pct_ge_4000": round(pct_ge(harm_base_lens, 4000), 2),
        },
    }

    prefix = args.output_prefix or os.path.join(root, "sweep_cell_pools")
    par = os.path.dirname(os.path.abspath(prefix))
    if par:
        os.makedirs(par, exist_ok=True)

    meta = {
        "description": "harm=基对Steer错; save=基错Steer对; both_bad=双错; 并集=任一run落入该格",
        "source_root": root,
        "expect_n": args.expect,
        "allow_partial": bool(args.allow_partial),
        "n_per_task": n_per,
        "harm_per_task": {n: sorted(cells[n]["harm"]) for n in task_names},
        "save_per_task": {n: sorted(cells[n]["save"]) for n in task_names},
        "both_bad_per_task": {n: sorted(cells[n]["both_bad"]) for n in task_names},
        "harm_union": sorted(harm_union),
        "save_union": sorted(save_union),
        "both_bad_union": sorted(both_bad_union),
        "union_three_cells": sorted(harm_union | save_union | both_bad_union),
        "harm_analysis": harm_analysis,
    }

    with open(prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    def write_idx(path: str, indices: list[int], label: str) -> None:
        payload = {
            "filter": label,
            "source_root": root,
            "n_indices": len(indices),
            "question_indices": indices,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    write_idx(prefix + "_harm.json", meta["harm_union"], "harm base_ok steer_wrong union across tasks")
    write_idx(prefix + "_save.json", meta["save_union"], "save base_wrong steer_ok union across tasks")
    write_idx(prefix + "_both_bad.json", meta["both_bad_union"], "both_bad union across tasks")
    write_idx(
        prefix + "_union_three.json",
        meta["union_three_cells"],
        "union of harm, save, both_bad across tasks",
    )

    print(json.dumps({"n_per_task": n_per, "harm_union": len(harm_union), "save_union": len(save_union), "both_bad_union": len(both_bad_union), "union_three": len(meta["union_three_cells"])}, indent=2, ensure_ascii=False))
    print("\n=== harm 跨 run 共性（摘要）===")
    print(f"  各 task 反损题数: {harm_analysis['harm_per_task_counts']}")
    print(f"  反损题号并集: {harm_analysis['harm_union_size']} 题")
    print(f"  四个 run 全是反损的交集: {harm_analysis['harm_intersection_all_runs_size']} 题 → {harm_analysis['harm_intersection_all_runs_indices']}")
    print(f"  反损出现次数分布（同一题在几个 run 里当反损）: {harm_analysis['harm_runs_histogram']}")
    print(f"  Steer 判题桶（harm 样本，按 run 展开）: {harm_analysis['harm_judge_steer_reason_bucket']}")
    print(f"  harm 的 len_steer: {harm_analysis['harm_len_steer']}")
    print("\n写出:")
    for suf in ("_meta.json", "_harm.json", "_save.json", "_both_bad.json", "_union_three.json"):
        print(" ", prefix + suf)


if __name__ == "__main__":
    main()
