#!/usr/bin/env python3
"""
汇总 run_basic_overwrite 在 --save_judge_reason 下写出的 question_{idx}_shard_{s}.json，
对 Steer 判错做分类：未抽取(parse 空) vs verify 认为不等价 vs 其它。

典型 judge_steer_reason 前缀（utils/rule_eval.evaluate_math）：
  rule_math:empty_pred_parse   — math_verify 从模型输出里抽不出可验证表达式（无「可验证答案」）
  rule_math:math_verify_false  — 抽到了，但与 gold 不等价
  rule_math:math_verify_missing  — 本机未装上 math_verify
  rule_math:no_gold — 数据无 gold

用法:
  python analyze_math500_steer_judge_json.py JUDGE_DIR
  python analyze_math500_steer_judge_json.py JUDGE_DIR --list-unverifiable
  python analyze_math500_steer_judge_json.py JUDGE_DIR --write-report report.json
  python analyze_math500_steer_judge_json.py JUDGE_DIR --merge-answers all_answers.json

说明: 历史作业若未开 --save_judge_reason，请重跑；run_math500_steer_overwrite.slurm 已默认 SAVE_JUDGE_REASON=1。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from glob import glob


def bucket_steer_reason(reason: str, steer_ok: bool) -> str:
    if steer_ok:
        return "steer_ok"
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


def steer_has_no_verifiable_answer(reason: str) -> bool:
    """math_verify 无法从 Steer 输出中 parse 出可验证对象（与判错标签 empty_pred_parse 一致）。"""
    r = (reason or "").strip()
    return "empty_pred_parse" in r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("judge_dir", help="含 question_*_shard_*.json 的目录（如 LOG_DIR/judge）")
    ap.add_argument("--list-empty", action="store_true", help="打印 empty_pred_parse 的题号")
    ap.add_argument("--list-false", action="store_true", help="打印 math_verify_false 的题号")
    ap.add_argument(
        "--list-unverifiable",
        action="store_true",
        help="打印 Steer 侧「无 math_verify 可解析答案」(empty_pred_parse) 的题号",
    )
    ap.add_argument(
        "--write-report",
        metavar="PATH",
        help="写出汇总 JSON（分类计数 + 不可验证题列表 + 可选摘要字段）",
    )
    ap.add_argument(
        "--merge-answers",
        metavar="PATH",
        help="合并目录下所有 judge JSON 为单个文件（含 response_base/steer 若存在）",
    )
    ap.add_argument(
        "--only-indices-json",
        metavar="PATH",
        help="仅统计题号在该 JSON 的 question_indices 中的 judge（如 export_math500_not_both_ok_indices.py 产出）",
    )
    args = ap.parse_args()

    only_q: set[int] | None = None
    if args.only_indices_json:
        ip = os.path.abspath(args.only_indices_json)
        with open(ip, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            only_q = set(int(x) for x in data)
        elif isinstance(data, dict) and "question_indices" in data:
            only_q = set(int(x) for x in data["question_indices"])
        else:
            raise SystemExit(f"--only-indices-json 需为数组或含 question_indices 的对象: {ip}")

    paths = sorted(glob(os.path.join(args.judge_dir, "question_*_shard_*.json")))
    if not paths:
        print(f"未找到 JSON: {args.judge_dir}")
        print("请确认已传 --save_judge_reason；run_math500_steer_overwrite.slurm 默认 SAVE_JUDGE_REASON=1。")
        return

    steer_bucket = Counter()
    base_steer_cell = Counter()  # (b_ok, s_ok) -> count
    lists: dict[str, list[int]] = defaultdict(list)
    records: list[dict] = []
    unverifiable_entries: list[dict] = []

    missing_base_meta = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        qid = d.get("question_idx")
        if qid is None:
            m = re.search(r"question_(\d+)_shard_", os.path.basename(p))
            if m:
                qid = int(m.group(1))
        if only_q is not None:
            if qid is None or qid not in only_q:
                continue
        records.append(d)
        s_ok = bool(d.get("steer_correct"))
        reason = d.get("judge_steer_reason") or ""
        if "base_correct" in d:
            b_ok = bool(d.get("base_correct"))
            base_steer_cell[(b_ok, s_ok)] += 1
        else:
            missing_base_meta += 1
        b = bucket_steer_reason(reason, s_ok)
        steer_bucket[b] += 1
        if b == "empty_pred_parse":
            lists["empty"].append(qid)
        if b == "math_verify_false":
            lists["false"].append(qid)
        if steer_has_no_verifiable_answer(reason):
            ent = {
                "question_idx": qid,
                "shard_json": os.path.basename(p),
                "judge_steer_reason": reason,
                "steer_correct": s_ok,
                "len_steer": d.get("len_steer"),
            }
            rs = d.get("response_steer")
            if isinstance(rs, str) and rs:
                ent["response_steer_preview_tail"] = rs[-400:] if len(rs) > 400 else rs
            unverifiable_entries.append(ent)

    n = len(records)
    print(f"JSON 文件数: {n} (目录: {args.judge_dir})")
    if only_q is not None:
        print(f"  （已按 --only-indices-json 过滤；目录下原共 {len(paths)} 个 judge 文件）")
    print()
    print("Steer 判题分类（在 steer_ok 之外，仅统计 Steer 为错的题）:")
    steer_wrong = n - steer_bucket["steer_ok"]
    if steer_wrong <= 0:
        print("  (无 Steer 错题)")
    else:
        for key in ("empty_pred_parse", "math_verify_false", "math_verify_missing", "no_gold", "other"):
            c = steer_bucket[key]
            if c:
                print(f"  {key}: {c} ({100 * c / steer_wrong:.1f}% of Steer错)")

    nu = len(unverifiable_entries)
    print()
    print(f"Steer 无 math_verify 可解析答案 (rule_math:empty_pred_parse): {nu} 题")

    print()
    if missing_base_meta:
        print(
            f"注意: {missing_base_meta} 条 JSON 无 base_correct（例如 steer_only 重跑默认不写 prior base 元数据）；"
            "列联表仅统计含 base_correct 的文件。需要列联请对 run_basic_overwrite 加 --save_judge_prior_base_meta。"
        )
        print()
    print("Base×Steer 列联（与 summarize_math500_steer_logs 一致）:")
    for b_ok in (True, False):
        for s_ok in (True, False):
            c = base_steer_cell[(b_ok, s_ok)]
            label = f"Base={'对' if b_ok else '错'} Steer={'对' if s_ok else '错'}"
            print(f"  {label}: {c}")

    if args.list_empty and lists["empty"]:
        print("\nempty_pred_parse 题号:", sorted(x for x in lists["empty"] if x is not None))
    if args.list_false and lists["false"]:
        print("\nmath_verify_false 题号:", sorted(x for x in lists["false"] if x is not None))
    if args.list_unverifiable:
        ids = sorted({e["question_idx"] for e in unverifiable_entries if e.get("question_idx") is not None})
        print("\nSteer 无可解析验证对象 (empty_pred_parse) 题号:", ids)

    if args.write_report:
        report = {
            "judge_dir": os.path.abspath(args.judge_dir),
            "n_json_files": n,
            "steer_bucket_counts": dict(steer_bucket),
            "base_steer_crosstab": {
                f"base_{b_ok}_steer_{s_ok}": base_steer_cell[(b_ok, s_ok)]
                for b_ok in (True, False)
                for s_ok in (True, False)
            },
            "steer_no_verifiable_parse": {
                "definition": "judge_steer_reason 含 empty_pred_parse：math_verify.parse(steer_output) 为空",
                "count_questions": nu,
                "question_indices": sorted(
                    {e["question_idx"] for e in unverifiable_entries if e.get("question_idx") is not None}
                ),
                "entries": sorted(unverifiable_entries, key=lambda e: (e.get("question_idx") is None, e.get("question_idx", -1))),
            },
        }
        _out = os.path.abspath(args.write_report)
        _par = os.path.dirname(_out)
        if _par:
            os.makedirs(_par, exist_ok=True)
        with open(_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n已写报告: {args.write_report}")

    if args.merge_answers:
        # 同 question_idx 多 shard 时不应发生；若发生保留首个
        by_q: dict[int, dict] = {}
        for d in sorted(records, key=lambda x: (x.get("question_idx") is None, x.get("question_idx", -1))):
            q = d.get("question_idx")
            if q is None:
                continue
            if q not in by_q:
                by_q[q] = d
        items = [by_q[k] for k in sorted(by_q.keys())]
        merged = {
            "meta": {
                "source_dir": os.path.abspath(args.judge_dir),
                "num_items": len(items),
                "note": "每题一条；字段来自 run_basic_overwrite --save_judge_reason [--save_judge_with_text]",
            },
            "items": items,
        }
        out_path = os.path.abspath(args.merge_answers)
        _mpar = os.path.dirname(out_path)
        if _mpar:
            os.makedirs(_mpar, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"已合并答案 JSON: {out_path} （{len(items)} 题）")


if __name__ == "__main__":
    main()
