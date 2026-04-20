#!/usr/bin/env python3
"""
从 run_basic_overwrite 写出的 judge/*.json 提取「非 (Base 与 Steer 都对)」的题号，
供 --example_idx 或 --example_idx_file 复跑。

用法:
  python export_math500_not_both_ok_indices.py JUDGE_DIR
  python export_math500_not_both_ok_indices.py JUDGE_DIR -o /path/prefix

默认要求 judge 内至少 500 道题有 verdict 才写出；未满会退出（避免未跑完就抽题单）。
若要未跑完也导出: 加 --allow-partial。其它题量: --require-n-verdicts N。

默认写出:
  <JUDGE_DIR>/../math500_not_both_ok_indices.json
  <JUDGE_DIR>/../math500_not_both_ok_indices.txt   （每行一题号，# 开头为注释）
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("judge_dir", help="含 question_*_shard_*.json 的目录")
    ap.add_argument(
        "-o",
        "--output-prefix",
        default=None,
        metavar="PREFIX",
        help="输出文件前缀（不写扩展名）。默认写到 judge 的父目录: math500_not_both_ok_indices",
    )
    ap.add_argument(
        "--require-n-verdicts",
        type=int,
        default=500,
        metavar="N",
        help="至少 N 道题有 verdict 才写出；默认 500（MATH-500）",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="未满 --require-n-verdicts 仍导出（会 union 缺题号，一般仅调试用）",
    )
    args = ap.parse_args()

    judge_dir = os.path.abspath(args.judge_dir)
    if not os.path.isdir(judge_dir):
        raise SystemExit(f"不是目录: {judge_dir}")

    paths = sorted(glob(os.path.join(judge_dir, "question_*_shard_*.json")))
    if not paths:
        raise SystemExit(f"未找到 JSON: {judge_dir}")

    verdicts = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        idx = d.get("question_idx")
        if idx is None:
            continue
        verdicts[idx] = (bool(d.get("base_correct")), bool(d.get("steer_correct")))

    n_v = len(verdicts)
    req = args.require_n_verdicts
    if req > 0 and n_v < req and not args.allow_partial:
        raise SystemExit(
            f"判题未满 {req} 道（当前有 verdict 的题号数={n_v}），请跑完再导出题单；"
            f"若确需半成品，请加 --allow-partial。"
        )

    both_ok = [i for i, (b, s) in sorted(verdicts.items()) if b and s]
    not_both_ok = [i for i, (b, s) in sorted(verdicts.items()) if not (b and s)]

    # 有 verdict 的题里未出现的 [0,500) 仍视为「需跑」（与 filter_from_judge_dir 一致）
    n_ds = 500
    missing = [i for i in range(n_ds) if i not in verdicts]
    if missing:
        not_both_ok = sorted(set(not_both_ok) | set(missing))

    prefix = args.output_prefix
    if not prefix:
        parent = os.path.dirname(judge_dir.rstrip(os.sep))
        prefix = os.path.join(parent, "math500_not_both_ok_indices")

    json_path = prefix + ".json"
    txt_path = prefix + ".txt"

    payload = {
        "filter": "not (base_correct and steer_correct); union missing_idx in [0,499) if any",
        "source_judge_dir": judge_dir,
        "n_verdicts_in_dir": len(verdicts),
        "n_both_ok": len(both_ok),
        "n_not_both_ok": len(not_both_ok),
        "n_missing_verdict": len(missing),
        "question_indices": not_both_ok,
        "question_indices_not_both_ok": not_both_ok,
        "question_indices_both_ok_only": both_ok,
        "run_basic_overwrite_example": (
            "python run_basic_overwrite.py --dataset math500 "
            f"--example_idx_file {json_path} ...  # 或使用 --example_idx 空格分隔题号"
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# not_both_ok count={len(not_both_ok)} judge={judge_dir}\n")
        for i in not_both_ok:
            f.write(f"{i}\n")

    print(json.dumps({k: payload[k] for k in ("n_verdicts_in_dir", "n_both_ok", "n_not_both_ok", "n_missing_verdict")}, indent=2))
    print("Wrote", json_path)
    print("Wrote", txt_path)


if __name__ == "__main__":
    main()
