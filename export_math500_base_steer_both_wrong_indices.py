#!/usr/bin/env python3
"""
从 run_basic_overwrite 写出的 judge/*.json 提取「Base 错 且 Steer 也错」(双错) 的题号，
供 --example_idx_file 在 max_token=4000 等设定下做子集实验。

要求每条 JSON 同时含 base_correct 与 steer_correct（全量 Base+Steer 跑出的 judge；steer_only 无 base 时会报错）。

用法:
  python export_math500_base_steer_both_wrong_indices.py JUDGE_DIR
  python export_math500_base_steer_both_wrong_indices.py JUDGE_DIR -o /path/prefix

默认写出:
  <JUDGE_DIR>/../math500_base_steer_both_wrong_indices.json
  <JUDGE_DIR>/../math500_base_steer_both_wrong_indices.txt
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
        help="输出文件前缀（不写扩展名）。默认写到 judge 父目录: math500_base_steer_both_wrong_indices",
    )
    args = ap.parse_args()

    judge_dir = os.path.abspath(args.judge_dir)
    if not os.path.isdir(judge_dir):
        raise SystemExit(f"不是目录: {judge_dir}")

    paths = sorted(glob(os.path.join(judge_dir, "question_*_shard_*.json")))
    if not paths:
        raise SystemExit(f"未找到 JSON: {judge_dir}")

    verdicts: dict[int, tuple[bool, bool]] = {}
    missing_fields: list[str] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        idx = d.get("question_idx")
        if idx is None:
            continue
        if "base_correct" not in d or "steer_correct" not in d:
            missing_fields.append(os.path.basename(p))
            continue
        verdicts[idx] = (bool(d["base_correct"]), bool(d["steer_correct"]))

    if missing_fields:
        raise SystemExit(
            f"以下 JSON 缺少 base_correct 或 steer_correct（steer_only 的 judge 不能用本脚本）:\n"
            + "\n".join(missing_fields[:20])
            + (f"\n... 共 {len(missing_fields)} 个" if len(missing_fields) > 20 else "")
        )

    both_wrong = [i for i, (b, s) in sorted(verdicts.items()) if (not b) and (not s)]
    both_ok = [i for i, (b, s) in sorted(verdicts.items()) if b and s]
    save_only = [i for i, (b, s) in sorted(verdicts.items()) if (not b) and s]
    hurt_only = [i for i, (b, s) in sorted(verdicts.items()) if b and (not s)]

    prefix = args.output_prefix
    if not prefix:
        parent = os.path.dirname(judge_dir.rstrip(os.sep))
        prefix = os.path.join(parent, "math500_base_steer_both_wrong_indices")

    json_path = prefix + ".json"
    txt_path = prefix + ".txt"

    payload = {
        "filter": "(not base_correct) and (not steer_correct); only indices with verdict JSON",
        "source_judge_dir": judge_dir,
        "n_verdicts_in_dir": len(verdicts),
        "n_both_wrong": len(both_wrong),
        "n_both_ok": len(both_ok),
        "n_save_only_base_wrong_steer_ok": len(save_only),
        "n_hurt_only_base_ok_steer_wrong": len(hurt_only),
        "question_indices": both_wrong,
        "question_indices_both_wrong": both_wrong,
        "run_basic_overwrite_example": (
            "python run_basic_overwrite.py --dataset math500 "
            f"--example_idx_file {json_path} ...  # 与全量相同的 max_token/reward 等"
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# base_wrong and steer_wrong count={len(both_wrong)} judge={judge_dir}\n")
        for i in both_wrong:
            f.write(f"{i}\n")

    print(
        json.dumps(
            {
                "n_verdicts_in_dir": len(verdicts),
                "n_both_wrong": len(both_wrong),
                "n_both_ok": len(both_ok),
                "n_save_only": len(save_only),
                "n_hurt_only": len(hurt_only),
            },
            indent=2,
        )
    )
    print("Wrote", json_path)
    print("Wrote", txt_path)


if __name__ == "__main__":
    main()
