#!/usr/bin/env python3
"""
根据当前 utils.rule_eval.extract_choice_letter / evaluate_gpqa，离线重写 judge_reasons 里
GPQA MCQ 相关字段（无需重跑 GPU）。

用法:
  cd thinking-llms-interp
  python scripts/refresh_gpqa_judge_json.py log2/gpqa_diamond_aime_mcq_130313/judge_reasons
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 仓库根目录；直接加载 rule_eval，避免 import utils → nnsight 等重型依赖。
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPORT_ERR: Exception | None = None
extract_choice_letter = evaluate_gpqa = None  # type: ignore
try:
    import importlib.util

    _path = os.path.join(_REPO, "utils", "rule_eval.py")
    _spec = importlib.util.spec_from_file_location("_rule_eval_refresh", _path)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        extract_choice_letter = _mod.extract_choice_letter
        evaluate_gpqa = _mod.evaluate_gpqa
except Exception as _e:
    _IMPORT_ERR = _e


def _refresh_payload(data: dict) -> bool:
    gold = data.get("gold_letter")
    if not gold:
        return False

    changed = False
    rb = data.get("response_base")
    if rb is not None:
        ex_b = extract_choice_letter(rb)
        ok_b, reason_b = evaluate_gpqa(rb, gold)
        if data.get("extracted_letter_base") != ex_b:
            data["extracted_letter_base"] = ex_b
            changed = True
        if data.get("judge_base_reason") != reason_b:
            data["judge_base_reason"] = reason_b
            changed = True
        if data.get("base_correct") != ok_b:
            data["base_correct"] = ok_b
            changed = True

    rs = data.get("response_steer")
    if rs is not None:
        ex_s = extract_choice_letter(rs)
        ok_s, reason_s = evaluate_gpqa(rs, gold)
        if data.get("extracted_letter_steer") != ex_s:
            data["extracted_letter_steer"] = ex_s
            changed = True
        if data.get("judge_steer_reason") != reason_s:
            data["judge_steer_reason"] = reason_s
            changed = True
        if data.get("steer_correct") != ok_s:
            data["steer_correct"] = ok_s
            changed = True

    return changed


def main() -> None:
    if _IMPORT_ERR is not None or extract_choice_letter is None:
        print(f"failed to load utils/rule_eval.py: {_IMPORT_ERR}", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser(description="Refresh GPQA MCQ judge JSON fields using current rule_eval.")
    p.add_argument(
        "judge_dir",
        nargs="?",
        default=None,
        help="judge_reasons 目录（默认可省略时用 cwd）",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印会改的文件，不写回")
    args = p.parse_args()

    judge_dir = args.judge_dir
    if not judge_dir:
        print("usage: python scripts/refresh_gpqa_judge_json.py /path/to/judge_reasons", file=sys.stderr)
        sys.exit(2)
    judge_dir = os.path.abspath(judge_dir)
    if not os.path.isdir(judge_dir):
        print(f"not a directory: {judge_dir}", file=sys.stderr)
        sys.exit(1)

    updated = 0
    for name in sorted(os.listdir(judge_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(judge_dir, name)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if _refresh_payload(data):
            updated += 1
            if args.dry_run:
                print(f"would update: {path}")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                print(f"updated: {path}")

    print(f"done. files changed: {updated}", file=sys.stderr)


if __name__ == "__main__":
    main()
