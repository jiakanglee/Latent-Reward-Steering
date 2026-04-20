#!/usr/bin/env python3
"""
从 log_worker_*.txt 解析 Base对Steer错、Base错Steer对 的题目 ID，
并可选输出 JSON（含 problem/solution/answer）供专门评估与调参。
用法: python extract_edge_cases.py --log_dir LOG_DIR [--output_json OUTPUT_JSON] [--dataset DATASET]
"""
import re
import argparse
import glob
import os
import json


def parse_per_question_with_ids(content: str) -> list[dict]:
    """解析所有题目，返回 [{question_id, base_correct, steer_correct}, ...]"""
    base_steer_pattern = re.compile(
        r"Base\s+\[(✅|❌)\]\s+Len:\s*\d+\s*\n\s*Steer\s+\[(✅|❌)\]\s+Len:\s*\d+",
        re.MULTILINE,
    )
    # 找所有 "======== Question N ========" 的位置
    question_markers = list(re.finditer(r"======== Question (\d+) ========", content))
    results = []
    for i, qm in enumerate(question_markers):
        qid = int(qm.group(1))
        start = qm.end()
        end = question_markers[i + 1].start() if i + 1 < len(question_markers) else len(content)
        block = content[start:end]
        sm = base_steer_pattern.search(block)
        if sm:
            base_ok = 1 if sm.group(1) == "✅" else 0
            steer_ok = 1 if sm.group(2) == "✅" else 0
            results.append({"question_id": qid, "base_correct": base_ok, "steer_correct": steer_ok})
    return results


def collect_from_log_dir(log_dir: str) -> tuple[list[dict], dict]:
    """从 log_dir 下的 log_worker_*.txt 收集所有题目结果（含 question_id）"""
    log_dir = os.path.abspath(log_dir)
    pattern = os.path.join(log_dir, "log_worker_*.txt")

    def _shard_num(p):
        m = re.search(r"log_worker_(\d+)\.txt", os.path.basename(p))
        return int(m.group(1)) if m else 999

    files = sorted(glob.glob(pattern), key=_shard_num)
    if not files:
        return [], {"error": f"No log_worker_*.txt in {log_dir}"}

    all_results = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        all_results.extend(parse_per_question_with_ids(content))

    # 按 question_id 排序去重（多 shard 不应重复，但保险起见取第一个）
    by_id = {}
    for r in all_results:
        qid = r["question_id"]
        if qid not in by_id:
            by_id[qid] = r
    all_results = [by_id[q] for q in sorted(by_id.keys())]

    agg = {
        "total": len(all_results),
        "base_right_steer_wrong_ids": [r["question_id"] for r in all_results if r["base_correct"] == 1 and r["steer_correct"] == 0],
        "base_wrong_steer_right_ids": [r["question_id"] for r in all_results if r["base_correct"] == 0 and r["steer_correct"] == 1],
        "base_right_steer_wrong": 0,
        "base_wrong_steer_right": 0,
    }
    agg["base_right_steer_wrong"] = len(agg["base_right_steer_wrong_ids"])
    agg["base_wrong_steer_right"] = len(agg["base_wrong_steer_right_ids"])
    agg["edge_case_ids"] = sorted(set(agg["base_right_steer_wrong_ids"]) | set(agg["base_wrong_steer_right_ids"]))

    return all_results, agg


def load_dataset_items(dataset_name: str, indices: list[int]) -> list[dict]:
    """从 HuggingFace 数据集加载指定索引的题目"""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    ds = load_dataset(dataset_name, split="test")
    items = []
    for i in indices:
        if 0 <= i < len(ds):
            items.append({"question_id": i, **ds[i]})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="包含 log_worker_*.txt 的目录")
    parser.add_argument("--output_json", type=str, default=None, help="输出 JSON 路径，含 problem/solution/answer")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", help="数据集名")
    parser.add_argument("--example_idx_only", action="store_true", help="仅打印 --example_idx 参数格式")
    args = parser.parse_args()

    all_results, agg = collect_from_log_dir(args.log_dir)
    if "error" in agg:
        print(agg["error"])
        return 1

    print("=" * 60)
    print("Base对Steer错 题目 ID:", agg["base_right_steer_wrong_ids"])
    print("Base错Steer对 题目 ID:", agg["base_wrong_steer_right_ids"])
    print("边例题目（共 24 题，用于调参）:", agg["edge_case_ids"])
    print("=" * 60)

    if args.example_idx_only:
        ids = agg["edge_case_ids"]
        print(f"\n--example_idx {' '.join(map(str, ids))}")

    if args.output_json:
        edge_ids = agg["edge_case_ids"]
        items = load_dataset_items(args.dataset, edge_ids)
        questions_dict = {}
        if items:
            questions_dict = {str(it["question_id"]): {"problem": it["problem"], "solution": it["solution"], "answer": it["answer"]} for it in items}
        out = {
            "base_right_steer_wrong_ids": agg["base_right_steer_wrong_ids"],
            "base_wrong_steer_right_ids": agg["base_wrong_steer_right_ids"],
            "edge_case_ids": edge_ids,
            "questions": questions_dict,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已保存到 {args.output_json}")

    return 0


if __name__ == "__main__":
    exit(main())
