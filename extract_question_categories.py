#!/usr/bin/env python3
"""
从 log_worker_*.txt 解析每道题的 Base/Steer 结果，输出：
- Base错Steer错（都错）
- Base对Steer错
- Base错Steer对
用法: python extract_question_categories.py --log_dir LOG_DIR [--num_total 500]
"""
import re
import argparse
import glob
import os


def parse_per_question_results(content: str) -> list[dict]:
    """解析所有 Base/Steer 对，返回 [{base_correct, steer_correct}, ...]"""
    pattern = re.compile(
        r"Base\s+\[(✅|❌)\]\s+Len:\s*\d+\s*\n\s*Steer\s+\[(✅|❌)\]\s+Len:\s*\d+",
        re.MULTILINE,
    )
    results = []
    for m in pattern.finditer(content):
        base_ok = 1 if m.group(1) == "✅" else 0
        steer_ok = 1 if m.group(2) == "✅" else 0
        results.append({"base_correct": base_ok, "steer_correct": steer_ok})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="包含 log_worker_*.txt 的目录")
    parser.add_argument("--num_total", type=int, default=500, help="总题目数")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径")
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    pattern = os.path.join(log_dir, "log_worker_*.txt")

    def _shard_num(p):
        m = re.search(r"log_worker_(\d+)\.txt", os.path.basename(p))
        return int(m.group(1)) if m else 999

    files = sorted(glob.glob(pattern), key=_shard_num)
    if not files:
        print(f"Error: No log_worker_*.txt in {log_dir}")
        return 1

    num_shards = len(files)
    # 每个 shard 处理的 indices: i % num_shards == shard_id
    all_results = {}  # idx -> {base_correct, steer_correct}

    for fp in files:
        m = re.search(r"log_worker_(\d+)\.txt", os.path.basename(fp))
        shard_id = int(m.group(1)) if m else 0
        indices = [i for i in range(args.num_total) if i % num_shards == shard_id]

        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        results = parse_per_question_results(content)

        if len(results) != len(indices):
            print(f"Warning: {fp} has {len(results)} results but expected {len(indices)} indices")
        for k, r in enumerate(results):
            idx = indices[k] if k < len(indices) else -1
            all_results[idx] = r

    # 按 idx 排序
    sorted_indices = sorted(all_results.keys())
    if -1 in all_results:
        del all_results[-1]

    both_wrong = [i for i in sorted_indices if all_results[i]["base_correct"] == 0 and all_results[i]["steer_correct"] == 0]
    base_ok_steer_wrong = [i for i in sorted_indices if all_results[i]["base_correct"] == 1 and all_results[i]["steer_correct"] == 0]
    base_wrong_steer_ok = [i for i in sorted_indices if all_results[i]["base_correct"] == 0 and all_results[i]["steer_correct"] == 1]

    print("=" * 60)
    print("500 道题分类结果")
    print("=" * 60)
    print(f"\n1. Base错Steer错（都错）: {len(both_wrong)} 题")
    print("   ", both_wrong)
    print(f"\n2. Base对Steer错: {len(base_ok_steer_wrong)} 题")
    print("   ", base_ok_steer_wrong)
    print(f"\n3. Base错Steer对: {len(base_wrong_steer_ok)} 题")
    print("   ", base_wrong_steer_ok)
    print("=" * 60)

    if args.output:
        import json
        out = {
            "both_wrong": both_wrong,
            "base_ok_steer_wrong": base_ok_steer_wrong,
            "base_wrong_steer_ok": base_wrong_steer_ok,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已保存到 {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
