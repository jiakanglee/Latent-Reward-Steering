#!/usr/bin/env python3
"""
从 MBPP log_worker_*.txt 解析出：
- Initial对 Steer错 (Base对Steer错)
- Initial错 Steer对 (Base错Steer对)
- 两个都错
的题号列表。用于后续在少量题目上做参数 sweep。
"""
import re
import argparse
import glob
import os


def parse_per_question_results_with_indices(content: str, shard_id: int, num_shards: int) -> list[dict]:
    """解析所有 Base/Steer 对，并计算对应的题号"""
    pattern = re.compile(
        r"Base\s+\[(✅|❌)\]\s+Len:\s*\d+\s*\n\s*Steer\s+\[(✅|❌)\]\s+Len:\s*\d+",
        re.MULTILINE,
    )
    results = []
    for k, m in enumerate(pattern.finditer(content)):
        base_ok = 1 if m.group(1) == "✅" else 0
        steer_ok = 1 if m.group(2) == "✅" else 0
        q_idx = k * num_shards + shard_id
        results.append({"q_idx": q_idx, "base_correct": base_ok, "steer_correct": steer_ok})
    return results


def collect_from_log_dir(log_dir: str, num_shards: int = 4) -> dict:
    """从 log_dir 收集所有题目结果，按类别分组题号"""
    log_dir = os.path.abspath(log_dir)

    def _shard_num(p):
        m = re.search(r"log_worker_(\d+)\.txt", os.path.basename(p))
        return int(m.group(1)) if m else 999

    files = sorted(glob.glob(os.path.join(log_dir, "log_worker_*.txt")), key=_shard_num)
    if not files:
        return {"error": f"No log_worker_*.txt in {log_dir}"}

    all_results = []
    for fp in files:
        shard_id = _shard_num(fp)
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        all_results.extend(parse_per_question_results_with_indices(content, shard_id, num_shards))

    # 按题号排序（多 shard 合并后可能乱序）
    all_results.sort(key=lambda x: x["q_idx"])

    base_right_steer_wrong = [r["q_idx"] for r in all_results if r["base_correct"] == 1 and r["steer_correct"] == 0]
    base_wrong_steer_right = [r["q_idx"] for r in all_results if r["base_correct"] == 0 and r["steer_correct"] == 1]
    both_wrong = [r["q_idx"] for r in all_results if r["base_correct"] == 0 and r["steer_correct"] == 0]
    both_right = [r["q_idx"] for r in all_results if r["base_correct"] == 1 and r["steer_correct"] == 1]

    return {
        "base_right_steer_wrong": base_right_steer_wrong,
        "base_wrong_steer_right": base_wrong_steer_right,
        "both_wrong": both_wrong,
        "both_right": both_right,
        "total": len(all_results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="包含 log_worker_*.txt 的目录")
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="输出题号到文件（--example_idx 格式）")
    parser.add_argument("--categories", type=str, nargs="+",
                        default=["base_right_steer_wrong", "base_wrong_steer_right", "both_wrong"],
                        help="要合并的类别")
    args = parser.parse_args()

    d = collect_from_log_dir(args.log_dir, args.num_shards)
    if "error" in d:
        print(d["error"])
        return 1

    print("=== MBPP 边例/错例题号 ===")
    print(f"Initial对 Steer错 (Base对Steer错): {len(d['base_right_steer_wrong'])} 题")
    print(d["base_right_steer_wrong"])
    print()
    print(f"Initial错 Steer对 (Base错Steer对): {len(d['base_wrong_steer_right'])} 题")
    print(d["base_wrong_steer_right"])
    print()
    print(f"两个都错: {len(d['both_wrong'])} 题")
    print(d["both_wrong"])
    print()
    print(f"两个都对: {len(d['both_right'])} 题 (不用于 sweep)")

    # 合并指定类别的题号，去重排序
    merged = []
    for cat in args.categories:
        if cat in d:
            merged.extend(d[cat])
    merged = sorted(set(merged))

    print()
    print(f"合并后共 {len(merged)} 题 (用于 sweep): {merged}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(" ".join(str(x) for x in merged))
        print(f"\n已写入 --example_idx 格式到 {args.output}")
        print(f"用法: --example_idx $(cat {args.output})")

    return 0


if __name__ == "__main__":
    exit(main())
