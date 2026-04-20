#!/usr/bin/env python3
"""
合并多个 log_worker_*.txt 到 logs/iter_steer_{job_id}.out，
并解析各 worker 的 Final Report 生成整合的 Final Report。
用法: python merge_worker_logs.py [--job_id JOB_ID] [--log_dir LOG_DIR]
"""
import re
import argparse
import glob
import os


def parse_final_report(content: str) -> dict | None:
    """从 worker 输出中解析 Final Report，返回 dict 或 None"""
    # 匹配 Final Report 块
    match = re.search(
        r"📊 Final Report \(Step=([\d.]+), Iter=(\d+)\)\s+Accuracy:\s+"
        r"Base : (\d+)/(\d+)\s+Steer: (\d+)/(\d+)\s+"
        r"Average Token Length \(Correct Answers Only\):\s+"
        r"Base : ([\d.]+) tokens\s+Steer: ([\d.]+) tokens",
        content,
        re.DOTALL,
    )
    if not match:
        return None
    return {
        "step_size": float(match.group(1)),
        "num_steps": int(match.group(2)),
        "base_correct": int(match.group(3)),
        "total": int(match.group(4)),
        "steer_correct": int(match.group(5)),
        "steer_total": int(match.group(6)),
        "avg_base_len": float(match.group(7)),
        "avg_steer_len": float(match.group(8)),
    }


def merge_logs(log_dir: str, job_id: str | None, output_path: str | None) -> str:
    """
    合并 log_worker_*.txt 到输出文件，并追加整合的 Final Report。
    返回输出文件路径。
    """
    log_dir = os.path.abspath(log_dir)
    if output_path is None:
        output_path = os.path.join(log_dir, f"iter_steer_{job_id or 'merged'}.out")

    # 按 shard 顺序收集 worker 文件
    pattern = os.path.join(log_dir, "log_worker_*.txt")
    def _shard_num(p):
        m = re.search(r"log_worker_(\d+)\.txt", os.path.basename(p))
        return int(m.group(1)) if m else 999
    files = sorted(glob.glob(pattern), key=_shard_num)

    if not files:
        raise FileNotFoundError(f"No log_worker_*.txt found in {log_dir}")

    reports = []
    with open(output_path, "w", encoding="utf-8") as out:
        for i, fp in enumerate(files):
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # 写入分片内容（带分隔头）
            shard_id = re.search(r"log_worker_(\d+)\.txt", fp)
            shard_id = shard_id.group(1) if shard_id else str(i)
            out.write("\n")
            out.write("=" * 60 + f"\n>>> Shard {shard_id} (from {os.path.basename(fp)})\n" + "=" * 60 + "\n\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")

            # 解析 Final Report
            r = parse_final_report(content)
            if r:
                r["shard_id"] = shard_id
                reports.append(r)

    # 整合 Final Report
    if reports:
        total_base = sum(r["base_correct"] for r in reports)
        total_steer = sum(r["steer_correct"] for r in reports)
        total_tasks = sum(r["total"] for r in reports)
        step_size = reports[0]["step_size"]
        num_steps = reports[0]["num_steps"]

        # 加权平均 token 长度（按正确数加权）
        if total_base > 0:
            avg_base = sum(r["avg_base_len"] * r["base_correct"] for r in reports) / total_base
        else:
            avg_base = 0.0
        if total_steer > 0:
            avg_steer = sum(r["avg_steer_len"] * r["steer_correct"] for r in reports) / total_steer
        else:
            avg_steer = 0.0

        delta = avg_steer - avg_base

        with open(output_path, "a", encoding="utf-8") as out:
            out.write("\n\n")
            out.write("=" * 60 + "\n")
            out.write("📊 COMBINED Final Report (all shards)\n")
            out.write("=" * 60 + "\n")
            out.write(f"Accuracy:\n")
            out.write(f"  Base : {total_base}/{total_tasks}\n")
            out.write(f"  Steer: {total_steer}/{total_tasks}\n")
            out.write(f"\nAverage Token Length (Correct Answers Only):\n")
            out.write(f"  Base : {avg_base:.1f} tokens\n")
            out.write(f"  Steer: {avg_steer:.1f} tokens\n")
            if delta > 0:
                out.write(f"📈 On average, steering increased thinking by +{delta:.1f} tokens.\n")
            else:
                out.write(f"📉 On average, steering decreased thinking by {delta:.1f} tokens.\n")
            out.write("=" * 60 + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default=None, help="SLURM job ID for output filename")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing log_worker_*.txt")
    parser.add_argument("--output", type=str, default=None, help="Output file path (overrides job_id)")
    args = parser.parse_args()

    job_id = args.job_id or os.environ.get("SLURM_JOB_ID")
    path = merge_logs(args.log_dir, job_id, args.output)
    print(f"✅ Merged logs saved to {path}")


if __name__ == "__main__":
    main()
