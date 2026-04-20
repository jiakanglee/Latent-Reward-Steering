#!/usr/bin/env python3
"""
从 log_worker_*.txt 或合并后的输出中解析每道题的结果，
统计 Base对Steer错、Base错Steer对 等数量。
用法: python collect_sweep_results.py --log_dir LOG_DIR [--output OUTPUT]
"""
import re
import argparse
import glob
import os


def parse_per_question_results(content: str) -> list[dict]:
    """解析所有 Base/Steer 对，返回 [{base_correct, steer_correct}, ...]"""
    # 匹配连续的 Base 和 Steer 行
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


def collect_from_log_dir(log_dir: str) -> dict:
    """从 log_dir 下的 log_worker_*.txt 收集所有题目结果"""
    log_dir = os.path.abspath(log_dir)
    pattern = os.path.join(log_dir, "log_worker_*.txt")

    def _shard_num(p):
        m = re.search(r"log_worker_(\d+)\.txt", os.path.basename(p))
        return int(m.group(1)) if m else 999

    files = sorted(glob.glob(pattern), key=_shard_num)
    if not files:
        return {"error": f"No log_worker_*.txt in {log_dir}"}

    all_results = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        all_results.extend(parse_per_question_results(content))

    return aggregate_results(all_results)


def aggregate_results(results: list[dict]) -> dict:
    """汇总结果"""
    if not results:
        return {
            "total": 0,
            "base_correct": 0,
            "steer_correct": 0,
            "base_right_steer_wrong": 0,  # Base对Steer错
            "base_wrong_steer_right": 0,  # Base错Steer对
        }
    total = len(results)
    base_correct = sum(r["base_correct"] for r in results)
    steer_correct = sum(r["steer_correct"] for r in results)
    base_right_steer_wrong = sum(1 for r in results if r["base_correct"] == 1 and r["steer_correct"] == 0)
    base_wrong_steer_right = sum(1 for r in results if r["base_correct"] == 0 and r["steer_correct"] == 1)
    return {
        "total": total,
        "base_correct": base_correct,
        "steer_correct": steer_correct,
        "base_right_steer_wrong": base_right_steer_wrong,
        "base_wrong_steer_right": base_wrong_steer_right,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="包含 log_worker_*.txt 的目录")
    parser.add_argument("--output", type=str, default=None, help="追加写入的结果文件")
    parser.add_argument("--label", type=str, default="", help="参数标签，如 r0.95_c0.5")
    args = parser.parse_args()

    d = collect_from_log_dir(args.log_dir)
    if "error" in d:
        print(d["error"])
        return 1

    # 表格行格式，便于汇总
    if args.label:
        parts = args.label.replace("[", "").replace("]", "").split("_")
        # 支持 t2500_s1.12_c0.72_r0.8_n4 格式（5 参数 sweep，含 reward）
        if len(parts) >= 5 and parts[0].startswith("t") and parts[1].startswith("s") and parts[2].startswith("c") and parts[3].startswith("r") and parts[4].startswith("n"):
            t_val = parts[0][1:] if parts[0].startswith("t") else ""
            s_val = parts[1][1:] if parts[1].startswith("s") else ""
            c_val = parts[2][1:] if parts[2].startswith("c") else ""
            r_val = parts[3][1:] if parts[3].startswith("r") else ""
            n_val = parts[4][1:] if parts[4].startswith("n") else ""
            table_line = f"{t_val:>6} | {s_val:>6} | {c_val:>6} | {r_val:>6} | {n_val:>6} | {d['total']:>5} | {d['base_correct']:>6} | {d['steer_correct']:>6} | {d['base_right_steer_wrong']:>13} | {d['base_wrong_steer_right']:>13}"
        # 支持 t2500_s1.12_c0.72_n4 格式（4 参数 sweep）
        elif len(parts) >= 4 and parts[0].startswith("t") and parts[1].startswith("s") and parts[2].startswith("c") and parts[3].startswith("n"):
            t_val = parts[0][1:] if parts[0].startswith("t") else ""
            s_val = parts[1][1:] if parts[1].startswith("s") else ""
            c_val = parts[2][1:] if parts[2].startswith("c") else ""
            n_val = parts[3][1:] if parts[3].startswith("n") else ""
            table_line = f"{t_val:>6} | {s_val:>6} | {c_val:>6} | {n_val:>6} | {d['total']:>5} | {d['base_correct']:>6} | {d['steer_correct']:>6} | {d['base_right_steer_wrong']:>13} | {d['base_wrong_steer_right']:>13}"
        elif len(parts) >= 3 and parts[0].startswith("t") and parts[1].startswith("s") and parts[2].startswith("n"):
            t_val = parts[0][1:] if parts[0].startswith("t") else ""
            s_val = parts[1][1:] if parts[1].startswith("s") else ""
            n_val = parts[2][1:] if parts[2].startswith("n") else ""
            table_line = f"{t_val:>6} | {s_val:>6} | {n_val:>6} | {d['total']:>5} | {d['base_correct']:>6} | {d['steer_correct']:>6} | {d['base_right_steer_wrong']:>13} | {d['base_wrong_steer_right']:>13}"
        else:
            # 兼容旧格式 r0.95_c0.5
            r_val = parts[0].replace("r", "") if len(parts) > 0 else ""
            c_val = parts[1].replace("c", "") if len(parts) > 1 else ""
            table_line = f"{r_val:>6} | {c_val:>6} | {d['total']:>5} | {d['base_correct']:>6} | {d['steer_correct']:>6} | {d['base_right_steer_wrong']:>13} | {d['base_wrong_steer_right']:>13}"
    else:
        table_line = f"{d['total']:5} | {d['base_correct']:6} | {d['steer_correct']:6} | {d['base_right_steer_wrong']:14} | {d['base_wrong_steer_right']:14}"

    human_line = (
        f"total={d['total']} | Base对={d['base_correct']} Steer对={d['steer_correct']} | "
        f"Base对Steer错={d['base_right_steer_wrong']} | Base错Steer对={d['base_wrong_steer_right']}"
    )
    print(human_line)

    if args.output:
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(table_line + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
