#!/usr/bin/env python3
"""
从 edge sweep 的 log_worker_*.txt 解析每道题的 Base/Steer 输出，
提取答案、对比变化，生成可读的 Markdown 报告。
便于分析：Base/Steer 做错的原因、steer 之后具体有哪些效果改变。

用法: python generate_edge_case_report.py --log_dir LOG_DIR [--edge_json EDGE_JSON] [--output OUTPUT]
"""
import re
import argparse
import glob
import os
import json


def _extract_braced(s: str, start: int) -> tuple[str, int]:
    """从 start 位置（刚读完 '{'）提取匹配的 {...} 内容，返回 (内容, 结束位置)"""
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    return s[start : i - 1].strip(), i


def extract_boxed_answer(text: str) -> str:
    """从输出中提取最终答案：优先 <answer>\\boxed{...}</answer>，否则取最后一个 \\boxed{...}（支持嵌套括号）"""
    if not text:
        return ""
    # 优先 <answer> 标签
    m = re.search(r"<answer>\s*\\?boxed\s*\{", text, re.IGNORECASE)
    if m:
        start = m.end()
        content, _ = _extract_braced(text, start)
        return content
    # 否则取最后一个 \boxed{
    boxes = list(re.finditer(r"\\boxed\s*\{", text))
    if boxes:
        m = boxes[-1]
        start = m.end()
        content, _ = _extract_braced(text, start)
        return content
    return ""


def parse_question_blocks(content: str) -> list[dict]:
    """解析每个题目块，提取 question_id, base_output, steer_output, base_ok, steer_ok, len_base, len_steer"""
    question_markers = list(re.finditer(r"======== Question (\d+) ========", content))
    base_steer_pattern = re.compile(
        r"Base\s+\[(✅|❌)\]\s+Len:\s*(\d+)\s*\n\s*Steer\s+\[(✅|❌)\]\s+Len:\s*(\d+)",
        re.MULTILINE,
    )
    # Base output: 从 [Baseline Output] 行后到 [Steered Output] 之前
    base_section = re.compile(
        r"\[Baseline Output\][^\n]*\n(.*?)(?=\[Steered Output\])",
        re.DOTALL,
    )
    # Steer output: 从 [Steered Output] 行后到 ====== 分隔线或 Base [ 之前
    steer_section = re.compile(
        r"\[Steered Output\][^\n]*\n(.*?)(?=\n={50,}\s*\n|Base\s+\[)",
        re.DOTALL,
    )
    results = []
    for i, qm in enumerate(question_markers):
        qid = int(qm.group(1))
        start = qm.end()
        end = question_markers[i + 1].start() if i + 1 < len(question_markers) else len(content)
        block = content[start:end]
        sm = base_steer_pattern.search(block)
        if not sm:
            continue
        base_ok = 1 if sm.group(1) == "✅" else 0
        steer_ok = 1 if sm.group(3) == "✅" else 0
        len_base = int(sm.group(2))
        len_steer = int(sm.group(4))
        base_text = ""
        steer_text = ""
        bm = base_section.search(block)
        if bm:
            base_text = bm.group(1).strip()
        sm2 = steer_section.search(block)
        if sm2:
            steer_text = sm2.group(1).strip()
        results.append({
            "question_id": qid,
            "base_correct": base_ok,
            "steer_correct": steer_ok,
            "len_base": len_base,
            "len_steer": len_steer,
            "base_output": base_text,
            "steer_output": steer_text,
            "base_answer": extract_boxed_answer(base_text),
            "steer_answer": extract_boxed_answer(steer_text),
        })
    return results


def collect_from_log_dir(log_dir: str) -> list[dict]:
    """从 log_dir 下的 log_worker_*.txt 收集所有题目"""
    log_dir = os.path.abspath(log_dir)
    pattern = os.path.join(log_dir, "log_worker_*.txt")

    def _shard_num(p):
        m = re.search(r"log_worker_(\d+)\.txt", os.path.basename(p))
        return int(m.group(1)) if m else 999

    files = sorted(glob.glob(pattern), key=_shard_num)
    if not files:
        return []

    all_results = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        all_results.extend(parse_question_blocks(content))

    by_id = {}
    for r in all_results:
        qid = r["question_id"]
        if qid not in by_id:
            by_id[qid] = r
    return [by_id[q] for q in sorted(by_id.keys())]


def load_edge_json(path: str) -> dict:
    """加载 edge_cases_24.json 获取 ground truth"""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_judge_reasons(judge_dir: str) -> dict[int, dict]:
    """从 judge_reasons 目录加载每道题的判题理由，返回 {question_id: {judge_base_reason, judge_steer_reason}}"""
    if not judge_dir or not os.path.isdir(judge_dir):
        return {}
    by_qid = {}
    for fn in glob.glob(os.path.join(judge_dir, "question_*_shard_*.json")):
        try:
            with open(fn, "r", encoding="utf-8") as f:
                d = json.load(f)
            qid = d.get("question_idx")
            if qid is not None:
                by_qid[qid] = {
                    "judge_base_reason": d.get("judge_base_reason", ""),
                    "judge_steer_reason": d.get("judge_steer_reason", ""),
                }
        except Exception:
            pass
    return by_qid


def generate_markdown_report(results: list[dict], edge_data: dict, judge_reasons: dict, conf: str, output_path: str) -> None:
    """生成 Markdown 报告"""
    questions = edge_data.get("questions", {})
    base_right_steer_wrong = set(edge_data.get("base_right_steer_wrong_ids", []))
    base_wrong_steer_right = set(edge_data.get("base_wrong_steer_right_ids", []))

    lines = [
        "# Edge Case 题目分析报告",
        "",
        f"**conf = {conf}** | 共 {len(results)} 题",
        "",
        "## 汇总",
        "",
        "| 类型 | 题目数 | 说明 |",
        "|------|--------|------|",
    ]
    b_r_s_w = sum(1 for r in results if r["base_correct"] and not r["steer_correct"])
    b_w_s_r = sum(1 for r in results if not r["base_correct"] and r["steer_correct"])
    lines.append(f"| Base对Steer错 | {b_r_s_w} | Base 答对、Steer 改错 |")
    lines.append(f"| Base错Steer对 | {b_w_s_r} | Base 答错、Steer 改对 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for r in results:
        qid = r["question_id"]
        qinfo = questions.get(str(qid), {})
        problem = qinfo.get("problem", "")[:200] + "..." if len(qinfo.get("problem", "")) > 200 else qinfo.get("problem", "")
        correct_ans = qinfo.get("answer", "?")
        case_type = "Base对Steer错" if (r["base_correct"] and not r["steer_correct"]) else ("Base错Steer对" if (not r["base_correct"] and r["steer_correct"]) else "其他")
        status_base = "✅" if r["base_correct"] else "❌"
        status_steer = "✅" if r["steer_correct"] else "❌"

        lines.append(f"## 题目 {qid} [{case_type}]")
        lines.append("")
        lines.append(f"- **正确答案**: `{correct_ans}`")
        lines.append(f"- **Base 提取答案**: `{r['base_answer'] or '(未提取到)'}` {status_base} | Len: {r['len_base']}")
        lines.append(f"- **Steer 提取答案**: `{r['steer_answer'] or '(未提取到)'}` {status_steer} | Len: {r['len_steer']}")
        delta = r["len_steer"] - r["len_base"]
        lines.append(f"- **Token 变化**: {delta:+d}")
        lines.append("")
        lines.append("**题目**: " + (problem or "(无)"))
        lines.append("")
        # 展示 Base / Steer 具体输出（尾部，含结论与答案）
        # Base 块可能混入 Steer 监控日志，截断到 "2️⃣ Generating" 之前
        TAIL_LEN = 1200
        base_raw = r.get("base_output") or ""
        if "2️⃣ Generating" in base_raw:
            base_raw = base_raw.split("2️⃣ Generating")[0].strip()
        base_tail = base_raw[-TAIL_LEN:].strip()
        steer_tail = (r.get("steer_output") or "")[-TAIL_LEN:].strip()
        lines.append("**Base 输出（尾部）**:")
        lines.append("```")
        lines.append(base_tail or "(无)")
        lines.append("```")
        lines.append("")
        lines.append("**Steer 输出（尾部）**:")
        lines.append("```")
        lines.append(steer_tail or "(无)")
        lines.append("```")
        lines.append("")
        if r["base_answer"] != r["steer_answer"]:
            lines.append("**答案变化**: Base → Steer 答案发生改变")
        jr = judge_reasons.get(qid, {})
        if jr.get("judge_base_reason") or jr.get("judge_steer_reason"):
            lines.append("")
            lines.append("**判题理由**:")
            if jr.get("judge_base_reason"):
                lines.append(f"- Base: {jr['judge_base_reason'][:300]}{'...' if len(jr.get('judge_base_reason',''))>300 else ''}")
            if jr.get("judge_steer_reason"):
                lines.append(f"- Steer: {jr['judge_steer_reason'][:300]}{'...' if len(jr.get('judge_steer_reason',''))>300 else ''}")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="sweep 日志目录，如 logs/edge_sweep_t3000_r0.8_c0.73")
    parser.add_argument("--edge_json", type=str, default="logs/edge_cases_24.json", help="edge cases JSON")
    parser.add_argument("--judge_dir", type=str, default=None, help="判题理由目录，默认 log_dir/judge_reasons")
    parser.add_argument("--output", type=str, default=None, help="输出 Markdown 路径，默认 log_dir/report.md")
    args = parser.parse_args()

    results = collect_from_log_dir(args.log_dir)
    if not results:
        print("未解析到任何题目，请检查 log_dir")
        return 1

    edge_data = load_edge_json(args.edge_json)
    judge_dir = args.judge_dir or os.path.join(os.path.abspath(args.log_dir), "judge_reasons")
    judge_reasons = load_judge_reasons(judge_dir)
    conf = "?"
    m = re.search(r"c0?\.?(\d+)", os.path.basename(os.path.abspath(args.log_dir)))
    if m:
        conf = "0." + m.group(1) if "." not in m.group(1) else m.group(1)

    out_path = args.output or os.path.join(os.path.abspath(args.log_dir), "report.md")
    generate_markdown_report(results, edge_data, judge_reasons, conf, out_path)
    return 0


if __name__ == "__main__":
    exit(main())
