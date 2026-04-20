#!/usr/bin/env python3
"""
Compare rolling JSONL files between current state and previous Git commit.
Output all responses where the judge answer changed.
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

FILES = [
    "results/rolling/rolling_qwen2.5-0.5b_gsm8k.jsonl",
    "results/rolling/rolling_qwen2.5-0.5b_gsm8k_0.jsonl",
    "results/rolling/rolling_qwen2.5-0.5b_math500.jsonl",
    "results/rolling/rolling_qwen2.5-0.5b_math500_0.jsonl",
    "results/rolling/rolling_qwen2.5-1.5b_gsm8k.jsonl",
    "results/rolling/rolling_qwen2.5-1.5b_math500.jsonl",
    "results/rolling/rolling_qwen2.5-32b-on-open-reasoner-zero-32b_gsm8k.jsonl",
    "results/rolling/rolling_qwen2.5-32b-on-open-reasoner-zero-32b_math500.jsonl",
    "results/rolling/rolling_qwen2.5-32b-on-open-reasoner-zero-32b_math500_0.jsonl",
    "results/rolling/rolling_qwen2.5-7b_gsm8k.jsonl",
    "results/rolling/rolling_qwen2.5-7b_gsm8k_0.jsonl",
    "results/rolling/rolling_qwen2.5-7b_math500.jsonl",
    "results/rolling/rolling_qwen2.5-7b_math500_0.jsonl",
]

MODEL_KEYS = ["base", "thinking", "hybrid"]

# Directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_previous_version(file_path: str) -> Optional[str]:
    """Get the file content from the previous commit using git show."""
    # Convert to path relative to repo root for git
    git_path = os.path.join("hybrid", file_path)
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD^:{git_path}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def read_current_version(file_path: str) -> Optional[str]:
    """Read the current file content."""
    full_path = os.path.join(SCRIPT_DIR, file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def parse_jsonl(content: str) -> List[dict]:
    """Parse JSONL content into a list of records."""
    records = []
    for line in content.strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def get_judge_correct(record: dict, model_key: str) -> Optional[bool]:
    """Extract the judge 'correct' value for a given model."""
    judges = record.get("judges", {})
    model_judge = judges.get(model_key, {})
    if isinstance(model_judge, dict):
        val = model_judge.get("correct")
        if isinstance(val, bool):
            return val
    return None


def get_judge_raw(record: dict, model_key: str) -> str:
    """Extract the judge 'raw' value for a given model."""
    judges = record.get("judges", {})
    model_judge = judges.get(model_key, {})
    if isinstance(model_judge, dict):
        return str(model_judge.get("raw", ""))
    return ""


def find_differences(
    prev_records: List[dict], curr_records: List[dict]
) -> List[Tuple[int, str, dict, dict]]:
    """
    Find records where the judge answer changed.

    Returns list of (index, model_key, prev_record, curr_record) tuples.
    """
    differences = []

    # Build a mapping from question to record for matching
    # Assuming records are in the same order, but we match by question for safety
    prev_by_question: Dict[str, dict] = {}
    for rec in prev_records:
        q = rec.get("question", "")
        prev_by_question[q] = rec

    for idx, curr_rec in enumerate(curr_records):
        q = curr_rec.get("question", "")
        prev_rec = prev_by_question.get(q)

        if prev_rec is None:
            # New record, not in previous version
            continue

        for model_key in MODEL_KEYS:
            prev_correct = get_judge_correct(prev_rec, model_key)
            curr_correct = get_judge_correct(curr_rec, model_key)

            # Check if the judgment changed
            if prev_correct != curr_correct:
                differences.append((idx, model_key, prev_rec, curr_rec))

    return differences


def format_difference(
    idx: int,
    model_key: str,
    prev_record: dict,
    curr_record: dict,
) -> str:
    """Format a single difference for output."""
    question = curr_record.get("question", "")
    gold_answer = curr_record.get("gold_answer", "")
    response = curr_record.get("answers", {}).get(model_key, "")
    prev_raw = get_judge_raw(prev_record, model_key)
    curr_raw = get_judge_raw(curr_record, model_key)

    lines = [
        f"Question: {question}",
        f"",
        f"Ground Truth: {gold_answer}",
        f"",
        f"Model: {model_key}",
        f"",
        "<<<RESPONSE START>>>",
        response,
        "<<<RESPONSE END>>>",
        f"",
        f"Previous raw judge: {prev_raw}",
        f"Current raw judge: {curr_raw}",
    ]
    return "\n".join(lines)


def process_file(file_path: str) -> Optional[str]:
    """Process a single file and return formatted differences."""
    prev_content = get_previous_version(file_path)
    curr_content = read_current_version(file_path)

    if prev_content is None:
        return None
    if curr_content is None:
        return None

    prev_records = parse_jsonl(prev_content)
    curr_records = parse_jsonl(curr_content)

    differences = find_differences(prev_records, curr_records)

    if not differences:
        return None

    # Format output
    file_name = file_path.split("/")[-1].replace(".jsonl", "")
    header_line = "=" * 80
    output_lines = [header_line, f"===== {file_name} =====", header_line, ""]

    separator = "\n".join([
        "",
        "",
        "################################################################################",
        "################################################################################",
        "",
        "",
    ])

    for i, (idx, model_key, prev_rec, curr_rec) in enumerate(differences):
        if i > 0:
            output_lines.append(separator)
        output_lines.append(format_difference(idx, model_key, prev_rec, curr_rec))

    return "\n".join(output_lines)


def main():
    all_outputs = []

    for file_path in FILES:
        result = process_file(file_path)
        if result:
            all_outputs.append(result)

    if not all_outputs:
        print("No differences found.", file=sys.stderr)
        final_output = "No differences found."
    else:
        file_separator = "\n\n\n" + ("*" * 80) + "\n" + ("*" * 80) + "\n" + ("*" * 80) + "\n\n\n"
        final_output = file_separator.join(all_outputs)

    # Write to output file in the hybrid folder
    output_path = os.path.join(SCRIPT_DIR, "judge-differences.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
