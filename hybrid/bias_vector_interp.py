#!/usr/bin/env python3

"""Summarize differences between baseline and bias-only hybrid responses.

This script samples paired responses from two rolling evaluation JSONL files, sends
batched comparison prompts to an LLM (``o4-mini`` by default), and aggregates the
partial descriptions into a final consolidated summary. All outputs are printed to
stdout and persisted to JSON for later analysis.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from openai import OpenAI


BASELINE_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "results",
    "rolling",
    "rolling_qwen2.5-32b_math500.jsonl",
)

BIAS_ONLY_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "results",
    "rolling",
    "rolling_qwen2.5-32b_math500_bias-only.jsonl",
)


@dataclass(frozen=True)
class PairedResponse:
    question: str
    baseline_base: str
    bias_only_hybrid: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and bias-only hybrid responses via LLM summarization.",
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        default=BASELINE_DEFAULT,
        help="Path to baseline rolling JSONL file.",
    )
    parser.add_argument(
        "--bias_only_path",
        type=str,
        default=BIAS_ONLY_DEFAULT,
        help="Path to bias-only rolling JSONL file.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1000,
        help="Maximum characters per response snippet shown to the evaluator model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of paired responses per comparison prompt.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o4-mini",
        help="Evaluator model used for generating descriptions.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=5000,
        help="Maximum tokens for each partial description request.",
    )
    parser.add_argument(
        "--final_max_output_tokens",
        type=int,
        default=5000,
        help="Maximum tokens for the final consolidation request.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="bias_vector_interp_results.json",
        help="Destination JSON file for saving intermediate and final descriptions.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="If set, process only the first batch and exit immediately.",
    )
    return parser.parse_known_args()[0]


def _load_jsonl(path: str) -> List[dict]:
    assert os.path.isfile(path), f"File not found: {path}"
    records: List[dict] = []
    with open(path, "r") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"JSON decoding failed at line {idx} in {path}") from exc
    assert len(records) > 0, f"No records loaded from {path}"
    return records


def _extract_answer(record: dict, index: int, path: str, key: str) -> str:
    answers = record.get("answers")
    assert isinstance(answers, dict), f"Missing answers dict at index {index} in {path}"
    value = answers.get(key)
    assert isinstance(value, str) and value.strip(), (
        f"Missing {key} response at index {index} in {path}"
    )
    return value.strip()


def _pair_records(baseline_records: Sequence[dict], bias_only_records: Sequence[dict]) -> List[PairedResponse]:
    baseline_map: Dict[str, dict] = {}
    for idx, record in enumerate(baseline_records):
        question = record.get("question")
        assert isinstance(question, str) and question.strip(), (
            f"Missing question in baseline record {idx}"
        )
        key = question.strip()
        assert key not in baseline_map, f"Duplicate question in baseline file: {key}"
        baseline_map[key] = record

    pairs: List[PairedResponse] = []
    unmatched_baseline: Dict[str, dict] = dict(baseline_map)

    for idx, record in enumerate(bias_only_records):
        question = record.get("question")
        assert isinstance(question, str) and question.strip(), (
            f"Missing question in bias-only record {idx}"
        )
        key = question.strip()
        baseline_record = baseline_map.get(key)
        if baseline_record is None:
            print(f"WARNING: Bias-only question without baseline match: {key}")
            continue
        base_answer = _extract_answer(
            baseline_record,
            idx,
            "baseline",
            "base",
        )
        hybrid_answer = _extract_answer(
            record,
            idx,
            "bias-only",
            "hybrid",
        )
        pairs.append(
            PairedResponse(
                question=key,
                baseline_base=base_answer,
                bias_only_hybrid=hybrid_answer,
            )
        )
        unmatched_baseline.pop(key, None)

    for key in unmatched_baseline:
        print(f"WARNING: Baseline question without bias-only match: {key}")

    assert len(pairs) > 0, "No matching questions found between baseline and bias-only files"
    return pairs


def _truncate(text: str, max_chars: int) -> str:
    snippet = text.strip()
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3] + "..."


def _batched(iterable: Sequence[PairedResponse], batch_size: int) -> Iterable[Sequence[PairedResponse]]:
    assert batch_size > 0, "batch_size must be positive"
    total = len(iterable)
    for start in range(0, total, batch_size):
        yield iterable[start : start + batch_size]


def _build_batch_prompt(batch: Sequence[PairedResponse], max_chars: int) -> str:
    lines = [
        "You are comparing responses from two model variants.",
        "Each numbered item contains the question followed by Model A and Model B outputs.",
        "Summarize the dominant qualitative differences you observe across this batch in 3-4 sentences.",
        "Focus on writing style, reasoning approach, structure, and qualitative behavior rather than numerical accuracy or correctness.",
        "Explicitly refrain from judging which model is correct; treat accuracy as unknown and out of scope.",
        "Number of pairs in this batch: {count}.".format(count=len(batch)),
        "",
    ]
    for idx, pair in enumerate(batch, start=1):
        lines.append(f"[{idx}] Question: {pair.question.strip()}\n")
        lines.append(f"    Model A:\n\n`{_truncate(pair.baseline_base, max_chars)}`\n")
        lines.append(f"    Model B:\n\n`{_truncate(pair.bias_only_hybrid, max_chars)}`\n")
        lines.append("")
    lines.append("")
    lines.append(
        "Respond with 3-4 sentences describing the consistent differences between Model A and Model B outputs in this batch."
    )
    return "\n".join(lines)


def _collect_partial_descriptions(
    client: OpenAI,
    pairs: Sequence[PairedResponse],
    batch_size: int,
    max_chars: int,
    model: str,
    max_output_tokens: int,
    debug: bool,
) -> List[str]:
    partials: List[str] = []
    for batch_index, batch in enumerate(_batched(pairs, batch_size)):
        prompt = _build_batch_prompt(batch, max_chars)
        if debug:
            print("--------------------------------")
            print("Prompt:")
            print(prompt)
            print("--------------------------------")
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        output = getattr(response, "output_text", None)
        assert isinstance(output, str) and output.strip(), "Empty response from evaluator model"
        text = output.strip()
        print(f"[Batch {batch_index + 1}] {text}\n")
        partials.append(text)
        if debug:
            break
    return partials


def _build_final_prompt(partials: Sequence[str]) -> str:
    lines = [
        "You previously analyzed several batches of model outputs.",
        "Each item below is a 3-4 sentence summary describing differences between Model A and Model B.",
        "Synthesize these notes into a single 3-4 sentence description that captures the most important recurring differences across all batches.",
        "Avoid repeating minor batch-specific details; instead, emphasize consistent trends in qualitative behavior.",
        "Do not speculate about which model is more accurate; focus purely on stylistic and behavioral differences.",
        "",
    ]
    for idx, summary in enumerate(partials, start=1):
        lines.append(f"[{idx}] {summary}\n")
    lines.append("")
    lines.append("Respond with exactly 3-4 sentences.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_records = _load_jsonl(args.baseline_path)
    bias_only_records = _load_jsonl(args.bias_only_path)
    paired = _pair_records(baseline_records, bias_only_records)
    assert len(paired) > 0, "No paired responses found"

    client = OpenAI()

    partial_descriptions = _collect_partial_descriptions(
        client=client,
        pairs=paired,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        debug=args.debug,
    )

    if args.debug:
        print("Debug mode enabled; skipping consolidation and JSON write.")
        return

    final_prompt = _build_final_prompt(partial_descriptions)
    print("--------------------------------")
    print("Final Prompt:")
    print(final_prompt)
    print("--------------------------------")
    final_response = client.responses.create(
        model=args.model,
        input=final_prompt,
        max_output_tokens=args.final_max_output_tokens,
    )
    final_text = getattr(final_response, "output_text", None)
    assert isinstance(final_text, str) and final_text.strip(), "Empty consolidated description"
    final_summary = final_text.strip()
    print(f"[Final] {final_summary}")

    output_payload = {
        "baseline_path": args.baseline_path,
        "bias_only_path": args.bias_only_path,
        "model": args.model,
        "max_chars": args.max_chars,
        "batch_size": args.batch_size,
        "max_output_tokens": args.max_output_tokens,
        "final_max_output_tokens": args.final_max_output_tokens,
        "partial_descriptions": partial_descriptions,
        "final_description": final_summary,
    }

    with open(args.output_path, "w") as handle:
        json.dump(output_payload, handle, indent=2)


if __name__ == "__main__":
    main()

