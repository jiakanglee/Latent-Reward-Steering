import argparse
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from tqdm.auto import tqdm
except ImportError as exc:
    raise ImportError("tqdm is required for progress reporting") from exc


MODEL_SPECS: List[Tuple[str, str]] = [
    ("thinking", "Thinking Model"),
    ("base", "Base Model"),
    ("hybrid", "Hybrid Model"),
]

# Qwen models that have OpenReasonerZero counterparts
# Note: For 32B, we specifically match the "on-open-reasoner-zero" variant,
# not the plain qwen2.5-32b which are different experiments (bias-only, random-vectors, etc.)
QWEN_ORZ_MODELS = [
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "qwen2.5-7b",
    "qwen2.5-32b-on-open-reasoner-zero",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate rolling results with LLM judges using OpenAI Batch API")
    parser.add_argument(
        "--prefix",
        required=False,
        help="Rolling file prefix. Accepts absolute path or name inside the rolling directory.",
    )
    parser.add_argument(
        "--rolling-dir",
        type=str,
        default=None,
        help="Directory containing rolling outputs (defaults to results/rolling next to this script).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2",
        help="Judge model to query via OpenAI Batch API.",
    )
    parser.add_argument(
        "--max-judge-tokens",
        type=int,
        default=2000,
        help="Maximum tokens returned by the judge model.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="qwen-orz",
        help="Filter which results to process: 'qwen-orz' (default, only Qwen models with ORZ counterparts), 'all' (all files), or a comma-separated list of model patterns.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Poll interval in seconds for checking batch status (default: 60).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Maximum requests per batch file (default: 5000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually running.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: run the full flow but only evaluate one record per file.",
    )
    parser.add_argument(
        "--only-finished-thinking",
        action="store_true",
        default=False,
        help="If set, only re-evaluate records where eos.thinking is true, and compute stats on that subset.",
    )
    parser.add_argument(
        "--recompute-only",
        action="store_true",
        help="Read existing judge results and print stats without re-running evaluation or modifying files.",
    )
    return parser.parse_known_args()[0]


def _default_rolling_dir() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "results", "rolling")


def _resolve_prefix(raw_prefix: str, rolling_dir: str) -> str:
    if raw_prefix is None:
        raise ValueError("raw_prefix must not be None")
    if os.path.isabs(raw_prefix):
        prefix = raw_prefix
    else:
        prefix = os.path.join(rolling_dir, raw_prefix)
    if prefix.endswith(".jsonl"):
        prefix = prefix[:-6]
    return prefix


def _split_prefix_parts(path: str) -> Tuple[str, Optional[int]]:
    """Split a rolling file path into its prefix and part number."""
    base = os.path.basename(path)
    parent = os.path.dirname(path)
    if base.endswith(".jsonl"):
        base = base[:-6]
    m = re.match(r"^(.*)_(\d+)$", base)
    if not m:
        return os.path.join(parent, base + ".jsonl"), None
    prefix = os.path.join(parent, m.group(1) + ".jsonl")
    return prefix, int(m.group(2))


def _list_ordered_files(prefix: str) -> List[str]:
    directory = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    assert base, "Prefix must include a filename component"

    files: List[str] = []
    legacy = os.path.join(directory, f"{base}.jsonl")
    if os.path.exists(legacy):
        files.append(legacy)

    part_pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.jsonl$")
    part_paths: List[str] = []
    for name in os.listdir(directory):
        match = part_pattern.match(name)
        if match:
            part_paths.append(os.path.join(directory, name))

    part_paths.sort(key=lambda path: int(part_pattern.match(os.path.basename(path)).group(1)))
    files.extend(part_paths)
    assert files, f"No rolling files found for prefix {prefix}"
    return files


def _extract_model_from_filename(filename: str) -> Optional[str]:
    """Extract the model name from a rolling filename like rolling_qwen2.5-1.5b_gsm8k.jsonl"""
    base = os.path.basename(filename)
    if not base.startswith("rolling_"):
        return None
    # Remove rolling_ prefix and .jsonl suffix
    rest = base[len("rolling_"):]
    if rest.endswith(".jsonl"):
        rest = rest[:-6]
    # Handle part numbers like _0, _1
    rest = re.sub(r"_\d+$", "", rest)
    # Extract model name (everything before the first _ that looks like a dataset)
    # Datasets are: gsm8k, math500, aime, mbpp, livecodebench
    parts = rest.split("_")
    # Find where the dataset starts
    datasets = {"gsm8k", "math500", "aime", "mbpp", "livecodebench", "medqa", "legalbench"}
    model_parts = []
    for part in parts:
        if part in datasets:
            break
        model_parts.append(part)
    return "_".join(model_parts) if model_parts else None


def _extract_dataset_from_filename(filename: str) -> Optional[str]:
    """Extract the dataset name from a rolling filename like rolling_qwen2.5-1.5b_gsm8k.jsonl"""
    base = os.path.basename(filename)
    if not base.startswith("rolling_"):
        return None
    # Remove rolling_ prefix and .jsonl suffix
    rest = base[len("rolling_"):]
    if rest.endswith(".jsonl"):
        rest = rest[:-6]
    # Handle part numbers like _0, _1
    rest = re.sub(r"_\d+$", "", rest)
    # Find dataset in the parts
    datasets = {"gsm8k", "math500", "aime", "mbpp", "livecodebench", "medqa", "legalbench"}
    parts = rest.split("_")
    for part in parts:
        if part in datasets:
            return part
    return None


CODING_DATASETS = {"mbpp", "livecodebench"}
MCQA_DATASETS = {"medqa"}  # Multiple choice QA datasets
TEXT_CLASSIFICATION_DATASETS = {"legalbench"}  # Text classification datasets

def _thinking_finished(record: Dict[str, Any]) -> bool:
    eos = record.get("eos", {})
    assert isinstance(eos, dict), "record['eos'] must be a dict when present"
    return bool(eos.get("thinking", False))


def _matches_filter(filename: str, filter_patterns: List[str], exclude_patterns: Optional[List[str]] = None) -> bool:
    """Check if a filename matches any of the filter patterns."""
    model = _extract_model_from_filename(filename)
    if model is None:
        return False
    model_lower = model.lower()
    # Check exclusions first
    if exclude_patterns:
        for pattern in exclude_patterns:
            if pattern.lower() in model_lower:
                return False
    # Check inclusions
    for pattern in filter_patterns:
        if pattern.lower() in model_lower:
            return True
    return False


def _list_all_rollings(
    rolling_dir: str,
    filter_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, List[str]]]:
    files: Dict[str, List[Any]] = {}
    for name in os.listdir(rolling_dir):
        if not name.endswith(".jsonl"):
            continue
        if not name.startswith("rolling_"):
            continue
        # Skip vector_stats files
        if "_vector_stats" in name:
            continue
        # Apply filter if specified
        if filter_patterns is not None:
            if not _matches_filter(name, filter_patterns, exclude_patterns):
                continue
        full_path = os.path.join(rolling_dir, name)
        prefix, part = _split_prefix_parts(full_path)
        files.setdefault(prefix, []).append(full_path if part is None else (part, full_path))

    grouped: Dict[str, List[str]] = {}
    for prefix, entries in files.items():
        sorted_paths: List[str] = []
        parts = [e for e in entries if isinstance(e, tuple)]
        legacy = [e for e in entries if isinstance(e, str)]
        if legacy:
            sorted_paths.extend(sorted(legacy))
        for _, path in sorted(parts, key=lambda x: x[0]):
            sorted_paths.append(path)
        grouped[prefix] = sorted_paths

    if not grouped:
        return []
    return sorted(grouped.items(), key=lambda kv: kv[0])


def clean_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_judge_prompt(
    question: str,
    correct_answer: str,
    model_answer: str,
    dataset_type: str = "math",
    test_list: Optional[List[str]] = None,
) -> str:
    if dataset_type == "coding":
        test_cases_str = "\n\n".join(test_list) if test_list else "No test cases provided"

        # Reference solution section (only for datasets like MBPP that provide one)
        if correct_answer:
            reference_section = f"\n\nReference solution:\n```python\n{correct_answer}\n```"
        else:
            reference_section = ""

        return (
            "Please evaluate whether the model's response contains a correct solution to this coding problem.\n\n"
            f"Problem: {question}\n\n"
            f"Model's response (including reasoning and code):\n{model_answer}"
            f"{reference_section}\n\n"
            f"Test cases that a correct solution should pass:\n{test_cases_str}\n\n"
            "Instructions for evaluation:\n"
            "1. Search the ENTIRE model response for any Python code that could solve the problem.\n"
            "2. IMPORTANT: If a correct implementation appears ANYWHERE in the response - whether in the final code block, "
            "during reasoning/planning, or in a \"let me try\" section - answer YES.\n"
            "3. The model gets credit if it wrote correct code at any point, even if:\n"
            "   - It appeared during an intermediate attempt\n"
            "   - The response continues with a different (possibly incorrect) version afterward\n"
            "   - There are multiple code attempts and at least one is correct\n"
            "4. Evaluate if the code logic would produce correct outputs for the test cases.\n"
            "5. Only answer NO if no correct implementation appears anywhere in the response.\n\n"
            "Just answer YES if a correct solution appears anywhere in the response, or NO if it doesn't. Nothing else.\n"
        )
    elif dataset_type == "mcqa":
        return (
            "Please evaluate whether the model arrived at the correct answer for this multiple choice question.\n\n"
            f"Question: {question}\n\n"
            f"Correct answer: {correct_answer}\n\n"
            f"Model's response (including reasoning): {model_answer}\n\n"
            "Instructions for evaluation:\n"
            f"1. First, identify the correct answer letter ({correct_answer}) from the \"Correct answer\" field.\n"
            "2. Search the ENTIRE model response (including all reasoning steps) for this correct answer.\n"
            "3. IMPORTANT: If the correct answer letter appears ANYWHERE in the model's response as the chosen/identified answer - "
            "whether in the final statement, during reasoning, or when eliminating options - answer YES.\n"
            "4. The model gets credit if it identified the correct answer at any point, even if:\n"
            "   - It appeared during \"let me check\" or \"verification\" steps\n"
            "   - The final stated answer is different (due to errors, continued generation, etc.)\n"
            "   - The response becomes garbled after stating the correct answer\n"
            f"5. Look for the correct letter (e.g., \"{correct_answer}\") being selected, chosen, or identified as correct.\n"
            "6. Only answer NO if the correct answer letter is never identified as the answer anywhere in the response.\n\n"
            "Just answer YES if the correct answer appears anywhere as the chosen answer, or NO if it doesn't. Nothing else.\n"
        )
    elif dataset_type == "classification":
        return (
            "Please evaluate whether the model arrived at the correct classification/answer for this task.\n\n"
            f"Question/Context: {question}\n\n"
            f"Correct answer: {correct_answer}\n\n"
            f"Model's response (including reasoning): {model_answer}\n\n"
            "Instructions for evaluation:\n"
            f"1. First, identify the correct answer/classification: \"{correct_answer}\"\n"
            "2. Search the ENTIRE model response (including all reasoning steps) for this correct answer.\n"
            "3. IMPORTANT: If the correct answer appears ANYWHERE in the model's response as the conclusion/classification - "
            "whether in the final statement, during analysis, or when considering options - answer YES.\n"
            "4. The model gets credit if it stated the correct answer at any point, even if:\n"
            "   - It appeared during intermediate reasoning\n"
            "   - The final stated answer is different (due to errors, continued generation, etc.)\n"
            "   - The response continues or becomes garbled after stating the correct answer\n"
            "5. The comparison should be case-insensitive and ignore minor formatting differences.\n"
            "6. Only answer NO if the correct answer/classification does not appear anywhere in the response as a conclusion.\n\n"
            "Just answer YES if the correct answer appears anywhere in the response, or NO if it doesn't. Nothing else.\n"
        )
    else:
        return (
            "Please evaluate whether the model arrived at the correct answer for this math problem.\n\n"
            f"Question: {question}\n\n"
            f"Correct answer: {correct_answer}\n\n"
            f"Model's response (including reasoning trace): {model_answer}\n\n"
            "Instructions for evaluation:\n"
            "1. First, identify the correct numerical answer from the \"Correct answer\" field.\n"
            "2. Search the ENTIRE model response (including all reasoning steps) for this correct answer.\n"
            "3. IMPORTANT: If the correct answer appears ANYWHERE in the model's response - whether in the final \\boxed{{}}, "
            "after ####, or even mentioned during intermediate reasoning steps - answer YES.\n"
            "4. The model gets credit if it computed or stated the correct answer at any point, even if:\n"
            "   - It appeared during \"checking\" or \"verification\" steps\n"
            "   - The final boxed answer is different (due to copying errors, continued generation, etc.)\n"
            "   - The response continues or becomes garbled after stating the correct answer\n"
            "5. Only answer NO if the correct numerical answer does not appear anywhere in the response.\n\n"
            "Just answer YES if the correct answer appears anywhere in the response, or NO if it doesn't. Nothing else.\n"
        )


def _get_model_id(model_name: str) -> str:
    """Strip provider prefix if present (e.g., 'openai/gpt-4.1' -> 'gpt-4.1')."""
    for sep in ["/", ":"]:
        if sep in model_name:
            prefix, model_id = model_name.split(sep, 1)
            if prefix.lower() in {"openai", "anthropic", "google", "mistral", "mistralai"}:
                return model_id
    return model_name


def _is_transient_error(e: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    status_code = getattr(e, "status_code", None)
    if isinstance(status_code, int) and status_code >= 500:
        return True
    response = getattr(e, "response", None)
    resp_status = getattr(response, "status_code", None)
    if isinstance(resp_status, int) and resp_status >= 500:
        return True
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    name = type(e).__name__.lower()
    if "timeout" in name or "connect" in name or "connection" in name:
        return True
    return False


class BatchEvaluator:
    """Handles batch submission and polling for OpenAI Batch API."""

    def __init__(
        self,
        judge_model: str,
        max_tokens: int,
        poll_interval: int = 60,
        batch_size: int = 5000,
    ):
        self.judge_model = judge_model
        self.model_id = _get_model_id(judge_model)
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.client = OpenAI()

    def submit_batches(self, prompts: List[str]) -> List[str]:
        """Submit prompts to OpenAI Batch API and return batch IDs."""
        total_items = len(prompts)
        total_batches = (total_items + self.batch_size - 1) // self.batch_size
        print(f"[BatchEvaluator] Submitting {total_items} prompts in {total_batches} batch(es)")

        batch_ids: List[str] = []
        for batch_idx, start_idx in enumerate(range(0, total_items, self.batch_size), start=1):
            end_idx = min(start_idx + self.batch_size, total_items)
            batch_prompts = prompts[start_idx:end_idx]

            # Build JSONL requests
            requests_list: List[Dict[str, Any]] = []
            for i, prompt in enumerate(batch_prompts):
                custom_id = f"req_{start_idx + i}"
                body = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": self.max_tokens,
                }
                requests_list.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                })

            print(f"[BatchEvaluator] Submitting batch {batch_idx}/{total_batches}: {len(requests_list)} requests (idx {start_idx}-{end_idx - 1})")

            # Write JSONL and upload
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                for req in requests_list:
                    json.dump(req, f)
                    f.write("\n")
                input_path = f.name

            try:
                with open(input_path, "rb") as f:
                    input_file = self.client.files.create(file=f, purpose="batch")

                batch = self.client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": "re_evaluate_answers_rolling"},
                )
            finally:
                os.unlink(input_path)

            batch_id = batch.id
            print(f"[BatchEvaluator] OPENAI_BATCH_ID={batch_id}")
            batch_ids.append(batch_id)

        return batch_ids

    def poll_batches(self, batch_ids: List[str], total_items: int) -> Dict[int, str]:
        """Poll batches until completion and return responses indexed by position."""
        print(f"[BatchEvaluator] Polling {len(batch_ids)} batch(es) every {self.poll_interval}s...")

        pending = set(batch_ids)
        responses: Dict[int, str] = {}

        while pending:
            status_counts: Dict[str, int] = {}
            completed_now: List[str] = []

            for batch_id in list(pending):
                try:
                    batch = self.client.batches.retrieve(batch_id)
                except Exception as e:
                    if _is_transient_error(e):
                        print(f"[BatchEvaluator] Transient error retrieving batch {batch_id}: {e}")
                        status_counts["in_progress"] = status_counts.get("in_progress", 0) + 1
                        continue
                    raise

                status = str(batch.status)
                status_counts[status] = status_counts.get(status, 0) + 1

                if status not in {"completed", "failed", "expired", "cancelled"}:
                    continue

                if status != "completed":
                    error_details = ""
                    if hasattr(batch, "errors") and batch.errors:
                        error_details = f", errors={batch.errors}"
                    if batch.error_file_id:
                        try:
                            err_content = self.client.files.content(batch.error_file_id)
                            error_details += f", error_file_content={err_content.text[:2000]}"
                        except Exception:
                            error_details += f", error_file_id={batch.error_file_id}"
                    raise RuntimeError(
                        f"OpenAI batch did not complete successfully: batch_id={batch_id}, "
                        f"status={status}{error_details}"
                    )

                if batch.output_file_id is None:
                    req_counts = getattr(batch, "request_counts", None)
                    all_failed = (
                        req_counts is not None
                        and getattr(req_counts, "completed", 0) == 0
                        and getattr(req_counts, "failed", 0) > 0
                    )
                    if all_failed:
                        raise RuntimeError(
                            f"OpenAI batch failed: all requests failed. batch_id={batch_id}"
                        )
                    print(f"[BatchEvaluator] Batch {batch_id} completed but output_file_id not ready, will retry")
                    continue

                # Process responses
                try:
                    file_response = self.client.files.content(batch.output_file_id)
                    for line in file_response.text.splitlines():
                        obj = json.loads(line)
                        custom_id = obj.get("custom_id")
                        assert isinstance(custom_id, str) and custom_id.startswith("req_")

                        if obj.get("error") is not None:
                            print(f"[BatchEvaluator] Request {custom_id} failed: {obj['error']}")
                            idx = int(custom_id.split("_", 1)[1])
                            responses[idx] = ""
                            continue

                        resp_body = obj.get("response", {}).get("body", {})
                        choices = resp_body.get("choices", [])
                        if not choices or "message" not in choices[0]:
                            print(f"[BatchEvaluator] Request {custom_id} missing choices/message")
                            idx = int(custom_id.split("_", 1)[1])
                            responses[idx] = ""
                            continue

                        idx = int(custom_id.split("_", 1)[1])
                        content = choices[0]["message"].get("content") or ""
                        responses[idx] = content

                    # Also check error file
                    if batch.error_file_id:
                        err_response = self.client.files.content(batch.error_file_id)
                        for line in err_response.text.splitlines():
                            obj = json.loads(line)
                            custom_id = obj.get("custom_id", "?")
                            if custom_id.startswith("req_"):
                                idx = int(custom_id.split("_", 1)[1])
                                if idx not in responses:
                                    responses[idx] = ""
                except Exception as e:
                    if _is_transient_error(e):
                        print(f"[BatchEvaluator] Transient error processing batch {batch_id}: {e}")
                        continue
                    raise

                pending.remove(batch_id)
                completed_now.append(batch_id)

            print(f"[BatchEvaluator] Batches pending={len(pending)} (completed={len(completed_now)}): {status_counts}")
            if pending:
                time.sleep(self.poll_interval)

        # Fill missing responses
        missing = [i for i in range(total_items) if i not in responses]
        if missing:
            print(f"[BatchEvaluator] WARNING: {len(missing)} responses missing, filling with empty strings")
            for idx in missing:
                responses[idx] = ""

        return responses


def collect_all_prompts(
    files: List[Tuple[str, List[str]]],
    max_records_per_file: Optional[int] = None,
    *,
    only_finished_thinking: bool = False,
) -> Tuple[List[str], List[Tuple[str, int, str]]]:
    """
    Collect all prompts from all files.

    Args:
        files: List of (prefix, paths) tuples
        max_records_per_file: If set, limit to this many records per file (for debug mode)

    Returns:
        prompts: List of all judge prompts
        prompt_mapping: List of (file_path, record_idx, model_key) for each prompt
    """
    prompts: List[str] = []
    prompt_mapping: List[Tuple[str, int, str]] = []

    for prefix, paths in tqdm(files, desc="Collecting prompts", unit="prefix"):
        for path in paths:
            # Determine dataset type from filename
            dataset = _extract_dataset_from_filename(path)
            if dataset in CODING_DATASETS:
                dataset_type = "coding"
            elif dataset in MCQA_DATASETS:
                dataset_type = "mcqa"
            elif dataset in TEXT_CLASSIFICATION_DATASETS:
                dataset_type = "classification"
            else:
                dataset_type = "math"

            with open(path, "r", encoding="utf-8") as src:
                records = [json.loads(line) for line in src if line.strip()]

            included_indices: List[int] = []
            for idx, record in enumerate(records):
                if only_finished_thinking and (not _thinking_finished(record)):
                    continue
                included_indices.append(idx)
            if max_records_per_file is not None:
                included_indices = included_indices[:max_records_per_file]

            for idx in included_indices:
                record = records[idx]
                question = str(record["question"])
                gold = str(record["gold_answer"])
                answers = record["answers"]

                # Get test_list for coding datasets (if available in record)
                test_list = record.get("test_list") if dataset_type == "coding" else None

                for key, _ in MODEL_SPECS:
                    answer_text = clean_answer(str(answers[key]))
                    prompt = _build_judge_prompt(
                        question, gold, answer_text,
                        dataset_type=dataset_type,
                        test_list=test_list,
                    )
                    prompts.append(prompt)
                    prompt_mapping.append((path, idx, key))

    return prompts, prompt_mapping


def update_all_files(
    files: List[Tuple[str, List[str]]],
    prompt_mapping: List[Tuple[str, int, str]],
    responses: Dict[int, str],
    *,
    only_finished_thinking: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Update all files with judge responses.

    Returns:
        per_prefix_stats: Dict mapping prefix to stats dict with:
            - total: number of records
            - correct: {model_key: count}
            - changed: {model_key: count}
            - eos: {model_key: count}
    """
    # Group responses by file
    file_updates: Dict[str, Dict[int, Dict[str, str]]] = {}
    for prompt_idx, (path, record_idx, model_key) in enumerate(prompt_mapping):
        response = responses.get(prompt_idx, "")
        file_updates.setdefault(path, {}).setdefault(record_idx, {})[model_key] = response

    # Track stats per prefix
    per_prefix_stats: Dict[str, Dict[str, Any]] = {}

    # Process each file group
    for prefix, paths in tqdm(files, desc="Updating files", unit="prefix"):
        prefix_total = 0
        prefix_correct: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
        prefix_changed: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
        prefix_eos: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}

        for path in paths:
            with open(path, "r", encoding="utf-8") as src:
                records = [json.loads(line) for line in src if line.strip()]

            updates = file_updates.get(path, {})
            included_indices = sorted(updates.keys())
            if only_finished_thinking:
                included_indices = [i for i in included_indices if _thinking_finished(records[i])]

            for idx, record in enumerate(records):
                if idx not in updates:
                    continue
                if idx not in included_indices:
                    continue

                existing_judges = record.get("judges", {})
                record.setdefault("judges", {})

                for key, _ in MODEL_SPECS:
                    raw = updates[idx].get(key, "")
                    is_correct = "yes" in raw.lower()

                    # Get previous value
                    prev_entry = existing_judges.get(key)
                    prev_correct = None
                    if isinstance(prev_entry, dict):
                        val = prev_entry.get("correct")
                        if isinstance(val, bool):
                            prev_correct = val

                    record["judges"][key] = {"correct": bool(is_correct), "raw": raw}

                    if is_correct:
                        prefix_correct[key] += 1

                    if prev_correct is None or bool(is_correct) != bool(prev_correct):
                        prefix_changed[key] += 1

            # Count EOS for included records in the file
            for i in included_indices:
                record = records[i]
                eos_data = record.get("eos", {})
                for key, _ in MODEL_SPECS:
                    if eos_data.get(key, False):
                        prefix_eos[key] += 1

            prefix_total += len(included_indices)

            # Write back
            temp_path = f"{path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as dst:
                for record in records:
                    dst.write(json.dumps(record) + "\n")
            os.replace(temp_path, path)

        per_prefix_stats[prefix] = {
            "total": prefix_total,
            "correct": prefix_correct,
            "changed": prefix_changed,
            "eos": prefix_eos,
        }

    return per_prefix_stats


def compute_stats_from_existing(
    files: List[Tuple[str, List[str]]],
    *,
    only_finished_thinking: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Read existing judge results from files and compute stats without modifying anything.

    Returns:
        per_prefix_stats: Dict mapping prefix to stats dict with:
            - total: number of records
            - correct: {model_key: count}
            - eos: {model_key: count}
    """
    per_prefix_stats: Dict[str, Dict[str, Any]] = {}

    for prefix, paths in tqdm(files, desc="Computing stats", unit="prefix"):
        prefix_total = 0
        prefix_correct: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
        prefix_eos: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}

        for path in paths:
            with open(path, "r", encoding="utf-8") as src:
                records = [json.loads(line) for line in src if line.strip()]

            for record in records:
                # Skip if only_finished_thinking and thinking didn't finish
                if only_finished_thinking and (not _thinking_finished(record)):
                    continue

                prefix_total += 1

                # Read existing judge results
                judges = record.get("judges", {})
                for key, _ in MODEL_SPECS:
                    judge_entry = judges.get(key)
                    if isinstance(judge_entry, dict) and judge_entry.get("correct"):
                        prefix_correct[key] += 1

                # Count EOS
                eos_data = record.get("eos", {})
                for key, _ in MODEL_SPECS:
                    if eos_data.get(key, False):
                        prefix_eos[key] += 1

        per_prefix_stats[prefix] = {
            "total": prefix_total,
            "correct": prefix_correct,
            "eos": prefix_eos,
        }

    return per_prefix_stats


def _print_stats(
    files: List[Tuple[str, List[str]]],
    per_prefix_stats: Dict[str, Dict[str, Any]],
    only_finished_thinking: bool,
    recompute_only: bool = False,
) -> None:
    """Print per-model results."""
    total_files = sum(len(paths) for _, paths in files)
    total_records = sum(s["total"] for s in per_prefix_stats.values())

    action = "Read" if recompute_only else "Re-evaluated"
    subset = "finished-thinking " if only_finished_thinking else ""
    print(f"\n{action} {total_records} {subset}records across {total_files} files.")

    for prefix, stats in per_prefix_stats.items():
        prefix_name = os.path.basename(prefix).replace(".jsonl", "")
        n = stats["total"]
        if n == 0:
            print(f"\n[WARNING] Skipping {prefix_name}: 0 records" + (" with finished thinking" if only_finished_thinking else ""))
            continue

        thinking_correct = stats["correct"]["thinking"]
        base_correct = stats["correct"]["base"]
        hybrid_correct = stats["correct"]["hybrid"]

        thinking_acc = thinking_correct / n * 100
        base_acc = base_correct / n * 100
        hybrid_acc = hybrid_correct / n * 100

        print(f"\n===== {prefix_name} =====")
        print(f"Thinking Model: {thinking_correct}/{n} correct ({thinking_acc:.1f}%)")
        print(f"Base Model: {base_correct}/{n} correct ({base_acc:.1f}%)")
        print(f"Hybrid Model: {hybrid_correct}/{n} correct ({hybrid_acc:.1f}%)")

        # Gap recovered by hybrid
        gap = abs(thinking_acc - base_acc)
        if gap > 0:
            recovered = (hybrid_acc - min(base_acc, thinking_acc)) / gap
            print(f"Gap recovered by hybrid: {max(0.0, recovered) * 100:.1f}% of |Thinking-Base|")
        else:
            print("Gap recovered by hybrid: n/a")

        # EOS endings
        eos_base = stats["eos"]["base"] / n * 100
        eos_thinking = stats["eos"]["thinking"] / n * 100
        eos_hybrid = stats["eos"]["hybrid"] / n * 100
        print(f"EOS endings: base {eos_base:.1f}, thinking {eos_thinking:.1f}, hybrid {eos_hybrid:.1f}")


def main() -> None:
    args = parse_args()
    rolling_dir = args.rolling_dir or _default_rolling_dir()
    assert os.path.isdir(rolling_dir), f"Rolling directory not found: {rolling_dir}"
    only_finished_thinking = bool(getattr(args, "only_finished_thinking", False))

    # Parse filter
    exclude_patterns: Optional[List[str]] = None
    if args.filter == "all":
        filter_patterns = None
    elif args.filter == "qwen-orz":
        filter_patterns = QWEN_ORZ_MODELS
        # Exclude variants that use different thinking models (e.g., DeepSeek)
        exclude_patterns = ["deepseek"]
    else:
        filter_patterns = [p.strip() for p in args.filter.split(",")]

    # Get files to process
    if args.prefix:
        prefix = _resolve_prefix(args.prefix, rolling_dir)
        files = [(prefix, _list_ordered_files(prefix))]
        print(f"Found {len(files[0][1])} files for prefix {prefix}")
    else:
        files = _list_all_rollings(rolling_dir, filter_patterns, exclude_patterns)
        if not files:
            print(f"No rolling files found in {rolling_dir} matching filter: {args.filter}")
            return
        total_files = sum(len(paths) for _, paths in files)
        print(f"Found {len(files)} rolling groups ({total_files} files) matching filter: {args.filter}")

    # Show what we're processing
    print("\nFiles to process:")
    for prefix, paths in files:
        print(f"  {os.path.basename(prefix)}: {len(paths)} file(s)")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above files. Exiting.")
        return

    # Recompute-only mode: read existing judge results and print stats
    if args.recompute_only:
        print("\n[RECOMPUTE-ONLY] Reading existing judge results...")
        per_prefix_stats = compute_stats_from_existing(
            files,
            only_finished_thinking=only_finished_thinking,
        )
        _print_stats(files, per_prefix_stats, only_finished_thinking, recompute_only=True)
        return

    if args.debug:
        print("\n[DEBUG MODE] Processing only 1 record per file.")

    # Collect all prompts
    print("\nPhase 1: Collecting all prompts...")
    max_records = 1 if args.debug else None
    prompts, prompt_mapping = collect_all_prompts(
        files,
        max_records_per_file=max_records,
        only_finished_thinking=only_finished_thinking,
    )
    print(f"Collected {len(prompts)} prompts total")

    if not prompts:
        print("No prompts to process.")
        return

    # Submit batches
    print("\nPhase 2: Submitting to OpenAI Batch API...")
    evaluator = BatchEvaluator(
        judge_model=args.judge_model,
        max_tokens=args.max_judge_tokens,
        poll_interval=args.poll_interval,
        batch_size=args.batch_size,
    )
    batch_ids = evaluator.submit_batches(prompts)

    # Poll until complete
    print("\nPhase 3: Polling for completion...")
    responses = evaluator.poll_batches(batch_ids, len(prompts))
    print(f"Received {len(responses)} responses")

    # Update all files
    print("\nPhase 4: Updating files...")
    per_prefix_stats = update_all_files(
        files,
        prompt_mapping,
        responses,
        only_finished_thinking=only_finished_thinking,
    )

    # Print per-model results
    _print_stats(files, per_prefix_stats, only_finished_thinking)


if __name__ == "__main__":
    main()
