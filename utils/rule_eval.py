"""
Rule-based / execution-based evaluation (no LLM judge).

- math: math_verify (parse + verify), for MATH-500 / AIME / GSM8K-style answers.
- mcqa: extract choice letter vs gold (GPQA-Diamond, MedQA-style).
- coding: extract Python + run MBPP test_list asserts in a subprocess (timeout).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

try:
    from math_verify import parse, verify
except ImportError as e:
    parse = None  # type: ignore
    verify = None  # type: ignore
    _MATH_VERIFY_IMPORT_ERROR = e
else:
    _MATH_VERIFY_IMPORT_ERROR = None


def _math_verify_available() -> bool:
    return parse is not None and verify is not None


def evaluate_math(
    model_answer: str,
    answer: Optional[str],
    solution: Optional[str],
) -> Tuple[bool, str]:
    """
    Gold: prefer short `answer` (numerical / boxed-style), else try full `solution` tex.
    Prediction: full model output (math_verify extracts \\boxed etc.).
    """
    if not _math_verify_available():
        return False, f"rule_math:math_verify_missing:{_MATH_VERIFY_IMPORT_ERROR}"

    candidates: List[str] = []
    for g in (answer, solution):
        if g is None:
            continue
        s = str(g).strip()
        if s and s not in candidates:
            candidates.append(s)

    if not candidates:
        return False, "rule_math:no_gold"

    pred_parsed = parse(model_answer or "")
    if not pred_parsed:
        return False, "rule_math:empty_pred_parse"

    for g in candidates:
        gp = parse(g)
        if not gp:
            continue
        try:
            if verify(gp, pred_parsed, raise_on_error=False):
                return True, "rule_math:math_verify"
        except Exception:
            continue
    return False, "rule_math:math_verify_false"


def extract_choice_letter(text: str, valid: Optional[str] = None) -> Optional[str]:
    """Extract A–Z choice from model output; **rightmost** high-confidence match wins."""
    if not text:
        return None
    t = text.upper()
    if valid is None:
        valid = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # LaTeX \\boxed{A}: use r"\\boxed" (one backslash in the regex, via pair `\\` in the raw string).
    # Match on **original** `text` with re.IGNORECASE — `text.upper()` turns "\\boxed" into "\\BOXED"
    # and a literal "boxed" pattern on `t` would miss.
    boxed = r"\\boxed\s*\{\s*([A-Za-z])\s*\}"
    # `<answer>` 内常有长句再接 `\boxed{X}`（非紧邻标签）。
    boxed_wrap = r"<answer>\s*(?:[\s\S]*?)\\boxed\s*\{\s*([A-Za-z])\s*\}"
    best_pos = -1
    best: Optional[str] = None

    def consider(m: re.Match[str]) -> None:
        nonlocal best_pos, best
        ch = m.group(1).upper()
        if ch in valid and m.start() >= best_pos:
            best_pos = m.start()
            best = ch

    for pat in (boxed, boxed_wrap):
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
            consider(m)

    patterns = [
        r"(?:FINAL\s+)?(?:ANSWER|CHOICE|OPTION)\s*[:.)-]\s*\(?([A-Z])\)?(?:\b|\.|,|\))",
        r"(?:FINAL\s+)?(?:ANSWER|CHOICE|OPTION)\s*[:.)-]\s*([A-Z])\b",
        r"\*\*([A-Z])\*\*(?=\s|$|[^A-Z*])",
        r"(?:SELECTED|CHOSEN)\s+OPTION\s+IS\s+([A-Z])\b",
        r"\bTHE\s+CORRECT\s+(?:ANSWER|OPTION)\s+IS\s+\(([A-Z])\)",
        r"\bTHE\s+CORRECT\s+(?:ANSWER|OPTION)\s+IS\s+([A-Z])\b",
        r"\bCORRECT\s+(?:ANSWER|OPTION)\s+IS\s+\(([A-Z])\)",
        r"\bCORRECT\s+(?:ANSWER|OPTION)\s+IS\s+([A-Z])\b",
        r"(?:THEREFORE|THUS)[,:]?\s+(?:THE\s+)?CORRECT\s+(?:ANSWER|OPTION)\s+IS\s+\(([A-Z])\)",
        r"(?:THEREFORE|THUS)[,:]?\s+(?:THE\s+)?CORRECT\s+(?:ANSWER|OPTION)\s+IS\s+([A-Z])\b",
        r"\(([A-Z])\)\s*(?:IS\s+)?CORRECT",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            consider(m)

    if best:
        return best

    # Last line: isolated letter
    for line in reversed(t.strip().splitlines()):
        s = line.strip().strip("*.-) ")
        if len(s) == 1 and s in valid:
            return s
    return None


def evaluate_gpqa(model_answer: str, gold_letter: str) -> Tuple[bool, str]:
    g = (gold_letter or "").strip().upper()[:1]
    if not g or g not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return False, "rule_mcqa:bad_gold"
    pred = extract_choice_letter(model_answer or "")
    if pred is None:
        return False, "rule_mcqa:no_letter"
    if pred == g:
        return True, "rule_mcqa:letter_match"
    return False, f"rule_mcqa:got_{pred}_want_{g}"


def extract_python_blocks(text: str) -> List[str]:
    if not text:
        return []
    blocks = re.findall(r"```\s*python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    blocks = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
    return [b.strip() for b in blocks if b.strip() and "def " in b]


def is_mbpp_executable_tests(test_list: Optional[List[str]]) -> bool:
    """MBPP 的 test_list 为 assert 语句；其它 benchmark 可能仅为自然语言描述。"""
    if not test_list:
        return False
    return any("assert" in str(t).lower() for t in test_list)


def evaluate_mbpp(
    model_answer: str,
    test_list: Optional[List[str]],
    timeout_sec: float = 15.0,
) -> Tuple[bool, str]:
    if not test_list:
        return False, "rule_mbpp:no_tests"
    candidates = extract_python_blocks(model_answer or "")
    if not candidates and "def " in (model_answer or ""):
        candidates = [(model_answer or "").strip()]

    if not candidates:
        return False, "rule_mbpp:no_code"

    tests_block = "\n".join(test_list)
    last_err = ""
    for code in reversed(candidates):
        src = f"{code}\n\n{tests_block}\n"
        ok, err = _run_python_src(src, timeout_sec)
        if ok:
            return True, "rule_mbpp:exec_pass"
        last_err = err[:400]
    return False, f"rule_mbpp:exec_fail:{last_err}"


def _run_python_src(src: str, timeout_sec: float) -> Tuple[bool, str]:
    fd, path = tempfile.mkstemp(suffix="_mbpp_eval.py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(src)
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if r.returncode == 0:
            return True, ""
        msg = (r.stderr or r.stdout or "") or f"exit_{r.returncode}"
        return False, msg
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
