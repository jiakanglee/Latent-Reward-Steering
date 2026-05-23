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
    # `\boxed{(A) 全名...}`：花括号内以括号+选项字母开头（常见 GPQA / 化学长名）。
    boxed_paren = r"\\boxed\s*\{\s*\(([A-Za-z])\)"
    boxed_wrap_paren = r"<answer>\s*(?:[\s\S]*?)\\boxed\s*\{\s*\(([A-Za-z])\)"
    best_pos = -1
    best: Optional[str] = None

    def consider(m: re.Match[str]) -> None:
        nonlocal best_pos, best
        ch = m.group(1).upper()
        if ch in valid and m.start() >= best_pos:
            best_pos = m.start()
            best = ch

    for pat in (boxed, boxed_wrap, boxed_paren, boxed_wrap_paren):
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


def ineqmath_gold_choice_letter(answer: Optional[str]) -> Optional[str]:
    """IneqMath relation 标答形如 ``(D) $<$``：取前导括号内选项字母 ``A``–``F``。"""
    if answer is None:
        return None
    m = re.match(r"^\(([A-F])\)", str(answer).strip(), flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


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


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_reasoner_tags_for_codegen(raw: str) -> str:
    """去掉 ORZ 的 redacted_thinking 标签段，避免把标签当 Python 执行触发 SyntaxError。"""
    if not raw:
        return raw
    s = raw
    if _THINK_CLOSE in s:
        s = s.rsplit(_THINK_CLOSE, 1)[-1]
    s = re.sub(
        re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE),
        "",
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )
    s = s.replace(_THINK_OPEN, "").replace(_THINK_CLOSE, "")
    return s.strip()


# MBPP：整段 fallback 时，丢弃首个 `def`/`async def` 之前的自然语言（常见「To solve...」）。
_DEF_LEAD = re.compile(
    r"(?m)^(\s*(?:@[\w\.]+\s*)*)(?:async\s+)?def\s+\w+\s*\(",
)


def slice_from_first_function_def(raw: str) -> str:
    if not raw:
        return raw
    m = _DEF_LEAD.search(raw)
    if m:
        return raw[m.start() :].lstrip()
    return raw


def strip_hf_answer_wrappers(raw: str) -> str:
    """去掉 ORZ 常见 `<answer>`, `<python>` 伪标签与 `<|endoftext|>`，避免 `</python>` 进执行文件。"""
    if not raw:
        return raw
    s = re.sub(r"<\|endoftext\|>[\s\S]*", "", raw)
    for pat in (
        r"<\s*answer\s*>",
        r"<\s*/\s*answer\s*>",
        r"<\s*python\s*>",
        r"<\s*/\s*python\s*>",
    ):
        s = re.sub(pat, "\n", s, flags=re.IGNORECASE)
    return s.strip()


def trim_mbpp_trailing_prose(code: str) -> str:
    """截断代码块后粘贴的 Markdown / 说明（Let's verify、###、闭合 fence 等）。"""
    if not code:
        return code
    for pat in (
        r"\n###\s",
        r"\n##\s",
        r"(?im)^let's verify\b",
        r"(?i)\nlet's verify\b",
        r"\n```\s*\n",
        r"\n```\s*$",
        r"(?i)</\s*python\s*>",
        r"(?i)</\s*answer\s*>",
        r"<\|endoftext\|>",
    ):
        m = re.search(pat, code)
        if m:
            code = code[: m.start()]
    return code.rstrip()


def unwrap_boxed_python(code: str) -> str:
    """去掉行首 `\\boxed{ ... }` 包裹（数学模板误套在整段代码上）。"""
    s = code.strip()
    while True:
        m = re.match(r"^\\boxed\s*\{", s)
        if not m:
            break
        start = m.end() - 1
        depth = 0
        j = start
        found_close = False
        while j < len(s):
            c = s[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    s = s[start + 1 : j].strip()
                    found_close = True
                    break
            j += 1
        if not found_close:
            # 常见截断：只有 `\\boxed{def ...` 没有闭合 `}`
            s = s[start + 1 :].strip()
        break
    return s


def normalize_mbpp_code_fragment(code: str) -> str:
    """对单段候选代码：去伪标签 / unwrap \\boxed、对齐到首个 def、截断尾部 Markdown。"""
    if not code:
        return code
    s = strip_hf_answer_wrappers(code)
    s = unwrap_boxed_python(s)
    if re.search(r"(?m)^(\s*(?:@[\w\.]+\s*)*)(?:async\s+)?def\s+\w+\s*\(", s):
        s = slice_from_first_function_def(s)
    s = trim_mbpp_trailing_prose(s)
    return s.strip()


def extract_python_blocks(text: str) -> List[str]:
    """抽取 Markdown 代码块；优先 ```python / ```py，开标签后允许无换行（与 prompt 约定一致）。"""
    if not text:
        return []
    blocks = re.findall(
        r"```\s*(?:python|py)\b\s*\n?(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not blocks:
        # 非 python 标记的 fence：避免与已匹配的 ```python 重复；且正文里须含 def（MBPP）
        blocks = re.findall(
            r"```(?!\s*(?:python|py)\b)(?:[^\n`]*)\s*\n?(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        blocks = [b for b in blocks if b.strip() and "def " in b]
    if not blocks:
        return []
    # 同长或无序时保留稳定顺序：先按长度降序去重
    out: List[str] = []
    seen = set()
    for b in sorted((x.strip() for x in blocks if x.strip()), key=len, reverse=True):
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _mbpp_entry_function_name(test_list: Optional[List[str]]) -> Optional[str]:
    """首条 assert 里的被测函数名，用于优先选对代码块。"""
    if not test_list:
        return None
    m = re.search(r"assert\s+([a-zA-Z_]\w*)\s*\(", str(test_list[0]))
    return m.group(1) if m else None


def _rank_mbpp_candidates_for_tests(candidates: List[str], test_list: Optional[List[str]]) -> List[str]:
    """含 `def <入口名>(` 的块优先，其次更长块（仍可能有多块时降低 NameError）。"""
    name = _mbpp_entry_function_name(test_list)
    if not name or not candidates:
        return candidates
    lead = re.compile(
        rf"(?m)^(\s*(?:@[\w\.]+\s*)*)(?:async\s+)?def\s+{re.escape(name)}\s*\(",
    )

    def key(b: str) -> tuple:
        has = bool(lead.search(b))
        return (0 if has else 1, -len(b))

    return sorted(candidates, key=key)


def is_mbpp_executable_tests(test_list: Optional[List[str]]) -> bool:
    """MBPP 的 test_list 为 assert 语句；其它 benchmark 可能仅为自然语言描述。"""
    if not test_list:
        return False
    return any("assert" in str(t).lower() for t in test_list)


# 在模型答案前注入：减少「忘写 import」导致的 NameError，使失败主要来自 assert 语义。
_MBPP_EVAL_GLOBAL_PREAMBLE = """from __future__ import annotations
import bisect
import collections
import copy
import functools
import heapq
import itertools
import math
import operator
import random
import re
import string
import statistics
from collections import Counter, defaultdict, deque
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
try:
    import numpy as np
except ImportError:
    pass
"""


def evaluate_mbpp(
    model_answer: str,
    test_list: Optional[List[str]],
    timeout_sec: float = 15.0,
) -> Tuple[bool, str]:
    if not test_list:
        return False, "rule_mbpp:no_tests"
    cleaned = strip_reasoner_tags_for_codegen(model_answer or "")
    cleaned = strip_hf_answer_wrappers(cleaned)
    # 必须先抽 fenced 块，再 slice：否则 slice 会从全文第一个 def 切到末尾，把思考区示例与 <answer> 尾标一并吃进去。
    candidates = extract_python_blocks(cleaned)
    if not candidates and "def " in cleaned:
        candidates = [slice_from_first_function_def(cleaned).strip()]
    candidates = _rank_mbpp_candidates_for_tests(candidates, test_list)

    if not candidates:
        return False, "rule_mbpp:no_code"

    tests_block = "\n".join(test_list)
    last_err = ""
    for code in candidates:
        code = normalize_mbpp_code_fragment(code)
        if not code:
            continue
        src = f"{_MBPP_EVAL_GLOBAL_PREAMBLE}\n\n{code}\n\n{tests_block}\n"
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
