import time
from openai import OpenAI

from .rule_eval import evaluate_gpqa, evaluate_math, evaluate_mbpp, is_mbpp_executable_tests

# =========================================================
# 1. 配置 DeepSeek 客户端
# =========================================================
# 建议使用环境变量 DEEPSEEK_API_KEY；若无则使用占位（请自行替换）
import os

DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "sk-86f11ad180ce4641913a75afa6f034ef",
)

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

# =========================================================
# 2. 核心函数：Safe Chat Batch (适配 DeepSeek)
# =========================================================
def safe_chat_batch(prompts, model_name="deepseek-reasoner", max_tokens=4000, **kwargs):
    """
    使用 DeepSeek API 进行判分。
    默认使用 'deepseek-reasoner' (R1) 以获得最强的逻辑判断能力。
    """
    results = []
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            content = response.choices[0].message.content
            results.append(content)
        except Exception as e:
            print(f"❌ DeepSeek API Error: {e}")
            results.append("")

    return results


def _evaluate_coding_llm(model_answer, correct_answer, question, model_name, test_list):
    """LiveCodeBench 等非 assert 测例：仍走 LLM（与旧版 coding prompt 一致）。"""
    test_cases_str = "\n\n".join(test_list) if test_list else "No test cases provided"
    reference_section = (
        f"""

Reference solution:
```python
{correct_answer}
```"""
        if correct_answer
        else ""
    )
    prompt = f"""Please evaluate whether the model's response contains a correct solution to this coding problem.

Problem: {question}

Model's response (including reasoning and code):
{model_answer}
{reference_section}

Test cases that a correct solution should pass:
{test_cases_str}

Instructions for evaluation:
1. Search the ENTIRE model response for any Python code that could solve the problem.
2. IMPORTANT: If a correct implementation appears ANYWHERE in the response - whether in the final code block, during reasoning/planning, or in a "let me try" section - answer YES.
3. The model gets credit if it wrote correct code at any point, even if:
   - It appeared during an intermediate attempt
   - The response continues with a different (possibly incorrect) version afterward
   - There are multiple code attempts and at least one is correct
4. Evaluate if the code logic would produce correct outputs for the test cases.
5. Only answer NO if no correct implementation appears anywhere in the response.

Just answer YES if a correct solution appears anywhere in the response, or NO if it doesn't. Nothing else.
"""
    max_retries = 3
    target_model = "deepseek-reasoner"
    for attempt in range(max_retries):
        try:
            response_list = safe_chat_batch([prompt], model_name=target_model)
            if response_list and response_list[0]:
                response = response_list[0].strip()
                return "yes" in response.lower(), response
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Judge API returned no response for {model_name}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            print(f"❌ Judge Failed: {e}")
            return False, "ERROR"
    return False, "ERROR"


# =========================================================
# 3. 评测主逻辑
# =========================================================
def evaluate_answer(
    model_answer,
    correct_answer,
    answer,
    question,
    model_name,
    dataset_type="math",
    test_list=None,
):
    """
    MATH / MCQA(GPQA 等) / Coding(MBPP)：默认使用 rule-based 或执行评测，无 API 费用。
    dataset_type == \"classification\"：仍使用 LLM judge。
    """
    if dataset_type == "math":
        return evaluate_math(model_answer, answer, correct_answer)
    if dataset_type == "mcqa":
        return evaluate_gpqa(model_answer, correct_answer)
    if dataset_type == "coding":
        if is_mbpp_executable_tests(test_list):
            return evaluate_mbpp(model_answer, test_list)
        return _evaluate_coding_llm(
            model_answer, correct_answer, question, model_name, test_list
        )

    if dataset_type != "classification":
        return evaluate_math(model_answer, answer, correct_answer)

    # --- Classification：仅此类别保留 LLM ---
    prompt = f"""Please evaluate whether the model arrived at the correct classification/answer for this task.

Question/Context: {question}

Correct answer: {correct_answer}

Model's response (including reasoning): {model_answer}

Instructions for evaluation:
1. First, identify the correct answer/classification: "{correct_answer}"
2. Search the ENTIRE model response (including all reasoning steps) for this correct answer.
3. IMPORTANT: If the correct answer appears ANYWHERE in the model's response as the conclusion/classification - whether in the final statement, during analysis, or when considering options - answer YES.
4. The model gets credit if it stated the correct answer at any point, even if:
   - It appeared during intermediate reasoning
   - The final stated answer is different (due to errors, continued generation, etc.)
   - The response continues or becomes garbled after stating the correct answer
5. The comparison should be case-insensitive and ignore minor formatting differences.
6. Only answer NO if the correct answer/classification does not appear anywhere in the response as a conclusion.

Just answer YES if the correct answer appears anywhere in the response, or NO if it doesn't. Nothing else.
"""

    max_retries = 3
    target_model = "deepseek-reasoner"

    for attempt in range(max_retries):
        try:
            response_list = safe_chat_batch([prompt], model_name=target_model)
            if response_list and response_list[0]:
                response = response_list[0].strip()
                is_correct = "yes" in response.lower()
                return is_correct, response
            if attempt < max_retries - 1:
                print(
                    f"Judge API returned no response for {model_name}, retrying ({attempt + 1}/{max_retries})..."
                )
                time.sleep(2**attempt)
                continue
            raise RuntimeError(
                f"Judge API returned no response for {model_name} after {max_retries} attempts"
            )
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"Judge API error for {model_name}: {e}, retrying ({attempt + 1}/{max_retries})..."
                )
                time.sleep(2**attempt)
                continue
            print(f"❌ Judge Failed: {e}")
            return False, "ERROR"

    return False, "ERROR"
