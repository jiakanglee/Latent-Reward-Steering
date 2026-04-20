#!/usr/bin/env python3
"""
Greedy-decode GPQA-Diamond 单题，打印并写入文件 —— 与 run_basic_overwrite.py 中
gpqa 分支一致：仅 user 消息为题干，apply_chat_template + add_generation_prompt，do_sample=False。

用法（在带 GPU 的机器上）:
  cd thinking-llms-interp
  python scripts/dump_gpqa_decode_sample.py --question_idx 0 --max_token 4000

无 GPU 时会直接退出并提示到计算节点跑。
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(
        description="Dump one GPQA-Diamond greedy decode (same as run_basic_overwrite / dataset gpqa_diamond)."
    )
    p.add_argument(
        "--model",
        default="Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
        help="与 run_basic_overwrite --model 一致",
    )
    p.add_argument("--question_idx", type=int, default=0)
    p.add_argument("--max_token", type=int, default=4000, help="max_new_tokens，对应 run_basic_overwrite --max_token")
    p.add_argument(
        "--out",
        default=None,
        help="输出文件（默认 log2/gpqa_decode_sample_q{idx}.txt）",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="cuda 或 cpu（7B CPU 极慢，仅调试用）",
    )
    p.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="与加载相关；无 flash_attn 时用 sdpa",
    )
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "FATAL: 本机无 CUDA。请在 GPU 节点上运行，例如: "
            "srun / apptainer / 登录到 rlab* 再执行。",
            file=sys.stderr,
        )
        sys.exit(1)

    ds = load_dataset("nichenshun/gpqa_diamond")["train"]
    if args.question_idx < 0 or args.question_idx >= len(ds):
        print(f"FATAL: question_idx 需在 [0, {len(ds) - 1}]", file=sys.stderr)
        sys.exit(1)

    row = ds[args.question_idx]
    question = row["question"]
    solution = row.get("solution") or ""

    attn_impl = None if args.attn == "sdpa" else args.attn
    load_kw: dict = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if args.device == "cuda":
        load_kw["device_map"] = "auto"
    if attn_impl:
        load_kw["attn_implementation"] = attn_impl

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print(f"⚠️ flash_attention_2 失败 ({e!r})，回退 sdpa。", file=sys.stderr)
            load_kw.pop("attn_implementation", None)
            load_kw["attn_implementation"] = "sdpa"
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
        else:
            raise

    if args.device == "cpu":
        model = model.to("cpu", dtype=torch.float32)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_token,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=False)

    lines = [
        f"model={args.model}",
        f"question_idx={args.question_idx}",
        f"max_new_tokens={args.max_token}",
        f"gold_solution_field={solution!r}",
        "",
        "=== USER MESSAGE ONLY (dataset question field) ===",
        question,
        "",
        "=== DECODED MODEL CONTINUATION ONLY (after prompt) ===",
        decoded,
        "",
    ]
    text = "\n".join(lines)
    print(text)

    out_path = args.out or os.path.join("log2", f"gpqa_decode_sample_q{args.question_idx}.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[wrote {os.path.abspath(out_path)}]", file=sys.stderr)


if __name__ == "__main__":
    main()
