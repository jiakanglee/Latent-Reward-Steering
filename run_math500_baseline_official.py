#!/usr/bin/env python3
"""MATH-500 baseline：evaluate_answer(math)。

- temperature > 0：do_sample=True，配合 top_p / top_k（top_k=0 表示不传 top_k）。
- temperature <= 0：贪心解码 do_sample=False（忽略 top_p / top_k）。
默认与旧版一致：T=1, top_p=1, top_k=0。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.llm_judge import evaluate_answer


def get_math500_fields(item):
    return item["problem"], item["solution"], item["answer"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
    p.add_argument("--max_token", type=int, default=2500)
    p.add_argument("--num_examples", type=int, default=None, help="只跑本分片前 N 题（试跑）")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--output_jsonl", default=None)
    p.add_argument("--print_response", action="store_true")
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=">0 时采样；<=0 时贪心解码 do_sample=False",
    )
    p.add_argument("--top_p", type=float, default=1.0, help="nucleus 采样 top_p")
    p.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="top_k；0 表示不把 top_k 传给 generate（与旧脚本一致）",
    )
    args = p.parse_args()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    n_full = len(ds)
    idxs = [j for j in range(n_full) if j % args.num_shards == args.shard_id]
    if args.num_examples is not None:
        idxs = idxs[: args.num_examples]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device

    greedy = float(args.temperature) <= 0.0
    if greedy:
        print(
            f"greedy do_sample=False (temperature<=0) | "
            f"shard {args.shard_id}/{args.num_shards} | {len(idxs)}/{n_full} questions"
        )
    else:
        tk_str = "omit" if args.top_k <= 0 else str(args.top_k)
        print(
            f"do_sample=True T={args.temperature} top_p={args.top_p} top_k={tk_str} | "
            f"shard {args.shard_id}/{args.num_shards} | {len(idxs)}/{n_full} questions"
        )
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    ok = 0
    for i in tqdm(idxs, desc="MATH-500"):
        q, solution, answer = get_math500_fields(ds[i])
        text = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        inp = {k: v.to(device) for k, v in tok(text, return_tensors="pt").items()}
        L = inp["input_ids"].shape[1]
        kw = dict(max_new_tokens=args.max_token, pad_token_id=tok.pad_token_id)
        if greedy:
            kw["do_sample"] = False
        else:
            kw["do_sample"] = True
            kw["temperature"] = float(args.temperature)
            kw["top_p"] = float(args.top_p)
            if args.top_k > 0:
                kw["top_k"] = int(args.top_k)
        with torch.no_grad():
            try:
                out = model.generate(**inp, **kw)
            except Exception:
                kw.pop("top_k", None)
                out = model.generate(**inp, **kw)
        res = tok.decode(out[0][L:], skip_special_tokens=False)
        if args.print_response:
            print(res[:4000])
        c, judge = evaluate_answer(
            res,
            solution,
            answer,
            q,
            "Baseline",
            dataset_type="math",
            test_list=None,
        )
        if c:
            ok += 1
        if args.output_jsonl:
            with open(args.output_jsonl, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"idx": i, "correct": bool(c), "judge": (judge or "")[:500]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    t = len(idxs)
    print(f"\ncorrect / total = {ok} / {t} = {(ok / t if t else 0):.4f}")


if __name__ == "__main__":
    main()
