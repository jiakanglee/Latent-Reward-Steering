#!/usr/bin/env python3
"""GPQA-Diamond baseline: do_sample, T=1, top_p=1, top_k=0; evaluate_answer(mcqa)."""

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


def get_gpqa_fields(item):
    q = item["question"]
    sol = item["solution"]
    ans = sol.replace("Answer:", "").strip().upper() if sol else ""
    if len(ans) > 1:
        ans = ans[0]
    return q, sol, ans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
    p.add_argument("--max_token", type=int, default=4000)
    p.add_argument("--num_examples", type=int, default=None, help="只跑本分片前 N 题（试跑）")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--output_jsonl", default=None)
    p.add_argument("--print_response", action="store_true")
    args = p.parse_args()

    ds = load_dataset("nichenshun/gpqa_diamond")["train"]
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

    print(
        f"do_sample=True T=1 top_p=1 top_k=0 | shard {args.shard_id}/{args.num_shards} | "
        f"{len(idxs)}/{n_full} questions"
    )
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    ok = 0
    for i in tqdm(idxs, desc="GPQA"):
        q, _s, ans = get_gpqa_fields(ds[i])
        text = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        inp = {k: v.to(device) for k, v in tok(text, return_tensors="pt").items()}
        L = inp["input_ids"].shape[1]
        kw = dict(
            max_new_tokens=args.max_token,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tok.pad_token_id,
        )
        with torch.no_grad():
            try:
                out = model.generate(**inp, **kw)
            except Exception:
                kw.pop("top_k", None)
                out = model.generate(**inp, **kw)
        res = tok.decode(out[0][L:], skip_special_tokens=False)
        if args.print_response:
            print(res[:4000])
        c, judge = evaluate_answer(res, ans, ans, q, "Baseline", dataset_type="mcqa", test_list=None)
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
