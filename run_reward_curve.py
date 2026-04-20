#!/usr/bin/env python3
"""
对 Base对Steer错 / Base错Steer对 的边例题目，分别记录 Base 和 Steer 生成时
reward model 的逐 token score，按每 10 token 取 mean 画曲线，每道题一张图。
用法: python run_reward_curve.py --example_idx 21 25 46 ... --output_dir logs/reward_curves
"""
import torch
import torch.nn as nn
import argparse
import sys
import os
import math
import json
import gc

sys.path.append(os.getcwd())

from datasets import load_dataset
from utils.utils import load_model, center_and_l2_normalize_torch
from utils.sae import load_sae

# 复用 run_basic_overwrite 的 LatentTransformer 和 IterativeTransformerSteeringHook
from run_basic_overwrite import LatentTransformer, IterativeTransformerSteeringHook, make_confidence_hook


class RecordRewardHook:
    """仅记录 reward，不 steering。用于 Base 生成时收集 reward 曲线。"""
    def __init__(self, sae, reward_model, sae_dim=10):
        self.sae = sae
        self.reward_model = reward_model
        self.sae_dim = sae_dim
        self.rewards = []  # 每 token 的 reward prob

        if hasattr(self.sae, "W_dec"):
            self.decoder_weight = self.sae.W_dec
        elif hasattr(self.sae, "decoder"):
            self.decoder_weight = self.sae.decoder
        else:
            raise ValueError("Could not find decoder weights in SAE.")
        if isinstance(self.decoder_weight, nn.Module):
            self.decoder_weight = self.decoder_weight.weight

    def _get_dense_latents(self, x):
        flat_x = x.view(-1, x.shape[-1])
        if hasattr(self.sae, "activation_mean"):
            pre_act = center_and_l2_normalize_torch(flat_x, self.sae.activation_mean)
        else:
            pre_act = flat_x - self.sae.b_dec
        top_acts, top_indices = self.sae.encode(pre_act)
        latents = torch.zeros((top_acts.shape[0], self.sae_dim), device=x.device, dtype=torch.float32)
        latents.scatter_(1, top_indices, top_acts.float())
        return latents

    def reset(self):
        self.rewards = []

    def __call__(self, module, inputs, outputs):
        original_act = outputs[0] if isinstance(outputs, tuple) else outputs
        batch_size, seq_len, _ = original_act.shape
        if seq_len > 1:
            return outputs
        x = original_act.to(self.sae.encoder.weight.dtype)
        with torch.no_grad():
            current_dense = self._get_dense_latents(x).view(batch_size, 1, -1)
            init_logits = self.reward_model(current_dense, return_logits=True)
            init_logit_val = init_logits[:, -1, 0].sum().item() / batch_size
            init_prob_val = 1 / (1 + math.exp(-init_logit_val)) if abs(init_logit_val) < 50 else (1.0 if init_logit_val > 0 else 0.0)
        self.rewards.append(init_prob_val)
        return outputs


def downsample_mean(rewards: list[float], window: int = 10) -> tuple[list[int], list[float]]:
    """每 window 个 token 取 mean，返回 (x 轴 token 索引, y 轴 mean 值)"""
    if not rewards:
        return [], []
    xs, ys = [], []
    for i in range(0, len(rewards), window):
        chunk = rewards[i : i + window]
        xs.append(i + (len(chunk) - 1) / 2)  # 窗口中心
        ys.append(sum(chunk) / len(chunk))
    return xs, ys


def plot_curves(base_rewards: list[float], steer_rewards: list[float], qid: int,
                base_correct: bool, steer_correct: bool, case_type: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_x, base_y = downsample_mean(base_rewards, 10)
    steer_x, steer_y = downsample_mean(steer_rewards, 10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(base_x, base_y, "b-", label="Base", linewidth=2)
    ax.plot(steer_x, steer_y, "r-", label="Steer", linewidth=2)
    ax.set_xlabel("Token index (每10 token mean)")
    ax.set_ylabel("Reward model score (prob)")
    ax.set_title(f"Q{qid} [{case_type}] Base{'✅' if base_correct else '❌'} Steer{'✅' if steer_correct else '❌'}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
    parser.add_argument("--sae_layer", type=int, default=20)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--reward_model_path", type=str, default="transformer_reward_model_best.pt")
    parser.add_argument("--example_idx", type=int, nargs="+", required=True, help="题目 ID 列表")
    parser.add_argument("--max_token", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default="logs/reward_curves")
    parser.add_argument("--edge_json", type=str, default="logs/edge_cases_24.json")
    parser.add_argument("--reward_threshold", type=float, default=0.8)
    parser.add_argument("--confidence_threshold", type=float, default=0.73)
    parser.add_argument("--step_size", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=5)
    args = parser.parse_args()

    print("Loading model, SAE, reward model...")
    model, tokenizer = load_model(model_name=args.model, device="cuda")
    sae, _ = load_sae("open-reasoner-zero-7b", args.sae_layer, args.n_clusters)
    sae = sae.to(model.device)
    sae.k = args.n_clusters

    reward_model = LatentTransformer(input_dim=args.n_clusters, d_model=128).to(model.device)
    reward_model.load_state_dict(torch.load(args.reward_model_path, map_location=model.device))
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # 加载 edge cases 以获取 base/steer 对错（若本地无结果则需跑一遍或从 sweep 解析）
    edge_data = {}
    if os.path.isfile(args.edge_json):
        with open(args.edge_json, "r", encoding="utf-8") as f:
            edge_data = json.load(f)
    base_right_steer_wrong = set(edge_data.get("base_right_steer_wrong_ids", []))
    base_wrong_steer_right = set(edge_data.get("base_wrong_steer_right_ids", []))
    edge_ids = set(edge_data.get("edge_case_ids", args.example_idx))

    os.makedirs(args.output_dir, exist_ok=True)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        lm_head = getattr(model.model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Could not find lm_head on model.")

    record_hook = RecordRewardHook(sae, reward_model, args.n_clusters)
    steer_hook = IterativeTransformerSteeringHook(
        sae, reward_model,
        step_size=args.step_size, num_steps=args.num_steps,
        sae_dim=args.n_clusters, monitor=False,
        reward_threshold=args.reward_threshold,
        confidence_threshold=args.confidence_threshold,
        lm_head=lm_head,
        record_tokens=True,
    )
    layer_module = model.model.layers[args.sae_layer]

    for i in args.example_idx:
        if i not in edge_ids:
            print(f"Skip Q{i} (not in edge cases)")
            continue
        try:
            item = dataset[i]
            question = item["problem"]
            messages = [{"role": "user", "content": question}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]

            # 1. Base：仅记录 reward
            print(f"Q{i}: Base...", end="", flush=True)
            record_hook.reset()
            h_base = layer_module.register_forward_hook(record_hook)
            with torch.no_grad():
                out_base = model.generate(**inputs, max_new_tokens=args.max_token, do_sample=False)
            h_base.remove()
            base_rewards = list(record_hook.rewards)
            len_base = out_base.shape[1] - input_len
            print(f" len={len_base} rewards={len(base_rewards)}")

            # 2. Steer：用完整 hook，record_tokens=True 会记录 reward_prob
            print(f"Q{i}: Steer...", end="", flush=True)
            steer_hook.reset()
            h_steer = layer_module.register_forward_hook(steer_hook)
            h_conf = lm_head.register_forward_hook(make_confidence_hook(steer_hook))
            try:
                with torch.no_grad():
                    out_steer = model.generate(**inputs, max_new_tokens=args.max_token, do_sample=False)
                steer_rewards = [r["reward_prob"] for r in steer_hook.token_records]
            finally:
                h_steer.remove()
                h_conf.remove()
            len_steer = out_steer.shape[1] - input_len
            print(f" len={len_steer} rewards={len(steer_rewards)}")

            if i in base_right_steer_wrong:
                base_correct, steer_correct = True, False
            elif i in base_wrong_steer_right:
                base_correct, steer_correct = False, True
            else:
                base_correct = steer_correct = None  # 其他边例

            case_type = "Base对Steer错" if (base_correct and not steer_correct) else ("Base错Steer对" if (not base_correct and steer_correct) else "其他")
            case_slug = "base_ok_steer_wrong" if case_type == "Base对Steer错" else ("base_wrong_steer_ok" if case_type == "Base错Steer对" else "other")
            out_path = os.path.join(args.output_dir, f"question_{i}_{case_slug}.png")
            plot_curves(base_rewards, steer_rewards, i, base_correct or False, steer_correct or False, case_type, out_path)
            print(f"  -> {out_path}")

        except Exception as e:
            print(f"Q{i} Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
