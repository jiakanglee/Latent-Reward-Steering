import torch
import torch.nn as nn
import argparse
import sys
import os
import math
import traceback
import numpy as np
import gc

sys.path.append(os.getcwd())

from datasets import load_dataset
from utils.utils import load_model, center_and_l2_normalize_torch
from utils.sae import load_sae
from utils.llm_judge import evaluate_answer


# =========================================================
# 1. Reward Model
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LatentTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def _build_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )

    def forward(self, x, return_logits=False):
        # x: [B, T, 10]
        x = self.input_norm(x)
        x = self.embedding(x)
        x = self.pos_encoder(x)

        causal_mask = self._build_causal_mask(x.size(1), x.device)
        x = self.transformer_encoder(x, mask=causal_mask)

        logits = self.head(x)  # [B, T, 1]
        if return_logits:
            return logits
        return torch.sigmoid(logits)


# =========================================================
# 2. Prefix-latent iterative steering hook
# =========================================================
class IterativeTransformerSteeringHook:
    def __init__(
        self,
        sae,
        reward_model,
        step_size=0.1,
        num_steps=1,
        sae_dim=10,
        delta_clip=0.2,
        reward_prefix_window=128,
        steer_after=32,
        steer_every=4,
        monitor=False
    ):
        self.sae = sae
        self.reward_model = reward_model
        self.step_size = step_size
        self.num_steps = num_steps
        self.sae_dim = sae_dim
        self.delta_clip = delta_clip
        self.reward_prefix_window = reward_prefix_window
        self.steer_after = steer_after
        self.steer_every = steer_every
        self.monitor = monitor

        self.step_count = 0
        self.latent_history = []

        if hasattr(self.sae, "W_dec"):
            self.decoder_weight = self.sae.W_dec
        elif hasattr(self.sae, "decoder"):
            self.decoder_weight = self.sae.decoder
        else:
            raise ValueError("Could not find decoder weights in SAE.")

        if isinstance(self.decoder_weight, nn.Module):
            self.decoder_weight = self.decoder_weight.weight

    def reset(self):
        self.step_count = 0
        self.latent_history = []
        gc.collect()
        torch.cuda.empty_cache()

    def _get_dense_latents(self, x):
        flat_x = x.view(-1, x.shape[-1])

        if hasattr(self.sae, "activation_mean"):
            pre_act = center_and_l2_normalize_torch(flat_x, self.sae.activation_mean)
        else:
            pre_act = flat_x - self.sae.b_dec

        top_acts, top_indices = self.sae.encode(pre_act)

        latents = torch.zeros(
            (top_acts.shape[0], self.sae_dim),
            device=x.device,
            dtype=torch.float32
        )
        latents.scatter_(1, top_indices, top_acts.float())
        return latents

    def _build_prefix_input(self, current_token_latent):
        # current_token_latent: [B, 1, D]
        if self.reward_prefix_window is None or self.reward_prefix_window <= 1:
            return current_token_latent

        keep_prev = self.reward_prefix_window - 1
        raw_prev = self.latent_history[-keep_prev:] if keep_prev > 0 else []

        prev = [
            p.detach().clone().to(
                device=current_token_latent.device,
                dtype=current_token_latent.dtype
            )
            for p in raw_prev
        ]

        if len(prev) == 0:
            return current_token_latent

        return torch.cat(prev + [current_token_latent], dim=1)

    def _should_steer_now(self):
        if self.step_count < self.steer_after:
            return False
        if self.steer_every <= 1:
            return True
        return ((self.step_count - self.steer_after) % self.steer_every) == 0

    def __call__(self, module, inputs, outputs):
        original_act = outputs[0] if isinstance(outputs, tuple) else outputs
        batch_size, seq_len, _ = original_act.shape

        sae_dtype = self.sae.encoder.weight.dtype
        x = original_act.to(sae_dtype)

        # Prompt 阶段：直接跳过
        if seq_len > 1:
            return outputs

        self.step_count += 1

        # 1) 当前 token latent
        with torch.no_grad():
            current_dense = self._get_dense_latents(x).view(batch_size, 1, -1)

        # 转成普通 tensor，避免 generate/inference_mode 里的 tensor 污染梯度
        current_dense = current_dense.detach().clone().float()

        # 2) 如果当前不该 steer，只记 history 然后返回
        if not self._should_steer_now():
            self.latent_history.append(current_dense.detach().clone().float())
            return outputs

        # 3) 监控初始 reward
        init_logit_val = None
        init_prob_val = None
        if self.monitor:
            with torch.no_grad():
                init_prefix = self._build_prefix_input(current_dense)
                init_logits = self.reward_model(init_prefix, return_logits=True)
                init_logit_val = init_logits[:, -1, 0].mean().item()
                init_prob_val = torch.sigmoid(init_logits[:, -1, 0]).mean().item()

        # 4) 优化当前 token latent
        base_latent = current_dense.detach().clone().float()
        optimized_latent = base_latent.clone()

        for step in range(self.num_steps):
            with torch.inference_mode(False):
                with torch.enable_grad():
                    target = optimized_latent.detach().clone().float().requires_grad_(True)

                    rm_input = self._build_prefix_input(target)
                    logits = self.reward_model(rm_input, return_logits=True)
                    target_score = logits[:, -1, 0].sum()

                    grad = torch.autograd.grad(
                        target_score,
                        target,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False
                    )[0]

            if grad is None:
                grad = torch.zeros_like(target)

            grad_norm = torch.norm(grad, dim=-1, keepdim=True).clamp_min(1e-8)
            updated = target + self.step_size * (grad / grad_norm)

            delta = updated - base_latent
            if self.delta_clip is not None and self.delta_clip > 0:
                delta = torch.clamp(delta, -self.delta_clip, self.delta_clip)

            optimized_latent = (base_latent + delta).detach()

            del logits, target_score, grad, target

        steered_dense = optimized_latent.detach().clone()

        # 5) 打印监控
        if self.monitor:
            with torch.no_grad():
                final_prefix = self._build_prefix_input(steered_dense)
                final_logits = self.reward_model(final_prefix, return_logits=True)
                final_logit_val = final_logits[:, -1, 0].mean().item()
                final_prob_val = torch.sigmoid(final_logits[:, -1, 0]).mean().item()

            print(
                f"    [Tok {self.step_count:4}] "
                f"Score: {init_logit_val:7.4f} -> {final_logit_val:7.4f} | "
                f"Prob: {init_prob_val:6.4f} -> {final_prob_val:6.4f}"
            )

            diff = (steered_dense[0, 0] - current_dense[0, 0]).detach().cpu().numpy()
            original = current_dense[0, 0].detach().cpu().numpy()

            print("      Initial Latents (Base):")
            print("      " + " ".join([f"{v:7.3f}" for v in original]))
            print("      Steering Diff (Add):")
            print("      " + " ".join([f"{v:+7.3f}" for v in diff]))
            print("      ---------------------------------------------------------------------")

        # 6) 保存 steered latent 到 history
        self.latent_history.append(steered_dense.detach().clone().float())

        # 7) 投影回 residual space
        with torch.no_grad():
            delta_latent = (steered_dense - current_dense).view(-1, self.sae_dim)
            W = self.decoder_weight.data
            if W.shape[0] != self.sae_dim:
                W = W.t()

            delta_act = delta_latent @ W
            final_act = x.view(-1, x.shape[-1]) + delta_act
            final_act = final_act.view_as(x)

        if isinstance(outputs, tuple):
            return (final_act.to(original_act.dtype),) + outputs[1:]
        return final_act.to(original_act.dtype)


# =========================================================
# 3. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Open-Reasoner-Zero/Open-Reasoner-Zero-7B')
    parser.add_argument('--sae_layer', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--reward_model_path', type=str, default='transformer_reward_model_prefix_best.pt')

    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--delta_clip', type=float, default=0.2)

    parser.add_argument('--reward_prefix_window', type=int, default=128)
    parser.add_argument('--steer_after', type=int, default=32)
    parser.add_argument('--steer_every', type=int, default=4)

    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--example_idx', type=int, nargs='+', default=None)

    parser.add_argument('--max_token', type=int, default=2000)
    parser.add_argument("--print_response", action="store_true")
    parser.add_argument('--monitor', action='store_true', default=False)

    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)

    args = parser.parse_args()

    print(f"🚀 Loading LLM: {args.model}...")
    model, tokenizer = load_model(model_name=args.model, device="cuda")

    print(f"🧩 Loading SAE...")
    sae, _ = load_sae("open-reasoner-zero-7b", args.sae_layer, args.n_clusters)
    sae = sae.to(model.device)
    sae.k = args.n_clusters

    print("🧠 Loading Prefix Reward Model...")
    reward_model = LatentTransformer(
        input_dim=args.n_clusters,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ).to(model.device)

    reward_model.load_state_dict(torch.load(args.reward_model_path, map_location=model.device))
    reward_model.eval()

    for p in reward_model.parameters():
        p.requires_grad_(False)

    print("✅ Reward Model loaded successfully.")

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    if args.example_idx is not None:
        all_indices = args.example_idx
        mode_str = "Mode 2 (Specific List)"
    else:
        all_indices = list(range(args.num_examples))
        mode_str = "Mode 1 (First N)"

    target_indices = [
        x for i, x in enumerate(all_indices)
        if i % args.num_shards == args.shard_id
    ]

    print(f"🎯 {mode_str} | Total: {len(all_indices)} | 🚀 Shard {args.shard_id}/{args.num_shards} claimed: {len(target_indices)} tasks")

    hook = IterativeTransformerSteeringHook(
        sae=sae,
        reward_model=reward_model,
        step_size=args.step_size,
        num_steps=args.num_steps,
        sae_dim=args.n_clusters,
        delta_clip=args.delta_clip,
        reward_prefix_window=args.reward_prefix_window,
        steer_after=args.steer_after,
        steer_every=args.steer_every,
        monitor=args.monitor
    )

    layer_module = model.model.layers[args.sae_layer]

    stats = {
        "base_correct": 0,
        "steer_correct": 0,
        "base_correct_lens": [],
        "steer_correct_lens": []
    }

    print(
        f"\n>>> Running Prefix Iterative Steering "
        f"(step={args.step_size}, steps={args.num_steps}, "
        f"window={args.reward_prefix_window}, after={args.steer_after}, every={args.steer_every})"
    )

    for i in target_indices:
        out_base = None
        out_steer = None
        inputs = None

        try:
            item = dataset[i]
            question = item['problem']
            solution = item['solution']
            answer = item['answer']

            print(f"\n======== Question {i} ========")

            messages = [{"role": "user", "content": question}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]

            # 1) Baseline
            print("1️⃣ Generating Baseline...", end="", flush=True)
            with torch.no_grad():
                out_base = model.generate(
                    **inputs,
                    max_new_tokens=args.max_token,
                    do_sample=False
                )

            len_base = out_base.shape[1] - input_len
            res_base = tokenizer.decode(out_base[0][input_len:], skip_special_tokens=False)
            print(f" Done. (Len: {len_base})")

            if args.print_response:
                print("\n" + "=" * 20 + " [Baseline Output] " + "=" * 20)
                print(res_base)
                print("=" * 60 + "\n")

            is_base_correct, _ = evaluate_answer(res_base, solution, answer, question, "Baseline")
            if is_base_correct:
                stats["base_correct"] += 1
                stats["base_correct_lens"].append(len_base)

            # 2) Steered
            print("2️⃣ Generating Steered...", end="", flush=True)
            hook.reset()
            handle = layer_module.register_forward_hook(hook)

            res_steer = "[FAILED]"
            len_steer = 0
            is_steer_correct = False

            try:
                out_steer = model.generate(
                    **inputs,
                    max_new_tokens=args.max_token,
                    do_sample=False
                )

                len_steer = out_steer.shape[1] - input_len
                res_steer = tokenizer.decode(out_steer[0][input_len:], skip_special_tokens=False)
                print(f" Done. (Len: {len_steer})")

                if args.print_response:
                    print("\n" + "=" * 20 + " [Steered Output] " + "=" * 20)
                    print(res_steer)
                    print("=" * 60 + "\n")

                is_steer_correct, _ = evaluate_answer(res_steer, solution, answer, question, "Steered")
                if is_steer_correct:
                    stats["steer_correct"] += 1
                    stats["steer_correct_lens"].append(len_steer)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n❌ OOM Error on idx {i}. Skipping this item.")
                else:
                    print(f"\n❌ Runtime Error: {e}")
                    traceback.print_exc()
            except Exception as e:
                print(f"\n❌ Error: {e}")
                traceback.print_exc()
            finally:
                handle.remove()

            # Report
            print("-" * 50)
            status_base = "✅" if is_base_correct else "❌"
            status_steer = "✅" if is_steer_correct else "❌"
            print(f"Base  [{status_base}] Len: {len_base}")
            print(f"Steer [{status_steer}] Len: {len_steer}")
            if (not is_base_correct) and is_steer_correct:
                print("🚀 SUCCESS! Steering fixed the error!")
            print("-" * 50)

        finally:
            if inputs is not None:
                del inputs
            if out_base is not None:
                del out_base
            if out_steer is not None:
                del out_steer
            torch.cuda.empty_cache()
            gc.collect()

    # Final report
    print("\n" + "=" * 50)
    print(
        f"📊 Final Report "
        f"(step={args.step_size}, steps={args.num_steps}, "
        f"window={args.reward_prefix_window}, after={args.steer_after}, every={args.steer_every})"
    )
    print("Accuracy:")
    print(f"  Base : {stats['base_correct']}/{len(target_indices)}")
    print(f"  Steer: {stats['steer_correct']}/{len(target_indices)}")

    print("\nAverage Token Length (Correct Answers Only):")
    avg_base_len = np.mean(stats["base_correct_lens"]) if stats["base_correct_lens"] else 0
    avg_steer_len = np.mean(stats["steer_correct_lens"]) if stats["steer_correct_lens"] else 0
    print(f"  Base : {avg_base_len:.1f} tokens")
    print(f"  Steer: {avg_steer_len:.1f} tokens")

    delta = avg_steer_len - avg_base_len
    if delta > 0:
        print(f"📈 On average, steering increased thinking by +{delta:.1f} tokens.")
    else:
        print(f"📉 On average, steering changed thinking by {delta:.1f} tokens.")
    print("=" * 50)


if __name__ == "__main__":
    main()