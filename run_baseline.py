import torch
import argparse
import sys
import os
import json
from tqdm import tqdm

# 确保逻辑目录可见
sys.path.append(os.getcwd())

from datasets import load_dataset
from utils.utils import load_model, center_and_l2_normalize_torch
from utils.sae import SAE 
from utils.llm_judge import evaluate_answer

# =========================================================
# Native Monitor & Steer Hook
# =========================================================
class NativeGlobalHook:
    def __init__(self, sae, multiplier, print_trigger=False):
        self.sae = sae
        self.multiplier = multiplier
        self.print_trigger = print_trigger # 动态开关
        
    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            original_act = outputs[0]
        else:
            original_act = outputs
            
        sae_dtype = self.sae.encoder.weight.dtype
        x = original_act.to(sae_dtype)
        
        with torch.no_grad():
            if hasattr(self.sae, "activation_mean") and self.sae.activation_mean is not None:
                pre_act = center_and_l2_normalize_torch(x, self.sae.activation_mean)
            else:
                pre_act = x - self.sae.b_dec

            # 1. 编码
            top_acts, top_indices = self.sae.encode(pre_act)
            
            # =========================================================
            # 🔍 [动态监控] 只有参数触发才打印详情
            # =========================================================
            if self.print_trigger:
                curr_acts = top_acts[0, -1]
                curr_inds = top_indices[0, -1]
                log_entries = [f"{idx.item()}:{val.item():.4f}" for idx, val in zip(curr_inds, curr_acts) if val.item() > 1e-5]
                if log_entries:
                    print(f"🧠 [SAE Monitor] {', '.join(log_entries)}")
            # =========================================================

            # 2. 干预逻辑
            new_acts = top_acts * self.multiplier
            # 锁定维度 4 = 0.1
            new_acts[top_indices == 4] = 0.1
            
            # 3. 重构
            recon_old = self.sae.decode(top_acts, top_indices)
            error_term = x - recon_old
            recon_new = self.sae.decode(new_acts, top_indices)
            new_act = recon_new + error_term
            
        return (new_act.to(original_act.dtype),) + outputs[1:] if isinstance(outputs, tuple) else new_act.to(original_act.dtype)

def get_real_tensor(saver_object):
    val = saver_object.value if hasattr(saver_object, 'value') else saver_object
    if isinstance(val, list): val = val[0]
    return val

# =========================================================
# 主程序
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    # 核心参数
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--sae_path', type=str, required=True)
    parser.add_argument('--sae_layer', type=int, required=True)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--multiplier', type=float, default=1.0)
    
    # 任务控制
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--example_idx', type=int, nargs='+', default=None)
    parser.add_argument('--max_new_tokens', type=int, default=300)
    
    # 动态打印与保存控制
    parser.add_argument('--output_file', type=str, default="results.jsonl")
    parser.add_argument('--print_response', action="store_true")
    parser.add_argument('--print_activations', action="store_true")

    args = parser.parse_args()

    # 1. 加载资源
    print(f"🚀 Loading Model: {args.model}")
    model, tokenizer = load_model(model_name=args.model, device="cuda")
    
    print(f"📂 Loading SAE Weight: {args.sae_path}")
    ckpt = torch.load(args.sae_path, map_location="cpu", weights_only=False)
    sae = SAE(ckpt['input_dim'], ckpt['num_latents'], k=ckpt.get('topk', 32)).to(model.device)
    sae.load_state_dict({k:v for k,v in ckpt.items() if k in sae.state_dict()}, strict=False)
    
    if ckpt.get("activation_mean") is not None:
        sae.activation_mean = torch.as_tensor(ckpt["activation_mean"]).to(model.device)
    sae.k = args.n_clusters
    
    # 2. 确定索引
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    indices = args.example_idx if args.example_idx else list(range(args.num_samples or 1))

    # 3. 设置 Hook
    hook = NativeGlobalHook(sae, args.multiplier, print_trigger=args.print_activations)
    layer_module = model.model.layers[args.sae_layer]

    stats = {"base_correct": 0, "steer_correct": 0}
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print(f"\n>>> Running Evaluation (N={len(indices)}, Multiplier={args.multiplier})")

    with open(args.output_file, "a", encoding="utf-8") as f_out:
        for idx in tqdm(indices, desc="Processing"):
            item = dataset[idx]
            question = item['problem']
            solution = item['solution']
            short_answer = item['answer']
            
            input_text = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
            prompt_len = tokenizer(input_text, return_tensors="pt").input_ids.shape[1]

            # --- Baseline ---
            with model.generate(input_text, max_new_tokens=args.max_new_tokens) as generator:
                out_base_saver = model.generator.output.save()
            base_ids = get_real_tensor(out_base_saver)
            res_base = tokenizer.decode(base_ids[prompt_len:], skip_special_tokens=True)

            # --- Steered ---
            handle = layer_module.register_forward_hook(hook)
            try:
                with model.generate(input_text, max_new_tokens=args.max_new_tokens) as generator:
                    out_steer_saver = model.generator.output.save()
            finally:
                handle.remove()
            steer_ids = get_real_tensor(out_steer_saver)
            res_steer = tokenizer.decode(steer_ids[prompt_len:], skip_special_tokens=True)

            # --- Judge ---
            is_base_correct, _ = evaluate_answer(
                res_base, solution, short_answer, question, "Baseline", dataset_type="math"
            )
            is_steer_correct, _ = evaluate_answer(
                res_steer, solution, short_answer, question, "Steered", dataset_type="math"
            )
            
            if is_base_correct: stats["base_correct"] += 1
            if is_steer_correct: stats["steer_correct"] += 1

            # --- 动态打印：仅在命令行有 flag 时执行 ---
            if args.print_response:
                print(f"\n{'-'*10} Sample {idx} {'-'*10}")
                print(f"【Baseline】(Correct: {is_base_correct}):\n{res_base}")
                print(f"【Steered 】(Correct: {is_steer_correct}):\n{res_steer}")

            # 🛠️ 修复 TypeError: 处理 args.num_samples 为 None 的情况
            total_task_count = args.num_samples if args.num_samples is not None else len(indices)
            
            # 只有在样本量较小时才把完整文本存入 JSON，防止文件体积爆炸
            text_to_save = res_steer if total_task_count < 10 else "too_long_skipped"

            f_out.write(json.dumps({
                "idx": idx, 
                "base_acc": is_base_correct, 
                "steer_acc": is_steer_correct,
                "text": text_to_save
            }, ensure_ascii=False) + "\n")
            f_out.flush()
            torch.cuda.empty_cache()

    print(f"\n📊 Final Accuracy: Base {stats['base_correct']}/{len(indices)}, Steer {stats['steer_correct']}/{len(indices)}")

if __name__ == "__main__":
    main()