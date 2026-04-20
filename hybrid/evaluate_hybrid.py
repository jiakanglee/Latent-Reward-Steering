# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

# Early Hugging Face token check
try:
    import os
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_HUB_TOKEN')
    if token:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            me = api.whoami(token=token)
            print('WHOAMI_OK', me.get('name') or me.get('id') or me)
        except Exception:
            pass
except Exception:
    pass

import torch
import torch.nn as nn
from datasets import load_dataset
import json
import random
from tqdm import tqdm
import numpy as np
import os
import utils
from collections import defaultdict
import gc
import torch.multiprocessing as mp
from functools import partial
import re

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate hybrid model performance on math problems")
parser.add_argument("--n_batches", type=int, default=60,
                    help="Number of batches to evaluate")
parser.add_argument("--model", type=str, default="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B",
                    help="Thinking model to use")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                    help="Base model to use")
parser.add_argument("--probe_layer", type=int, default=10,
                    help="Layer to use for probe")
parser.add_argument('--top_k', type=int, default=1,
                    help='Top-k SAE latents to combine for steering')
parser.add_argument("--load_in_8bit", type=bool, default=False,
                    help="Load in 8bit")
parser.add_argument("--max_new_tokens", type=int, default=2000,
                    help="Max new tokens")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--n_gpus", type=int, default=1,
                    help="Number of GPUs to use")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for parallel generation")
args, _ = parser.parse_known_args()

# ==========================================
# 1. 内置 LinearProbe 类 (防止 Import Error)
# ==========================================
class LinearProbe(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
    def forward(self, x):
        return self.linear(x)

def extract_answer(response):
    """Extract the final answer from the model's response."""
    try:
        answer = response.split("</think>")
        if len(answer) > 1:
            answer = answer[-1].strip()
            if answer == "": answer = None
        else: answer = None
        return answer
    except: return None

def get_batched_question_ids(tokenizer, questions):
    # 简单的 batch 处理
    input_ids = []
    for q in questions:
        # 使用 chat template 或者直接 encode
        try:
            val = tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")
        except:
            # Fallback for models without chat template
            val = tokenizer.encode(q, return_tensors="pt")
        input_ids.append(val[0])
    
    # Pad to max length
    max_len = max([len(x) for x in input_ids])
    padded = torch.ones((len(input_ids), max_len), dtype=torch.long) * tokenizer.pad_token_id
    for i, x in enumerate(input_ids):
        padded[i, :len(x)] = x
    return padded.to("cuda")

def evaluate_answer(question, model_answer, correct_answer):
    final_answer = model_answer["response"] if model_answer["extracted_answer"] is None else model_answer["extracted_answer"]
    return utils.evaluate_answer(question, model_answer, correct_answer)

# ==========================================
# 核心区域: process_gpu_chunk
# ==========================================
def process_gpu_chunk(chunk_id, dataset_chunk, gpu_id, args):
    print(f"🚀 [GPU {gpu_id}] Starting process with {len(dataset_chunk)} examples")
    
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    random.seed(args.seed + gpu_id)
    torch.manual_seed(args.seed + gpu_id)
    
    # 1. 加载模型
    print("📦 Loading models...")
    thinking_model, thinking_tokenizer = utils.load_model(device=device, model_name=args.model)
    base_model, base_tokenizer = utils.load_model(device=device, model_name=args.base_model)
    
    # =======================================================
# =======================================================
    # 2. 智能筛选 Vectors (拨乱反正：精准锁定 Math 向量)
    # =======================================================
    print("🏹 Auto-loading matching Steering Vectors...")
    
    vector_dir = "/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/train-vectors/results/vars/optimized_vectors"
    vectors = {}
    
    target_hidden_size = thinking_model.config.hidden_size # 1536
    
    # 获取映射后的“真名”
    # 根据表：DeepSeek-R1-Distill-Qwen-1.5B -> Qwen2.5-Math-1.5B
    mapping_full_name = utils.model_mapping.get(args.model, "")
    mapping_name = mapping_full_name.split('/')[-1].lower() if mapping_full_name else ""
    
    if not mapping_name:
        # 强制保底：如果表里没查到，但我们知道是这个 DeepSeek 模型
        if "deepseek-r1-distill-qwen-1.5b" in args.model.lower():
            mapping_name = "qwen2.5-math-1.5b"

    print(f"🎯 Target Model: {args.model}")
    print(f"📖 Correct Mapping Target: {mapping_name}")

    if os.path.exists(vector_dir):
        import glob
        all_files = glob.glob(os.path.join(vector_dir, "*.pt"))
        
        for f in all_files:
            filename = os.path.basename(f).lower()
            
            # 【核心逻辑】只认准 mapping_name (qwen2.5-math-1.5b)
            if mapping_name in filename:
                try:
                    data = torch.load(f, map_location=device)
                    vec_tensor = data[list(data.keys())[0]] if isinstance(data, dict) else data
                    
                    # 检查维度 (1536)
                    if vec_tensor is not None and vec_tensor.shape[0] == target_hidden_size:
                        idx_match = re.search(r'idx(\d+)', filename)
                        if idx_match:
                            idx_key = f"idx{idx_match.group(1)}"
                            vectors[idx_key] = vec_tensor.to(device=device, dtype=thinking_model.dtype)
                            vectors[idx_key + "_linear"] = vectors[idx_key]
                except Exception as e:
                    pass

    unique_keys = sorted([k for k in vectors.keys() if not k.endswith('_linear')], key=lambda x: int(x[3:]))
    if not vectors:
        print(f"❌ ERROR: Still no vectors found for {mapping_name}! Check if the .pt files exist in {vector_dir}")
    else:
        print(f"✅ Successfully loaded {len(unique_keys)} vectors.")
        print(f"📊 Final Vector Keys: {unique_keys}")
    # 3. 加载 Probe
    print("🔍 Loading probe...")
    model_id = args.model.split('/')[-1].lower()
    probe_filename = f"probe_layer{args.probe_layer}_{model_id}.pt"
    # 搜索路径
    possible_paths = [
        f"results/vars/{probe_filename}",
        f"../results/vars/{probe_filename}",
        f"/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/hybrid/results/vars/{probe_filename}"
    ]
    
    probe_path = None
    for p in possible_paths:
        if os.path.exists(p):
            probe_path = p
            print(f"✅ Found probe at: {p}")
            break
            
    if not probe_path:
        print(f"❌ Critical: Could not find probe file {probe_filename}")
        return defaultdict(list)

    # 4. 智能加载 Probe (自动检测维度)
    checkpoint = torch.load(probe_path, map_location=device)
    
    if 'probe_state_dict' in checkpoint:
        state_dict = checkpoint['probe_state_dict']
        label_to_idx = checkpoint.get('label_to_idx', {})
    else:
        state_dict = checkpoint
        label_to_idx = {}

    # 从权重推断维度
    saved_weight = state_dict['linear.weight']
    num_labels_loaded = saved_weight.shape[0]
    hidden_size_loaded = saved_weight.shape[1]
    
    print(f"📊 Probe Config: Hidden={hidden_size_loaded}, NumLabels={num_labels_loaded}")
    
    probe = LinearProbe(hidden_size_loaded, num_labels_loaded).to(device)
    probe.load_state_dict(state_dict)
    
    # [Fix Dtype] 强制转换 Probe 精度
    probe.to(dtype=thinking_model.dtype)
    probe.eval()
    
    # 5. 建立标签映射
    if not label_to_idx:
        label_to_idx = {f"class_{i}": i for i in range(num_labels_loaded)}
        
    print(f"🏷️ Labels: {list(label_to_idx.keys())}")
    forcing_labels = list(label_to_idx.keys())

    gpu_results = defaultdict(list)
    
    # 6. 开始循环
    pbar = tqdm(range(0, len(dataset_chunk), args.batch_size), desc=f"GPU {gpu_id}")
    for i in pbar:
        batch = dataset_chunk[i:i + args.batch_size]
        if not batch: continue
        
        questions = [ex["problem"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
        
        # --- Hybrid Model Config ---
        hybrid_config = {
            "probe": probe,
            "label_to_idx": label_to_idx,
            "probe_layer": args.probe_layer,
            "forcing": forcing_labels,
            "top_k": args.top_k,
            "strength": 0.05,
            "vectors": vectors  # <--- 将筛选好的 Correct Vectors 传进去
        }
        
        base_input_ids = get_batched_question_ids(base_tokenizer, questions)

        try:
            output_ids, _, _, _ = utils.custom_hybrid_generate(
                thinking_model=thinking_model,
                base_model=base_model,
                tokenizer=base_tokenizer,
                input_ids=base_input_ids,
                max_new_tokens=args.max_new_tokens,
                baseline_config=hybrid_config,
                baseline_method="probe"
            )
            
            for j, out_tokens in enumerate(output_ids):
                resp = base_tokenizer.decode(out_tokens, skip_special_tokens=True)
                ans = extract_answer(resp)
                
                is_cor = False 
                if ans and answers[j]:
                    is_cor = (ans.strip() == answers[j].strip())
                
                gpu_results["hybrid"].append({
                    "response": resp,
                    "correct": is_cor,
                    "question": questions[j]
                })
        except Exception as e:
            print(f"Error in batch: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    return gpu_results

def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Loading dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        dataset = list(dataset)[:args.n_batches * args.batch_size]
    except:
        print("⚠️ Failed to load HF dataset, using dummy data.")
        dataset = [{"problem": "1+1=?", "answer": "2"}] * (args.n_batches * args.batch_size)
    
    print(f"Loaded {len(dataset)} examples")
    
    # 单 GPU 模式
    gpu_results = process_gpu_chunk(0, dataset, 0, args)
    
    # Save
    os.makedirs("results/vars", exist_ok=True)
    result_file = f"results/vars/hybrid_results_{args.model.split('/')[-1].lower()}.json"
    
    # Metrics
    metrics = {}
    if gpu_results["hybrid"]:
        correct = sum(1 for r in gpu_results["hybrid"] if r["correct"])
        metrics["hybrid"] = {"accuracy": correct / len(gpu_results["hybrid"])}
        
    with open(result_file, "w") as f:
        json.dump({"metrics": metrics, "results": gpu_results}, f, indent=2)
        
    print(f"✅ Done! Results saved to {result_file}")

if __name__ == "__main__":
    main()