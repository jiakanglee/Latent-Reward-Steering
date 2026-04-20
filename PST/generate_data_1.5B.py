import torch
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from utils.utils import load_model, center_and_l2_normalize_torch
from utils.sae import load_sae 
from utils.llm_judge import evaluate_answer

# =========================================================
# 📸 序列采集 Hook (保持不变)
# =========================================================
class SequenceCollector:
    def __init__(self, sae, sae_dim):
        self.sae = sae
        self.sae_dim = sae_dim
        self.current_seq = [] 
    
    def clear(self):
        self.current_seq = []

    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple): act = outputs[0]
        else: act = outputs
            
        sae_dtype = self.sae.encoder.weight.dtype
        x = act.to(sae_dtype)
        
        with torch.no_grad():
            # [Batch, Seq, Hidden] -> [Batch*Seq, Hidden]
            flat_x = x.view(-1, x.shape[-1])
            
            if hasattr(self.sae, "activation_mean"):
                pre_act = center_and_l2_normalize_torch(flat_x, self.sae.activation_mean)
            else:
                pre_act = flat_x - self.sae.b_dec

            top_acts, top_indices = self.sae.encode(pre_act)
            
            dense = torch.zeros((top_acts.shape[0], self.sae_dim), device=x.device, dtype=torch.float32)
            dense.scatter_(1, top_indices, top_acts.float())
            
            # 存入列表 (CPU)
            self.current_seq.append(dense.cpu())

        return outputs

    def get_trajectory(self):
        if not self.current_seq:
            return None
        # 拼接 [Total_Tokens, 15]
        return torch.cat(self.current_seq, dim=0)

# =========================================================
# 主程序
# =========================================================
def load_dataset_by_name(dataset_name):
    """Load dataset by name. Returns (dataset, source_name) for each item."""
    if dataset_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [(ds[i], "math500") for i in range(len(ds))]
    elif dataset_name == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024")["train"]
        return [(ds[i], "aime24") for i in range(len(ds))]
    elif dataset_name == "aime25":
        ds = load_dataset("yentinglin/aime_2025")["train"]
        return [(ds[i], "aime25") for i in range(len(ds))]
    elif dataset_name == "aime24_aime25":
        ds24 = load_dataset("HuggingFaceH4/aime_2024")["train"]
        ds25 = load_dataset("yentinglin/aime_2025")["train"]
        items = [(ds24[i], "aime24") for i in range(len(ds24))]
        items += [(ds25[i], "aime25") for i in range(len(ds25))]
        return items
    elif dataset_name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "full")["train"]  # 374 题
        return [(ds[i], "mbpp") for i in range(len(ds))]
    elif dataset_name == "mbpp_train_test":
        ds_train = load_dataset("google-research-datasets/mbpp", "full")["train"]  # 374
        ds_test = load_dataset("google-research-datasets/mbpp", "full")["test"]    # 500
        items = [(ds_train[i], "mbpp_train") for i in range(len(ds_train))]
        items += [(ds_test[i], "mbpp_test") for i in range(len(ds_test))]
        return items  # 874 题
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B')
    parser.add_argument('--sae_layer', type=int, default=8)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--num_examples', type=int, default=None,
        help='Max examples to collect (default: all for aime, 200 for math500)')
    parser.add_argument('--max_token', type=int, default=4000) 
    parser.add_argument('--dataset', type=str, choices=['math500', 'aime24', 'aime25', 'aime24_aime25', 'mbpp', 'mbpp_train_test'],
        default='aime24_aime25', help='Dataset to collect from')
    parser.add_argument('--output_file', type=str, default=None,
        help='Output file (default: collected_sae_latents_5dim_{max_token}_{dataset}.pt)')
    parser.add_argument('--shard_id', type=int, default=0, help='当前分片 ID（多卡并行时使用）')
    parser.add_argument('--num_shards', type=int, default=1, help='总分片数（= GPU 数）')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8-bit to save VRAM (for A4000 etc)')
    args = parser.parse_args()

    print(f"Loading Model & SAE...")
    model, tokenizer = load_model(model_name=args.model, device="cuda", load_in_8bit=args.load_in_8bit)
    
    think_start_str = "<think>"
    eos_ids = tokenizer.encode(think_start_str, add_special_tokens=False)
    print(f"🧐 Tokenizer check: '{think_start_str}' encodes to IDs: {eos_ids}")
    
    sae, _ = load_sae("open-reasoner-zero-1.5b", args.sae_layer, args.n_clusters)
    real_dim = sae.encoder.weight.shape[0] if hasattr(sae, 'encoder') else args.n_clusters
    if real_dim != args.n_clusters:
        print(f"⚠️ Warning: Arg n_clusters={args.n_clusters} but Model dim={real_dim}. Fixing...")
        args.n_clusters = real_dim
        
    sae_dim = real_dim
    sae.k = sae_dim 
    sae = sae.to(model.device)
    
    # Load dataset
    dataset_items = load_dataset_by_name(args.dataset)
    num_examples = args.num_examples if args.num_examples is not None else len(dataset_items)
    num_examples = min(num_examples, len(dataset_items))
    
    if args.output_file is None:
        args.output_file = f"collected_sae_latents_5dim_{args.max_token}_{args.dataset}.pt"
    if not args.output_file.endswith('.pt'):
        args.output_file += '.pt'

    # 分片：只处理 i % num_shards == shard_id 的样本
    all_indices = list(range(num_examples))
    target_indices = [i for i in all_indices if i % args.num_shards == args.shard_id]
    
    print(f"Dataset: {args.dataset} | Examples: {num_examples}/{len(dataset_items)} | max_token={args.max_token}")
    print(f"🚀 Shard {args.shard_id}/{args.num_shards} claimed: {len(target_indices)} tasks")
    
    collector = SequenceCollector(sae, sae_dim)
    model.model.layers[args.sae_layer].register_forward_hook(collector)

    dataset_out = []

    CODING_TASK_PREFIX = "Task: Write a single Python function for the following problem. Do not include tests or examples in your output."

    def get_item_fields(item, dataset_name):
        if dataset_name in ("mbpp", "mbpp_train_test"):
            text = item["text"]
            test_list = item.get("test_list", [])
            tests_section = "\n\nPublic Tests:\n" + "\n".join(test_list) if test_list else ""
            question = f"{CODING_TASK_PREFIX}\n\nProblem: {text}{tests_section}"
            solution = str(item.get("code", ""))
            answer = solution
            return question, solution, answer, "coding", test_list
        else:
            question = item.get("problem", "")
            solution = str(item.get("solution", ""))
            answer = str(item.get("answer", ""))
            return question, solution, answer, "math", None

    # 多卡时每个 shard 写入单独文件，最后需合并
    if args.num_shards > 1:
        base = args.output_file[:-3]  # 去掉 .pt
        args.output_file = f"{base}_shard_{args.shard_id}.pt"

    print(f"\n>>> Start Collecting Sequences with Indexing (<think>)...")

    for i in target_indices:
        item, source = dataset_items[i]
        question, solution, answer, dataset_type, test_list = get_item_fields(item, args.dataset)
        
        collector.clear()

        messages = [{"role": "user", "content": question}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_token, do_sample=False)
        
        traj = collector.get_trajectory()
        generated_ids = out[0][prompt_len:] 
        
        full_gen_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        think_pos = full_gen_text.find(think_start_str)
        
        if think_pos != -1:
            think_part = full_gen_text[:think_pos + len(think_start_str)]
            think_part_ids = tokenizer.encode(think_part, add_special_tokens=False)
            think_idx = len(think_part_ids) - 1
        else:
            think_idx = -1 
            
        gen_len = len(generated_ids)
        if traj.shape[0] >= gen_len:
             final_traj = traj[-gen_len:]
        else:
             final_traj = traj

        res_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        is_correct, _ = evaluate_answer(res_text, solution, answer, question, "Collector", dataset_type=dataset_type, test_list=test_list)
        
        if traj is not None:
            data_point = {
                "latent_seq": final_traj,   
                "label": 1.0 if is_correct else 0.0,
                "length": final_traj.shape[0],
                "think_idx": think_idx,
                "source": source,
            }
            dataset_out.append(data_point)
            
            status = "✅" if is_correct else "❌"
            print(f"Generated: {res_text[:2000]}")  # 只打印前300字符避免太长
            print(f"Sample {i+1}/{num_examples} [{source}] | GenLen: {gen_len} | ThinkIdx: {think_idx} | {status}")

    print(f"\n💾 Saving {len(dataset_out)} trajectories to {args.output_file}...")
    torch.save(dataset_out, args.output_file)

if __name__ == "__main__":
    main()