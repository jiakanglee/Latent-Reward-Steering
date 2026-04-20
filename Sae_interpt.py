import torch
import argparse
import sys
import os
import gc
import traceback

sys.path.append(os.getcwd())
from datasets import load_dataset
from utils.utils import load_model, center_and_l2_normalize_torch
from utils.sae import load_sae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Open-Reasoner-Zero/Open-Reasoner-Zero-7B')
    parser.add_argument('--sae_layer', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--top_k_examples', type=int, default=15)
    args = parser.parse_args()

    print(f"🚀 Loading LLM via custom load_model...")
    model, tokenizer = load_model(model_name=args.model, device="cuda")

    # 🔥 防漏杀招 1：彻底剥离 NNsight 外壳，拿到最纯正的 PyTorch 模型
    # 这样 NNsight 就绝对无法在后台偷偷吃显存了
    if hasattr(model, '_model'):
        core_model = model._model
    else:
        core_model = model

    print(f"🧩 Loading SAE...")
    sae, _ = load_sae("open-reasoner-zero-7b", args.sae_layer, args.n_clusters)
    sae = sae.to(core_model.device)
    sae.k = 1 

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    top_activations = {i: [] for i in range(args.n_clusters)}

    # ==== Hook 逻辑 ====
    captured_acts = {}
    def hook_fn(module, inputs, outputs):
        original_act = outputs[0] if isinstance(outputs, tuple) else outputs
        captured_acts['hidden_states'] = original_act.detach().clone()

    # 注意：这里改用 core_model
    layer_module = core_model.model.layers[args.sae_layer]
    handle = layer_module.register_forward_hook(hook_fn)

    print(f"\n🧠 开始收集激活值 (共 {args.num_samples} 题，为了防 OOM 截断至 256 Tokens)...")

    for i in range(args.num_samples):
        try:
            item = dataset[i]
            question = item['problem']
            solution = item['solution']

            messages = [{"role": "user", "content": question}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = input_text + solution
            
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1000).to(core_model.device)
            
            with torch.inference_mode():
                # 🔥 防漏杀招 2：显式接收输出，不再使用隐式的 _
                outputs = core_model(**inputs)
                x = captured_acts['hidden_states']
                
                flat_x = x.view(-1, x.shape[-1])
                sae_dtype = sae.encoder.weight.dtype if hasattr(sae, 'encoder') else torch.float32
                flat_x = flat_x.to(sae_dtype)

                if hasattr(sae, "activation_mean"):
                    pre_act = center_and_l2_normalize_torch(flat_x, sae.activation_mean)
                else:
                    pre_act = flat_x - sae.b_dec
                    
                top_acts, top_indices = sae.encode(pre_act)
                latents = torch.zeros((top_acts.shape[0], args.n_clusters), device=x.device, dtype=torch.float32)
                latents.scatter_(1, top_indices.long(), top_acts.float())

            input_ids = inputs.input_ids[0]
            prompt_len = len(tokenizer(input_text).input_ids)
            start_pos = min(prompt_len, len(input_ids) - 1)

            for pos in range(start_pos, len(input_ids)):
                token_str = tokenizer.decode(input_ids[pos:pos+1])
                for dim in range(args.n_clusters):
                    act_val = latents[pos, dim].item()
                    if act_val > 0.1: 
                        prefix_ids = input_ids[max(0, pos-30):pos]
                        prefix_str = tokenizer.decode(prefix_ids)
                        highlighted_context = f"{prefix_str}\033[91m{token_str}\033[0m"
                        top_activations[dim].append((act_val, highlighted_context))

            print(f"  ✓ 题 {i} 处理完成 (Seq_len: {len(input_ids)})")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ 题 {i} OOM 爆显存，跳过。")
            else:
                print(f"  ❌ 题 {i} 运行时错误: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"  ❌ 题 {i} 出错: {e}")
            traceback.print_exc()
        finally:
            # 🔥 防漏杀招 3：核弹级清理，把所有涉及 GPU 的中间变量全部枪毙
            if 'inputs' in locals(): del inputs
            if 'outputs' in locals(): del outputs  # 删掉那个包含十几万词表的巨大张量
            if 'x' in locals(): del x
            if 'flat_x' in locals(): del flat_x
            if 'pre_act' in locals(): del pre_act
            if 'top_acts' in locals(): del top_acts
            if 'top_indices' in locals(): del top_indices
            if 'latents' in locals(): del latents
            
            captured_acts.clear() # 清空 Hook 字典，彻底切断对上一题的图引用
            
            torch.cuda.empty_cache()
            gc.collect()

    handle.remove()

    print("\n" + "="*50)
    print("📊 SAE 10维特征解释报告 (Max Activating Contexts)")
    print("="*50)

    for dim in range(args.n_clusters):
        print(f"\n🔥 维度 {dim} (Dimension {dim}):")
        sorted_acts = sorted(top_activations[dim], key=lambda x: x[0], reverse=True)
        if not sorted_acts:
            print("  [空] 该维度在测试样本中未被显著激活。")
            continue
        for rank, (act_val, highlighted_context) in enumerate(sorted_acts[:args.top_k_examples]):
            print(f"  Top {rank+1} (激活值: {act_val:.4f})")
            print(f"  上下文: ...{highlighted_context}")
        print("-" * 30)

if __name__ == "__main__":
    main()