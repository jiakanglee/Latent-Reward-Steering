import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import re
from tqdm import tqdm
import numpy as np
import os
import utils

# ==========================================
# 1. 参数解析
# ==========================================
parser = argparse.ArgumentParser(description="Train probes for hybrid models")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to train probes for")
parser.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for training")
parser.add_argument("--num_samples", type=int, default=2000,
                    help="Number of samples to train on")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--probe_layer", type=int, default=10,
                    help="Model layer to use for probe")
parser.add_argument("--load_in_8bit", type=bool, default=False,
                    help="Load model in 8-bit mode")
args, _ = parser.parse_known_args()

# ==========================================
# 2. 模型结构定义
# ==========================================
class LinearProbe(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
    def forward(self, x):
        return self.linear(x)

# ==========================================
# 3. 核心工具函数
# ==========================================
def get_char_to_token_map(text, tokenizer):
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
    return char_to_token

def get_activations(text, tokenizer, model):
    """原生 PyTorch 获取所有层激活值"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, inputs.input_ids[0]

def find_label_positions(annotated_response, original_text, tokenizer, label_to_idx):
    """
    根据标注文本找到对应的 token 位置。
    修复版：兼容 [\"6.56:idx13\"] 这种转义和时间戳格式。
    """
    label_positions = {}
    # 正则：匹配 ["任何内容"] 后面直到下一个标签的内容
    raw_pattern = r'\["([^"]+)"\](.*?)(?=\["[^"]+"\]|$)'
    matches = list(re.finditer(raw_pattern, annotated_response, re.DOTALL))
    
    char_to_token = get_char_to_token_map(original_text, tokenizer)
    
    for match in matches:
        raw_tag = match.group(1) # 例如 "452.26:idx4"
        
        # 提取真正的 idxN
        idx_match = re.search(r'idx(\d+)', raw_tag)
        if idx_match:
            label = f"idx{idx_match.group(1)}"
        else:
            continue
            
        if label not in label_to_idx:
            continue
            
        text_content = match.group(2).strip()
        if not text_content: continue
            
        # 在原始文本中定位这段话
        text_pos = original_text.find(text_content)
        if text_pos >= 0:
            if label not in label_positions: label_positions[label] = []
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text_content), None)
            if token_start is not None and token_end is not None:
                label_positions[label].append((token_start, token_end))
                
    return label_positions

def create_training_examples(layer_output, label_positions, label_to_idx):
    if not label_positions: return None, None
    examples, labels = [], []
    
    # 将 layer_output 转为 [seq_len, hidden_size]
    current_tensor = layer_output.squeeze(0) if layer_output.dim() == 3 else layer_output
    current_tensor = current_tensor.to(torch.float32)

    for label, positions in label_positions.items():
        label_idx = label_to_idx[label]
        for start, end in positions:
            if start >= end or end > current_tensor.shape[0]: continue
            
            # 取该片段的平均激活值作为特征
            example = current_tensor[start:end].mean(dim=0)
            if torch.isnan(example).any(): continue
            
            examples.append(example.unsqueeze(0))
            
            # Multi-label 格式 (One-hot)
            label_tensor = torch.zeros(len(label_to_idx))
            label_tensor[label_idx] = 1
            labels.append(label_tensor.unsqueeze(0))
    
    if not examples: return None, None
    return torch.cat(examples, dim=0), torch.cat(labels, dim=0)

def train_probe(model, tokenizer, responses, labels_list, layer_idx=10, epochs=20, batch_size=32, lr=1e-3):
    device = model.device
    num_labels = len(labels_list)
    label_to_idx = {label: i for i, label in enumerate(labels_list)}
    
    probe = LinearProbe(model.config.hidden_size, num_labels).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    
    print(f"⌛ Processing activations for {len(responses)} samples...")
    all_x, all_y = [], []
    
    for r in tqdm(responses):
        try:
            # 获取完整文本
            full_resp = r.get("full_response") or (r.get("annotated_thinking", "") + r.get("response", ""))
            if not full_resp: continue
            
            # 获取激活值
            hidden_states, _ = get_activations(full_resp, tokenizer, model)
            target_layer_states = hidden_states[layer_idx + 1] # +1 修正 embedding 层偏移
            
            # 提取位置并生成样本
            pos = find_label_positions(r["annotated_thinking"], full_resp, tokenizer, label_to_idx)
            ex, lbl = create_training_examples(target_layer_states, pos, label_to_idx)
            
            if ex is not None:
                all_x.append(ex)
                all_y.append(lbl)
        except Exception:
            continue
            
    if not all_x: raise ValueError("No valid training examples found! Check regex/data.")
    
    train_x = torch.cat(all_x, dim=0).to(device)
    train_y = torch.cat(all_y, dim=0).to(device)
    
    print(f"✅ Final Training Set: {len(train_x)} samples.")
    for i, label in enumerate(labels_list):
        print(f"  - {label}: {int(train_y[:, i].sum())} instances")

    # 训练循环
    for epoch in range(epochs):
        probe.train()
        indices = torch.randperm(len(train_x))
        epoch_loss = 0
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            optimizer.zero_grad()
            logits = probe(train_x[batch_idx])
            loss = F.binary_cross_entropy_with_logits(logits, train_y[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_x):.4f}")
        
    return probe, label_to_idx

# ==========================================
# 4. 主程序
# ==========================================
def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs('results/vars', exist_ok=True)

    # 1. 加载模型
    print(f"📦 Loading model: {args.model}")
    model, tokenizer = utils.load_model(model_name=args.model, load_in_8bit=args.load_in_8bit)

    # 2. 加载标注数据
    data_path = f"/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/generate-responses/results/vars/annotated_responses_{args.model.split('/')[-1].lower()}.json"
    print(f"📖 Loading data from: {data_path}")
    with open(data_path, 'r') as f:
        responses = json.load(f)
    
    responses = responses[:args.num_samples]
    
    # 3. 动态提取标签 (修复版)
    print("🎯 Scanning for all unique labels (idx0-idx14)...")
    all_labels = set()
    # 只要包含 idx 后面跟数字的都抓
    idx_scanner = re.compile(r'idx(\d+)')
    
    for r in responses:
        text = r.get("annotated_thinking", "")
        found = idx_scanner.findall(text)
        for num in found:
            all_labels.add(f"idx{num}")
    
    # 按照数字大小排序: idx0, idx1... idx10, idx11...
    sorted_labels = sorted(list(all_labels), key=lambda x: int(re.search(r'\d+', x).group()))
    print(f"✅ Detected {len(sorted_labels)} labels: {sorted_labels}")

    # 4. 训练
    probe, label_to_idx = train_probe(
        model, tokenizer, responses, sorted_labels, 
        layer_idx=args.probe_layer, epochs=args.epochs, 
        batch_size=args.batch_size, lr=args.lr
    )

    # 5. 保存
    model_short = args.model.split('/')[-1].lower()
    save_path = f"results/vars/probe_layer{args.probe_layer}_{model_short}.pt"
    torch.save({
        'probe_state_dict': probe.state_dict(),
        'label_to_idx': label_to_idx,
        'config': {
            'hidden_size': model.config.hidden_size,
            'num_labels': len(sorted_labels),
            'layer': args.probe_layer
        }
    }, save_path)
    print(f"🚀 Probe saved successfully to: {save_path}")

if __name__ == "__main__":
    main()