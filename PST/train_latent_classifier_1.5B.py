import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import argparse
import os
import math

# ==========================================
# 1. Dataset (保持你的原始逻辑不变)
# ==========================================
class ValueDatasetLegacy(Dataset):
    def __init__(self, data_path, device="cuda"):
        print(f"📂 Loading data from {data_path}...")
        
        try:
            raw_data = torch.load(data_path)
        except Exception as e:
            print(f"❌ Error opening file: {e}")
            self.data = []
            return

        self.data = []
        valid_count = 0
        
        for i, item in enumerate(raw_data):
            full_seq = None
            label = None
            
            # 1. 找序列 Tensor
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 2 and v.shape[1] == 5:
                        full_seq = v
                        break
            
            # 兜底查找
            if full_seq is None:
                if 'latent_seq' in item: full_seq = item['latent_seq']
                elif 'sae_latents' in item: full_seq = item['sae_latents']

            # 2. 找 Label
            if 'label' in item: label = item['label']
            elif 'is_correct' in item: label = item['is_correct']
            
            if full_seq is None or label is None:
                continue

            think_idx = item.get('think_idx', 0)
            if think_idx == -1: think_idx = 0
            
            reasoning_seq = full_seq[think_idx:]
            if len(reasoning_seq) == 0: continue
            
            label_float = 1.0 if label else 0.0
            
            self.data.append({
                "input": reasoning_seq.to(device),
                "target": torch.full((len(reasoning_seq), 1), label_float).to(device),
                "raw_label": label_float
            })
            valid_count += 1
            
        print(f"✅ Loaded {valid_count} items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 2. Transformer 模型 (🔥核心修复：加入 LayerNorm)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LatentTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 🔥【关键修复】输入归一化
        # SAE 的激活值通常极小(0.01)或忽大忽小，这会导致信号被 Positional Encoding 淹没
        # LayerNorm 把它强行拉回标准分布 (Mean=0, Std=1)，让模型能看清特征
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [Batch, Seq, 10]
        
        # 1. 先做归一化！(这是解决 0.9960 不动的核心)
        x = self.input_norm(x)
        
        # 2. 再进 Embedding 和 Transformer
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        probs = self.head(x) 
        return probs

# ==========================================
# 3. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True) 
    parser.add_argument('--save_path', type=str, default='transformer_reward_model.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    dataset = ValueDatasetLegacy(args.data_file, device=device)
    if len(dataset) == 0:
        print("❌ No data loaded.")
        return

    # Class Balancing (解决数据不平衡)
    print("⚖️  Calculating Class Weights...")
    targets = [item['raw_label'] for item in dataset.data]
    class_counts = {0.0: 0, 1.0: 0}
    for t in targets:
        class_counts[t] += 1
    
    print(f"   Pos (Correct): {class_counts[1.0]}")
    print(f"   Neg (Wrong)  : {class_counts[0.0]}")
    
    if class_counts[0.0] == 0:
        print("⚠️ Warning: No negative samples! Training might fail.")
        sampler = None
    else:
        weight_pos = 1.0 / class_counts[1.0]
        weight_neg = 1.0 / class_counts[0.0]
        sample_weights = [weight_pos if t == 1.0 else weight_neg for t in targets]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(dataset), 
            replacement=True
        )
        print("✅ WeightedRandomSampler activated.")

    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    # 2. Init Model
    model = LatentTransformer(
        input_dim=5,
        d_model=args.hidden_dim, 
        nhead=4, 
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    print(f"\n🔥 Start Training Transformer (with LayerNorm fix)...")
    model.train()
    
    best_loss = float('inf')
    best_model_path = args.save_path.replace('.pt', '_best.pt') # 保存为 transformer_reward_model_best.pt

    for epoch in range(args.epochs):
        total_loss = 0
        correct_trends = 0
        count = 0
        
        for batch in dataloader:
            inputs = batch['input']
            targets = batch['target']
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Accuracy Check
            final_pred = preds[0, -1, 0].item()
            final_label = targets[0, -1, 0].item()
            
            if (final_pred > 0.5 and final_label == 1.0) or (final_pred < 0.5 and final_label == 0.0):
                correct_trends += 1
            count += 1
        
        avg_loss = total_loss / count
        acc = correct_trends / count
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Trend Acc: {acc:.2%}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  🌟 New Best! Saved to {best_model_path}")
        else:
            print("") # 换行

    torch.save(model.state_dict(), args.save_path)
    print(f"\n💾 Transformer Model saved to {args.save_path}")

if __name__ == "__main__":
    main()