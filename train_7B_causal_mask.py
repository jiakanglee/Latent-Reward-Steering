import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import argparse
import math
import numpy as np
import os

# ==========================================
# 1. Dataset & Collate
# ==========================================
class ValueDataset(Dataset):
    def __init__(self, data_path):
        print(f"📂 Loading data from {data_path}...")
        raw_data = torch.load(data_path, map_location="cpu")
        self.data = []
        for item in raw_data:
            # --- 修改后的逻辑：显式检查 None，避免对 Tensor 做布尔判断 ---
            seq = item.get('latent_seq')
            if seq is None:
                seq = item.get('sae_latents')
            if seq is None:
                seq = next((v for v in item.values() if isinstance(v, torch.Tensor) and v.dim() == 2), None)
            
            label = item.get('label')
            if label is None:
                label = item.get('is_correct')
            # -------------------------------------------------------

            if seq is not None and label is not None:
                idx = max(0, item.get('think_idx', 0))
                if len(seq[idx:]) > 0:
                    self.data.append({
                        "input": seq[idx:].float(), 
                        "label": 1.0 if label else 0.0
                    })
        print(f"✅ Total Samples: {len(self.data)}")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    inputs = pad_sequence([item['input'] for item in batch], batch_first=True)
    targets = torch.stack([torch.full((inputs.size(1), 1), item['label']) for item in batch])
    return inputs, targets

# ==========================================
# 2. Causal Transformer Model
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class LatentTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        el = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(el, num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        # x shape: [Batch, SeqLen, Dim]
        seq_len = x.size(1)
        device = x.device
        
        # 1. 预处理
        x = self.pos_encoder(self.embedding(self.input_norm(x)))
        
        # 2. 🔥 核心修复：生成下三角掩码
        # generate_square_subsequent_mask 会生成一个上三角为 -inf，下三角为 0 的矩阵
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # 3. 传入 mask 和 is_causal 提示
        x = self.transformer_encoder(x, mask=mask, is_causal=True)
        
        return self.head(x)

# ==========================================
# 3. Main Training Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--hidden_dim', type=int, default=128) # 兼容你之前的参数名
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists('logs'): os.makedirs('logs')

    ds = ValueDataset(args.data_file)
    labels = [item['label'] for item in ds.data]
    class_counts = np.bincount(np.array(labels).astype(int))
    weights = 1. / class_counts
    sampler = WeightedRandomSampler([weights[int(l)] for l in labels], len(labels))
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)

    model = LatentTransformer(d_model=args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_loss = float('inf')
    best_acc = 0.0

    print(f"🔥 Start Training (Causal Mask Active)...")
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        all_preds, all_labels = [], []

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            with torch.no_grad():
                all_preds.extend((preds[:, -1, 0] > 0.5).cpu().numpy())
                all_labels.extend(targets[:, -1, 0].cpu().numpy())

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}", end="")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "transformer_reward_model_best_loss.pt")
            print(" | 📉 Best Loss!", end="")
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), "transformer_reward_model_best_acc.pt")
            print(" | 🎯 Best Acc!", end="")
        
        print("")

    print(f"\n✅ Done! Best Loss: {best_loss:.4f}, Best Acc: {best_acc:.2%}")

if __name__ == "__main__":
    main()