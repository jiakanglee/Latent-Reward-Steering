import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os

# ==========================================
# 1. 兼容型数据集定义 (Legacy Compatible)
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
        
        print(f"   Raw data count: {len(raw_data)}")
        
        if len(raw_data) > 0:
            print(f"   Sample keys: {raw_data[0].keys()}")

        for i, item in enumerate(raw_data):
            # --- 兼容性处理核心 (Fix applied here) ---
            
            # 1. 获取序列 (增加 'latent_seq' 的检查)
            if 'sae_latents' in item:
                full_seq = item['sae_latents']
            elif 'latents' in item:
                full_seq = item['latents']
            elif 'latent_seq' in item:  # <--- 🔥 加上了这一行！
                full_seq = item['latent_seq']
            else:
                # 找不到数据，打印一下第一条失败的原因方便调试
                if i == 0: print(f"⚠️ Warning: Key mismatch. Available: {item.keys()}")
                continue

            # 2. 获取 Label
            if 'label' in item:
                label = item['label']
            elif 'is_correct' in item:
                label = 1.0 if item['is_correct'] else 0.0
            else:
                continue

            # 3. 处理 think_idx
            # 你的老数据里其实有 think_idx 字段！日志里显示了！
            # 我们可以优先用它，如果它是 -1 再从头开始
            think_idx = item.get('think_idx', -1)
            
            if think_idx == -1:
                think_idx = 0 # 降级：如果没有有效的 think_idx，就从头开始 (包含 Prompt)
            
            # 4. 切片
            reasoning_seq = full_seq[think_idx:] 
            
            # 确保序列不为空
            if reasoning_seq.shape[0] == 0:
                continue

            # 5. 构造 Target
            seq_len = reasoning_seq.shape[0]
            target = torch.full((seq_len, 1), float(label), dtype=torch.float32)
            
            self.data.append({
                "input": reasoning_seq.to(device),
                "target": target.to(device)
            })
            valid_count += 1
            
        print(f"✅ Successfully loaded {valid_count} items (Fixed Key Check).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 2. 模型定义 (保持不变)
# ==========================================
class ValueLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features, _ = self.lstm(x)
        probs = self.head(features)
        return probs

# ==========================================
# 3. 训练主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='final_sae_data.pt') 
    parser.add_argument('--save_path', type=str, default='value_lstm_legacy.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用修正后的 Dataset
    dataset = ValueDatasetLegacy(args.data_file, device=device)
    
    if len(dataset) == 0:
        print("❌ CRITICAL ERROR: Still no data loaded. Please double check keys.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = ValueLSTM(input_dim=10, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    print(f"\n🔥 Start Training (Legacy Mode)...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        correct_trends = 0
        
        for batch in dataloader:
            inputs = batch['input']
            targets = batch['target']
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            final_pred = preds[0, -1, 0].item()
            final_label = targets[0, -1, 0].item()
            if (final_pred > 0.5 and final_label == 1.0) or (final_pred < 0.5 and final_label == 0.0):
                correct_trends += 1
        
        avg_loss = total_loss / len(dataloader)
        acc = correct_trends / len(dataloader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Trend Acc: {acc:.2%}")

    torch.save(model.state_dict(), args.save_path)
    print(f"\n💾 Model saved to {args.save_path}")

if __name__ == "__main__":
    main()