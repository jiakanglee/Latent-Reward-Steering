import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import numpy as np

# ==========================================
# 1. 模型结构 (必须与训练代码 V2 完全一致)
# ==========================================
class AdvancedLSTMRewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.5):
        super().__init__()
        # 结构必须严格对应：Bi-LSTM, 2 Layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, 
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output_packed, (hn, cn) = self.lstm(x_packed)
        # 取最后一层 (layer_idx = -1 和 -2)
        final_fwd = hn[-2, :, :]
        final_bwd = hn[-1, :, :]
        final_embedding = torch.cat((final_fwd, final_bwd), dim=1)
        return self.head(final_embedding)

# ==========================================
# 2. 数据处理
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        print(f"📂 Loaded {len(self.data)} samples from {data_path}")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]['latent_seq'], self.data[idx]['label']

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return padded_seqs, labels, lengths

# ==========================================
# 3. 评测主逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # 你的新数据集路径
    parser.add_argument('--test_data', type=str, default='collected_sae_latents_10dim_1300.pt')
    # 你训练好的模型路径 (默认是用 V2 练出来的)
    parser.add_argument('--model_path', type=str, default='lstm_reward_model.pth')
    args = parser.parse_args()

    if not os.path.exists(args.test_data):
        print(f"❌ Error: Test data {args.test_data} not found.")
        return

    # 1. 加载数据
    dataset = TrajectoryDataset(args.test_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 自动获取维度
    input_dim = dataset[0][0].shape[-1]
    print(f"🧐 Data Dimension: {input_dim}")

    # 2. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Loading model from {args.model_path}...")
    
    try:
        model = AdvancedLSTMRewardModel(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"\n❌ Model Loading Failed: {e}")
        print("💡 Hint: Did you train with 'AdvancedLSTMRewardModel' (Bi-LSTM)? Make sure class definition matches.")
        return

    # 3. 开始推理
    correct_preds = 0
    total_samples = 0
    
    true_pos = 0 # 猜对的正样本
    true_neg = 0 # 猜对的负样本
    real_pos = 0 # 真实的正样本总数
    real_neg = 0 # 真实的负样本总数

    print("\n🚀 Start Evaluating...")
    with torch.no_grad():
        for X, y, lens in loader:
            X, y = X.to(device), y.to(device)
            
            logits = model(X, lens)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            correct_preds += (preds == y).sum().item()
            total_samples += y.size(0)
            
            # 统计召回率
            true_pos += ((preds == 1) & (y == 1)).sum().item()
            true_neg += ((preds == 0) & (y == 0)).sum().item()
            real_pos += (y == 1).sum().item()
            real_neg += (y == 0).sum().item()

    acc = correct_preds / total_samples
    
    print(f"\n📊 === Evaluation Report ===")
    print(f"Total Samples: {total_samples}")
    print(f"Overall Accuracy: {acc:.2%}  <-- 你的最终得分")
    
    print("-" * 30)
    if real_pos > 0:
        print(f"✅ Correct Recall: {true_pos/real_pos:.2%} ({true_pos}/{real_pos})")
        print("   (How many correct answers did we successfully identify?)")
    else:
        print("✅ Correct Recall: N/A (No positive samples)")
        
    if real_neg > 0:
        print(f"❌ Wrong Recall:   {true_neg/real_neg:.2%} ({true_neg}/{real_neg})")
        print("   (How many wrong answers did we successfully catch?)")
    else:
        print("❌ Wrong Recall: N/A (No negative samples)")
    print("-" * 30)

if __name__ == "__main__":
    main()