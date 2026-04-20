import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import argparse
import math


# ==========================================
# 1. Dataset
#    现在只保留 prefix sequence + sequence-level label
# ==========================================
class ValueDatasetPrefix(Dataset):
    def __init__(self, data_path):
        print(f"📂 Loading data from {data_path}...")

        try:
            raw_data = torch.load(data_path, map_location="cpu")
        except Exception as e:
            print(f"❌ Error opening file: {e}")
            self.data = []
            return

        self.data = []
        valid_count = 0

        for item in raw_data:
            full_seq = None
            label = None

            # 1) 找 latent sequence
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 2 and v.shape[1] == 10:
                        full_seq = v
                        break

            if full_seq is None:
                if 'latent_seq' in item:
                    full_seq = item['latent_seq']
                elif 'sae_latents' in item:
                    full_seq = item['sae_latents']

            # 2) 找 label
            if 'label' in item:
                label = item['label']
            elif 'is_correct' in item:
                label = item['is_correct']

            if full_seq is None or label is None:
                continue

            think_idx = item.get('think_idx', 0)
            if think_idx == -1:
                think_idx = 0

            reasoning_seq = full_seq[think_idx:]
            if len(reasoning_seq) == 0:
                continue

            label_float = 1.0 if bool(label) else 0.0

            self.data.append({
                "input": reasoning_seq.float(),                     # [T, 10]
                "label": torch.tensor(label_float, dtype=torch.float32)
            })
            valid_count += 1

        print(f"✅ Loaded {valid_count} items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==========================================
# 2. Model
#    causal transformer + last-token logit
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
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
        # 上三角为 -inf，表示未来不可见
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )
        return mask

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


# ==========================================
# 3. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='transformer_reward_model_prefix.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load data
    dataset = ValueDatasetPrefix(args.data_file)
    if len(dataset) == 0:
        print("❌ No data loaded.")
        return

    # 2) Weighted sampler
    print("⚖️ Calculating Class Weights...")
    labels = [item["label"].item() for item in dataset.data]
    class_counts = {0.0: 0, 1.0: 0}
    for y in labels:
        class_counts[y] += 1

    print(f"   Pos (Correct): {class_counts[1.0]}")
    print(f"   Neg (Wrong)  : {class_counts[0.0]}")

    sampler = None
    if class_counts[0.0] > 0 and class_counts[1.0] > 0:
        weight_pos = 1.0 / class_counts[1.0]
        weight_neg = 1.0 / class_counts[0.0]
        sample_weights = [weight_pos if y == 1.0 else weight_neg for y in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        print("✅ WeightedRandomSampler activated.")
    else:
        print("⚠️ Warning: one class is missing.")

    # batch_size 先保持 1，避免 variable length collate 麻烦
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=(sampler is None))

    # 3) Model
    model = LatentTransformer(
        input_dim=10,
        d_model=args.hidden_dim,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    best_model_path = args.save_path.replace(".pt", "_best.pt")

    print("\n🔥 Start Training Prefix Reward Model...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            inputs = batch["input"].to(device)   # [1, T, 10]
            labels = batch["label"].to(device)   # [1]

            optimizer.zero_grad()

            logits = model(inputs, return_logits=True)[:, -1, 0]  # 只取最后一个位置
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

        avg_loss = total_loss / max(len(dataloader), 1)
        acc = correct / max(total, 1)

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | Final Acc: {acc:.2%}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  🌟 New Best! Saved to {best_model_path}")

    torch.save(model.state_dict(), args.save_path)
    print(f"\n💾 Final model saved to {args.save_path}")


if __name__ == "__main__":
    main()