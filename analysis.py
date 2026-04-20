import torch, numpy as np
import matplotlib.pyplot as plt

data = torch.load("collected_sae_latents_10dim_2000.pt")
samples = [s for s in data if int(s["length"]) != 2000]

dim4, lengths = [], []
for s in samples:
    T = int(s["length"])
    z = s["latent_seq"][:T]
    dim4.append(z.mean(dim=0)[4].item())
    lengths.append(T)

dim4 = np.array(dim4)
lengths = np.array(lengths)

plt.figure(figsize=(6,5))
plt.scatter(dim4, lengths, alpha=0.6)
plt.xlabel("Mean activation of latent dim 4")
plt.ylabel("Token length")
plt.title("Latent dim 4 vs. reasoning length")
plt.tight_layout()
plt.show()
plt.savefig("latent_dim4_vs_length.png")
