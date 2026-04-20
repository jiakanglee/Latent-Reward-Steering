import torch.nn as nn
import os
import torch
import pickle

def load_sae(model_id, layer, n_clusters, load_base_decoder=False, require_activation_mean: bool = True):
    # Try several plausible locations for the trained SAE checkpoints. Historically the
    # repo referenced '../train-saes' but in this workspace train-saes lives under
    # the project root 'train-saes' (i.e. './train-saes'). Try both.
    candidates = [
        f'../train-saes/results/vars/saes/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt',
        f'./train-saes/results/vars/saes/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt',
        f'train-saes/results/vars/saes/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt',
    ]
    sae_path = None
    for c in candidates:
        if os.path.exists(c):
            sae_path = c
            break
    if sae_path is None:
        # Try to help by listing available SAE files in the most-likely directory
        likely_dir = None
        for d in ["./train-saes/results/vars/saes", "../train-saes/results/vars/saes", "train-saes/results/vars/saes"]:
            if os.path.isdir(d):
                likely_dir = d
                break
        available = []
        if likely_dir is not None:
            try:
                available = os.listdir(likely_dir)
            except Exception:
                available = []
        raise FileNotFoundError(
            f"SAE model not found for model_id={model_id}, layer={layer}, clusters={n_clusters}.\n"
            f"Tried paths: {candidates}\n"
            f"Available files in {likely_dir or 'none'}: {available[:50]}"
        )
        
    checkpoint = torch.load(sae_path, weights_only=False)
    
    # Create SAE model
    sae = SAE(checkpoint['input_dim'], checkpoint['num_latents'], k=checkpoint.get('topk', 3))
    
    # Load weights
    sae.encoder.weight.data = checkpoint['encoder_weight']
    sae.encoder.bias.data = checkpoint['encoder_bias']
    sae.W_dec.data = checkpoint['decoder_weight']
    sae.b_dec.data = checkpoint['b_dec']

    if require_activation_mean:
        # Require activation mean for downstream parity with SAE training.
        assert "activation_mean" in checkpoint, (
            "SAE checkpoint is missing 'activation_mean'. This repo now requires SAE checkpoints to embed the "
            "centering mean used for activation-cache construction so downstream usage can reproduce "
            "centered+L2-normalized activations."
        )
        activation_mean = checkpoint["activation_mean"]
        assert isinstance(activation_mean, torch.Tensor), f"activation_mean must be torch.Tensor, got {type(activation_mean)}"
        assert activation_mean.ndim == 1, f"activation_mean must be 1D, got shape {tuple(activation_mean.shape)}"
        assert activation_mean.shape == (int(checkpoint["input_dim"]),), (
            f"activation_mean shape mismatch: {tuple(activation_mean.shape)} vs expected {(int(checkpoint['input_dim']),)}"
        )
        assert torch.isfinite(activation_mean).all(), "Non-finite values in activation_mean"
        sae.activation_mean.copy_(activation_mean.to(dtype=torch.float32, device=sae.activation_mean.device))

        # Sanity-check metadata when present.
        if "activation_mean_model_id" in checkpoint:
            assert checkpoint["activation_mean_model_id"] == model_id, (
                f"activation_mean_model_id mismatch: {checkpoint['activation_mean_model_id']} vs {model_id}"
            )
        if "activation_mean_layer" in checkpoint:
            assert int(checkpoint["activation_mean_layer"]) == int(layer), (
                f"activation_mean_layer mismatch: {checkpoint['activation_mean_layer']} vs {layer}"
            )

        # Extra safety: if a mean file exists alongside the cached activations, assert it matches the checkpoint.
        # (This catches accidental mismatches when multiple cache builds exist for the same model/layer.)
        if "activation_mean_n_examples" in checkpoint:
            n_examples = int(checkpoint["activation_mean_n_examples"])
            mean_pkl = f"../generate-responses/results/vars/activations_{model_id}_{n_examples}_{layer}_mean.pkl"
            if os.path.exists(mean_pkl):
                with open(mean_pkl, "rb") as f:
                    payload = pickle.load(f)
                assert isinstance(payload, dict), f"Bad mean payload type: {type(payload)}"
                mean_file = payload.get("activation_mean", None)
                assert mean_file is not None, f"Mean payload missing 'activation_mean': keys={list(payload.keys())}"
                mean_file_t = torch.as_tensor(mean_file, dtype=torch.float32, device="cpu").reshape(-1)
                mean_ckpt_t = activation_mean.detach().cpu().to(torch.float32).reshape(-1)
                assert mean_file_t.shape == mean_ckpt_t.shape, (
                    f"Mean shape mismatch: file {tuple(mean_file_t.shape)} vs ckpt {tuple(mean_ckpt_t.shape)}"
                )
                assert torch.equal(mean_file_t, mean_ckpt_t), (
                    "Activation mean mismatch between mean file and SAE checkpoint.\n"
                    f"mean_file={mean_pkl}\n"
                    f"sae_ckpt={sae_path}"
                )
    
    print(f"Loaded SAE model from {sae_path}")

    return sae, checkpoint

class SAE(nn.Module):
    def __init__(self, d_in, num_latents, k=1):
        super().__init__()
        self.encoder = nn.Linear(d_in, num_latents, bias=True)
        self.encoder.bias.data.zero_()
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Mean used for centering before L2-normalization (persisted in SAE checkpoints).
        self.register_buffer("activation_mean", torch.zeros(d_in, dtype=torch.float32), persistent=False)
        self.k = k
        self.set_decoder_norm_to_unit_norm()
        
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + 1e-5
        
    def encode(self, x):
        forward = self.encoder(x - self.b_dec)
        top_acts, top_indices = forward.topk(self.k, dim=-1)
        return top_acts, top_indices
        
    def decode(self, top_acts, top_indices):
        batch_size = top_indices.shape[0]
        
        # Reshape for embedding_bag
        top_acts_flat = top_acts.view(-1)
        top_indices_flat = top_indices.view(-1)
        
        # For embedding_bag we need offsets that point to the start of each sample
        offsets = torch.arange(0, batch_size, device=top_indices.device) * self.k
        
        # Use embedding_bag
        res = nn.functional.embedding_bag(
            top_indices_flat, self.W_dec, offsets=offsets, 
            per_sample_weights=top_acts_flat, mode="sum"
        )
        
        return res + self.b_dec
        
    def forward(self, x):
        top_acts, top_indices = self.encode(x)
        return self.decode(top_acts, top_indices)