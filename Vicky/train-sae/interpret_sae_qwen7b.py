"""
SAE Interpretation for DeepSeek-R1-Distill-Qwen-7B

For each (layer, n_clusters), load the trained SAE and cached activations,
then find the top-K max-activating sentences for each SAE feature/dimension.
Results are saved to results/vars/sae_topk_results_*_layer*.json.

Run from: thinking-llms-interp/Vicky/train-sae/
  python interpret_sae_qwen7b.py --layer 4 --n_clusters 10 20 30
"""

import argparse
import json
import os
import sys
import pickle
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.sae import SAE

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
parser.add_argument('--layer', type=int, required=True)
parser.add_argument('--n_clusters', type=int, nargs='+',
                    default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
parser.add_argument('--n_examples', type=int, default=12032)
parser.add_argument('--top_k', type=int, default=20,
                    help='Number of top activating examples to store per feature')
args = parser.parse_args()

MODEL_ID = args.model.split('/')[-1].lower()
ACTS_PKL = f'results/vars/activations_{MODEL_ID}_{args.n_examples}_{args.layer}.pkl'
RESULTS_DIR = 'results/vars'


# ── Load cached activations ────────────────────────────────────────────────────
print(f'Loading cached activations from {ACTS_PKL} ...')
assert os.path.exists(ACTS_PKL), f'Activation cache not found: {ACTS_PKL}'
with open(ACTS_PKL, 'rb') as f:
    all_activations, all_texts = pickle.load(f)

# all_activations: np.ndarray (N, d_model), already centered + L2-normalized
# all_texts:       list[str], length N
all_activations = np.asarray(all_activations, dtype=np.float32)
N, d_model = all_activations.shape
print(f'  {N} sentences, d_model={d_model}')

acts_tensor = torch.from_numpy(all_activations)  # (N, d_model), stays on CPU


# ── Process each cluster size ──────────────────────────────────────────────────
for n_clusters in args.n_clusters:
    sae_path = f'results/vars/saes/sae_{MODEL_ID}_layer{args.layer}_clusters{n_clusters}.pt'
    results_path = f'{RESULTS_DIR}/sae_topk_results_{MODEL_ID}_layer{args.layer}.json'

    if not os.path.exists(sae_path):
        print(f'[SKIP] SAE not found: {sae_path}')
        continue

    print(f'\n── Layer {args.layer}, clusters={n_clusters} ──')

    # Load SAE checkpoint
    ckpt = torch.load(sae_path, weights_only=False, map_location='cpu')
    sae = SAE(ckpt['input_dim'], ckpt['num_latents'], k=ckpt.get('topk', 3))
    sae.encoder.weight.data = ckpt['encoder_weight']
    sae.encoder.bias.data   = ckpt['encoder_bias']
    sae.W_dec.data          = ckpt['decoder_weight']
    sae.b_dec.data          = ckpt['b_dec']
    sae.eval()

    # Run all activations through SAE encoder in one shot (CPU, no grad)
    # SAE.encode(x) computes encoder(x - b_dec) then topk
    # The cached activations are already the input space the SAE was trained on.
    with torch.no_grad():
        # encoder_acts: (N, n_clusters) — full encoder output before topk
        encoder_acts = sae.encoder(acts_tensor - sae.b_dec).numpy()  # (N, n_clusters)

    # For each feature dimension, find top-K sentences by activation value
    per_feature_results = []
    for dim in range(n_clusters):
        feature_acts = encoder_acts[:, dim]          # (N,)
        top_indices  = np.argsort(feature_acts)[::-1][:args.top_k]

        examples = []
        for idx in top_indices:
            act_val = float(feature_acts[idx])
            if act_val <= 0:
                break  # activations are sparse; stop once we hit non-positive
            examples.append({
                'activation': round(act_val, 5),
                'text': all_texts[idx],
            })

        per_feature_results.append({
            'feature_id': dim,
            'top_examples': examples,
        })

    # Load existing results JSON and update the entry for this cluster size
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results_data = json.load(f)
    else:
        results_data = {
            'clustering_method': 'sae_topk',
            'model_id': args.model,
            'layer': args.layer,
            'results_by_cluster_size': {},
        }

    results_data['results_by_cluster_size'][str(n_clusters)] = {
        'all_results':     per_feature_results,
        'avg_final_score': 0.0,
        'statistics':      {},
    }

    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f'  Saved {n_clusters}-cluster results → {results_path}')

print('\nDone.')
