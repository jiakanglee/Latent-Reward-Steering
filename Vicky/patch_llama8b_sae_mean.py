"""
Patch DeepSeek-R1-Distill-Llama-8B SAE checkpoints with activation_mean.

The original SAE checkpoints were trained before activation_mean was embedded
into checkpoints. This script:
  1. Loads Llama-8B and runs forward passes over the responses JSON
  2. Computes the per-layer running mean (same logic as process_saved_responses)
  3. Saves mean pkl files (so load_sae's cross-check passes)
  4. Patches all existing SAE checkpoints to include activation_mean

Run from: thinking-llms-interp/
  python Vicky/patch_llama8b_sae_mean.py
"""

import torch
import pickle
import json
import os
import sys
import random
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.utils import load_model, get_char_to_token_map, split_into_sentences
from utils.responses import extract_thinking_process

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_ID     = "deepseek-r1-distill-llama-8b"
LAYERS       = [6, 10, 14, 18, 22, 26]
N_EXAMPLES   = 100000   # matches original SAE training; all 12k responses used
RESPONSES_JSON = "generate-responses/results/vars/responses_deepseek-r1-distill-llama-8b.json"
MEAN_DIR     = "generate-responses/results/vars"
SAE_DIR      = "train-saes/results/vars/saes"
MAX_SEQ_LEN  = 2048     # truncate long responses to avoid OOM
# ────────────────────────────────────────────────────────────────────────────


def mean_pkl_path(layer):
    return os.path.join(MEAN_DIR, f"activations_{MODEL_ID}_{N_EXAMPLES}_{layer}_mean.pkl")


def compute_layer_means(model, tokenizer):
    """
    One pass over the responses JSON, collecting running mean per layer.
    Mirrors the logic in utils.utils.process_saved_responses.
    """
    print(f"Loading responses from {RESPONSES_JSON} ...")
    with open(RESPONSES_JSON) as f:
        responses_data = json.load(f)

    random.shuffle(responses_data)
    responses_data = responses_data[:N_EXAMPLES]
    print(f"Using {len(responses_data)} responses")

    hidden_size = model.config.hidden_size
    mean_by_layer  = {l: torch.zeros(1, hidden_size) for l in LAYERS}
    count_by_layer = {l: 0 for l in LAYERS}

    for response_data in tqdm(responses_data, desc="Forward passes"):
        thinking_process = extract_thinking_process(response_data["full_response"])
        if not thinking_process:
            continue

        full_response = response_data["full_response"]
        sentences = split_into_sentences(thinking_process)
        if not sentences:
            continue

        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        if input_ids.shape[1] > MAX_SEQ_LEN:
            input_ids = input_ids[:, :MAX_SEQ_LEN]

        # Collect activations at all layers in a single trace
        layer_outputs = {}
        with model.trace({
            "input_ids": input_ids,
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        }):
            for layer in LAYERS:
                layer_outputs[layer] = model.model.layers[layer].output.save()

        for layer in LAYERS:
            layer_outputs[layer] = layer_outputs[layer].detach().cpu().to(torch.float32)

        char_to_token = get_char_to_token_map(full_response, tokenizer)

        for layer in LAYERS:
            layer_output = layer_outputs[layer]
            min_tok = float('inf')
            max_tok = -float('inf')

            for sentence in sentences:
                text_pos = full_response.find(sentence)
                if text_pos < 0:
                    continue
                token_start = char_to_token.get(text_pos, None)
                token_end   = char_to_token.get(text_pos + len(sentence), None)
                if token_start is not None and token_end is not None and token_start < token_end:
                    min_tok = min(min_tok, token_start)
                    max_tok = max(max_tok, token_end)

            if min_tok < layer_output.shape[1] and max_tok > 0:
                vector = layer_output[:, min_tok:max_tok, :].mean(dim=1).cpu()
                n = count_by_layer[layer]
                mean_by_layer[layer] = mean_by_layer[layer] + (vector - mean_by_layer[layer]) / (n + 1)
                count_by_layer[layer] += 1

    return mean_by_layer, count_by_layer


def save_mean_pkls(mean_by_layer, count_by_layer):
    os.makedirs(MEAN_DIR, exist_ok=True)
    for layer in LAYERS:
        mean_np = mean_by_layer[layer].cpu().numpy().reshape(-1).astype(np.float32)
        path = mean_pkl_path(layer)
        with open(path, "wb") as f:
            pickle.dump({
                "model_id":        MODEL_ID,
                "layer":           int(layer),
                "n_examples":      int(N_EXAMPLES),
                "count_vectors":   int(count_by_layer[layer]),
                "activation_mean": mean_np,
            }, f)
        print(f"Saved mean pkl → {path}  (count={count_by_layer[layer]})")


def patch_sae_checkpoints(mean_by_layer):
    sae_files = [
        f for f in os.listdir(SAE_DIR)
        if f.startswith(f"sae_{MODEL_ID}_layer") and f.endswith(".pt")
    ]
    print(f"\nFound {len(sae_files)} SAE checkpoints to patch.")

    for fname in sorted(sae_files):
        # Parse layer from filename: sae_deepseek-r1-distill-llama-8b_layer22_clusters30.pt
        parts = fname.replace(".pt", "").split("_layer")
        layer_str = parts[1].split("_clusters")[0]
        layer = int(layer_str)

        if layer not in mean_by_layer:
            print(f"  SKIP {fname} (layer {layer} not in computed layers)")
            continue

        path = os.path.join(SAE_DIR, fname)
        ckpt = torch.load(path, weights_only=False)

        if "activation_mean" in ckpt:
            print(f"  SKIP {fname} (already has activation_mean)")
            continue

        mean_t = mean_by_layer[layer].cpu().to(torch.float32).reshape(-1)
        assert mean_t.shape == (int(ckpt["input_dim"]),), \
            f"Shape mismatch: mean {tuple(mean_t.shape)} vs input_dim {ckpt['input_dim']}"

        ckpt["activation_mean"]          = mean_t
        ckpt["activation_mean_model_id"] = MODEL_ID
        ckpt["activation_mean_layer"]    = layer
        ckpt["activation_mean_n_examples"] = int(N_EXAMPLES)

        torch.save(ckpt, path)
        print(f"  PATCHED {fname}")


def main():
    print("Loading model ...")
    model, tokenizer = load_model(model_name=MODEL_NAME, device="auto")

    mean_by_layer, count_by_layer = compute_layer_means(model, tokenizer)

    print("\nSaving mean pkl files ...")
    save_mean_pkls(mean_by_layer, count_by_layer)

    print("\nPatching SAE checkpoints ...")
    patch_sae_checkpoints(mean_by_layer)

    print("\nDone. All SAE checkpoints patched.")


if __name__ == "__main__":
    main()
