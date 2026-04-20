# %%
"""interactive_hybrid_sentence.py

A minimal interactive demo for the sentence-level hybrid generation
approach. After each *thinking*-model sentence we detect the dominant
latent, generate the corresponding *base*-model sentence (steered by a
user-chosen coefficient) and let the user decide whether to keep it,
retry with a different coefficient, or fall back to an un-steered
sentence. The loop continues until an EOS token is produced or the
maximum number of new tokens is reached.

USAGE (example):
    python interactive_hybrid_sentence.py --example_idx 0 --dataset gsm8k \
        --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --base_model meta-llama/Llama-3.1-8B

The script deliberately keeps its dependency footprint tiny by re-using
utility helpers already present in *hybrid_sentence.py*.
"""
import dotenv
dotenv.load_dotenv("../.env")

import argparse
import gc
import math
import os
import re
import sys
from typing import Tuple

import torch
from datasets import load_dataset

# Project-local helpers --------------------------------------------------------
from utils.sae import load_sae
from utils.utils import (
    load_model,
    center_and_l2_normalize_torch,
    load_steering_vectors as _load_all_steering_vectors,
)
from utils.clustering import get_latent_descriptions

# -----------------------------------------------------------------------------
# Utility functions (mostly copied verbatim from hybrid_sentence.py)
# -----------------------------------------------------------------------------

def is_sentence_end(token_str: str) -> bool:
    """Heuristic to decide whether *token_str* ends a sentence."""
    stripped = token_str.strip()
    return bool(re.search(r"[.!?]$", stripped)) or stripped == "\n"


def get_next_token(logits, temperature: float):
    """Greedy or temperature sampling for one token."""
    if temperature > 0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    return torch.argmax(logits).item()


def get_token_and_str(logits, tokenizer, temperature: float):
    tok = get_next_token(logits, temperature)
    return tok, tokenizer.decode(tok)


def get_perplexity(token_str: str, logits, model) -> float:
    token_id = model.tokenizer.encode(token_str, return_tensors="pt", add_special_tokens=False).to(model.device)
    if token_id.shape[1] == 0 or token_id[0, 0].item() >= logits.shape[-1]:
        return float("inf")
    log_prob = torch.log_softmax(logits[0, -1], dim=-1)[token_id[0, 0]].item()
    return math.exp(-log_prob)

# -----------------------------------------------------------------------------
# Model & vector loading helpers
# -----------------------------------------------------------------------------

def load_steering_vectors(base_model_id: str, thinking_model_id: str, sae_layer: int, n_clusters: int):
    """Return mapping latent-key → steering-vector for the requested setup."""
    hyperparams_dir_abs = os.path.join(os.path.dirname(__file__), "../train-vectors/results/vars/hyperparams")
    vectors_dir_abs = os.path.join(os.path.dirname(__file__), "../train-vectors/results/vars/optimized_vectors")

    all_vectors = _load_all_steering_vectors(
        hyperparams_dir=hyperparams_dir_abs,
        vectors_dir=vectors_dir_abs,
        verbose=False,
    )

    descriptions = get_latent_descriptions(thinking_model_id, sae_layer, n_clusters)
    steering_vectors = {}
    for desc in descriptions.values():
        latent_key = desc.get("key", "")
        if latent_key:
            slug = latent_key.lower().replace(" ", "-")
            if slug in all_vectors:
                steering_vectors[latent_key] = all_vectors[slug]
    return steering_vectors, descriptions

# -----------------------------------------------------------------------------
# Sentence-level generation building blocks
# -----------------------------------------------------------------------------

def thinking_generate_sentence(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    sae_layer: int,
    sae,
    temperature: float = 0.0,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Generate **one** sentence with *model* and return:
    1. the decoded sentence string
    2. the averaged SAE activation for that sentence (latent-space)
    3. the updated *input_ids* (original prompt + new sentence)
    """

    sentence_tokens = []
    sentence_activations = []

    while True:
        # Forward pass for logits & last-token activation
        with torch.no_grad():
            with model.trace(input_ids) as tracer:
                logits = model.lm_head.output.save()
                act = model.model.layers[sae_layer].output[0][0, -1, :].save()

        # Sample next token
        next_tok_id, next_tok_str = get_token_and_str(logits[0, -1], tokenizer, temperature)
        tok_ids = tokenizer.encode(next_tok_str, return_tensors="pt", add_special_tokens=False).to(model.device)

        # Update containers & sequences
        input_ids = torch.cat([input_ids, tok_ids.to(torch.long)], dim=1)
        sentence_tokens.append(next_tok_str)
        sentence_activations.append(act.detach().clone())

        # Sentence boundary?
        if is_sentence_end(next_tok_str) or next_tok_id == tokenizer.eos_token_id:
            break

    # Average activations
    stacked = torch.stack(sentence_activations, dim=0)
    avg_activation = torch.mean(stacked, dim=0)

    # SAE latent vector
    x_norm = center_and_l2_normalize_torch(avg_activation, sae.activation_mean)
    latent_acts = sae.encoder(x_norm - sae.b_dec)

    return "".join(sentence_tokens), latent_acts, input_ids


def base_generate_sentence(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    coefficient: float,
    steering_layer: int,
    temperature: float = 0.0,
) -> Tuple[str, torch.Tensor]:
    """Generate a sentence with (optionally) applied steering.

    Returns the decoded sentence string and the updated *input_ids*.
    """

    sentence_tokens = []

    while True:
        with torch.no_grad():
            with model.trace(input_ids) as tracer:
                # Apply steering to the *current* hidden states (last 50 tokens)
                if steering_vector is not None and coefficient != 0.0:
                    model.model.layers[steering_layer].input[:, -1:, :] += coefficient * steering_vector
                logits = model.lm_head.output.save()

        next_tok_id, next_tok_str = get_token_and_str(logits[0, -1], tokenizer, temperature)
        tok_ids = tokenizer.encode(next_tok_str, return_tensors="pt", add_special_tokens=False).to(model.device)
        input_ids = torch.cat([input_ids, tok_ids.to(torch.long)], dim=1)
        sentence_tokens.append(next_tok_str)

        if is_sentence_end(next_tok_str) or next_tok_id == tokenizer.eos_token_id:
            break

    return "".join(sentence_tokens), input_ids

# -----------------------------------------------------------------------------
# Helper: compute activations for an arbitrary sentence passed through the
# thinking model (mirrors logic from thinking_generate_sentence)
# -----------------------------------------------------------------------------

def compute_sentence_activation(
    model,
    tokenizer,
    sae_layer: int,
    sae,
    prefix_ids: torch.Tensor,
    sentence_str: str,
):
    """Return SAE latent vector for *sentence_str* when appended after *prefix_ids*.

    The function feeds *sentence_str* token-by-token through *model*, collects the
    hidden activations from *sae_layer*, averages them over the sentence and
    finally projects into SAE latent space – identical to the procedure used in
    *thinking_generate_sentence*.
    """

    # Tokenise the sentence (no special tokens) and send to correct device
    sentence_token_ids = tokenizer.encode(
        sentence_str,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device).to(torch.long)

    # Keep a copy of the prefix to avoid side effects
    input_ids = prefix_ids.clone()
    token_activations = []

    for i in range(sentence_token_ids.shape[1]):
        # Append the next token
        input_ids = torch.cat([input_ids, sentence_token_ids[:, i : i + 1]], dim=1)
        # Forward pass and record activation for the **current** token
        with torch.no_grad():
            with model.trace(input_ids) as tracer:
                act = model.model.layers[sae_layer].output[0][0, -1, :].save()
        token_activations.append(act.detach().clone())

    if not token_activations:
        # Empty sentence – return zeros of correct length
        return torch.zeros(sae.encoder.out_features, device=model.device)

    stacked = torch.stack(token_activations, dim=0)
    avg_act = torch.mean(stacked, dim=0)
    x_norm = center_and_l2_normalize_torch(avg_act, sae.activation_mean)
    latent_acts = sae.encoder(x_norm - sae.b_dec)
    return latent_acts

# -----------------------------------------------------------------------------
# Main interactive loop
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive hybrid sentence-level generation")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500", "aime"])
    parser.add_argument("--example_idx", type=int, default=21)
    parser.add_argument("--thinking_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--steering_layer", type=int, default=12)
    parser.add_argument("--sae_layer", type=int, default=6)
    parser.add_argument("--n_clusters", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--default_coeff", type=float, default=0.5, help="Default steering coefficient")

    args, _ = parser.parse_known_args()

    # ---------------------------------------------------------------------
    # Dataset & example selection
    # ---------------------------------------------------------------------
    print(f"Loading dataset {args.dataset}…")
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")["test"]  # type: ignore
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]  # type: ignore
    else:  # aime
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]  # type: ignore

    if args.example_idx >= len(dataset):
        sys.exit(f"example_idx {args.example_idx} is out of range!")

    if args.dataset == "gsm8k":
        question = dataset[args.example_idx]["question"]
        correct_answer = dataset[args.example_idx]["answer"]
    else:
        question = dataset[args.example_idx]["problem"]
        correct_answer = dataset[args.example_idx]["answer"]

    print("\n===== Task =====\n")
    print(question)

    # ---------------------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------------------
    print("\nLoading models… (this can take a while)")
    thinking_model, thinking_tok = load_model(model_name=args.thinking_model)
    thinking_model.tokenizer = thinking_tok  # convenience
    base_model, base_tok = load_model(model_name=args.base_model)

    # Sampling config
    if args.temperature > 0:
        thinking_model.generation_config.do_sample = True
        base_model.generation_config.do_sample = True

    # ---------------------------------------------------------------------
    # SAE & steering vectors
    # ---------------------------------------------------------------------
    thinker_id = args.thinking_model.split("/")[-1].lower()
    base_id = args.base_model.split("/")[-1].lower()
    sae, _ = load_sae(thinker_id, args.sae_layer, args.n_clusters)
    sae = sae.to(thinking_model.device)

    steering_vectors, latent_descs = load_steering_vectors(base_id, thinker_id, args.sae_layer, args.n_clusters)

    # ---------------------------------------------------------------------
    # Initial prompts
    # ---------------------------------------------------------------------
    thinking_ids = thinking_tok.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(thinking_model.device).to(torch.long)

    base_prompt = (
        f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n"
    )
    base_ids = base_tok.encode(base_prompt, return_tensors="pt").to(base_model.device).to(torch.long)

    # ---------------------------------------------------------------------
    # Interactive generation loop
    # ---------------------------------------------------------------------
    generated_tokens = 0
    print("\n===== Interactive Generation =====\n")
    coeff = args.default_coeff

    while generated_tokens < args.max_new_tokens:
        # (1) Thinking model – produce one sentence
        sent_think, latent_acts, thinking_ids = thinking_generate_sentence(
            thinking_model,
            thinking_tok,
            thinking_ids,
            args.sae_layer,
            sae,
            temperature=args.temperature,
        )
        print(f"\n[Thinking] {sent_think}")

        # Latent detection
        # --- obtain top-3 latent categories by activation ---
        top_vals, top_ids_tensor = torch.topk(latent_acts, k=min(3, latent_acts.shape[0]))
        top_ids = top_ids_tensor.tolist()
        top_vals = top_vals.tolist()

        print("Top 3 latent categories (thinking model):")
        for rank, (lid, val) in enumerate(zip(top_ids, top_vals), 1):
            title = latent_descs[lid]["title"]
            print(f"  {rank}. {title} (id={lid}, activation={val:.3f})")

        # Primary category (rank-1) used for steering
        latent_id = top_ids[0]
        latent_key = latent_descs[latent_id]["key"]
        steering_vec = steering_vectors.get(latent_key)

        if steering_vec is None:
            print(f"No steering vector available for primary category '{latent_descs[latent_id]['title']}'.")

        # ------------------------------------------------------------------
        # (2) Base model – ALWAYS show an un-steered sentence first
        # ------------------------------------------------------------------
        base_ids_pre = base_ids.clone()  # keep original state intact
        sent_unsteered, base_ids_unsteered = base_generate_sentence(
            base_model,
            base_tok,
            base_ids_pre,
            steering_vector=None,
            coefficient=0.0,
            steering_layer=args.steering_layer,
            temperature=args.temperature,
        )
        print(f"[Base • unsteered] {sent_unsteered}")

        # --------------------------------------------------------------
        # Analyse un-steered base sentence with the thinking model
        # --------------------------------------------------------------
        unsteered_latent_acts = compute_sentence_activation(
            thinking_model,
            thinking_tok,
            args.sae_layer,
            sae,
            thinking_ids,
            sent_unsteered,
        )

        print("Comparison with un-steered base sentence:")
        diff_vals = []  # store Δ for default coefficients
        for rank, lid in enumerate(top_ids, 1):
            think_val = top_vals[rank - 1]
            base_val = unsteered_latent_acts[lid].item()
            diff_val = think_val - base_val  # positive => base lower than thinking
            diff_vals.append(diff_val)
            title = latent_descs[lid]["title"]
            print(
                f"  {rank}. {title}: thinking={think_val:.3f}, base={base_val:.3f}, Δ={diff_val:+.3f}"
            )

        # ------------------------
        # Determine the latent with the largest |Δ| that has a steering vector
        # ------------------------
        best_idx = None
        best_abs = 0.0
        best_vec = None
        for idx, (lid, delta_val) in enumerate(zip(top_ids, diff_vals)):
            vec = steering_vectors.get(latent_descs[lid]["key"])
            if vec is None:
                continue
            if abs(delta_val) > best_abs:
                best_abs = abs(delta_val)
                best_idx = idx
                best_vec = vec

        # Default selection variables
        chosen_coeff: float = 0.0
        sent_base: str = sent_unsteered
        base_ids_final = base_ids_unsteered

        if best_vec is not None:
            # Interactive loop for single coefficient
            target_title = latent_descs[top_ids[best_idx]]["title"]
            default_coeff = diff_vals[best_idx]

            while True:
                prompt_msg = (
                    f"Enter steering coefficient for category '{target_title}' "
                    f"(blank=Δ {default_coeff:+.3f}, 'k' to keep, 'q' to quit): "
                )
                user_in = input(prompt_msg).strip().lower()

                if user_in in ("k", "keep"):
                    break  # keep unsteered
                if user_in == "q":
                    print("Exiting interactive loop.")
                    return

                if user_in == "":
                    coeff_val = default_coeff
                else:
                    try:
                        coeff_val = float(user_in)
                    except ValueError:
                        print("  Invalid input – please enter a number, blank for default, 'k' to keep, or 'q' to quit.")
                        continue

                # Regenerate steered variant
                sent_steered, base_ids_steered = base_generate_sentence(
                    base_model,
                    base_tok,
                    base_ids.clone(),
                    best_vec.to(base_model.device),
                    coeff_val,
                    args.steering_layer,
                    temperature=args.temperature,
                )

                print(f"[Base • steered coeff={coeff_val:+.3f}] {sent_steered}")
                accept = input("Accept steered sentence? (y/[n]): ").strip().lower()
                if accept in ("y", "yes", ""):
                    chosen_coeff = coeff_val
                    sent_base = sent_steered
                    base_ids_final = base_ids_steered
                    break
                # else loop and ask again

        # Finalise selection for this iteration
        base_ids = base_ids_final
        generated_tokens += len(base_tok.encode(sent_base))
        print(f"[Base] (coeff={chosen_coeff:+.3f}) {sent_base}")

        # Keep both models in sync: append the *accepted* base sentence to thinking_ids
        think_append = thinking_tok.encode(
            sent_base,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(thinking_model.device).to(torch.long)
        thinking_ids = torch.cat([thinking_ids, think_append], dim=1)

        # EOS?
        if base_ids[0, -1].item() == base_tok.eos_token_id:
            print("\n<EOS reached – stopping>")
            break

        # Optionally allow user to continue or stop
        cont = input("Continue? (y/[n]): ").strip().lower()
        if cont not in ("y", "yes", ""):
            break

        # House-keeping
        torch.cuda.empty_cache()
        gc.collect()

    # ---------------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------------
    answer = base_tok.decode(base_ids[0][len(base_prompt.split()):], skip_special_tokens=True)
    print("\n===== Final Hybrid Answer =====\n")
    print(answer)
    print("\nReference answer (for convenience):\n", correct_answer)


if __name__ == "__main__":
    main() 
# %%
