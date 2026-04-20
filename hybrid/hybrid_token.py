# %%
import dotenv
dotenv.load_dotenv("../.env")

import sys
import torch
import json
import zlib
import base64
import pickle
from utils.sae import load_sae
from utils.utils import load_model, center_and_l2_normalize_torch
from utils.clustering import get_latent_descriptions
from utils.utils import chat, chat_batch
from utils.utils import load_steering_vectors as _load_all_steering_vectors
import os
import time
import gc
import colorsys
import math
import matplotlib.pyplot as plt
import re
import argparse
from typing import List, Optional, Tuple
from collections import Counter
from matplotlib.patches import Rectangle
try:
    from tqdm.auto import tqdm  # progress bar
except Exception:
    tqdm = None


from datasets import load_dataset

CODING_DATASETS = {"mbpp", "livecodebench"}
MCQA_DATASETS = {"medqa"}  # Multiple choice QA datasets
TEXT_CLASSIFICATION_DATASETS = {"legalbench"}  # Text classification datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate hybrid model on datasets (token-level steering)')
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'math500', "aime24", "aime25", "mbpp", "livecodebench", "medqa", "legalbench"], default='aime24',
                      help='Dataset to evaluate on (gsm8k, math500, aime24, aime25, mbpp, livecodebench, medqa, or legalbench)')
    parser.add_argument('--thinking_model', type=str, default='Qwen/QwQ-32B',
                      help='Model for thinking/perplexity')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-32B',
                      help='Model for base generation')
    parser.add_argument('--steering_layer', type=int, default=24,
                      help='Layer to steer in the base model')
    parser.add_argument('--sae_layer', type=int, default=27,
                      help='Layer to read from in the thinking model for SAE projection')
    parser.add_argument('--n_clusters', type=int, default=10,
                      help='Number of clusters for SAE')
    parser.add_argument(
        '--disable-sae-mean',
        action='store_true',
        default=False,
        help='If set, do NOT mean-center or L2-normalize thinking activations before SAE projection; feed raw activations into the SAE encoder.',
    )
    parser.add_argument('--n_tasks', type=int, default=500,
                      help='Number of tasks to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=5000,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--max_thinking_tokens', type=int, default=5000,
                      help='Maximum number of tokens for the thinking model only')
    parser.add_argument(
        '--only-finished-thinking',
        action='store_true',
        default=False,
        help='If set, skip base+hybrid generation when the thinking model does not end with EOS; still record the attempt and exclude such examples from printed stats.',
    )
    parser.add_argument('--eval_start_idx', type=int, default=0,
                      help='Starting index in the dataset')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Temperature for sampling in standalone thinking/base generation (not used by token-level hybrid generation).')
    parser.add_argument(
        '--steered-temperature',
        type=float,
        default=0.0,
        help='Temperature for sampling ONLY on tokens where hybrid steering is applied (default: 0.0 = greedy). Tokens without steering remain greedy.',
    )
    parser.add_argument('--coefficients', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='List of steering coefficients to evaluate per-token under the guardrail')
    parser.add_argument('--token_windows', type=int, nargs='+', default=[1],
                        help='List of token windows (negative = last N tokens) to apply steering to; 0 or [1] means all tokens; e.g., -1 applies only to the last token')
    parser.add_argument('--n_cold_start_tokens', type=int, default=0,
                      help='Number of initial tokens from the thinking model to prepend as a cold-start prefix')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument(
        '--results-suffix',
        type=str,
        default='',
        help='Optional custom suffix appended to all result filenames/prefixes (before extension). Example: --results-suffix exp1',
    )
    parser.add_argument('--example_idx', type=int, default=0,
                      help='Index of example to run')
    parser.add_argument('--use_perplexity_guardrail', action='store_true', default=True,
                      help='If set, select among steered candidates based on thinking-model perplexity')
    parser.add_argument('--run_example', action='store_true', default=False,
                      help='Run a single example before evaluation')
    parser.add_argument('--show_progress', action='store_true', default=True,
                      help='Show tqdm progress during hybrid token generation')
    parser.add_argument('--disable_disagreement_only', action='store_true', default=False,
                      help='Disable optimization that only steers when base vs steered disagree')
    parser.add_argument('--store_per_token_details', action='store_true', default=True,
                      help='Keep per-token arrays in RAM during eval (uses more memory)')
    parser.add_argument('--only-bias', action='store_true', default=False,
                      help='If set, use only the bias vector for steering (no other latents)')
    parser.add_argument('--random-firing', action='store_true', default=False,
                      help='If set, randomly select which latent to fire each token (bypass oracle)')
    parser.add_argument('--random-vectors', action='store_true', default=False,
                      help='If set, steer using random unit vectors (correct shape), ignoring trained vectors')
    parser.add_argument('--top_k', type=int, default=1,
                      help='Select top-k SAE latents (k=1 keeps original behaviour)')
    parser.add_argument('--top_k_weighting', type=str, default='softmax',
                      help='Weighting method to combine top-k latents. Options: "softmax", "normalized", "uniform"')
    parser.add_argument(
        '--disable-steering-in-code-blocks',
        action='store_true',
        default=False,
        help='If set, hybrid token steering is disabled inside fenced python code blocks: after generating "```python" steering is off until the next "```".',
    )
    args, unknown = parser.parse_known_args()
    assert not unknown, f"Unknown arguments: {unknown}"
    # Special handling: if [1] is provided, treat as "all tokens"
    if isinstance(args.token_windows, list) and len(args.token_windows) == 1 and int(args.token_windows[0]) == 1:
        args.token_windows = [0]
    return args

def _result_suffix(args):
    s = ""
    if getattr(args, "only_bias", False):
        s += "_bias-only"
    if getattr(args, "random_firing", False):
        s += "_random-firing"
    if getattr(args, "random_vectors", False):
        s += "_random-vectors"
    custom = getattr(args, "results_suffix", "")
    if custom:
        assert isinstance(custom, str)
        assert custom == custom.strip(), "--results-suffix must not have leading/trailing whitespace"
        assert re.match(r"^[A-Za-z0-9._-]+$", custom), "--results-suffix must match ^[A-Za-z0-9._-]+$"
        if not custom.startswith("_"):
            s += "_" + custom
        else:
            s += custom
    return s

def _is_ablation(args):
    return bool(getattr(args, "only_bias", False) or getattr(args, "random_firing", False) or getattr(args, "random_vectors", False))

def _ablation_flags_str(args):
    return (
        f"only-bias={bool(getattr(args, 'only_bias', False))}, "
        f"random-firing={bool(getattr(args, 'random_firing', False))}, "
        f"random-vectors={bool(getattr(args, 'random_vectors', False))}"
    )

def get_next_token(logits, temperature, model, input_ids=None):
    """Get next token from logits using temperature sampling or greedy decoding (repetition penalty removed)"""
    if isinstance(logits, torch.Tensor):
        logits = logits.to(dtype=torch.float32)
    if temperature > 0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        del logits, probs
        return token
    else:
        token = torch.argmax(logits).item()
        del logits
        return token

def get_token_and_string(logits, temperature, tokenizer, input_ids=None):
    """Get token ID and string from logits"""
    token = get_next_token(logits, temperature, tokenizer, input_ids)
    token_string = tokenizer.decode(token)
    return token, token_string

def get_perplexity(token_string, logits, model):
    """Calculate perplexity of a token string under the given logits.

    Accepts either full logits [batch, seq, vocab], [seq, vocab], or last-step [vocab].
    Works on CPU tensors to reduce GPU memory pressure.
    """
    token_id = model.tokenizer.encode(token_string, return_tensors="pt", add_special_tokens=False).to(torch.long)
    # Normalize to last step logits vector
    if isinstance(logits, torch.Tensor):
        if logits.dim() == 1:
            last_logits = logits
        elif logits.dim() == 2:
            last_logits = logits[-1]
        elif logits.dim() == 3:
            last_logits = logits[0, -1]
        else:
            del token_id
            return float('inf')
    else:
        del token_id
        return float('inf')

    if token_id.shape[1] == 0:
        del token_id
        return float('inf')
    idx = int(token_id[0, 0].item())
    if idx < 0 or idx >= last_logits.shape[-1]:
        del token_id
        return float('inf')
    log_prob = torch.log_softmax(last_logits, dim=-1)[idx].item()
    perplexity = math.exp(-log_prob)
    del token_id
    return perplexity

# NEW: Helper to identify sentence boundaries (unused but handy for analysis)
def is_sentence_end(token_str: str) -> bool:
    stripped = token_str.strip()
    return bool(re.search(r'[.!?]$', stripped)) or stripped == "\n"

# NEW: Prepare cold-start inputs using the already generated thinking tokens
def prepare_cold_start(
    thinking_outputs: torch.Tensor,
    thinking_input_ids: torch.Tensor,
    base_input_ids: torch.Tensor,
    *,
    thinking_tokenizer,
    base_tokenizer,
    n_cold_start_tokens: int,
):
    if n_cold_start_tokens <= 0:
        return base_input_ids, thinking_input_ids, ""

    gen_slice = thinking_outputs[
        :,
        thinking_input_ids.shape[1] : thinking_input_ids.shape[1] + n_cold_start_tokens,
    ]

    cold_start_text: str = thinking_tokenizer.decode(gen_slice[0], skip_special_tokens=True)

    thinking_with_cold = torch.cat([thinking_input_ids, gen_slice.to(torch.long)], dim=1)

    base_cold_ids = (
        base_tokenizer.encode(cold_start_text, return_tensors="pt")
        .to(base_input_ids.device)
        .to(torch.long)
    )
    base_with_cold = torch.cat([base_input_ids, base_cold_ids], dim=1)

    del gen_slice, base_cold_ids
    torch.cuda.empty_cache()

    return base_with_cold, thinking_with_cold, cold_start_text

def _update_code_fence_state(
    text: str,
    *,
    in_code_fence: bool,
    search_pos: int,
    start_marker: str = "```python",
    end_marker: str = "```",
) -> Tuple[bool, int]:
    """Update (in_code_fence, search_pos) by scanning `text` from `search_pos`."""
    assert isinstance(text, str)
    assert isinstance(in_code_fence, bool)
    assert isinstance(search_pos, int) and search_pos >= 0
    assert isinstance(start_marker, str) and start_marker
    assert isinstance(end_marker, str) and end_marker

    while True:
        if not in_code_fence:
            idx = text.find(start_marker, search_pos)
            if idx == -1:
                break
            in_code_fence = True
            search_pos = idx + len(start_marker)
            continue

        idx = text.find(end_marker, search_pos)
        if idx == -1:
            break
        in_code_fence = False
        search_pos = idx + len(end_marker)

    return in_code_fence, search_pos

# ---------------------------------------------------------------------------------
# Token-level hybrid generation
# ---------------------------------------------------------------------------------

def hybrid_generate_token(
    thinking_model,
    base_model,
    base_tokenizer,
    thinking_input_ids,
    base_input_ids,
    max_new_tokens,
    steering_layer,
    sae_layer,
    sae,
    steering_vectors,
    latent_descriptions,
    *,
    coefficient: float = 1.0,
    steered_temperature: float = 0.0,
    disable_sae_mean: bool = False,
    disable_steering_in_code_blocks: bool = False,
    initial_generated_text: str = "",
    verbose: bool = False,
    use_perplexity_guardrail: bool = False,
    coefficients: Optional[List[float]] = None,
    token_windows: Optional[List[int]] = None,
    show_progress: bool = False,
    disagreement_only: bool = True,
    collect_details: bool = True,
    only_bias: bool = False,
    random_firing: bool = False,
    random_vectors: bool = False,
    top_k: int = 1,
    top_k_weighting: str = 'softmax',
):
    """Per-token variant of hybrid generation.

    For each output token:
      1) Use thinking model's last-token hidden state at `sae_layer` to choose the dominant latent and its steering vector.
      2) For each (coefficient, token_window) candidate, compute steered logits on base model.
      3) Select the next token by minimum perplexity under the thinking model (guardrail). If guardrail disabled, use the first provided candidate.
    """
    assert float(steered_temperature) >= 0.0, "steered_temperature must be >= 0"
    assert isinstance(disable_sae_mean, bool)
    assert isinstance(disable_steering_in_code_blocks, bool)
    assert isinstance(initial_generated_text, str)

    # Normalize special-case: [1] means all tokens
    if token_windows is not None and isinstance(token_windows, list) and len(token_windows) == 1:
        try:
            if int(token_windows[0]) == 1:
                token_windows = [0]
        except Exception:
            pass

    # Clone inputs so we do not modify the originals in-place
    base_output_ids = base_input_ids.clone()
    thinking_output_ids = thinking_input_ids.clone()
    del base_input_ids, thinking_input_ids
    torch.cuda.empty_cache()

    token_latent_info = [] if collect_details else None
    per_token_perplexity = [] if collect_details else None
    token_position = [] if collect_details else None
    steering_selection = []

    generated_tokens = 0
    ended_by_eos = False
    generated_text = initial_generated_text
    in_code_fence = False
    code_fence_search_pos = 0
    if disable_steering_in_code_blocks and generated_text:
        in_code_fence, code_fence_search_pos = _update_code_fence_state(
            generated_text,
            in_code_fence=in_code_fence,
            search_pos=code_fence_search_pos,
        )
        if in_code_fence:
            print("[CodeFence] Detected ```python in prefix; steering disabled until closing ```.")

    # Access bias vector if present
    bias_vector = steering_vectors.get("bias", None)
    if only_bias:
        assert bias_vector is not None, "--only-bias requires an available 'bias' steering vector"

    # Precompute latent key mappings and available keys for random-firing
    key_to_idx = {desc["key"]: idx for idx, desc in latent_descriptions.items()}
    available_steer_keys = [k for k in steering_vectors.keys() if k != "bias" and k in key_to_idx]
    if random_firing:
        assert len(available_steer_keys) > 0, "random-firing requires at least one available steering latent key"

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=max_new_tokens, desc="Hybrid tokens", leave=False)

    while generated_tokens < max_new_tokens:
        # 1) THINKING MODEL — derive steering vector from current position (skip activation trace if random_firing)
        latent_acts = None
        if not random_firing:
            with torch.inference_mode():
                with thinking_model.trace(thinking_output_ids) as tracer:
                    activation_curr = thinking_model.model.layers[sae_layer].output[0, -1, :].save()
            activation_curr = activation_curr.detach().clone()
            if disable_sae_mean:
                x_in = activation_curr.to(dtype=torch.float32)
                assert x_in.ndim == 1
                assert x_in.shape == sae.b_dec.shape, f"Bad shapes: x_in.shape={tuple(x_in.shape)} b_dec.shape={tuple(sae.b_dec.shape)}"
                latent_acts = sae.encoder(x_in - sae.b_dec)
                del x_in
            else:
                x_norm = center_and_l2_normalize_torch(activation_curr, sae.activation_mean)
                latent_acts = sae.encoder(x_norm - sae.b_dec)
                del x_norm
            del activation_curr
            torch.cuda.empty_cache()

        if random_firing:
            print("Ablation flag active: random-firing=True (sampling latent uniformly)")
            latent_key = available_steer_keys[int(torch.randint(low=0, high=len(available_steer_keys), size=(1,)).item())]
            assert latent_key in key_to_idx, f"Selected latent key {latent_key} not in descriptions"
            latent_id = int(key_to_idx[latent_key])
            activation_value = 0.0
            selected_latent_ids = [latent_id]
            selected_weights = [1.0]
            # steering_vector will be picked below from steering_vectors
        else:
            assert latent_acts is not None
            # Determine top-k indices & weights
            k = max(1, int(top_k))
            k = min(k, int(latent_acts.numel()))
            values, indices = torch.topk(latent_acts, k=k)
            # Compute weights according to chosen scheme
            if top_k_weighting == 'softmax':
                weights = torch.softmax(values, dim=0)
            elif top_k_weighting == 'normalized':
                vals = torch.clamp(values, min=0.0)
                s = vals.sum()
                if s > 0:
                    weights = vals / s
                else:
                    weights = torch.ones_like(vals) / vals.numel()
            else:  # uniform
                weights = torch.ones_like(values) / values.numel()

            selected_latent_ids = [int(i.item()) for i in indices]
            selected_weights = [float(w.item()) for w in weights]

            # Primary latent (for legacy fields)
            latent_id = int(indices[0].item())
            activation_value = float(values[0].item())
            latent_key = latent_descriptions[latent_id]["key"]
        latent_title = latent_descriptions[latent_id]["title"]
        # Access steering vector for selected latent(s) and combine if top_k>1
        steering_vector = None
        try:
            base_device = base_model.device
        except Exception:
            base_device = torch.device('cpu')

        # If using only bias, steering handled later via bias_vec
        if not only_bias and (not random_vectors):
            # If we have multiple selected latents, sum weighted vectors; otherwise pick single
            collected = None
            for lid, w in zip(selected_latent_ids, selected_weights):
                key = latent_descriptions[lid].get('key')
                if key is None:
                    continue
                vec = steering_vectors.get(key)
                if vec is None:
                    continue
                # Ensure vector on base device and correct dtype
                try:
                    vec = vec.to(device=base_device)
                except Exception:
                    vec = vec
                if collected is None:
                    collected = (w * vec).clone()
                else:
                    collected = collected + (w * vec)
            if collected is not None:
                steering_vector = collected
            else:
                # Fallback: try single top key
                steering_vector = steering_vectors.get(latent_key)
        elif random_vectors and (not only_bias):
            # random_vectors handled in load_models_and_sae by replacing steering_vectors
            steering_vector = steering_vectors.get(latent_key)
        del latent_acts
        torch.cuda.empty_cache()

        if verbose and (generated_tokens % 20 == 0):
            print(f"Token {generated_tokens}: latent={latent_title} (value={activation_value:.3f})")

        # 2) BASE MODEL — build candidate tokens across (coef, window)
        candidate_tokens = []
        # Precompute shapes and vectors outside tracing context to avoid Proxy conditionals
        hidden_size_expected = int(getattr(base_model.config, "hidden_size", 0))
        bias_vec = (
            bias_vector
            if (
                bias_vector is not None
                and hasattr(bias_vector, "shape")
                and bias_vector.shape[-1] == hidden_size_expected
            )
            else None
        )
        if only_bias:
            assert bias_vec is not None, "Bias vector missing or wrong hidden size for base model"
        steer_vec = (
            None if only_bias else (
                steering_vector
                if (
                    isinstance(steering_vector, torch.Tensor)
                    and steering_vector.shape[-1] == hidden_size_expected
                )
                else None
            )
        )
        # First compute the unsteered base token (save only last-step logits, then move to CPU)
        with torch.inference_mode():
            with base_model.trace(base_output_ids) as tracer:
                _last_logits_unsteered = base_model.lm_head.output[0, -1].save()
        _last_logits_unsteered = _last_logits_unsteered.detach().to("cpu")
        # Greedy token for disagreement gating (deterministic)
        base_pred_tok = int(torch.argmax(_last_logits_unsteered).item())
        # Actual next-token candidate according to chosen temperature
        tok_unsteered, tok_unsteered_str = get_token_and_string(
            _last_logits_unsteered,
            0.0,  # Always greedy when not steering
            base_tokenizer,
            base_output_ids,
        )
        del _last_logits_unsteered

        # Compute thinking model logits at current position (for gating and guardrail)
        with torch.inference_mode():
            with thinking_model.trace(thinking_output_ids) as tracer:
                last_logits_thinking = thinking_model.lm_head.output[0, -1].save()
        last_logits_thinking = last_logits_thinking.detach().to("cpu")
        thinking_pred_tok = int(torch.argmax(last_logits_thinking).item())

        # Decide whether to perform full steering based on base vs thinking disagreement
        perform_steering = True
        if disagreement_only and thinking_pred_tok == base_pred_tok:
            perform_steering = False
        if disable_steering_in_code_blocks and in_code_fence:
            perform_steering = False

        if not perform_steering:
            # Choose unsteered token and compute its perplexity for logging
            p = get_perplexity(tok_unsteered_str, last_logits_thinking, thinking_model)
            next_tok = tok_unsteered
            next_tok_str = tok_unsteered_str
            token_perpl = p
            chosen = "unsteered"
            chosen_coef = None
            chosen_window = None
            del candidate_tokens
            del last_logits_thinking
        else:
            # Build candidate tokens; compute initial steered candidate after gating
            if use_perplexity_guardrail:
                coef_list = coefficients if (coefficients is not None and len(coefficients) > 0) else [coefficient]
                win_list = token_windows if (token_windows is not None and len(token_windows) > 0) else [-1]
                c0 = float(coef_list[0])
                w0 = int(win_list[0])
            else:
                c0 = float((coefficients[0] if (coefficients is not None and len(coefficients) > 0) else coefficient))
                w0 = int((token_windows[0] if (token_windows is not None and len(token_windows) > 0) else -1))
            window_size0 = abs(int(w0)) if int(w0) != 0 else 0
            with torch.inference_mode():
                with base_model.trace(base_output_ids) as tracer:
                    full_out0 = base_model.model.layers[steering_layer].output.save()
                    assert full_out0.dim() == 3
                    assert full_out0.shape[0] >= 1
                    assert full_out0.shape[-1] == hidden_size_expected
                    new_full0 = full_out0.clone()
                    if window_size0 > 0:
                        if bias_vec is not None:
                            new_full0[0, -window_size0:, :] += c0 * bias_vec
                        if steer_vec is not None:
                            new_full0[0, -window_size0:, :] += c0 * steer_vec
                    else:
                        if bias_vec is not None:
                            new_full0[0, :, :] += c0 * bias_vec
                        if steer_vec is not None:
                            new_full0[0, :, :] += c0 * steer_vec
                    base_model.model.layers[steering_layer].output = new_full0
                    _last_logits_steered0 = base_model.lm_head.output[0, -1].save()
            # Initial steered candidate
            _last_logits_steered0 = _last_logits_steered0.detach().to("cpu")
            tok_steered0, tok_steered0_str = get_token_and_string(
                _last_logits_steered0,
                steered_temperature,
                base_tokenizer,
                base_output_ids,
            )
            del _last_logits_steered0

            # Seed candidate list
            candidate_tokens.append({
                "type": "steered",
                "coef": float(c0),
                "window": int(w0),
                "tok": tok_steered0,
                "tok_str": tok_steered0_str,
            })
            if use_perplexity_guardrail:
                for coef in coef_list:
                    for win in win_list:
                        if float(coef) == c0 and int(win) == w0:
                            continue
                        window_size = abs(int(win)) if int(win) != 0 else 0
                        with torch.inference_mode():
                            with base_model.trace(base_output_ids) as tracer:
                                full_out_c = base_model.model.layers[steering_layer].output.save()
                                assert full_out_c.dim() == 3
                                assert full_out_c.shape[0] >= 1
                                assert full_out_c.shape[-1] == hidden_size_expected
                                new_full_c = full_out_c.clone()
                                if window_size > 0:
                                    if bias_vec is not None:
                                        new_full_c[0, -window_size:, :] += float(coef) * bias_vec
                                    if steer_vec is not None:
                                        new_full_c[0, -window_size:, :] += float(coef) * steer_vec
                                else:
                                    if bias_vec is not None:
                                        new_full_c[0, :, :] += float(coef) * bias_vec
                                    if steer_vec is not None:
                                        new_full_c[0, :, :] += float(coef) * steer_vec
                                base_model.model.layers[steering_layer].output = new_full_c
                                _last_logits_steered_c = base_model.lm_head.output[0, -1].save()
                        _last_logits_steered_c = _last_logits_steered_c.detach().to("cpu")
                        tok_c, tok_c_str = get_token_and_string(
                            _last_logits_steered_c,
                            steered_temperature,
                            base_tokenizer,
                            base_output_ids,
                        )
                        del _last_logits_steered_c
                        candidate_tokens.append({
                            "type": "steered",
                            "coef": float(coef),
                            "window": int(win),
                            "tok": tok_c,
                            "tok_str": tok_c_str,
                        })
            else:
                # No guardrail; use the initial steered candidate directly
                p = get_perplexity(tok_steered0_str, last_logits_thinking, thinking_model)
                next_tok = tok_steered0
                next_tok_str = tok_steered0_str
                token_perpl = p
                chosen = "steered"
                chosen_coef = float(c0)
                chosen_window = int(w0)
                del candidate_tokens
                del last_logits_thinking

        # 3) Evaluate perplexity of each candidate under thinking model and pick best (if steering performed and guardrail enabled)
        if perform_steering and use_perplexity_guardrail:
            best = None  # tuple(perplexity, index)
            for idx, cand in enumerate(candidate_tokens):
                p = get_perplexity(cand["tok_str"], last_logits_thinking, thinking_model)
                cand["perplexity"] = p
                if best is None or p < best[0]:
                    best = (p, idx)
            assert best is not None
            chosen_cand = candidate_tokens[best[1]]
            next_tok = chosen_cand["tok"]
            next_tok_str = chosen_cand["tok_str"]
            token_perpl = chosen_cand["perplexity"]
            chosen = chosen_cand["type"]
            chosen_coef = chosen_cand.get("coef", None)
            chosen_window = chosen_cand.get("window", None)
            del candidate_tokens
            del last_logits_thinking
            torch.cuda.empty_cache()


        # 4) Append chosen token to both sequences
        base_tok_ids = base_tokenizer.encode(
            next_tok_str, return_tensors="pt", add_special_tokens=False
        ).to(base_model.device).to(torch.long)
        thinking_tok_ids = thinking_model.tokenizer.encode(
            next_tok_str, return_tensors="pt", add_special_tokens=False
        ).to(thinking_model.device).to(torch.long)

        base_output_ids = torch.cat([base_output_ids, base_tok_ids], dim=1)
        thinking_output_ids = torch.cat([thinking_output_ids, thinking_tok_ids], dim=1)
        del base_tok_ids, thinking_tok_ids
        torch.cuda.empty_cache()

        # 5) Book-keeping
        steering_selection.append(chosen)
        if collect_details:
            logged_latent_title = ("Random Vector" if (chosen == "steered" and random_vectors) else latent_title)
            per_token_perplexity.append(token_perpl)
            token_position.append(len(token_latent_info))
            token_latent_info.append(
                {
                    "token": next_tok_str,
                    "latent_id": latent_id if chosen == "steered" else None,
                    "latent_ids": (selected_latent_ids if chosen == "steered" else None),
                    "latent_weights": (selected_weights if chosen == "steered" else None),
                    "latent_title": logged_latent_title if chosen == "steered" else "No Steering",
                    "latent_key": latent_key if chosen == "steered" else None,
                    "activation_value": activation_value if chosen == "steered" else 0.0,
                    "perplexity": token_perpl,
                    "coefficient": chosen_coef if chosen == "steered" else None,
                    "window": chosen_window if chosen == "steered" else None,
                    "selection": chosen,
                    "future_token": None,
                }
            )

        generated_tokens += 1

        if pbar is not None:
            pbar.update(1)

        if disable_steering_in_code_blocks:
            prev_in_code_fence = in_code_fence
            generated_text += next_tok_str
            in_code_fence, code_fence_search_pos = _update_code_fence_state(
                generated_text,
                in_code_fence=in_code_fence,
                search_pos=code_fence_search_pos,
            )
            if (not prev_in_code_fence) and in_code_fence:
                print("[CodeFence] Detected ```python; steering disabled until closing ```.")
            elif prev_in_code_fence and (not in_code_fence):
                print("[CodeFence] Detected closing ```; steering re-enabled.")

        # Periodic cleanup to reduce fragmentation
        if (generated_tokens % 8) == 0:
            gc.collect()
            torch.cuda.empty_cache()
        # Drop temporary references
        try:
            del bias_vec
            del steer_vec
        except Exception:
            pass

        # Stop if EOS
        if next_tok == base_tokenizer.eos_token_id:
            ended_by_eos = True
            break

    # Final cleanup
    if pbar is not None:
        pbar.close()
    try:
        del steering_vector
    except Exception:
        pass
    try:
        del bias_vector
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    return (
        base_output_ids,
        (token_latent_info if collect_details else []),
        (per_token_perplexity if collect_details else []),
        (token_position if collect_details else []),
        steering_selection,
        ended_by_eos,
    )

def load_steering_vectors(model_id, thinking_model_id, sae_layer, n_clusters):
    """Load steering vectors for the specific base model from train_vectors output.

    This filters vector files by the provided base-model `model_id` (e.g., "qwen2.5-32b"),
    avoiding collisions with vectors trained for other architectures (e.g., Llama),
    which can cause hidden-size mismatches.
    """
    hyperparams_dir_abs = os.path.join(os.path.dirname(__file__), "../train-vectors/results/vars/hyperparams")
    vectors_dir_abs = os.path.join(os.path.dirname(__file__), "../train-vectors/results/vars/optimized_vectors")

    if model_id == "qwen2.5-32b" and thinking_model_id == "deepseek-r1-distill-qwen-32b":
        model_id = "qwen2.5-32b-on-deepseek-r1-distill-qwen-32b"

    # Build a mapping {category (e.g., "idx7" or "bias"): tensor} only for this model_id
    model_specific_vectors = {}
    try:
        for fname in os.listdir(hyperparams_dir_abs):
            if not fname.startswith(f"steering_vector_hyperparams_{model_id}_"):
                continue
            hp_path = os.path.join(hyperparams_dir_abs, fname)
            try:
                with open(hp_path, "r") as f:
                    hp = json.load(f)
                category = hp.get("category")  # "idxN" or "bias"
                if not category:
                    continue
                # Derive the matching vector file name
                idx_stub = fname.split(f"steering_vector_hyperparams_{model_id}_", 1)[1].rsplit(".json", 1)[0]
                vec_path = os.path.join(vectors_dir_abs, f"{model_id}_{idx_stub}.pt")
                if not os.path.exists(vec_path):
                    continue
                vec_obj = torch.load(vec_path, map_location="cpu")
                if isinstance(vec_obj, dict):
                    vector_tensor = vec_obj.get(category)
                    if vector_tensor is None and len(vec_obj) == 1:
                        # Fallback for older files: single-entry dict
                        vector_tensor = next(iter(vec_obj.values()))
                else:
                    vector_tensor = vec_obj
                if vector_tensor is None:
                    continue
                model_specific_vectors[category] = vector_tensor
            except Exception:
                continue
    except FileNotFoundError:
        pass

    # Map SAE latent keys ("idxN") to the corresponding vectors
    descriptions = get_latent_descriptions(thinking_model_id, sae_layer, n_clusters)
    steering_vectors = {}
    for _, desc in descriptions.items():
        latent_key = desc.get("key", "")  # e.g., "idx7"
        if not latent_key:
            continue
        key = latent_key.lower().replace(" ", "-")
        if key in model_specific_vectors:
            steering_vectors[key] = model_specific_vectors[key]

    # Include general bias vector (if present for this model)
    if "bias" in model_specific_vectors and "bias" not in steering_vectors:
        steering_vectors["bias"] = model_specific_vectors["bias"]
    else:
        print("No bias vector found")

    print(steering_vectors.keys())

    return steering_vectors

def generate_latent_colors(latent_descriptions):
    colors = {}
    unique_latents = set([desc["title"] for desc in latent_descriptions.values()])
    num_colors = len(unique_latents)
    for i, latent_title in enumerate(unique_latents):
        hue = i / num_colors
        saturation = 0.7
        value = 0.9
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        colors[latent_title] = hex_color
    colors["Cold Start"] = "#808080"
    colors["Fallback"] = "#CCCCCC"
    colors["No Steering"] = "#111111"
    colors["Random Vector"] = "#ff00ff"
    return colors

def visualize_generation_results(token_latent_info, steering_selection, per_token_perplexity, token_position, latent_colors):
    tokens = [info["token"] for info in token_latent_info]
    fig = plt.figure(figsize=(14, 10))
    plt.subplot(4, 1, 1)
    plt.axis('off')
    for i, info in enumerate(token_latent_info):
        token = info["token"].replace('$', '\\$')
        latent_title = info["latent_title"]
        color = latent_colors[latent_title]
        plt.text(i, 0, token, color=color, fontsize=10, ha='center')
    plt.xlim(-1, len(tokens))
    plt.title("Generated Text (Colored by Latent)")
    plt.subplot(4, 1, 2)
    plt.plot(token_position, per_token_perplexity, marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.yscale('log')
    plt.title("Perplexity by Token Position")
    plt.subplot(4, 1, 3)
    choices = []
    for choice in steering_selection:
        if choice == "steered":
            choices.append(1)
        elif choice == "unsteered":
            choices.append(0)
        else:
            choices.append(-1)
    plt.imshow([choices], cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    escaped_tokens = [t.replace('$', '\\$') for t in tokens]
    plt.xticks(range(len(tokens)), escaped_tokens, rotation=90, fontsize=8)
    plt.yticks([])
    cbar = plt.colorbar(orientation="horizontal", pad=0.1, ticks=[-1, 0, 1])
    cbar.set_ticklabels(["None", "No Steering", "Steering Used"])
    plt.title("Steering Selection")
    plt.subplot(4, 1, 4)
    latent_titles = [info.get("latent_title", "None") for info in token_latent_info]
    unique_latents = sorted(set(latent_titles))
    latent_map = {latent: i for i, latent in enumerate(unique_latents)}
    latent_values = [latent_map[latent] for latent in latent_titles]
    plt.imshow([latent_values], cmap="tab20", aspect="auto")
    plt.xticks(range(len(tokens)), escaped_tokens, rotation=90, fontsize=8)
    plt.yticks([])
    handles = []
    for latent in unique_latents:
        color = latent_colors.get(latent, "#000000")
        patch = Rectangle((0, 0), 1, 1, fc=color)
        handles.append(patch)
    plt.legend(handles, unique_latents, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(5, len(unique_latents)), frameon=False)
    plt.title("Latent Features")
    plt.tight_layout()
    plt.show()

def load_models_and_sae(args):
    thinking_model_id = args.thinking_model.split('/')[-1].lower()
    base_model_id = args.base_model.split('/')[-1].lower()
    print(f"Loading models {args.thinking_model} and {args.base_model}...")
    thinking_model, thinking_tokenizer = load_model(model_name=getattr(args, 'thinking_model', args.base_model if hasattr(args, 'base_model') else None))
    thinking_model.tokenizer = thinking_tokenizer
    if getattr(args, 'temperature', 0.0) > 0:
        try:
            thinking_model.generation_config.do_sample = True
        except Exception:
            pass
    base_model, base_tokenizer = load_model(model_name=getattr(args, 'base_model', None))
    if getattr(args, 'temperature', 0.0) > 0:
        try:
            base_model.generation_config.do_sample = True
        except Exception:
            pass
    print(f"Loading SAE for model {thinking_model_id}, layer {args.sae_layer}...")
    # Debug: print whether the disable_sae_mean flag is present at runtime
    try:
        dbg_disable = bool(getattr(args, "disable_sae_mean", False))
    except Exception:
        dbg_disable = 'UNAVAILABLE'
    print(f"DEBUG: args.disable_sae_mean = {dbg_disable}")
    sae, _ = load_sae(
        thinking_model_id,
        args.sae_layer,
        args.n_clusters,
        require_activation_mean=(not bool(getattr(args, "disable_sae_mean", False))),
    )
    sae = sae.to(thinking_model.device)
    print(f"Loading steering vectors and layer effects...")
    descriptions = get_latent_descriptions(thinking_model_id, args.sae_layer, args.n_clusters)
    steering_vectors = load_steering_vectors(base_model_id, thinking_model_id, args.sae_layer, args.n_clusters)
    # If random-vectors ablation is active, replace latent vectors with fixed random ones per latent key
    if bool(getattr(args, "random_vectors", False)):
        print("Ablation flag active: random-vectors=True (fixed random vector per latent)")
        hidden_size_expected = int(getattr(base_model.config, "hidden_size", 0))
        assert hidden_size_expected > 0, "Base model hidden_size must be > 0"
        new_vectors = {}
        for _, desc in descriptions.items():
            latent_key = desc.get("key", "")
            if not latent_key:
                continue
            key = latent_key.lower().replace(" ", "-")
            vec = torch.randn(hidden_size_expected, device="cpu", dtype=torch.float32)
            vec = vec / (torch.norm(vec) + 1e-12)
            new_vectors[key] = vec
        # Preserve bias vector if present; otherwise ignore
        if "bias" in steering_vectors:
            new_vectors["bias"] = steering_vectors["bias"]
        steering_vectors = new_vectors
    # Move steering vectors to base model device and dtype
    base_device = base_model.device
    base_dtype = next(base_model.parameters()).dtype if hasattr(base_model, "parameters") else torch.float32
    for k, v in list(steering_vectors.items()):
        if isinstance(v, torch.Tensor):
            steering_vectors[k] = v.to(device=base_device, dtype=base_dtype, non_blocking=True)
    return thinking_model, thinking_tokenizer, base_model, base_tokenizer, sae, steering_vectors, descriptions, thinking_model_id, base_model_id

CODING_TASK_PREFIX = "Task: Write a single Python function for the following problem. Do not include tests or examples in your output."
CODING_BASE_SUFFIX = "Algorithmic steps to solve this problem, followed by the Python function:\n"

def run_example(thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
               sae, steering_vectors, descriptions, args, dataset):
    sample_idx = args.example_idx
    for i, item in enumerate(dataset):
        if i == sample_idx:
            if args.dataset == "gsm8k":
                example = {
                    "question": item["question"],
                    "answer": item["answer"]
                }
            elif args.dataset == "math500":
                example = {
                    "question": item["problem"],
                    "answer": item["answer"]
                }
            elif args.dataset in ("aime24", "aime25"):
                example = {
                    "question": item["problem"],
                    "answer": item["answer"]
                }
            elif args.dataset == "mbpp":
                example = {
                    "question": item["text"],
                    "answer": item["code"],
                    "test_list": item["test_list"]
                }
            elif args.dataset == "livecodebench":
                # Format public test cases as strings
                public_tests = item.get("public_test_cases", [])
                test_list = [f"# Test {i+1}:\n- Input:\n{t['input']}\n- Output:\n{t['output']}" for i, t in enumerate(public_tests)] if public_tests else []
                starter_code = item.get("starter_code", "")
                example = {
                    "question": item["question_content"],
                    "answer": "",  # No reference solution provided
                    "test_list": test_list,
                    "starter_code": starter_code
                }
            elif args.dataset == "medqa":
                # MedQA multiple choice - format options into question
                options = item["options"]
                options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
                example = {
                    "question": f"{item['question']}\n\nOptions:\n{options_str}",
                    "answer": item["answer_idx"],  # Use letter (A/B/C/D) as canonical answer
                    "answer_text": item["answer"],  # Full text for context
                    "options": options
                }
            elif args.dataset == "legalbench":
                # LegalBench - variable format across subsets
                # Get the text content to insert into the prompt
                if "text" in item:
                    text_content = item["text"]
                else:
                    text_content = str(item)
                example = {
                    "question": text_content,  # Raw text (will be inserted into base_prompt template)
                    "answer": str(item.get("answer", "")),
                    "subset": item.get("subset", "unknown"),
                    "base_prompt_template": item.get("base_prompt", None)  # Task-specific prompt template from GitHub
                }
            break

    question = example["question"]
    answer = example["answer"]
    test_list = example.get("test_list", None)
    starter_code = example.get("starter_code", "")

    print("\n===== Example =====")
    print(f"Question: {question}")

    # Build prompts based on dataset type
    if args.dataset == "mbpp":
        test_cases_hint = "\n".join(test_list) if test_list else ""
        tests_section = f"\n\nPublic Tests:\n{test_cases_hint}" if test_cases_hint else ""
        thinking_prompt = f"{CODING_TASK_PREFIX}\n\nProblem: {question}{tests_section}"
        base_prompt = f"{CODING_TASK_PREFIX}\n\nProblem: {question}{tests_section}\n\n{CODING_BASE_SUFFIX}"
    elif args.dataset == "livecodebench":
        test_cases_hint = "\n\n".join(test_list) if test_list else ""
        tests_section = f"\n\nPublic Tests:\n\n{test_cases_hint}" if test_cases_hint else ""
        starter_hint = f"\n\nStarter code:\n```python\n{starter_code}\n```" if starter_code else ""
        thinking_prompt = f"{CODING_TASK_PREFIX}\n\nProblem: {question}{starter_hint}{tests_section}"
        base_prompt = f"{CODING_TASK_PREFIX}\n\nProblem: {question}{starter_hint}{tests_section}\n\n{CODING_BASE_SUFFIX}"
    elif args.dataset == "medqa":
        # Multiple choice medical question - ask for the answer letter
        thinking_prompt = f"{question}\n\nPlease select the correct answer (A, B, C, or D) and explain your reasoning."
        base_prompt = f"Task: Answer the following medical question by selecting the correct option (A, B, C, or D). Explain your reasoning step by step.\n\n{question}\n\nStep by step answer:\n"
    elif args.dataset == "legalbench":
        # Legal reasoning - use task-specific prompt template from GitHub
        base_prompt_template = example.get("base_prompt_template")
        task_prefix = "Task: Answer the following legal question. Explain your reasoning step by step.\n\nContext:\n"
        # Template uses {{text}} placeholder and already ends with answer cue (A: or Label:)
        full_prompt = task_prefix + base_prompt_template.replace("{{text}}", question)
        thinking_prompt = full_prompt
        base_prompt = full_prompt
    else:
        thinking_prompt = question
        base_prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n"

    thinking_input_ids = thinking_tokenizer.apply_chat_template(
        [{"role": "user", "content": thinking_prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(thinking_model.device).to(torch.long)

    base_input_ids = base_tokenizer.encode(
        base_prompt,
        return_tensors="pt"
    ).to(base_model.device).to(torch.long)
    
    # Generate with thinking model
    print("\n===== Generating with Thinking Model =====")
    with thinking_model.generate(thinking_input_ids, max_new_tokens=args.max_thinking_tokens, temperature=args.temperature, pad_token_id=thinking_tokenizer.eos_token_id) as gen:
        thinking_outputs = thinking_model.generator.output.save()
    thinking_response = thinking_tokenizer.decode(thinking_outputs[0][len(thinking_input_ids[0]):], skip_special_tokens=True)
    print(thinking_response)

    try:
        thinking_eos_end = bool(int(thinking_outputs[0, -1].item()) == int(thinking_tokenizer.eos_token_id))
    except Exception:
        thinking_eos_end = False

    if bool(getattr(args, "only_finished_thinking", False)) and (not thinking_eos_end):
        print("[Skip] Thinking model did not end with EOS for this example; skipping base and hybrid generation.")
        return thinking_response, "", "", [], []
    
    # Generate with base model
    print("\n===== Generating with Base Model =====")
    base_input_with_cold_start, _, cold_start_text = prepare_cold_start(
        thinking_outputs,
        thinking_input_ids,
        base_input_ids,
        thinking_tokenizer=thinking_tokenizer,
        base_tokenizer=base_tokenizer,
        n_cold_start_tokens=args.n_cold_start_tokens,
    )
    
    with base_model.generate(base_input_with_cold_start, max_new_tokens=100, temperature=args.temperature, pad_token_id=base_tokenizer.eos_token_id) as gen:
        base_outputs = base_model.generator.output.save()
    base_response = f"{cold_start_text}{base_tokenizer.decode(base_outputs[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
    print(base_response)
    
    # Clean up base model outputs
    del base_outputs
    torch.cuda.empty_cache()
    
    # Generate with hybrid approach
    print("\n===== Generating with Hybrid Approach (Token-Level) =====")
    base_input_with_cold_start, thinking_input_with_cold_start, cold_start_text = prepare_cold_start(
        thinking_outputs,
        thinking_input_ids,
        base_input_ids,
        thinking_tokenizer=thinking_tokenizer,
        base_tokenizer=base_tokenizer,
        n_cold_start_tokens=args.n_cold_start_tokens,
    )
    
    hybrid_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection, _ = hybrid_generate_token(
        thinking_model=thinking_model,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        thinking_input_ids=thinking_input_with_cold_start,
        base_input_ids=base_input_with_cold_start,
        max_new_tokens=args.max_new_tokens,
        steering_layer=args.steering_layer,
        sae_layer=args.sae_layer,
        sae=sae,
        steering_vectors=steering_vectors,
        latent_descriptions=descriptions,
        steered_temperature=float(args.steered_temperature),
        disable_steering_in_code_blocks=bool(getattr(args, "disable_steering_in_code_blocks", False)),
        initial_generated_text=cold_start_text,
        coefficient=(args.coefficients[0] if args.coefficients else 0.3),
        coefficients=args.coefficients,
        token_windows=args.token_windows,
        verbose=False,
        use_perplexity_guardrail=args.use_perplexity_guardrail,
        show_progress=args.show_progress,
        disagreement_only=(not args.disable_disagreement_only),
        collect_details=True,
        only_bias=bool(args.only_bias),
        random_firing=bool(args.random_firing),
        random_vectors=bool(args.random_vectors),
    )
    hybrid_response = f"{cold_start_text}{base_tokenizer.decode(hybrid_output_ids[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
    print(hybrid_response)
    
    # Clean up hybrid outputs
    del hybrid_output_ids
    torch.cuda.empty_cache()
    
    # Print correct answer for reference
    print("\n===== Correct Answer =====")
    print(answer)
    
    # Visualize results
    latent_colors = generate_latent_colors(descriptions)
    visualize_generation_results(token_latent_info, steering_selection, per_token_perplexity, token_position, latent_colors)
    
    # Clean up example-specific variables
    del latent_colors, per_token_perplexity, token_position
    
    return thinking_response, base_response, hybrid_response, token_latent_info, steering_selection

def clean_answer(text):
    return re.sub(r'\s+', ' ', text).strip()

def safe_chat_batch(prompts, model_name: str = "openai/gpt-5.2", max_tokens: int = 2000, **kwargs):
    import asyncio
    import concurrent.futures
    async def _run():
        return await chat_batch(
            prompts,
            model=model_name,
            max_tokens=max_tokens,
            **kwargs,
        )
    try:
        loop = asyncio.get_running_loop()
        def _thread_runner():
            return asyncio.run(_run())
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(_thread_runner)
            return fut.result()
    except RuntimeError:
        return asyncio.run(_run())

def quick_judge_api_test():
    """Run a fast connectivity test against the judge chat API and print status.

    Uses a tiny prompt and small max_tokens to keep cost minimal. This does not
    affect program flow; it only reports readiness early.
    """
    test_prompt = "Reply with YES."
    try:
        responses = safe_chat_batch([test_prompt], model_name="openai/gpt-5.2", max_tokens=5)
        ok = isinstance(responses, (list, tuple)) and len(responses) > 0 and isinstance(responses[0], str)
        if ok:
            print("Judge API test: OK")
        else:
            print("Judge API test: FAILED (no response). Check OPENAI_API_KEY/OPENAI_PROJECT.")
    except Exception as e:
        print(f"Judge API test: FAILED ({e}). Check OPENAI_API_KEY/OPENAI_PROJECT.")

def evaluate_answer(model_answer, correct_answer, question, model_name, dataset_type="math", test_list=None):
    """Delegate to utils.llm_judge（math/mcqa/coding 规则或执行；classification 仍为 LLM）。"""
    from utils.llm_judge import evaluate_answer as _central_eval

    return _central_eval(
        model_answer,
        correct_answer,
        correct_answer,
        question,
        model_name,
        dataset_type=dataset_type,
        test_list=test_list,
    )


ROLLING_MAX_BYTES = 90 * 1024 * 1024  # 100 MB hard cap per rolling part file

def _rolling_prefix(args, base_model_id: str, thinking_model_id: str) -> str:
    """Return the base prefix (without .jsonl or part suffix) for rolling outputs.

    Splitting/part management is performed by helper functions below, keeping callers agnostic.
    """
    os.makedirs(f"{args.results_dir}/rolling", exist_ok=True)
    base_id_for_files = _normalized_base_id_for_filenames(base_model_id, thinking_model_id)
    suffix = _result_suffix(args)
    return f"{args.results_dir}/rolling/rolling_{base_id_for_files}_{args.dataset}{suffix}"

def _normalized_base_id_for_filenames(base_model_id: str, thinking_model_id: str) -> str:
    """Return base-model id to use in filenames.

    Includes the special "-on-<thinking>" suffix when the base is qwen2.5-32b
    and the thinking model is deepseek-r1-distill-qwen-32b or open-reasoner-zero-32b.
    """
    if base_model_id == "qwen2.5-32b" and thinking_model_id in (
        "deepseek-r1-distill-qwen-32b",
        "open-reasoner-zero-32b",
    ):
        return f"{base_model_id}-on-{thinking_model_id}"
    return base_model_id

def _list_rolling_files(prefix: str):
    """Return (legacy_file, part_files_sorted) for a given prefix.

    - legacy_file: prefix + ".jsonl" if it exists, else None
    - part_files_sorted: list of files matching prefix_#.jsonl sorted by # ascending
    """
    directory = os.path.dirname(prefix)
    base = os.path.basename(prefix)
    legacy = os.path.join(directory, base + ".jsonl")
    legacy_file = legacy if os.path.exists(legacy) else None
    part_files = []
    try:
        for fname in os.listdir(directory):
            if not fname.startswith(base + "_") or not fname.endswith(".jsonl"):
                continue
            m = re.match(rf"^{re.escape(base)}_(\\d+)\\.jsonl$", fname)
            if not m:
                continue
            idx = int(m.group(1))
            part_files.append((idx, os.path.join(directory, fname)))
    except FileNotFoundError:
        part_files = []
    part_files_sorted = [p for _, p in sorted(part_files, key=lambda x: x[0])]
    return legacy_file, part_files_sorted

def _next_part_path(prefix: str, existing_parts: list) -> str:
    """Return a new part path with next monotonically increasing index."""
    if not existing_parts:
        next_idx = 0
    else:
        # existing_parts are full paths; extract max index
        directory = os.path.dirname(prefix)
        base = os.path.basename(prefix)
        max_idx = -1
        for p in existing_parts:
            fname = os.path.basename(p)
            m = re.match(rf"^{re.escape(base)}_(\\d+)\\.jsonl$", fname)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        next_idx = max_idx + 1
    return f"{prefix}_{next_idx}.jsonl"

def _migrate_legacy_to_parts(prefix: str, legacy_file: str, parts: list) -> list:
    """Migrate legacy file into the part series when both exist (e.g., after restart).
    
    Always creates a new part file for the legacy content to avoid potential duplicates.
    Returns updated parts list after migration.
    """
    print(f"[Rolling] Detected mixed state (legacy + parts) for {prefix}. Migrating legacy to new part...")
    
    # Check if legacy is empty
    legacy_size = os.path.getsize(legacy_file)
    if legacy_size == 0:
        os.remove(legacy_file)
        print(f"[Rolling] Removed empty legacy file: {legacy_file}")
        return parts
    
    # Always create a new part for legacy to avoid duplicates (legacy might contain data also in latest part)
    new_part = _next_part_path(prefix, parts if parts else [])
    os.rename(legacy_file, new_part)
    parts_updated = parts + [new_part] if parts else [new_part]
    print(f"[Rolling] Migrated legacy -> {new_part} ({legacy_size / (1024*1024):.1f} MB)")
    
    return parts_updated


def _count_completed_tasks(args, base_model_id: str, thinking_model_id: str) -> int:
    """Return number of already completed tasks by counting lines across rolling JSONL parts."""
    prefix = _rolling_prefix(args, base_model_id, thinking_model_id)
    legacy_file, parts = _list_rolling_files(prefix)
    
    # Handle mixed state (legacy + parts) from interrupted runs
    if legacy_file is not None and len(parts) > 0:
        parts = _migrate_legacy_to_parts(prefix, legacy_file, parts)
        legacy_file = None
    
    total = 0
    files = ([] if legacy_file is None else [legacy_file]) + parts
    for path in files:
        try:
            with open(path, "r") as f:
                for _ in f:
                    total += 1
        except FileNotFoundError:
            continue
    return total


def append_rolling_result(record: dict, args, base_model_id: str, thinking_model_id: str):
    """Append a record to the current rolling file, splitting into parts to enforce 100MB cap."""
    assert isinstance(record, dict)
    prefix = _rolling_prefix(args, base_model_id, thinking_model_id)
    legacy_file, parts = _list_rolling_files(prefix)
    
    # Handle mixed state (legacy + parts) from interrupted runs
    if legacy_file is not None and len(parts) > 0:
        parts = _migrate_legacy_to_parts(prefix, legacy_file, parts)
        legacy_file = None

    serialized = json.dumps(record)
    line_bytes = len((serialized + "\n").encode("utf-8"))

    # Choose target file with preference: latest part; else legacy; else create legacy first
    if parts:
        target_path = parts[-1]
        current_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
        if current_size + line_bytes > ROLLING_MAX_BYTES:
            target_path = _next_part_path(prefix, parts)
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(serialized + "\n")
        return

    # No parts exist
    if legacy_file is None:
        # Start with a single legacy file (no _0 suffix)
        legacy_file = prefix + ".jsonl"
        # It will remain single-file unless/ until it exceeds the cap during a future append
        with open(legacy_file, "a", encoding="utf-8") as f:
            f.write(serialized + "\n")
        return

    # We only have a legacy file
    current_size = os.path.getsize(legacy_file) if os.path.exists(legacy_file) else 0
    if current_size + line_bytes <= ROLLING_MAX_BYTES:
        with open(legacy_file, "a", encoding="utf-8") as f:
            f.write(serialized + "\n")
        return

    # Exceeding cap: migrate legacy to parts to avoid mixing
    # Move legacy -> _0.jsonl, then write to _1.jsonl (or next as needed)
    existing_parts = []
    part0 = _next_part_path(prefix, existing_parts)
    os.rename(legacy_file, part0)
    part1 = _next_part_path(prefix, [part0])
    with open(part1, "a", encoding="utf-8") as f:
        f.write(serialized + "\n")


def _load_prev_counts(args, base_model_id: str, thinking_model_id: str):
    """Load existing rolling results across all parts and return
    (n_completed, correct_counts, eos_true_counts, eos_known_counts).

    For EOS, only count entries that explicitly recorded EOS for each model into the known denominator.
    """
    prefix = _rolling_prefix(args, base_model_id, thinking_model_id)
    legacy_file, parts = _list_rolling_files(prefix)
    
    # Handle mixed state (legacy + parts) from interrupted runs
    if legacy_file is not None and len(parts) > 0:
        parts = _migrate_legacy_to_parts(prefix, legacy_file, parts)
        legacy_file = None
    
    files = ([] if legacy_file is None else [legacy_file]) + parts
    if not files:
        return 0, {"thinking": 0, "base": 0, "hybrid": 0}, {"thinking": 0, "base": 0, "hybrid": 0}, {"thinking": 0, "base": 0, "hybrid": 0}
    n = 0
    counts = {"thinking": 0, "base": 0, "hybrid": 0}
    eos_counts = {"thinking": 0, "base": 0, "hybrid": 0}
    eos_known = {"thinking": 0, "base": 0, "hybrid": 0}
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    eos = rec.get("eos", {})
                    if bool(getattr(args, "only_finished_thinking", False)) and (not bool(eos.get("thinking", False))):
                        continue
                    judges = rec["judges"]
                    for k in ("thinking", "base", "hybrid"):
                        c = judges[k]["correct"]
                        assert isinstance(c, bool)
                        if c:
                            counts[k] += 1
                    for k in ("thinking", "base", "hybrid"):
                        if k in eos:
                            eos_known[k] += 1
                            if bool(eos.get(k, False)):
                                eos_counts[k] += 1
                    n += 1
        except FileNotFoundError:
            continue
    return n, counts, eos_counts, eos_known

def analyze_hybrid_stats(token_latent_info, steering_selection):
    steered_count = steering_selection.count("steered")
    unsteered_count = steering_selection.count("unsteered")
    total = steered_count + unsteered_count
    no_steering_fraction = unsteered_count / total if total > 0 else 0
    latent_counts = Counter()
    for info in token_latent_info:
        if info["latent_title"] != "No Steering":
            latent_counts[info["latent_title"]] += 1
    latent_percentages = {}
    if steered_count > 0:
        for latent, count in latent_counts.items():
            latent_percentages[latent] = (count / steered_count) * 100
    steering_stats = {
        "steered_count": steered_count,
        "unsteered_count": unsteered_count,
        "total_tokens": total,
        "steering_fraction": steered_count / total if total > 0 else 0,
        "no_steering_fraction": no_steering_fraction
    }
    return no_steering_fraction, latent_counts, latent_percentages, steering_stats

def save_detailed_results(results, args, thinking_model_id, base_model_id):
    os.makedirs(f"{args.results_dir}/detailed", exist_ok=True)
    base_id_for_files = _normalized_base_id_for_filenames(base_model_id, thinking_model_id)
    suffix = _result_suffix(args)
    filename = f"{args.results_dir}/detailed/hybrid_stats_{base_id_for_files}_{args.dataset}{suffix}.json"
    avg_steering_stats = {
        "steered_count": sum(stat["steered_count"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "unsteered_count": sum(stat["unsteered_count"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "total_tokens": sum(stat["total_tokens"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "steering_fraction": sum(stat["steering_fraction"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "no_steering_fraction": sum(stat["no_steering_fraction"] for stat in results["steering_stats"]) / len(results["steering_stats"])
    }
    detailed_data = {
        "metadata": {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "dataset": args.dataset,
            "temperature": args.temperature,
            "coefficients": args.coefficients,
            "n_tasks": len(results["questions"])
        },
        "answer_lengths": {
            "base_model": results["base_lengths"],
            "thinking_model": results["thinking_lengths"],
            "hybrid_model": results["hybrid_lengths"],
            "avg_base": sum(results["base_lengths"]) / len(results["base_lengths"]) if results["base_lengths"] else 0,
            "avg_thinking": sum(results["thinking_lengths"]) / len(results["thinking_lengths"]) if results["thinking_lengths"] else 0,
            "avg_hybrid": sum(results["hybrid_lengths"]) / len(results["hybrid_lengths"]) if results["hybrid_lengths"] else 0
        },
        "steering_stats": {
            "no_steering_fractions": results["no_steering_fractions"],
            "avg_no_steering": sum(results["no_steering_fractions"]) / len(results["no_steering_fractions"]) if results["no_steering_fractions"] else 0,
            "detailed_stats": results["steering_stats"],
            "average_stats": avg_steering_stats
        },
        "latent_usage": results["latent_usage"],
        "accuracies": {
            "base_model": results["base_correct"] / len(results["questions"]) * 100,
            "thinking_model": results["thinking_correct"] / len(results["questions"]) * 100,
            "hybrid_model": results["hybrid_correct"] / len(results["questions"]) * 100
        }
    }
    with open(filename, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    print(f"Detailed results saved to {filename}")
    return detailed_data

def run_evaluation(thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
                  sae, steering_vectors, descriptions, args, dataset, thinking_model_id, base_model_id,
                  prev_completed: int = 0, prev_counts: Optional[dict] = None):

    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    only_finished_thinking = bool(getattr(args, "only_finished_thinking", False))
    if only_finished_thinking:
        assert not _is_ablation(args), "--only-finished-thinking is incompatible with ablation modes (which skip standalone thinking generation)"

    if _is_ablation(args):
        print(f"Ablation active: skipping standalone base/thinking generations for all tasks. Flags: {_ablation_flags_str(args)}")
        assert int(args.n_cold_start_tokens) == 0, "Ablation runs require --n_cold_start_tokens 0"

    results = {
        "base_correct": 0,
        "thinking_correct": 0,
        "hybrid_correct": 0,
        "base_answers": [],
        "thinking_answers": [],
        "hybrid_answers": [],
        "questions": [],
        "correct_answers": [],
        "base_lengths": [],
        "thinking_lengths": [],
        "hybrid_lengths": [],
        "thinking_eos": [],
        "base_eos": [],
        "hybrid_eos": [],
        "no_steering_fractions": [],
        "latent_usage": [],
        "steering_stats": [],
        "token_latent_info": [],
        "steering_selection": []
    }
    
    task_counter = 0
    included_counter = 0
    skipped_unfinished_thinking = 0
    for i, item in enumerate(dataset):
        if i < args.eval_start_idx:
            continue
        if task_counter >= args.n_tasks:
            break
        task_counter += 1
        print(f"\n===== Processing Task {task_counter}/{args.n_tasks} (dataset idx {i}) =====")
        
        if args.dataset == "gsm8k":
            question = item["question"]
            correct_answer = item["answer"]
            test_list = None
            starter_code = ""
        elif args.dataset in ("aime24", "aime25"):
            question = item["problem"]
            correct_answer = item["answer"]
            test_list = None
            starter_code = ""
        elif args.dataset == "math500":
            question = item["problem"]
            correct_answer = item["answer"]
            test_list = None
            starter_code = ""
        elif args.dataset == "mbpp":
            question = item["text"]
            correct_answer = item["code"]
            test_list = item["test_list"]
            starter_code = ""
        elif args.dataset == "livecodebench":
            question = item["question_content"]
            correct_answer = ""  # No reference solution provided
            # Parse public tests (shown in prompt) - stored as JSON string
            public_tests_raw = item.get("public_test_cases", "[]")
            public_tests = json.loads(public_tests_raw) if public_tests_raw else []
            public_test_list = [f"# Test {i+1}:\n- Input:\n{t['input']}\n- Output:\n{t['output']}" for i, t in enumerate(public_tests)] if public_tests else []
            # NOTE: Private tests disabled for LLM-based judging.
            # LiveCodeBench private tests can be enormous (up to 18M chars / 4.5M tokens
            # for a single test case). 150 problems have first private test > 100k chars.
            # This exceeds GPT-4's 1M token context limit. For proper evaluation,
            # use execution-based judging instead of LLM-based judging.
            # private_tests_raw = item.get("private_test_cases", "")
            # if private_tests_raw:
            #     decompressed = zlib.decompress(base64.b64decode(private_tests_raw.encode("utf-8")))
            #     private_tests = json.loads(pickle.loads(decompressed))
            # else:
            #     private_tests = []
            # num_public = len(public_tests)
            # private_test_list = [f"# Test {num_public + i + 1}:\n- Input:\n{t['input']}\n- Output:\n{t['output']}" for i, t in enumerate(private_tests)] if private_tests else []
            # test_list = public_test_list + private_test_list
            test_list = public_test_list
            starter_code = item.get("starter_code", "")
        elif args.dataset == "medqa":
            # MedQA multiple choice - format options into question
            options = item["options"]
            options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
            question = f"{item['question']}\n\nOptions:\n{options_str}"
            correct_answer = item["answer_idx"]  # Letter (A/B/C/D)
            test_list = None
            starter_code = ""
        elif args.dataset == "legalbench":
            # LegalBench - variable format across subsets
            # Get raw text content (will be inserted into prompt template)
            if "text" in item:
                question = item["text"]
            else:
                question = str(item)
            correct_answer = str(item.get("answer", ""))
            base_prompt_template = item.get("base_prompt", None)  # Task-specific prompt template from GitHub
            test_list = None
            starter_code = ""

        # Build prompts based on dataset type
        if args.dataset == "mbpp":
            # Code generation prompt
            test_cases_hint = "\n".join(test_list) if test_list else ""
            tests_section = f"\n\nPublic Tests:\n{test_cases_hint}" if test_cases_hint else ""
            thinking_prompt = f"{CODING_TASK_PREFIX}\n\nProblem: {question}{tests_section}"
            base_prompt = f"{CODING_TASK_PREFIX}\n\nProblem: {question}{tests_section}\n\n{CODING_BASE_SUFFIX}"
        elif args.dataset == "livecodebench":
            # LiveCodeBench prompt (only public tests in hint, private tests are for judge only)
            test_cases_hint = "\n\n".join(public_test_list) if public_test_list else ""
            tests_section = f"\n\nPublic Tests:\n\n{test_cases_hint}" if test_cases_hint else ""
            starter_hint = f"\n\nStarter code:\n```python\n{starter_code}\n```" if starter_code else ""
            thinking_prompt = f"{CODING_TASK_PREFIX}\n\nProblem: {question}{starter_hint}{tests_section}"
            base_prompt = f"{CODING_TASK_PREFIX}\n\nProblem: {question}{starter_hint}{tests_section}\n\n{CODING_BASE_SUFFIX}"
        elif args.dataset == "medqa":
            # Multiple choice medical question - ask for the answer letter
            thinking_prompt = f"{question}\n\nPlease select the correct answer (A, B, C, or D) and explain your reasoning."
            base_prompt = f"Task: Answer the following medical question by selecting the correct option (A, B, C, or D). Explain your reasoning step by step.\n\n{question}\n\nStep by step answer:\n"
        elif args.dataset == "legalbench":
            # Legal reasoning - use task-specific prompt template from GitHub
            task_prefix = "Task: Answer the following legal question. Explain your reasoning step by step.\n\nContext:\n"
            # Template uses {{text}} placeholder and already ends with answer cue (A: or Label:)
            full_prompt = task_prefix + base_prompt_template.replace("{{text}}", question)
            thinking_prompt = full_prompt
            base_prompt = full_prompt
        else:
            # Math problem prompt (original)
            thinking_prompt = question
            base_prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n"

        thinking_input_ids = thinking_tokenizer.apply_chat_template(
            [{"role": "user", "content": thinking_prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(thinking_model.device).to(torch.long)

        base_input_ids = base_tokenizer.encode(
            base_prompt,
            return_tensors="pt"
        ).to(base_model.device).to(torch.long)
        
        # Thinking model (skip in ablation)
        print("\n" + "-" * 80)
        print("PROMPT (for hybrid/base):")
        print("-" * 80)
        print(base_prompt)
        print("-" * 80)
        if _is_ablation(args):
            print(f"Ablation: skipping thinking model generation ({_ablation_flags_str(args)})")
            thinking_outputs = None
            thinking_response = ""
            thinking_tokens = 0
            thinking_eos_end = False
        else:
            print("Generating with Thinking Model...")
            clear_gpu_memory()
            with thinking_model.generate(thinking_input_ids, max_new_tokens=args.max_thinking_tokens, temperature=args.temperature, pad_token_id=thinking_tokenizer.eos_token_id) as gen:
                thinking_outputs = thinking_model.generator.output.save()
            thinking_tokens = len(thinking_outputs[0]) - len(thinking_input_ids[0])
            thinking_response = thinking_tokenizer.decode(thinking_outputs[0][len(thinking_input_ids[0]):], skip_special_tokens=True)
            # Track EOS termination
            try:
                thinking_eos_end = bool(int(thinking_outputs[0, -1].item()) == int(thinking_tokenizer.eos_token_id))
            except Exception:
                thinking_eos_end = False

        if (not _is_ablation(args)) and only_finished_thinking and (not bool(thinking_eos_end)):
            skipped_unfinished_thinking += 1
            print(
                f"[Skip] Thinking model did not end with EOS (generated {thinking_tokens}/{int(args.max_thinking_tokens)} tokens). "
                "Skipping base and hybrid generation for this input."
            )
            rolling_record = {
                "ts": time.time(),
                "dataset": args.dataset,
                "question": question,
                "gold_answer": correct_answer,
                "test_list": test_list,
                "answers": {
                    "thinking": thinking_response,
                    "base": "",
                    "hybrid": "",
                },
                "judges": {
                    "thinking": {"correct": False, "raw": "SKIPPED_UNFINISHED_THINKING"},
                    "base": {"correct": False, "raw": "SKIPPED_UNFINISHED_THINKING"},
                    "hybrid": {"correct": False, "raw": "SKIPPED_UNFINISHED_THINKING"},
                },
                "eos": {
                    "thinking": False,
                },
                "skipped": {
                    "reason": "thinking_no_eos",
                },
            }
            append_rolling_result(rolling_record, args, base_model_id, thinking_model_id)
            del thinking_input_ids, base_input_ids, thinking_outputs
            del thinking_response
            torch.cuda.empty_cache()
            gc.collect()
            continue

        included_counter += 1
        results["questions"].append(question)
        results["correct_answers"].append(correct_answer)
        results["thinking_answers"].append(thinking_response)
        results["thinking_lengths"].append(len(thinking_response.split()) if thinking_response else 0)
        results["thinking_eos"].append(bool(thinking_eos_end))

        # Base model (skip in ablation)
        if _is_ablation(args):
            print(f"Ablation: skipping base model generation ({_ablation_flags_str(args)})")
            base_response = ""
            base_tokens = 0
            base_eos_end = False
        else:
            print("Generating with Base Model...")
            clear_gpu_memory()
            base_input_with_cold_start, _, cold_start_text = prepare_cold_start(
                thinking_outputs,
                thinking_input_ids,
                base_input_ids,
                thinking_tokenizer=thinking_tokenizer,
                base_tokenizer=base_tokenizer,
                n_cold_start_tokens=args.n_cold_start_tokens,
            )
            with base_model.generate(base_input_with_cold_start, max_new_tokens=args.max_new_tokens, temperature=args.temperature, pad_token_id=base_tokenizer.eos_token_id) as gen:
                base_outputs = base_model.generator.output.save()
            base_tokens = len(base_outputs[0]) - len(base_input_with_cold_start[0])
            # Track EOS termination
            try:
                base_eos_end = bool(int(base_outputs[0, -1].item()) == int(base_tokenizer.eos_token_id))
            except Exception:
                base_eos_end = False
            base_response = f"{cold_start_text}{base_tokenizer.decode(base_outputs[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
            del base_outputs, base_input_with_cold_start
            clear_gpu_memory()
        results["base_answers"].append(base_response)
        results["base_lengths"].append(len(base_response.split()) if base_response else 0)
        results["base_eos"].append(bool(base_eos_end))
        
        # Hybrid token-level
        print("Generating with Hybrid Approach (Token-Level)...")
        clear_gpu_memory()
        base_input_with_cold_start, thinking_input_with_cold_start, cold_start_text = prepare_cold_start(
            thinking_outputs,
            thinking_input_ids,
            base_input_ids,
            thinking_tokenizer=thinking_tokenizer,
            base_tokenizer=base_tokenizer,
            n_cold_start_tokens=args.n_cold_start_tokens,
        )
        hybrid_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection, hybrid_eos_end = hybrid_generate_token(
            thinking_model=thinking_model,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            thinking_input_ids=thinking_input_with_cold_start,
            base_input_ids=base_input_with_cold_start,
            max_new_tokens=args.max_new_tokens,
            steering_layer=args.steering_layer,
            sae_layer=args.sae_layer,
            sae=sae,
            steering_vectors=steering_vectors,
            latent_descriptions=descriptions,
            steered_temperature=float(args.steered_temperature),
            disable_steering_in_code_blocks=bool(getattr(args, "disable_steering_in_code_blocks", False)),
            initial_generated_text=cold_start_text,
            coefficient=(args.coefficients[0] if args.coefficients else 0.3),
            coefficients=args.coefficients,
            token_windows=args.token_windows,
            verbose=False,
            use_perplexity_guardrail=args.use_perplexity_guardrail,
            show_progress=args.show_progress,
            disagreement_only=(not args.disable_disagreement_only),
            collect_details=bool(args.store_per_token_details),
            only_bias=bool(args.only_bias),
            random_firing=bool(args.random_firing),
            random_vectors=bool(args.random_vectors),
        )
        hybrid_tokens = len(hybrid_output_ids[0]) - len(base_input_with_cold_start[0])
        hybrid_response = f"{cold_start_text}{base_tokenizer.decode(hybrid_output_ids[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
        del hybrid_output_ids, base_input_with_cold_start, thinking_input_with_cold_start
        clear_gpu_memory()
        results["hybrid_answers"].append(hybrid_response)
        results["hybrid_lengths"].append(len(hybrid_response.split()))
        results["hybrid_eos"].append(bool(hybrid_eos_end))
        
        # Store token latent info and steering selection (optional to reduce RAM)
        if args.store_per_token_details:
            results["token_latent_info"].append(token_latent_info)
            results["steering_selection"].append(steering_selection)
        
        # Analyze and store steering statistics
        no_steering_fraction, latent_counts, latent_percentages, steering_stats = analyze_hybrid_stats(token_latent_info, steering_selection)
        results["no_steering_fractions"].append(no_steering_fraction)
        results["latent_usage"].append(latent_percentages)
        results["steering_stats"].append(steering_stats)
        
        # Clean and evaluate answers
        clean_thinking_answer = clean_answer(thinking_response)
        clean_base_answer = clean_answer(base_response)
        clean_hybrid_answer = clean_answer(hybrid_response)

        # Determine dataset type for evaluation
        if args.dataset in CODING_DATASETS:
            dataset_type = "coding"
        elif args.dataset in MCQA_DATASETS:
            dataset_type = "mcqa"
        elif args.dataset in TEXT_CLASSIFICATION_DATASETS:
            dataset_type = "classification"
        else:
            dataset_type = "math"

        # Display all responses clearly before evaluation
        print("\n" + "=" * 80)
        print("QUESTION:")
        print("=" * 80)
        print(question)
        if correct_answer:
            print("\n" + "-" * 80)
            print("CORRECT ANSWER:")
            print("-" * 80)
            print(correct_answer)
        print("\n" + "-" * 80)
        print(f"THINKING MODEL RESPONSE ({thinking_tokens} tokens):")
        print("-" * 80)
        print(thinking_response)
        print("\n" + "-" * 80)
        print(f"BASE MODEL RESPONSE ({base_tokens} tokens):")
        print("-" * 80)
        print(base_response)
        print("\n" + "-" * 80)
        print(f"HYBRID MODEL RESPONSE ({hybrid_tokens} tokens):")
        print("-" * 80)
        print(hybrid_response)
        print("\n" + "=" * 80)

        # Evaluate answers
        print("\nEvaluating answers...")
        if _is_ablation(args):
            print(f"Ablation: evaluating hybrid only ({_ablation_flags_str(args)})")
            thinking_correct, thinking_judge_raw = False, "SKIPPED"
            base_correct, base_judge_raw = False, "SKIPPED"
            hybrid_correct, hybrid_judge_raw = evaluate_answer(clean_hybrid_answer, correct_answer, question, "Hybrid Model", dataset_type=dataset_type, test_list=test_list)
        else:
            thinking_correct, thinking_judge_raw = evaluate_answer(clean_thinking_answer, correct_answer, question, "Thinking Model", dataset_type=dataset_type, test_list=test_list)
            base_correct, base_judge_raw = evaluate_answer(clean_base_answer, correct_answer, question, "Base Model", dataset_type=dataset_type, test_list=test_list)
            hybrid_correct, hybrid_judge_raw = evaluate_answer(clean_hybrid_answer, correct_answer, question, "Hybrid Model", dataset_type=dataset_type, test_list=test_list)
        
        if thinking_correct:
            results["thinking_correct"] += 1
        if base_correct:
            results["base_correct"] += 1
        if hybrid_correct:
            results["hybrid_correct"] += 1
        
        # Rolling save for this task
        rolling_record = {
            "ts": time.time(),
            "dataset": args.dataset,
            "question": question,
            "gold_answer": correct_answer,
            "test_list": test_list,  # For code benchmarks (MBPP), None for math
            "answers": {
                "thinking": thinking_response,
                "base": base_response,
                "hybrid": hybrid_response,
            },
            "judges": {
                "thinking": {"correct": bool(thinking_correct), "raw": thinking_judge_raw},
                "base": {"correct": bool(base_correct), "raw": base_judge_raw},
                "hybrid": {"correct": bool(hybrid_correct), "raw": hybrid_judge_raw},
            },
            "eos": {
                "thinking": bool(thinking_eos_end) if (not _is_ablation(args)) else False,
                "base": bool(base_eos_end) if (not _is_ablation(args)) else False,
                "hybrid": bool(hybrid_eos_end),
            },
            "hybrid_details": {
                "per_token": token_latent_info,
                "steering_selection": steering_selection,
                "coefficients": args.coefficients,
                "token_windows": args.token_windows,
                "sae_layer": args.sae_layer,
                "steering_layer": args.steering_layer,
            },
        }
        append_rolling_result(rolling_record, args, base_model_id, thinking_model_id)

        # Print current results
        if prev_counts is None:
            prev_counts = {"thinking": 0, "base": 0, "hybrid": 0}
        so_far_cum = prev_completed + included_counter
        cum_thinking = prev_counts["thinking"] + results["thinking_correct"]
        cum_base = prev_counts["base"] + results["base_correct"]
        cum_hybrid = prev_counts["hybrid"] + results["hybrid_correct"]

        if only_finished_thinking:
            print(f"\nCurrent Results after {so_far_cum} finished-thinking tasks:")
        else:
            print(f"\nCurrent Results after {so_far_cum} tasks:")
        if not _is_ablation(args):
            print(f"Thinking Model: {cum_thinking}/{so_far_cum} correct ({(cum_thinking/so_far_cum)*100:.1f}%)")
            print(f"Base Model: {cum_base}/{so_far_cum} correct ({(cum_base/so_far_cum)*100:.1f}%)")
        print(f"Hybrid Model: {cum_hybrid}/{so_far_cum} correct ({(cum_hybrid/so_far_cum)*100:.1f}%)")
        # Concise gap recovery and EOS summary so far
        so_far = so_far_cum
        base_acc_now = (cum_base / so_far * 100) if not _is_ablation(args) else 0.0
        thinking_acc_now = (cum_thinking / so_far * 100) if not _is_ablation(args) else 0.0
        hybrid_acc_now = (cum_hybrid / so_far * 100)
        gap_now = abs(thinking_acc_now - base_acc_now) if not _is_ablation(args) else 0.0
        if gap_now > 0:
            recovered_now = (hybrid_acc_now - min(base_acc_now, thinking_acc_now)) / gap_now
            print(f"Gap recovered by hybrid: {max(0.0, recovered_now)*100:.1f}% of |Thinking-Base|")
        else:
            print("Gap recovered by hybrid: n/a")
        # EOS percentages: report combined across previous + current only
        if _is_ablation(args):
            cum_den = prev_eos_known.get('hybrid', 0) + included_counter
            cum_hybrid_eos = prev_eos_counts.get('hybrid', 0) + sum(results['hybrid_eos'])
            cum_pct = (cum_hybrid_eos / cum_den) * 100 if cum_den > 0 else 0.0
            print(f"EOS endings (% across all {so_far_cum} tasks): hybrid {cum_pct:.1f}")
        else:
            cum_den_base = prev_eos_known.get('base', 0) + included_counter
            cum_den_thinking = prev_eos_known.get('thinking', 0) + included_counter
            cum_den_hybrid = prev_eos_known.get('hybrid', 0) + included_counter
            cum_base_eos = prev_eos_counts.get('base', 0) + sum(results['base_eos'])
            cum_thinking_eos = prev_eos_counts.get('thinking', 0) + sum(results['thinking_eos'])
            cum_hybrid_eos = prev_eos_counts.get('hybrid', 0) + sum(results['hybrid_eos'])
            cum_base_pct = (cum_base_eos / cum_den_base) * 100 if cum_den_base > 0 else 0.0
            cum_thinking_pct = (cum_thinking_eos / cum_den_thinking) * 100 if cum_den_thinking > 0 else 0.0
            cum_hybrid_pct = (cum_hybrid_eos / cum_den_hybrid) * 100 if cum_den_hybrid > 0 else 0.0
            print(f"EOS endings (% across all {so_far_cum} tasks): base {cum_base_pct:.1f}, thinking {cum_thinking_pct:.1f}, hybrid {cum_hybrid_pct:.1f}")
        
        # Clean up to prevent memory leaks
        del thinking_input_ids, base_input_ids, thinking_outputs
        try:
            del token_latent_info, per_token_perplexity, token_position
        except Exception:
            pass
        del steering_selection
        del thinking_response, base_response, hybrid_response
        del clean_thinking_answer, clean_base_answer, clean_hybrid_answer
        del latent_counts, latent_percentages, steering_stats
        del cold_start_text
        torch.cuda.empty_cache()
        gc.collect()

    assert included_counter > 0, "No tasks included in stats; possibly the thinking model never ended with EOS under --only-finished-thinking."
    thinking_accuracy = results["thinking_correct"] / included_counter * 100
    base_accuracy = results["base_correct"] / included_counter * 100
    hybrid_accuracy = results["hybrid_correct"] / included_counter * 100

    print("\n===== Final Results =====")
    if only_finished_thinking and skipped_unfinished_thinking > 0:
        print(f"Excluded {skipped_unfinished_thinking} inputs where the thinking model did not end with EOS (this run).")
    if not _is_ablation(args):
        print(f"Thinking Model: {results['thinking_correct']}/{included_counter} correct ({thinking_accuracy:.1f}%)")
        print(f"Base Model: {results['base_correct']}/{included_counter} correct ({base_accuracy:.1f}%)")
    print(f"Hybrid Model: {results['hybrid_correct']}/{included_counter} correct ({hybrid_accuracy:.1f}%)")
    # Concise end-of-run gap and EOS summary
    gap_final = abs(thinking_accuracy - base_accuracy) if not _is_ablation(args) else 0.0
    if gap_final > 0:
        recovered_final = (hybrid_accuracy - min(base_accuracy, thinking_accuracy)) / gap_final
        print(f"Gap recovered by hybrid: {max(0.0, recovered_final)*100:.1f}% of |Thinking-Base|")
    else:
        print("Gap recovered by hybrid: n/a")
    # EOS endings combined across previous + this run
    if _is_ablation(args):
        cum_den_hybrid = prev_eos_known.get('hybrid', 0) + included_counter
        cum_hybrid_eos = prev_eos_counts.get('hybrid', 0) + sum(results['hybrid_eos'])
        cum_hybrid_pct = (cum_hybrid_eos / cum_den_hybrid) * 100 if cum_den_hybrid > 0 else 0.0
        print(f"EOS endings (% across all {prev_completed + included_counter} tasks): hybrid {cum_hybrid_pct:.1f}")
    else:
        cum_den_base = prev_eos_known.get('base', 0) + included_counter
        cum_den_thinking = prev_eos_known.get('thinking', 0) + included_counter
        cum_den_hybrid = prev_eos_known.get('hybrid', 0) + included_counter
        cum_base_eos = prev_eos_counts.get('base', 0) + sum(results['base_eos'])
        cum_thinking_eos = prev_eos_counts.get('thinking', 0) + sum(results['thinking_eos'])
        cum_hybrid_eos = prev_eos_counts.get('hybrid', 0) + sum(results['hybrid_eos'])
        cum_base_pct = (cum_base_eos / cum_den_base) * 100 if cum_den_base > 0 else 0.0
        cum_thinking_pct = (cum_thinking_eos / cum_den_thinking) * 100 if cum_den_thinking > 0 else 0.0
        cum_hybrid_pct = (cum_hybrid_eos / cum_den_hybrid) * 100 if cum_den_hybrid > 0 else 0.0
        print(f"EOS endings (% across all {prev_completed + included_counter} tasks): base {cum_base_pct:.1f}, thinking {cum_thinking_pct:.1f}, hybrid {cum_hybrid_pct:.1f}")

    plt.figure(figsize=(10, 6))
    model_names = ["Base", "Thinking", "Hybrid"]
    if _is_ablation(args):
        accuracies = [0.0, 0.0, hybrid_accuracy]
        colors = ["#bdc3c7", "#bdc3c7", "#2ecc71"]
    else:
        accuracies = [base_accuracy, thinking_accuracy, hybrid_accuracy]
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
    plt.bar(model_names, accuracies, color=colors)
    plt.title(f"Model Accuracy on {included_counter} {args.dataset} Tasks")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 2, f"{accuracy:.1f}%", ha='center')
    plt.tight_layout()
    suffix = _result_suffix(args)
    base_id_for_files = _normalized_base_id_for_filenames(base_model_id, thinking_model_id)
    plt.savefig(f"{args.results_dir}/accuracy_{base_id_for_files}_{args.dataset}{suffix}.png")
    plt.show()

    benchmark_data = {
        "metadata": {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "n_tasks": included_counter,
        },
        "results": {
            "accuracy": {
                "base_model": base_accuracy,
                "thinking_model": thinking_accuracy,
                "hybrid_model": hybrid_accuracy
            },
            "correct_count": {
                "base_model": results["base_correct"],
                "thinking_model": results["thinking_correct"],
                "hybrid_model": results["hybrid_correct"]
            }
        },
        "tasks": []
    }
    for i in range(included_counter):
        task_data = {
            "question": results["questions"][i],
            "correct_answer": results["correct_answers"][i],
            "model_answers": {
                "base_model": results["base_answers"][i],
                "thinking_model": results["thinking_answers"][i],
                "hybrid_model": results["hybrid_answers"][i]
            }
        }
        benchmark_data["tasks"].append(task_data)
    suffix = _result_suffix(args)
    json_path = f"{args.results_dir}/benchmark_results_{base_id_for_files}_{args.dataset}{suffix}.json"
    with open(json_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Benchmark results saved to {json_path}")
    return results

if __name__ == "__main__":
    # Get command line arguments when running this file as a script
    args = parse_args()
    if bool(getattr(args, "results_suffix", "")):
        print(f"[Results] results-suffix active: {getattr(args, 'results_suffix')}")
    if bool(getattr(args, "disable_steering_in_code_blocks", False)):
        print("[CodeFence] disable-steering-in-code-blocks enabled: steering disabled inside ```python ... ```.")
    if bool(getattr(args, "disable_sae_mean", False)):
        print("\n" + "!" * 90)
        print("!!! WARNING: --disable-sae-mean is enabled !!!")
        print("This will feed RAW thinking-model activations into the SAE encoder (no mean-centering, no L2-normalization).")
        print("Results may not be comparable to standard runs. Use only if you know what you're doing.")
        print("!" * 90 + "\n")

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
if __name__ == "__main__":
    os.makedirs(f"{args.results_dir}/vars", exist_ok=True)

    # Quick judge API connectivity test
    quick_judge_api_test()

    # %% Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'gsm8k':
        dataset = load_dataset("openai/gsm8k", "main")["test"]  # type: ignore
    elif args.dataset == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]  # type: ignore
    elif args.dataset == "aime25":
        dataset = load_dataset("yentinglin/aime_2025")["train"]  # type: ignore
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]  # type: ignore
    elif args.dataset == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "full")["test"]  # type: ignore
    elif args.dataset == "livecodebench":
        # Use bzantium/livecodebench - compatible with datasets 4.0+ (uses JSONL instead of deprecated loading script)
        dataset = load_dataset("bzantium/livecodebench", "release_v5")["test"]  # type: ignore
    elif args.dataset == "medqa":
        # MedQA USMLE 4-option multiple choice - medical licensing exam questions (first 500)
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")["test"].select(range(500))  # type: ignore
    elif args.dataset == "legalbench":
        # LegalBench - legal reasoning benchmark with 162 subsets
        # Load first 5 short examples from each subset (loading script deprecated, fetch TSV directly)
        import requests
        import pandas as pd
        from io import StringIO

        LEGALBENCH_EXAMPLES_PER_SUBSET = 5
        LEGALBENCH_MAX_CHARS = 4000  # ~1000 tokens
        # Skip subsets where all examples exceed max length (e.g. full tax statutes ~30k chars each)
        LEGALBENCH_SKIP_SUBSETS = {"sara_numeric"}

        def _legalbench_example_length(row):
            """Calculate total character length of text fields in a row."""
            length = 0
            if 'text' in row:
                length += len(str(row['text']))
            if 'question' in row:
                length += len(str(row['question']))
            # Also check other common text fields
            for col in ['contract', 'policy', 'claim', 'bill_summary', 'Paragraph']:
                if col in row:
                    length += len(str(row[col]))
            return length

        print("Loading LegalBench subsets (this may take a moment)...")
        resp = requests.get('https://huggingface.co/api/datasets/nguha/legalbench/tree/main/data')
        folders = [f['path'].replace('data/', '') for f in resp.json() if f['type'] == 'directory' and '.ipynb' not in f['path']]

        # Cache for base prompts (fetched from GitHub)
        base_prompts = {}

        def _get_base_prompt(subset):
            """Fetch base_prompt.txt from GitHub for a subset."""
            if subset in base_prompts:
                return base_prompts[subset]
            prompt_url = f'https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks/{subset}/base_prompt.txt'
            try:
                resp = requests.get(prompt_url, timeout=5)
                if resp.status_code == 200:
                    base_prompts[subset] = resp.text.strip()
                else:
                    base_prompts[subset] = None
            except:
                base_prompts[subset] = None
            return base_prompts[subset]

        all_examples = []
        skipped_long = 0
        for subset in folders:
            if subset in LEGALBENCH_SKIP_SUBSETS:
                continue
            url = f'https://huggingface.co/datasets/nguha/legalbench/resolve/main/data/{subset}/test.tsv'
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    df = pd.read_csv(StringIO(resp.text), sep='\t')
                    # Fetch base prompt for this subset
                    base_prompt = _get_base_prompt(subset)
                    # Take first N short examples from each subset
                    subset_count = 0
                    for _, row in df.iterrows():
                        if subset_count >= LEGALBENCH_EXAMPLES_PER_SUBSET:
                            break
                        if _legalbench_example_length(row) > LEGALBENCH_MAX_CHARS:
                            skipped_long += 1
                            continue
                        example = dict(row)
                        example['subset'] = subset
                        example['base_prompt'] = base_prompt
                        all_examples.append(example)
                        subset_count += 1
            except Exception as e:
                print(f"Warning: Failed to load {subset}: {e}")

        print(f"Loaded {len(all_examples)} LegalBench examples ({LEGALBENCH_EXAMPLES_PER_SUBSET} per subset from {len(folders)} subsets, skipped {skipped_long} long examples)")
        dataset = all_examples  # type: ignore

    # %% Load models and SAE
    thinking_model, thinking_tokenizer, base_model, base_tokenizer, sae, steering_vectors, descriptions, thinking_model_id, base_model_id = load_models_and_sae(args)

    # %% Auto-resume from rolling results if present
    completed = _count_completed_tasks(args, base_model_id, thinking_model_id)
    if completed > 0:
        total_available = len(dataset)
        assert completed <= total_available, "Rolling file has more entries than dataset size"
        if completed >= int(args.n_tasks):
            print(f"Resume: {completed} tasks already completed (>= n_tasks {args.n_tasks}). Nothing to do.")
            sys.exit(0)
        print(f"Resume: found {completed} completed tasks. Starting from index {completed} and running {int(args.n_tasks) - int(completed)} more.")
        args.eval_start_idx = int(completed)
        args.n_tasks = int(args.n_tasks) - int(completed)

    # %% Run an example (optional)
    if args.run_example:
        print("\n===== Running Example =====")
        thinking_response, base_response, hybrid_response, token_latent_info, steering_selection = run_example(
            thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
            sae, steering_vectors, descriptions, args, dataset
        )

        # Analyze example stats
        no_steering_fraction, latent_counts, latent_percentages, steering_stats = analyze_hybrid_stats(token_latent_info, steering_selection)

        print("\n===== Example Statistics =====")
        print(f"No-steering fraction: {no_steering_fraction:.2f}")
        print("Latent usage (top 5):")
        for latent, count in latent_counts.most_common(5):
            print(f"  {latent}: {count} tokens ({latent_percentages[latent]:.1f}%)")

    # %% Run evaluation
    print("\n===== Running Evaluation =====")
    # Load previous cumulative counts for printing
    prev_completed_count, prev_correct_counts, prev_eos_counts, prev_eos_known = _load_prev_counts(args, base_model_id, thinking_model_id)
    results = run_evaluation(
        thinking_model, thinking_tokenizer, base_model, base_tokenizer,
        sae, steering_vectors, descriptions, args, dataset, thinking_model_id, base_model_id,
        prev_completed=prev_completed_count, prev_counts=prev_correct_counts,
    )

    # Save detailed results
    detailed_data = save_detailed_results(results, args, thinking_model_id, base_model_id)

    # Plot additional statistics
    if results["no_steering_fractions"]:
        plt.figure(figsize=(10, 6))
        plt.hist(results["no_steering_fractions"], bins=10, color="#2ecc71", alpha=0.7)
        plt.title("Distribution of No-Steering Fraction")
        plt.xlabel("Fraction of Tokens Using No Steering")
        plt.ylabel("Number of Tasks")
        plt.axvline(x=detailed_data["steering_stats"]["avg_no_steering"], color='red', linestyle='--', 
                    label=f"Average: {detailed_data['steering_stats']['avg_no_steering']:.2f}")
        plt.legend()
        plt.tight_layout()
        suffix = _result_suffix(args)
        base_id_for_files = _normalized_base_id_for_filenames(base_model_id, thinking_model_id)
        plt.savefig(f"{args.results_dir}/no_steering_distribution_{base_id_for_files}_{args.dataset}{suffix}.png")
        plt.show()

    # Clean up
    print("Evaluation complete. Cleaning up...")
    del thinking_model, base_model, sae
    torch.cuda.empty_cache()
    gc.collect()
