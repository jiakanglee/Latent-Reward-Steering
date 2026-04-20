#!/usr/bin/env python
# %%
"""
Bug analysis: quantify how often SAE argmax latents change when using:

  (A) cached activations that were centered + L2-normalized (as produced by
      `utils.utils.process_saved_responses`, which backs `train-saes/evaluate_trained_clustering.py`)
  vs
  (B) recomputed *raw* activations (sentence-mean residual stream vectors) without centering/L2-normalization
      (as used by `generate-responses/annotate_thinking.py` and parts of `hybrid/`).

We enforce **token-span parity** with a strict alignment check:
- We reconstruct token spans exactly using the same `full_response.find(sentence)` + `get_char_to_token_map`
  logic used by `process_saved_responses`.
- We recompute centered+L2-normalized vectors for each sampled sentence and require they match the cached vectors
  with very high cosine similarity. This ensures any disagreement is really due to normalization, not misalignment.

Run (repo root):
  uv run python bug-analysis.py
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm


# %%
# -----------------------------
# Config (ORZ 0.5B)
# -----------------------------

MODEL_NAME = "Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B"
MODEL_ID = MODEL_NAME.split("/")[-1].lower()  # must match repo naming convention

# Choose a layer that has both cached activations and trained SAEs on disk.
LAYER = 8

# Used only to locate the cached activations file on disk. This is a filename tag, not a guarantee.
ACTIVATIONS_CACHE_N_EXAMPLES_TAG = 100000

# Choose one SAE size for argmax analysis.
N_CLUSTERS = 15

# Sampling + determinism
N_SENTENCES = 5000
SEED = 0

# Strict alignment check:
# cosine(sim(recomputed_centered_l2, cached_centered_l2)) must exceed this threshold.
ALIGNMENT_COSINE_THRESHOLD = 0.999

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %%
def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# Import repo utilities (match the project’s exact behavior).
# We add repo root so `import utils...` resolves.
sys.path.insert(0, _repo_root())
from utils.responses import extract_thinking_process  # noqa: E402
from utils.sae import SAE  # noqa: E402
from utils.utils import get_char_to_token_map, split_into_sentences, load_model  # noqa: E402


# %%
def _paths(model_id: str, layer: int) -> tuple[str, str, str]:
    root = _repo_root()
    responses_path = os.path.join(root, "generate-responses", "results", "vars", f"responses_{model_id}.json")
    cache_path = os.path.join(
        root,
        "generate-responses",
        "results",
        "vars",
        f"activations_{model_id}_{ACTIVATIONS_CACHE_N_EXAMPLES_TAG}_{layer}.pkl",
    )
    sae_path = os.path.join(
        root,
        "train-saes",
        "results",
        "vars",
        "saes",
        f"sae_{model_id}_layer{layer}_clusters{N_CLUSTERS}.pt",
    )
    return responses_path, cache_path, sae_path


@dataclass(frozen=True)
class SentenceOccurrence:
    resp_idx: int
    sentence: str


# %%
def argmax_latent_from_activation(*, sae: SAE, x: torch.Tensor, hidden_size: int, n_clusters: int) -> int:
    """
    x: (d_model,) float32
    Returns: int latent index in [0, n_clusters).
    """
    assert x.shape == (hidden_size,), f"Expected x.shape={(hidden_size,)}, got {tuple(x.shape)}"
    x = x.to(dtype=torch.float32, device="cpu")
    with torch.no_grad():
        logits = sae.encoder(x - sae.b_dec)  # (n_clusters,)
    assert logits.shape == (n_clusters,), f"Bad SAE logits shape: {tuple(logits.shape)}"
    return int(torch.argmax(logits).item())


# %%
# -----------------------------
# Load cached activations (centered + L2-normalized)
# -----------------------------

responses_path, cache_path, sae_path = _paths(MODEL_ID, LAYER)

assert os.path.exists(cache_path), f"Missing cached activations: {cache_path}"
with open(cache_path, "rb") as f:
    cached_acts_norm, cached_sentences = pickle.load(f)

cached_acts_norm = np.asarray(cached_acts_norm)
assert cached_acts_norm.ndim == 2, f"Expected (n_sentences, d_model), got {cached_acts_norm.shape}"
assert len(cached_sentences) == cached_acts_norm.shape[0]

hidden_size = int(cached_acts_norm.shape[1])
print(f"Loaded cached activations: acts={cached_acts_norm.shape}, sentences={len(cached_sentences)}")


# %%
# -----------------------------
# Load SAE (trained on centered + L2-normalized activations)
# -----------------------------

assert os.path.exists(sae_path), f"Missing SAE checkpoint: {sae_path}"
ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)

sae = SAE(int(ckpt["input_dim"]), int(ckpt["num_latents"]), k=int(ckpt.get("topk", 3)))
sae.encoder.weight.data = ckpt["encoder_weight"]
sae.encoder.bias.data = ckpt["encoder_bias"]
sae.W_dec.data = ckpt["decoder_weight"]
sae.b_dec.data = ckpt["b_dec"]
sae.eval()

assert sae.b_dec.shape == (hidden_size,), f"b_dec shape mismatch: {sae.b_dec.shape} vs hidden_size={hidden_size}"
assert sae.encoder.weight.shape[1] == hidden_size
assert sae.encoder.bias.shape[0] == N_CLUSTERS
print(f"Loaded SAE: layer={LAYER}, n_clusters={N_CLUSTERS}, d_model={hidden_size}")


# %%
# -----------------------------
# Load responses JSON (source of truth for full_response text)
# -----------------------------

assert os.path.exists(responses_path), f"Missing responses JSON: {responses_path}"
with open(responses_path, "r", encoding="utf-8") as f:
    responses_data = json.load(f)
assert isinstance(responses_data, list) and len(responses_data) > 0
print(f"Loaded responses: {len(responses_data)} examples from {responses_path}")

# Critical assumption for the strict alignment check:
# `process_saved_responses` does `random.shuffle(responses_data); responses_data = responses_data[:n_examples]`.
# If `n_examples < len(responses_data)`, the centering mean depends on that shuffle and cannot be reproduced
# without the exact seed/subset. We fail fast in that case.
n_examples_used_for_cache = min(ACTIVATIONS_CACHE_N_EXAMPLES_TAG, len(responses_data))
assert n_examples_used_for_cache == len(responses_data), (
    "This notebook cannot reliably reproduce the cache's centering mean when the cache was built from a "
    "random subset of a larger responses JSON (because `process_saved_responses` shuffles before slicing). "
    f"Here: cache tag n_examples={ACTIVATIONS_CACHE_N_EXAMPLES_TAG}, responses_json_len={len(responses_data)}.\n"
    "To proceed, regenerate the cache with n_examples >= responses_json_len, or save the exact subset/seed "
    "used to build the cache and add it here."
)

# Cache for the dataset mean vector used for centering (expensive to recompute).
mean_cache_path = os.path.join(
    _repo_root(),
    "generate-responses",
    "results",
    "vars",
    f"overall_mean_{MODEL_ID}_layer{LAYER}_n{len(responses_data)}_tag{ACTIVATIONS_CACHE_N_EXAMPLES_TAG}.pkl",
)


# %%
# -----------------------------
# Step 1: sample sentences that are unique in the cache
# -----------------------------

rng = random.Random(SEED)
sentence_counts = Counter(cached_sentences)
unique_cache_indices = [i for i, s in enumerate(cached_sentences) if sentence_counts[s] == 1]
assert len(unique_cache_indices) >= N_SENTENCES, (
    f"Not enough unique cached sentences ({len(unique_cache_indices)}) to sample {N_SENTENCES}"
)

sampled_cache_indices = rng.sample(unique_cache_indices, N_SENTENCES)
sampled_sentences = [cached_sentences[i] for i in sampled_cache_indices]

# Map sentence -> cache_idx (unique by construction)
sentence_to_cache_idx = {cached_sentences[i]: i for i in sampled_cache_indices}
assert len(sentence_to_cache_idx) == N_SENTENCES
print(f"Sampled {N_SENTENCES} sentences (unique in cached list).")


# %%
# -----------------------------
# Step 2: locate each sampled sentence in the responses JSON
#         (matching the *project* notion of "a sentence")
# -----------------------------

def find_unique_occurrences_for_sample(
    sampled_sentences: list[str],
    responses_data: list[dict],
) -> dict[str, int | None]:
    """
    Single-pass Step 2 to avoid O(N_SENTENCES * N_RESPONSES).

    Semantics match the cache pipeline (`process_saved_responses`):
      - A sampled sentence "matches" a response iff it appears as an element of
        `split_into_sentences(extract_thinking_process(full_response))`.
      - If it matches 0 responses -> None
      - If it matches >= 2 responses -> None (ambiguous)
    """
    sampled_set = set(sampled_sentences)
    found: dict[str, int | None] = {s: None for s in sampled_sentences}
    found_count: dict[str, int] = {s: 0 for s in sampled_sentences}

    for resp_idx, resp in enumerate(tqdm(responses_data, desc="Step 2: locating sampled sentences")):
        full = resp["full_response"]
        thinking = extract_thinking_process(full)
        if not thinking:
            continue
        sentences = split_into_sentences(thinking)
        if not sentences:
            continue

        present = sampled_set.intersection(set(sentences))
        if not present:
            continue

        for s in present:
            found_count[s] += 1
            if found_count[s] == 1:
                found[s] = resp_idx
            else:
                found[s] = None  # ambiguous across responses

    for s in sampled_sentences:
        if found_count[s] != 1:
            found[s] = None
    return found


sentence_to_resp_idx = find_unique_occurrences_for_sample(sampled_sentences, responses_data)

occurrences: list[SentenceOccurrence] = []
missing_or_ambiguous: list[str] = []
for s in sampled_sentences:
    idx = sentence_to_resp_idx[s]
    if idx is None:
        missing_or_ambiguous.append(s)
    else:
        occurrences.append(SentenceOccurrence(resp_idx=idx, sentence=s))

assert len(occurrences) == N_SENTENCES, (
    f"Could only uniquely locate {len(occurrences)}/{N_SENTENCES} sampled sentences in responses JSON. "
    f"(missing/ambiguous={len(missing_or_ambiguous)})\n"
    f"Tip: rerun with a different SEED, or relax uniqueness constraints."
)
print("All sampled sentences have unique (response_idx, provenance) in responses JSON.")


# %%
# -----------------------------
# Step 3: load ORZ 0.5B and recompute raw activations + centering mean
# -----------------------------

torch.set_grad_enabled(False)
print(f"About to load model for recomputation: {MODEL_NAME}")
model, tokenizer = load_model(model_name=MODEL_NAME, load_in_8bit=False)
model.eval()

tok_id = getattr(tokenizer, "name_or_path", "") or ""
assert MODEL_ID in tok_id.lower(), (
    "Loaded an unexpected tokenizer/model. "
    f"Expected tokenizer.name_or_path to contain '{MODEL_ID}', but got '{tok_id}'. "
    "Double-check you are running `uv run python bug-analysis.py` (not another script)."
)
print(f"Loaded nnsight model on {DEVICE}: tokenizer.name_or_path={tok_id!r}")


def _extract_sentence_token_span(full_response: str, sentence: str) -> tuple[int, int]:
    """
    Match `process_saved_responses` exactly:
      - char position via full_response.find(sentence)
      - token_start = char_to_token[text_pos]
      - token_end   = char_to_token[text_pos + len(sentence)]
      - segment uses token_start-1 : token_end
    Returns (token_start, token_end) where token_end is exclusive.
    """
    text_pos = full_response.find(sentence)
    assert text_pos >= 0, "Sentence not found in full_response (unexpected after Step 2)"
    char_to_token = get_char_to_token_map(full_response, tokenizer)
    token_start = char_to_token.get(text_pos, None)
    token_end = char_to_token.get(text_pos + len(sentence), None)
    assert token_start is not None and token_end is not None
    assert 0 < token_start < token_end, f"Bad token span: start={token_start}, end={token_end}"
    return int(token_start), int(token_end)


def _charpos_to_token_idx(offset_mapping: list[tuple[int, int]], pos: int) -> int | None:
    """
    Return token index i such that offset_mapping[i][0] <= pos < offset_mapping[i][1], else None.

    This is a lightweight alternative to `get_char_to_token_map` that avoids building an O(len(text))
    dictionary. It relies on tokenizer `offset_mapping` produced by fast tokenizers.
    """
    # NOTE: offset_mapping often contains (0, 0) for special tokens; those will never match.
    starts = [s for (s, _e) in offset_mapping]
    i = bisect_right(starts, pos) - 1
    if i < 0:
        return None
    s, e = offset_mapping[i]
    if s <= pos < e and e > s:
        return i
    return None


def _tokenize_with_offsets(full_response: str):
    """
    Single tokenization call that provides:
      - input_ids (torch.LongTensor [1, seq])
      - attention_mask (torch.LongTensor [1, seq])
      - offset_mapping (list[(start,end)] length seq)
    """
    enc = tokenizer(
        full_response,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    # `offset_mapping` is a (1, seq, 2) tensor for fast tokenizers.
    offset_t = enc["offset_mapping"][0]
    offset_mapping = [(int(s.item()), int(e.item())) for (s, e) in offset_t]
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    return input_ids, attention_mask, offset_mapping


def _forward_layer_output(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Match `utils.utils.process_saved_responses`:
      - trace with attention_mask
      - read `model.model.layers[LAYER].output`

    Returns layer output on CPU float32 with shape (1, seq_len, d_model).
    """
    with torch.no_grad():
        with model.trace({"input_ids": input_ids, "attention_mask": attention_mask}) as _tracer:
            saved_output = model.model.layers[LAYER].output.save()
    out = saved_output.detach().cpu().to(torch.float32)
    assert out.ndim == 3 and out.shape[0] == 1 and out.shape[2] == hidden_size, f"Bad out shape: {tuple(out.shape)}"
    del saved_output
    return out


# %%
# Recompute (or load) overall mean (centering vector) exactly like `process_saved_responses`.

# Group sampled occurrences by resp_idx so we can efficiently extract their sentence activations.
occ_by_resp: dict[int, list[str]] = defaultdict(list)
for occ in occurrences:
    occ_by_resp[occ.resp_idx].append(occ.sentence)

raw_sentence_vecs: dict[str, torch.Tensor] = {}
recomputed_centered_l2_vecs: dict[str, torch.Tensor] = {}

overall_mean: torch.Tensor
count_vectors: int

if os.path.exists(mean_cache_path):
    with open(mean_cache_path, "rb") as f:
        mean_cache = pickle.load(f)
    assert isinstance(mean_cache, dict)
    assert mean_cache.get("model_id") == MODEL_ID
    assert int(mean_cache.get("layer")) == int(LAYER)
    assert int(mean_cache.get("n_responses")) == int(len(responses_data))
    assert int(mean_cache.get("cache_tag")) == int(ACTIVATIONS_CACHE_N_EXAMPLES_TAG)
    mean_np = np.asarray(mean_cache.get("overall_mean"))
    count_vectors = int(mean_cache.get("count_vectors"))
    assert mean_np.shape == (hidden_size,), f"Cached mean shape mismatch: {mean_np.shape} vs {(hidden_size,)}"
    assert count_vectors > 0
    overall_mean = torch.from_numpy(mean_np).to(torch.float32)
    print(f"Loaded cached overall mean from {mean_cache_path} (count_vectors={count_vectors}).")

    # Only forward the (small) subset of responses needed to get raw vectors for sampled sentences.
    for resp_idx in tqdm(sorted(occ_by_resp.keys()), desc="Forward pass (sampled sentences only)"):
        resp = responses_data[resp_idx]
        full_response: str = resp["full_response"]
        thinking = extract_thinking_process(full_response)
        assert thinking, "Response used for a sampled sentence has empty thinking; unexpected."
        sentences = split_into_sentences(thinking)
        assert sentences, "Response used for a sampled sentence has no sentences; unexpected."
        sentences_set = set(sentences)

        input_ids, attention_mask, _offsets = _tokenize_with_offsets(full_response)
        layer_out = _forward_layer_output(input_ids, attention_mask)  # (1, seq, d_model) on CPU
        for target_sentence in occ_by_resp[resp_idx]:
            assert target_sentence in sentences_set, (
                "Sampled sentence is not present in this response's extracted thinking sentences; "
                "token-span parity with cached activations would be broken."
            )
            ts, te = _extract_sentence_token_span(full_response, target_sentence)
            seg = layer_out[:, ts - 1 : te, :]  # (1, n_tokens, d_model)
            assert seg.shape[0] == 1 and seg.shape[2] == hidden_size and seg.shape[1] > 0
            raw_vec = seg.mean(dim=1).squeeze(0)  # (d_model,)
            assert raw_vec.shape == (hidden_size,)
            raw_sentence_vecs[target_sentence] = raw_vec

        del layer_out
        del input_ids, attention_mask, _offsets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

else:
    sum_vectors = torch.zeros((1, hidden_size), dtype=torch.float32)
    count_vectors = 0

    for resp_idx, resp in enumerate(tqdm(responses_data, desc="Forward pass (mean + sampled sentences)")):
        full_response: str = resp["full_response"]
        thinking = extract_thinking_process(full_response)
        if not thinking:
            continue

        sentences = split_into_sentences(thinking)
        if not sentences:
            continue

        # Match `process_saved_responses` min/max span logic over all sentences in the response
        input_ids, attention_mask, offsets = _tokenize_with_offsets(full_response)
        layer_out = _forward_layer_output(input_ids, attention_mask)  # (1, seq, d_model) on CPU

        min_token_start = float("inf")
        max_token_end = -float("inf")

        # Precompute spans for all sentences to get min/max
        for s in sentences:
            text_pos = full_response.find(s)
            if text_pos < 0:
                continue
            token_start = _charpos_to_token_idx(offsets, text_pos)
            token_end = _charpos_to_token_idx(offsets, text_pos + len(s))
            if token_start is None or token_end is None or token_start >= token_end:
                continue
            min_token_start = min(min_token_start, token_start)
            max_token_end = max(max_token_end, token_end)

        if min_token_start < layer_out.shape[1] and max_token_end > 0:
            vec = layer_out[:, int(min_token_start) : int(max_token_end), :].mean(dim=1)  # (1, d_model)
            assert vec.shape == (1, hidden_size)
            sum_vectors += vec
            count_vectors += 1

        # If this response contains any sampled sentences, compute their raw mean vectors.
        if resp_idx in occ_by_resp:
            sentences_set = set(sentences)
            for target_sentence in occ_by_resp[resp_idx]:
                assert target_sentence in sentences_set, (
                    "Sampled sentence was found via Step 2 sentence-matching, but is missing from this response's "
                    "split_into_sentences(thinking) list at recompute-time; token-span parity would be broken."
                )
                ts, te = _extract_sentence_token_span(full_response, target_sentence)
                seg = layer_out[:, ts - 1 : te, :]  # (1, n_tokens, d_model)
                assert seg.shape[0] == 1 and seg.shape[2] == hidden_size and seg.shape[1] > 0
                raw_vec = seg.mean(dim=1).squeeze(0)  # (d_model,)
                assert raw_vec.shape == (hidden_size,)
                raw_sentence_vecs[target_sentence] = raw_vec

        del layer_out, input_ids, attention_mask, offsets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert count_vectors > 0, "No responses contributed to overall mean; unexpected."
    overall_mean = (sum_vectors / count_vectors).squeeze(0)  # (d_model,)
    assert overall_mean.shape == (hidden_size,)
    print(f"Computed overall mean from {count_vectors} responses.")

    with open(mean_cache_path, "wb") as f:
        pickle.dump(
            {
                "model_id": MODEL_ID,
                "layer": int(LAYER),
                "n_responses": int(len(responses_data)),
                "cache_tag": int(ACTIVATIONS_CACHE_N_EXAMPLES_TAG),
                "count_vectors": int(count_vectors),
                "overall_mean": overall_mean.detach().cpu().numpy(),
            },
            f,
        )
    print(f"Saved overall mean cache to {mean_cache_path}.")

assert len(raw_sentence_vecs) == N_SENTENCES, (
    f"Expected raw vectors for all sampled sentences; got {len(raw_sentence_vecs)}/{N_SENTENCES}"
)


# %%
# Recompute centered+L2-normalized vectors for sampled sentences, and compare to cached.

for s in sampled_sentences:
    raw_vec = raw_sentence_vecs[s]  # (d_model,)
    centered = raw_vec - overall_mean
    denom = torch.norm(centered)
    assert float(denom.item()) > 0.0
    centered_l2 = centered / denom
    assert centered_l2.shape == (hidden_size,)
    recomputed_centered_l2_vecs[s] = centered_l2

    cache_idx = sentence_to_cache_idx[s]
    cached_vec = torch.from_numpy(cached_acts_norm[cache_idx]).to(torch.float32)
    assert cached_vec.shape == (hidden_size,)

    cos = torch.dot(centered_l2, cached_vec) / (torch.norm(centered_l2) * torch.norm(cached_vec))
    if float(cos.item()) < ALIGNMENT_COSINE_THRESHOLD:
        raise AssertionError(
            f"Alignment check failed for sentence (cos={cos.item():.6f} < {ALIGNMENT_COSINE_THRESHOLD}).\n"
            f"Sentence: {s[:200]!r}"
        )

print(f"Alignment check passed for all {N_SENTENCES} sentences (cos >= {ALIGNMENT_COSINE_THRESHOLD}).")


# %%
# Compute argmax latents and agreement metrics.

latent_cached_norm: list[int] = []
latent_raw: list[int] = []
latent_recomputed_norm: list[int] = []

for s in sampled_sentences:
    cache_idx = sentence_to_cache_idx[s]
    cached_vec = torch.from_numpy(cached_acts_norm[cache_idx]).to(torch.float32)
    raw_vec = raw_sentence_vecs[s].to(torch.float32)
    rec_norm_vec = recomputed_centered_l2_vecs[s].to(torch.float32)

    latent_cached_norm.append(argmax_latent_from_activation(sae=sae, x=cached_vec, hidden_size=hidden_size, n_clusters=N_CLUSTERS))
    latent_raw.append(argmax_latent_from_activation(sae=sae, x=raw_vec, hidden_size=hidden_size, n_clusters=N_CLUSTERS))
    latent_recomputed_norm.append(argmax_latent_from_activation(sae=sae, x=rec_norm_vec, hidden_size=hidden_size, n_clusters=N_CLUSTERS))

latent_cached_norm_arr = np.asarray(latent_cached_norm)
latent_raw_arr = np.asarray(latent_raw)
latent_recomputed_norm_arr = np.asarray(latent_recomputed_norm)
assert latent_cached_norm_arr.shape == (N_SENTENCES,)
assert latent_raw_arr.shape == (N_SENTENCES,)
assert latent_recomputed_norm_arr.shape == (N_SENTENCES,)

agree_cached_vs_raw = float(np.mean(latent_cached_norm_arr == latent_raw_arr))
agree_cached_vs_recomputed_norm = float(np.mean(latent_cached_norm_arr == latent_recomputed_norm_arr))

print("\n=== RESULTS ===")
print(f"Model: {MODEL_NAME} (id={MODEL_ID}), layer={LAYER}, n_clusters={N_CLUSTERS}")
print(f"Sentences sampled: {N_SENTENCES} (seed={SEED})")
print(f"Agreement (cached centered+L2 vs raw): {agree_cached_vs_raw:.3f}")
print(f"Agreement (cached centered+L2 vs recomputed centered+L2): {agree_cached_vs_recomputed_norm:.3f}  [token-span sanity]")


# %%
# Optional: show a few disagreements for inspection

disagree_idxs = np.where(latent_cached_norm_arr != latent_raw_arr)[0].tolist()
print(f"\nDisagreements (cached vs raw): {len(disagree_idxs)}/{N_SENTENCES}")

for j in disagree_idxs[:10]:
    s = sampled_sentences[j]
    print("-" * 80)
    print(f"cached_norm latent: {latent_cached_norm_arr[j]} | raw latent: {latent_raw_arr[j]}")
    print(f"sentence: {s}")

