# %%
import dotenv
dotenv.load_dotenv("../.env")

import argparse
import json
import os
import random
import sys
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports (cot-interp utils live one directory up)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils  # noqa: E402
from utils import steering_opt  # noqa: E402

# We re-use helper utilities from evaluate_steering_vectors.py to avoid code
# duplication. Importing the module does *not* execute its main() entry-point
# because it is guarded by `if __name__ == "__main__":`.
from evaluate_steering_vectors import (
    extract_eval_examples_for_category,
    generate_steered_completion,
)

# ---------------------------------------------------------------------------
# Helper: generate an *un-steered* continuation (baseline)
# ---------------------------------------------------------------------------

def generate_base_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
) -> str:
    """Return the raw continuation from *model* without any steering."""
    out_tokens = model.generate(
        **tokenizer(prompt, return_tensors="pt").to(model.device),
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        suppress_tokens=[tokenizer.eos_token_id],  # avoid immediate stop
    )
    full = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)[0]
    return full[len(prompt) :]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Preview steering vectors on random examples")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated list of category indices (e.g. 0,4,7)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B", help="Base model name or path")
    parser.add_argument("--vectors_dir", type=str, default="results/vars/optimized_vectors", help="Directory containing <model>_idx{n}.pt vector files")
    parser.add_argument("--n_examples", type=int, default=1, help="Number of random examples to preview per category")
    parser.add_argument("--context_sentences", type=int, default=0, help="Number of additional context sentences to append")
    parser.add_argument("--steering_token_window", type=int, default=50, help="Window size used when applying steering vector")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Tokens to generate for each completion")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Load base model in 8-bit mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_activation_perplexity_selection", action="store_true", default=False, help="Use activation→perplexity selection when sampling examples (same as optimisation script)")

    args, _ = parser.parse_known_args(argv)

    # ---------------------------------------------------------------------
    # Setup & model loading
    # ---------------------------------------------------------------------

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        dtype=torch.bfloat16,
    )
    torch.set_default_device(base_model.device)
    for p in base_model.parameters():
        p.requires_grad = False

    # ---------------------------------------------------------------------
    # Resolve paths & load supporting artefacts
    # ---------------------------------------------------------------------

    script_dir = os.path.dirname(__file__)
    vectors_dir = os.path.join(script_dir, args.vectors_dir)

    # Hyper-parameters JSON (aggregated) – optional
    aggregated_hparams: Dict[str, Dict[str, dict]] = {}
    model_short = args.model.split("/")[-1].lower()

    # Directory containing per-vector hyperparameter JSONs
    hyperparams_dir = os.path.join(script_dir, "results", "vars", "hyperparams")
    indices_available: List[int] = []
    if os.path.isdir(hyperparams_dir):
        for fn in os.listdir(hyperparams_dir):
            if fn.startswith(f"steering_vector_hyperparams_{model_short}_") and fn.endswith(".json"):
                try:
                    idx_val = int(fn.rsplit("_", 1)[-1].split(".")[0])
                    indices_available.append(idx_val)
                except ValueError:
                    continue

    model_hparams = aggregated_hparams.get(model_short, {})

    # ---------------------------------------------------------------------
    # Load responses & annotations (for example sampling)
    # ---------------------------------------------------------------------

    thinking_model_name = utils.model_mapping.get(args.model, model_short)
    thinking_short = thinking_model_name.split("/")[-1].lower()
    responses_path = os.path.join(script_dir, "..", "generate-responses", "results", "vars", f"responses_{thinking_short}.json")
    annotated_path = os.path.join(script_dir, "..", "generate-responses", "results", "vars", f"annotated_responses_{thinking_short}.json")

    if not os.path.exists(responses_path) or not os.path.exists(annotated_path):
        parser.error("Responses or annotated responses not found – please generate & annotate first.")

    with open(responses_path, "r") as f:
        responses_data = json.load(f)
    with open(annotated_path, "r") as f:
        annotated_data = json.load(f)

    # merge responses with their annotations (same logic as other scripts)
    valid_responses = []
    for i, resp in enumerate(responses_data):
        if i < len(annotated_data):
            ann = annotated_data[i]
            if (
                resp.get("question_id") == ann.get("question_id")
                and resp.get("dataset_name") == ann.get("dataset_name")
                and ann.get("annotated_thinking")
            ):
                merged = resp.copy()
                merged["annotated_thinking"] = ann["annotated_thinking"]
                valid_responses.append(merged)
    if not valid_responses:
        parser.error("No valid annotated responses found.")

    # ---------------------------------------------------------------------
    # Iterate over requested indices
    # ---------------------------------------------------------------------

    # Determine which indices to process
    if args.indices:
        indices: List[int] = [int(idx.strip()) for idx in args.indices.split(",") if idx.strip()]
    else:
        if model_hparams:
            indices = sorted(int(k) for k in model_hparams.keys())
        elif indices_available:
            indices = sorted(indices_available)
        else:
            parser.error("Could not determine available category indices automatically – please specify --indices.")

    for idx in indices:

        # ---------------------- Load hyper-parameters -------------------- #
        hp_entry = model_hparams.get(str(idx))

        if hp_entry:
            category = hp_entry["category"]
            layer = hp_entry["hyperparameters"].get("layer", 0)
        else:
            # fallback: read per-vector hyperparam file
            per_hp_path = os.path.join(
                script_dir,
                "results",
                "vars",
                "hyperparams",
                f"steering_vector_hyperparams_{model_short}_{idx}.json",
            )
            if os.path.isfile(per_hp_path):
                with open(per_hp_path, "r") as f:
                    tmp = json.load(f)
                category = tmp.get("category", f"<unknown-{idx}>")
                layer = tmp.get("hyperparameters", {}).get("layer", 0)
            else:
                print(f"[WARN] Hyper-parameters for idx {idx} not found – skipping.")
                continue

        print(f"\n==============================\n# CATEGORY {category}\n==============================")

        # ---------------------- Load steering vector -------------------- #
        vec_path = os.path.join(vectors_dir, f"{model_short}_idx{idx}.pt")
        if not os.path.isfile(vec_path):
            print(f"[WARN] Vector file {vec_path} not found – skipping.")
            continue
        vec_dict = torch.load(vec_path, map_location="cpu")
        vector = next(iter(vec_dict.values())).to(base_model.device).to(base_model.dtype)

        # ---------------------- Sample examples ------------------------- #
        examples = extract_eval_examples_for_category(
            valid_responses,
            category,
            tokenizer,
            base_model,
            n_examples=args.n_examples,
            context_sentences=args.context_sentences,
            use_activation_perplexity_selection=args.use_activation_perplexity_selection,
        )
        if not examples:
            print("No examples found for this category – skipping.")
            continue

        for ex_num, ex in enumerate(random.sample(examples, min(args.n_examples, len(examples))), 1):
            prompt = ex["prompt"]
            target_completion = ex["target_completion"]

            print(f"\n--- Example {ex_num}/{args.n_examples} ")
            print("Prompt (last 120 chars): …" + prompt[-120:])

            # Base completion
            base_out = generate_base_completion(
                base_model, tokenizer, prompt, max_new_tokens=args.max_new_tokens
            )
            print("\n[BASE] Completion:\n" + base_out.strip())

            # Steered completion
            steered_out = generate_steered_completion(
                base_model,
                tokenizer,
                vector,
                layer,
                prompt,
                target_completion,
                max_new_tokens=args.max_new_tokens,
                steering_token_window=args.steering_token_window,
            )
            print("\n[STEERED] Completion:\n" + steered_out.strip())

            # Optionally, also show reference target
            print("\n[TARGET] Reference completion:\n" + target_completion.strip())

    print("\nDone.")


if __name__ == "__main__":
    main()
# %% 