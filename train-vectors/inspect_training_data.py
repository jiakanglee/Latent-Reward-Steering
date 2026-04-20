# %%
import argparse
import os
import sys
import json
import re
import random

import dotenv
dotenv.load_dotenv("../.env")

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from utils.responses import extract_thinking_process

# Ensure we can import utilities from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils  # noqa: E402


# Patterns matching the annotation format used during optimisation
ANNOTATION_PATTERN = re.compile(r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)
CATEGORY_PATTERN = re.compile(r'\["[\d.]+:(\S+?)"\]')


def get_label_positions(annotated_thinking: str, response_text: str, tokenizer, context_sentences: int = 0):
    """Parse SAE annotations and find token positions for each label.

    Returns a dict: {label: List[(token_start, token_end, text_with_optional_context, activation, text_char_pos)]}
    """
    label_positions = {}

    matches = list(ANNOTATION_PATTERN.finditer(annotated_thinking))
    char_to_token = utils.get_char_to_token_map(response_text, tokenizer)
    sentences = utils.split_into_sentences(response_text, min_words=0)

    for match in matches[:-1]:
        activation_str = match.group(1).strip()
        label = match.group(2).strip()
        text = match.group(3).strip()

        try:
            activation = float(activation_str)
        except ValueError:
            continue

        if not text:
            continue

        pattern = r'(?:[.?!;\n]|\n\n)\s*(' + re.escape(text) + ')'
        match = re.search(pattern, response_text)
        text_pos = match.start(1) if match else -1
        if text_pos < 0:
            continue

        token_start = char_to_token.get(text_pos, None)
        token_end = char_to_token.get(text_pos + len(text) - 1, None)
        if token_end is not None:
            token_end += 1
        if token_start is None or token_end is None or token_start >= token_end:
            continue

        target_sentence_idx = -1
        for i, sentence in enumerate(sentences):
            if text in sentence:
                target_sentence_idx = i
                break
        if target_sentence_idx == -1:
            continue

        additional_context = ""
        if context_sentences > 0 and target_sentence_idx < len(sentences) - 1:
            end_idx = min(target_sentence_idx + context_sentences + 1, len(sentences))
            additional_sentences = sentences[target_sentence_idx + 1:end_idx]
            if additional_sentences:
                text_end_pos = text_pos + len(text)
                next_sentence_start = response_text.find(additional_sentences[0], text_end_pos)
                if next_sentence_start > text_end_pos:
                    original_whitespace = response_text[text_end_pos:next_sentence_start]
                    additional_context = original_whitespace + original_whitespace.join(additional_sentences)
                else:
                    additional_context = " " + " ".join(additional_sentences)

            if additional_context:
                context_end_pos = text_pos + len(text) + len(additional_context)
                context_token_end = char_to_token.get(context_end_pos - 1, None)
                if context_token_end is not None:
                    token_end = context_token_end + 1

        if label not in label_positions:
            label_positions[label] = []
        label_positions[label].append((token_start, token_end, text + additional_context, activation, text_pos))

    return label_positions


def get_sorted_categories(responses_data):
    """Extract unique categories from annotated responses and return them sorted by idx number."""
    categories = set()
    for resp in responses_data:
        if not resp.get('annotated_thinking'):
            continue
        matches = CATEGORY_PATTERN.finditer(resp['annotated_thinking'])
        for match in matches:
            categories.add(match.group(1).strip())

    def extract_idx(category_name: str):
        if category_name.startswith('idx'):
            try:
                return int(category_name[3:])
            except ValueError:
                return float('inf')
        return float('inf')

    return sorted(list(categories), key=extract_idx)


def load_category_metadata(model_id: str, layer: int):
    """Load category metadata (title/description) per idx from train-saes results.

    Returns dict[int, {"title": str, "description": str}].
    """
    root = os.path.join(os.path.dirname(__file__), "..", "train-saes", "results", "vars")
    meta = {}
    if not os.path.isdir(root):
        return meta

    candidate_ids = list({model_id, model_id.replace("-", "_"), model_id.replace("_", "-")})

    def collect_from_mapping(mapping):
        # Collect entries with immediate title/description
        for k, v in mapping.items():
            if isinstance(v, dict):
                title = v.get("title") if "title" in v else None
                desc = v.get("description") if "description" in v else None
                if title or desc:
                    try:
                        idx = int(k)
                        meta[idx] = {"title": title, "description": desc}
                    except Exception:
                        pass
                # Recurse into nested 'detailed_results' if present
                if "detailed_results" in v and isinstance(v["detailed_results"], dict):
                    collect_from_mapping(v["detailed_results"])
                # Also recurse into any nested dicts just in case
                for sub_key, sub_val in v.items():
                    if isinstance(sub_val, dict):
                        collect_from_mapping(sub_val)

    for fn in os.listdir(root):
        if not any(fn.endswith(f"_{mid}_layer{layer}.json") for mid in candidate_ids):
            continue
        try:
            with open(os.path.join(root, fn), "r") as f:
                data = json.load(f)
        except Exception:
            continue
        detailed = data.get("detailed_results", {})
        if isinstance(detailed, dict):
            collect_from_mapping(detailed)
    return meta


def extract_examples_for_category(responses_data, category_name: str, tokenizer, n_examples: int, model, context_sentences: int, use_activation_perplexity_selection: bool):
    """Extract examples for a given category using the same filters as optimisation.

    Returns a list of dicts with keys: prompt, target_completion, original_question.
    """
    examples_for_category = []

    for resp in tqdm(responses_data, desc=f"Scanning: {category_name}"):
        if not resp.get('annotated_thinking'):
            continue

        thinking_process = extract_thinking_process(resp["full_response"])

        full_text = (
            f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\n"
            f"Question:\n{resp['original_message']['content']}\n\n"
            f"Step by step answer:\n{thinking_process}"
        )

        if category_name not in resp['annotated_thinking']:
            continue

        label_positions = get_label_positions(resp['annotated_thinking'], full_text, tokenizer, context_sentences)
        if category_name not in label_positions:
            continue

        for _, _, text, activation, text_pos in label_positions[category_name]:
            context = full_text[:text_pos]
            if not context:
                continue
            # Match punctuation boundary filters used by the optimiser
            ctx_stripped = context.strip()
            if len(context) < 2:
                continue
            if (context[-1] not in ['.', '?', '!', ';', '\n', '\n\n'] and
                context[-2] not in ['.', '?', '!', ';', '\n', '\n\n'] and
                ctx_stripped and ctx_stripped[-1] not in ['.', '?', '!', ';', '\n', '\n\n']):
                continue

            if len(text.strip().split()) < 10:
                continue

            examples_for_category.append({
                'prompt': context,
                'target_completion': text,
                'original_question': resp['original_message']['content'],
                'activation': activation,
            })

    if not examples_for_category:
        return []

    total_needed = n_examples
    if not use_activation_perplexity_selection:
        if len(examples_for_category) > total_needed:
            selected = random.sample(examples_for_category, total_needed)
        else:
            selected = examples_for_category.copy()
        random.shuffle(selected)
        return selected[:total_needed]

    # Activation pre-filter then perplexity ranking
    sorted_by_act = sorted(examples_for_category, key=lambda x: x['activation'], reverse=True)
    sample_size = min(len(sorted_by_act), total_needed * 16)
    sampled = sorted_by_act[:sample_size]

    results = []
    for ex in tqdm(sampled, desc=f"Perplexity: {category_name}"):
        try:
            full_text = ex['prompt'] + ex['target_completion']
            inputs = tokenizer(full_text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs.to(model.device))
                logits = outputs.logits[0]
                prompt_len = len(tokenizer(ex['prompt'], return_tensors='pt')['input_ids'][0])
                shift_logits = logits[:-1, :].contiguous()
                shift_labels = inputs['input_ids'][0][1:].contiguous()
                tgt_start = prompt_len - 1
                tgt_end = len(shift_labels)
                if tgt_start >= tgt_end or tgt_start < 0:
                    continue
                tgt_logits = shift_logits[tgt_start:tgt_end, :]
                tgt_labels = shift_labels[tgt_start:tgt_end]
                if tgt_logits.size(0) == 0:
                    continue
                loss = torch.nn.functional.cross_entropy(tgt_logits, tgt_labels, reduction='mean')
                ppl = torch.exp(loss).item()
                results.append({**ex, 'perplexity': ppl})
        except Exception:
            continue

    if not results:
        if len(examples_for_category) > total_needed:
            return random.sample(examples_for_category, total_needed)
        return examples_for_category

    top = sorted(results, key=lambda x: x['perplexity'], reverse=True)[:total_needed]
    for ex in top:
        ex.pop('perplexity', None)
        ex.pop('activation', None)
    return top


def generate_base_completion(model, tokenizer, prompt: str, max_new_tokens: int):
    tokens = model.generate(
        **tokenizer(prompt, return_tensors='pt').to(model.device),
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        suppress_tokens=[tokenizer.eos_token_id]
    )
    full_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return full_text[len(prompt):]


def main():
    parser = argparse.ArgumentParser(description="Inspect training examples per category with base model completions")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B", help="HF model to load for completions")
    parser.add_argument("--n_per_category", type=int, default=16, help="Number of examples to display per category")
    parser.add_argument("--category_idx", type=str, default=None, help="Comma-separated indices of categories to inspect; if omitted, inspect all")
    parser.add_argument("--context_sentences", type=int, default=0, help="Additional sentences after target to include in target_completion")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens for base completion generation")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Load model in 8-bit mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_activation_perplexity_selection", action="store_true", default=True, help="Use activation+perplexity selection instead of random sampling")
    parser.add_argument("--save_json_path", type=str, default="./results/vars/raw_training_examples.json", help="Optional path to save a JSON dump of inspected examples. Defaults to results/vars/inspections/<auto>.json")
    parser.add_argument("--layer", type=int, default=6, help="Layer index to load category titles/descriptions (if available)")
    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        dtype=torch.bfloat16,
    )
    torch.set_default_device(model.device)
    for p in model.parameters():
        p.requires_grad = False

    # Map to the correct responses files
    model_name_short = args.model.split('/')[-1].lower()
    thinking_model_name = utils.model_mapping.get(args.model, model_name_short)
    if thinking_model_name is None:
        thinking_model_name = model_name_short
    thinking_model_short = thinking_model_name.split('/')[-1].lower()

    responses_json_path = f"../generate-responses/results/vars/responses_{thinking_model_short}.json"
    annotated_responses_json_path = f"../generate-responses/results/vars/annotated_responses_{thinking_model_short}.json"

    if not os.path.exists(responses_json_path):
        raise FileNotFoundError(f"Responses file not found at {responses_json_path}")
    if not os.path.exists(annotated_responses_json_path):
        raise FileNotFoundError(f"Annotated responses file not found at {annotated_responses_json_path}")

    print(f"Loading responses from {responses_json_path}")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)
    print(f"Loading annotations from {annotated_responses_json_path}")
    with open(annotated_responses_json_path, 'r') as f:
        annotated_responses_data = json.load(f)

    # Merge by index with validation
    valid_responses = []
    for i, resp in enumerate(responses_data):
        if i < len(annotated_responses_data):
            ann = annotated_responses_data[i]
            if (
                resp['question_id'] == ann.get('question_id') and
                resp['dataset_name'] == ann.get('dataset_name') and
                ann.get('annotated_thinking')
            ):
                merged = resp.copy()
                merged['annotated_thinking'] = ann['annotated_thinking']
                valid_responses.append(merged)

    print(f"Found {len(valid_responses)} responses with annotations out of {len(responses_data)} total responses")

    categories = get_sorted_categories(valid_responses)
    if not categories:
        print("No categories found in annotations.")
        return

    print("\nAvailable categories with indices:")
    for idx, cat in enumerate(categories):
        print(f"  [{idx}] {cat}")

    target_indices = None
    if args.category_idx:
        target_indices = [int(x.strip()) for x in args.category_idx.split(',') if x.strip()]
        for ti in target_indices:
            if ti < 0 or ti >= len(categories):
                raise ValueError(f"Invalid category index {ti}; must be in [0, {len(categories)-1}]")

    chosen = [(i, c) for i, c in enumerate(categories) if (target_indices is None or i in target_indices)]

    # Prepare output aggregation for JSON
    inspection_output = {
        "model": args.model,
        "context_sentences": args.context_sentences,
        "max_new_tokens": args.max_new_tokens,
        "use_activation_perplexity_selection": args.use_activation_perplexity_selection,
        "n_per_category": args.n_per_category,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "categories": []
    }

    # Attempt to load metadata (title/description) for categories if available
    # Convert model short to the ID used in results filenames (replace '-' with '_')
    model_id_for_meta = thinking_model_short.replace("-", "_")
    category_meta_by_idx = load_category_metadata(model_id_for_meta, args.layer)

    for idx, category in chosen:
        # Resolve title/description first so we can include them inside the header block
        cat_idx_num = None
        if isinstance(category, str) and category.startswith('idx'):
            try:
                cat_idx_num = int(category[3:])
            except Exception:
                cat_idx_num = None
        meta_title = None
        meta_desc = None
        if cat_idx_num is not None and cat_idx_num in category_meta_by_idx:
            meta_title = category_meta_by_idx[cat_idx_num].get("title")
            meta_desc = category_meta_by_idx[cat_idx_num].get("description")

        # Header with title/description inline if available
        print("\n" + "=" * 80)
        if meta_title or meta_desc:
            title_inline = f" — {meta_title}" if meta_title else ""
            print(f"Category [{idx}]: {category}{title_inline}")
            if meta_desc:
                print(meta_desc)
        else:
            print(f"Category [{idx}]: {category}")
        print("=" * 80)

        examples = extract_examples_for_category(
            valid_responses,
            category,
            tokenizer,
            args.n_per_category,
            model,
            context_sentences=args.context_sentences,
            use_activation_perplexity_selection=args.use_activation_perplexity_selection,
        )

        if not examples:
            print("No examples found after filtering.")
            inspection_output["categories"].append({
                "category_idx": idx,
                "category_name": category,
                "num_examples": 0,
                "examples": []
            })
            continue

        cat_entry = {
            "category_idx": idx,
            "category_name": category,
            "category_title": meta_title,
            "category_description": meta_desc,
            "num_examples": len(examples),
            "examples": []
        }

        for j, ex in enumerate(examples, start=1):
            print("-" * 80)
            print(f"Example {j}/{len(examples)}")
            print("-- Context (prompt) --")
            print(ex['prompt'])
            print("\n-- Target completion --")
            print(ex['target_completion'])
            print("\n-- Base model completion --")
            try:
                gen = generate_base_completion(model, tokenizer, ex['prompt'], args.max_new_tokens)
                print(gen)
            except Exception as e:
                print(f"[Generation error] {e}")
                gen = None
            print()

            cat_entry["examples"].append({
                "original_question": ex.get('original_question'),
                "prompt": ex['prompt'],
                "target_completion": ex['target_completion'],
                "base_completion": gen,
            })

        inspection_output["categories"].append(cat_entry)

    # Save JSON output if requested (or to default path)
    try:
        out_path = args.save_json_path
        if out_path is None:
            out_dir = os.path.join("results", "vars", "inspections")
            os.makedirs(out_dir, exist_ok=True)
            model_short = args.model.split('/')[-1].lower()
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"inspection_{model_short}_{ts}.json")
        else:
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

        with open(out_path, 'w') as f:
            json.dump(inspection_output, f, indent=2)
        print(f"\nSaved inspection JSON to: {out_path}")
    except Exception as e:
        print(f"\n[Warning] Failed to save JSON output: {e}")


if __name__ == "__main__":
    main()


# %%
