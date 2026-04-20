# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
import random
import json
import sys
import os
import pickle
# Add the parent directory to the path to ensure we import from the correct utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
import gc
from utils import steering_opt
from tqdm import tqdm
import traceback
import wandb
from utils.responses import extract_thinking_process

# %% Parse arguments
parser = argparse.ArgumentParser(description="Optimize steering vectors from annotated responses")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model to train steering vectors for")
parser.add_argument("--thinking_model", type=str, default=None,
                    help="Model to train steering vectors on")
parser.add_argument("--n_training_examples", type=int, default=8,
                    help="Number of training examples to use per category")
parser.add_argument("--n_eval_examples", type=int, default=0,
                    help="Number of evaluation examples to use per category (0 disables eval)")
parser.add_argument("--save_path", type=str, default="results/vars/optimized_vectors",
                    help="Path to save optimized vectors")
parser.add_argument("--layer", type=int, default=6,
                    help="Layer to optimise the steering vector for")
parser.add_argument("--max_iters", type=int, required=True,
                    help="Maximum optimization iterations")
parser.add_argument("--lr", type=str, default="1e-1",
                    help="Learning rate(s) for optimization. Can be a single value or comma-separated list")
parser.add_argument("--min_lr", type=float, default=0,
                    help="Minimum learning rate for optimization")
parser.add_argument("--warmup_iters", type=int, default=0,
                    help="Number of warmup iterations")
parser.add_argument("--context_sentences", type=int, default=0,
                    help="Number of additional sentences to include after target completion")
parser.add_argument("--test_max_tokens", type=int, default=30,
                    help="Maximum tokens to generate when testing")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--optim_minibatch_size", type=int, default=6,
                    help="Size of minibatches for optimization loop")
parser.add_argument("--base_gen_minibatch_size", type=int, default=6,
                    help="Size of minibatches for base completion generation")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--steering_vector_idx", type=int, default=0,
                    help="Index of the specific steering vector to optimize")
parser.add_argument("--grad_clip", type=float, default=None,
                    help="Maximum L2 norm of gradients for gradient clipping. None means no clipping.")
parser.add_argument("--steering_token_window", type=int, default=None,
                    help="Number of previous tokens in the target completion to apply the steering vector to (None means all)")
parser.add_argument("--use_wandb", action="store_true", default=False,
                    help="Use wandb for logging")
parser.add_argument("--wandb_project", type=str, default="optimize-steering-vectors",
                    help="Wandb project name")
parser.add_argument("--use_activation_perplexity_selection", action="store_true", default=False,
                    help="If set, use activation/perplexity-based selection. If not set, use random sampling over the full set.")
parser.add_argument("--use_synthetic_examples", action="store_true", default=False,
                    help="If set, use synthetic training examples from synthetic_training_examples.json instead of generated responses")
parser.add_argument("--steering_type", type=str, choices=["linear", "adaptive_linear", "resid_lora"], default="linear",
                    help="Type of steering to optimize: 'linear' steering vector, 'adaptive_linear' MLP-gated vector, or 'resid_lora' low-rank adapter")
parser.add_argument("--lora_rank", type=int, default=1,
                    help="Rank r for LoRA adapter when --steering_type=lora")
parser.add_argument("--adaptive_hidden", type=int, default=128,
                    help="Hidden dimension for adaptive_linear MLP gate")
args, _ = parser.parse_known_args()

# At module level
ANNOTATION_PATTERN = re.compile(r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)
CATEGORY_PATTERN = re.compile(r'\["[\d.]+:(\S+?)"\]')

def get_label_positions(annotated_thinking, response_text, tokenizer, context_sentences=0):
    """Parse SAE annotations and find token positions for each label"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments in the format [activation:category-name] text [end-section]
    # Now supporting activation strength values in the format [56.86:category-name]
    matches = list(ANNOTATION_PATTERN.finditer(annotated_thinking))
    
    # Create character to token mapping once
    char_to_token = utils.get_char_to_token_map(response_text, tokenizer)
    
    # Split response into sentences for context
    sentences = utils.split_into_sentences(response_text, min_words=0)
    
    for match in matches[:-1]:
        activation_str = match.group(1).strip()
        label = match.group(2).strip()
        text = match.group(3).strip()
        
        try:
            activation = float(activation_str)
        except ValueError:
            print(f"Warning: Could not parse activation value '{activation_str}' for category '{label}'")
            continue
            
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        pattern = r'(?:[.?!;\n]|\n\n)\s*(' + re.escape(text) + ')'
        match = re.search(pattern, response_text)
        text_pos = match.start(1) if match else -1
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            
            # Adjust token_end to include the entire token
            if token_end is not None:
                token_end += 1

            if token_start is None or token_end is None or token_start >= token_end:
                continue
            
            # Find the sentence containing our target text
            target_sentence_idx = -1
            for i, sentence in enumerate(sentences):
                if text in sentence:
                    target_sentence_idx = i
                    break
            
            if target_sentence_idx == -1:
                continue
                
            # Get additional context sentences if requested
            additional_context = ""
            if context_sentences > 0 and target_sentence_idx < len(sentences) - 1:
                end_idx = min(target_sentence_idx + context_sentences + 1, len(sentences))
                additional_sentences = sentences[target_sentence_idx + 1:end_idx]
                
                # Find the original whitespace between sentences
                if additional_sentences:
                    # Get the text up to the end of our target text
                    text_end_pos = text_pos + len(text)
                    # Get the text up to the start of the next sentence
                    next_sentence_start = response_text.find(additional_sentences[0], text_end_pos)
                    if next_sentence_start > text_end_pos:
                        # Extract the original whitespace
                        original_whitespace = response_text[text_end_pos:next_sentence_start]
                        # Use the original whitespace to join sentences
                        additional_context = original_whitespace + original_whitespace.join(additional_sentences)
                    else:
                        # Fallback to space if we can't find the original whitespace
                        additional_context = " " + " ".join(additional_sentences)
                
                # Update token_end to include additional context
                if additional_context:
                    context_end_pos = text_pos + len(text) + len(additional_context)
                    context_token_end = char_to_token.get(context_end_pos - 1, None)
                    if context_token_end is not None:
                        token_end = context_token_end + 1
            
            # If we found valid token positions
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end, text + additional_context, activation, text_pos))
    
    return label_positions

def extract_examples_for_category(responses_data, category_name, tokenizer, n_training_examples, n_eval_examples, model, model_name_short):
    """Extract training and evaluation examples for `category_name`.

    Returns (training_examples, eval_examples, max_activation).
    If `--use_activation_perplexity_selection` is off, we random-sample for train and then
    sample eval from the remaining pool. If on, we activation pre-filter, compute perplexities,
    then take top-K where K = train + eval; the first N train, next M eval.
    """
    examples_for_category = []
    
    # Process each response to extract labeled segments for the specified category
    n_annotated_thinking_containing_category = 0
    for resp in tqdm(responses_data, desc="Extracting examples for category"):
        if not resp.get('annotated_thinking'):
            continue

        thinking_process = extract_thinking_process(resp["full_response"])

        full_text = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{resp['original_message']['content']}\n\nStep by step answer:\n{thinking_process}"
        
        # Look for the specific category in the annotated thinking
        if category_name not in resp['annotated_thinking']:
            continue
            
        n_annotated_thinking_containing_category += 1
        label_positions = get_label_positions(resp['annotated_thinking'], full_text, tokenizer, args.context_sentences)
        
        if category_name in label_positions:
            for start, end, text, activation, text_pos in label_positions[category_name]:
                # Get the text up to this point using the saved text_pos
                context = full_text[:text_pos]

                # Check if context ends properly
                if context[-1] not in ['.', '?', '!', ';', '\n', '\n\n'] and context[-2] not in ['.', '?', '!', ';', '\n', '\n\n'] and context.strip()[-1] not in ['.', '?', '!', ';', '\n', '\n\n']:
                    continue
                
                # Filter out target sequences with fewer than 3 words
                word_count = len(text.strip().split())
                if word_count < 7:
                    continue
                
                examples_for_category.append({
                    'prompt': context,
                    'target_completion': text,
                    'original_question': resp['original_message']['content'],
                    'full_thinking': extract_thinking_process(resp["full_response"]),
                    'activation': activation
                })
    
    if not examples_for_category:
        print(f"No valid examples found for category {category_name}. Exiting.")
        return [], 0.0

    print(f"Found {n_annotated_thinking_containing_category} annotated thinking containing category {category_name}")
    print(f"Found {len(examples_for_category)} examples for category {category_name}")

    # Total needed to cover both train and eval
    total_examples_needed = n_training_examples + n_eval_examples

    if not args.use_activation_perplexity_selection:
        # Random sampling separately for train and eval (no overlap)
        pool = examples_for_category.copy()
        random.shuffle(pool)
        training_examples = pool[:min(n_training_examples, len(pool))]
        remaining = pool[len(training_examples):]
        eval_examples = remaining[:min(n_eval_examples, len(remaining))]
        print(f"Final selection (random): {len(training_examples)} training examples, {len(eval_examples)} eval examples")
        return training_examples, eval_examples, 1.0

    # --- CORRECTED LOGIC: Activation pre-filter, then select by perplexity ---
    # 1. Sort all examples by activation (descending)
    examples_for_category_sorted = sorted(examples_for_category, key=lambda x: x['activation'], reverse=True)
    # 2. Take the top 4x the data needed
    sample_size = min(len(examples_for_category_sorted), max(1, total_examples_needed) * 4)
    sampled_examples = examples_for_category_sorted[:sample_size]

    print(f"Top {sample_size} examples by activation selected for perplexity ranking")

    # 3. Calculate perplexity for each of these examples, with caching by steering index
    vector_id_base = "bias" if args.steering_vector_idx == -1 else f"idx{args.steering_vector_idx}"
    cache_dir = "results/vars/perplexity"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/perplexities_{model_name_short}_{vector_id_base}.pkl"

    examples_with_metrics = []
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            examples_with_metrics = pickle.load(f)
        assert isinstance(examples_with_metrics, list) and all('perplexity' in ex for ex in examples_with_metrics), "Cached perplexities invalid format"
        assert len(examples_with_metrics) >= sample_size, "Cached perplexities insufficient; delete the cache file to recompute"
        print(f"Loaded cached perplexities from {cache_path}")

    if not examples_with_metrics:
        print("Calculating perplexity for sampled examples...")
        for example in tqdm(sampled_examples, desc="Computing perplexity"):
            try:
                # Tokenize the full sequence (prompt + target completion)
                full_text = example['prompt'] + example['target_completion']
                inputs = tokenizer(full_text, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0]  # Remove batch dimension
                    prompt_tokens = tokenizer(example['prompt'], return_tensors='pt')['input_ids'][0]
                    prompt_len = len(prompt_tokens)
                    shift_logits = logits[:-1, :].contiguous()
                    shift_labels = inputs['input_ids'][0][1:].contiguous()
                    target_start = prompt_len - 1
                    target_end = len(shift_labels)
                    if target_start >= target_end or target_start < 0:
                        continue
                    target_logits = shift_logits[target_start:target_end, :]
                    target_labels = shift_labels[target_start:target_end]
                    if target_logits.size(0) == 0:
                        continue
                    loss = torch.nn.functional.cross_entropy(target_logits, target_labels, reduction='mean')
                    perplexity = torch.exp(loss).item()
                    examples_with_metrics.append({
                        **example,
                        'perplexity': perplexity,
                        'activation': example['activation']
                    })
            except Exception as e:
                print(f"Error calculating perplexity for example: {e}")
                continue
        with open(cache_path, "wb") as f:
            pickle.dump(examples_with_metrics, f)
        print(f"Saved perplexities to {cache_path}")

    if not examples_with_metrics:
        print("No valid examples with perplexity found. Falling back to random sampling.")
        # Fall back to random sampling if perplexity calculation fails
        pool = examples_for_category.copy()
        random.shuffle(pool)
        training_examples = pool[:min(n_training_examples, len(pool))]
        remaining = pool[len(training_examples):]
        eval_examples = remaining[:min(n_eval_examples, len(remaining))]
        return training_examples, eval_examples, 1.0

    # 4. From the top N (train + test) by perplexity (descending), split into train and test
    final_examples = sorted(examples_with_metrics, key=lambda x: x['perplexity'], reverse=True)[:max(0, total_examples_needed)]

    # 5. Split into train and eval (no overlap)
    training_examples = final_examples[:min(n_training_examples, len(final_examples))]
    eval_examples = final_examples[len(training_examples):len(training_examples) + n_eval_examples]

    print(f"Final selection (perplexity): {len(training_examples)} training examples, {len(eval_examples)} eval examples")

    # Calculate max activation before removing the field
    max_activation = max(ex['activation'] for ex in final_examples) if final_examples else 1.0

    # Remove perplexity/activation from the selected examples only
    for ex in training_examples:
        ex.pop('perplexity', None)
        ex.pop('activation', None)
    for ex in eval_examples:
        ex.pop('perplexity', None)
        ex.pop('activation', None)

    return training_examples, eval_examples, max_activation

def generate_bias_examples(responses_data, tokenizer, model, n_training_examples, n_eval_examples, model_name_short):
    """Generate examples for the *bias* vector.
    Each example uses the *entire* thinking process as the target completion, with the task + question
    (everything *before* the reasoning) as the prompt.

    Selection logic mirrors `extract_examples_for_category` except that we have no activation scores,
    so we optionally rank purely by perplexity.
    """
    all_examples = []

    for resp in tqdm(responses_data, desc="Preparing bias examples"):
        if not resp.get('thinking_process'):
            continue

        # Build prompt exactly as in other parts of the script
        prompt = (
            f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{resp['original_message']['content']}\n\nStep by step answer:\n"
        )
        target_completion = resp['thinking_process'].strip()

        # Simple quality filter – skip very short reasoning chains
        if len(target_completion.split()) < 20:
            continue

        all_examples.append({
            'prompt': prompt,
            'target_completion': target_completion,
            'original_question': resp['original_message']['content'],
        })

    if not all_examples:
        print("No valid bias examples found. Exiting.")
        return []

    print(f"Collected {len(all_examples)} candidate bias examples")

    total_examples_needed = n_training_examples

    # If the user disabled perplexity selection, just random-sample
    if not args.use_activation_perplexity_selection:
        pool = all_examples.copy()
        random.shuffle(pool)
        train_sel = pool[:min(n_training_examples, len(pool))]
        remaining = pool[len(train_sel):]
        eval_sel = remaining[:min(n_eval_examples, len(remaining))]
        print(f"Selected {len(train_sel)} random training and {len(eval_sel)} eval examples (no perplexity ranking)")
        return train_sel, eval_sel

    # Otherwise: random pre-sample 2×, then rank by perplexity
    sample_size = min(len(all_examples), max(1, total_examples_needed) * 4)
    presampled = random.sample(all_examples, sample_size)

    print(f"Presampled {sample_size} examples for perplexity ranking; computing perplexities…")
    examples_with_metrics = []

    # Caching for bias perplexities as well (steering idx -1)
    vector_id_base = "bias" if args.steering_vector_idx == -1 else f"idx{args.steering_vector_idx}"
    cache_dir = "results/vars/perplexity"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/perplexities_{model_name_short}_{vector_id_base}.pkl"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            examples_with_metrics = pickle.load(f)
        assert isinstance(examples_with_metrics, list) and all('perplexity' in ex for ex in examples_with_metrics), "Cached perplexities invalid format"
        assert len(examples_with_metrics) >= sample_size, "Cached perplexities insufficient; delete the cache file to recompute"
        print(f"Loaded cached perplexities from {cache_path}")

    if not examples_with_metrics:
        for ex in tqdm(presampled, desc="Computing perplexity"):
            try:
                full_txt = ex['prompt'] + ex['target_completion']
                inputs = tokenizer(full_txt, return_tensors='pt')
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
                    examples_with_metrics.append({**ex, 'perplexity': ppl})
            except Exception as e:
                print(f"Error computing perplexity: {e}")
                continue
        with open(cache_path, "wb") as f:
            pickle.dump(examples_with_metrics, f)
        print(f"Saved perplexities to {cache_path}")

    if not examples_with_metrics:
        print("Perplexity computation failed for all examples; falling back to random selection.")
        pool = all_examples.copy()
        random.shuffle(pool)
        train_sel = pool[:min(n_training_examples, len(pool))]
        remaining = pool[len(train_sel):]
        eval_sel = remaining[:min(n_eval_examples, len(remaining))]
        return train_sel, eval_sel

    # Sort by perplexity descending and take top N
    final_examples = sorted(examples_with_metrics, key=lambda x: x['perplexity'], reverse=True)[:max(0, total_examples_needed)]
    train_sel = final_examples[:min(n_training_examples, len(final_examples))]
    eval_sel = final_examples[len(train_sel):len(train_sel) + n_eval_examples]
    for ex in train_sel:
        ex.pop('perplexity', None)
    for ex in eval_sel:
        ex.pop('perplexity', None)
    print(f"Selected {len(train_sel)} training and {len(eval_sel)} eval examples after perplexity ranking")
    return train_sel, eval_sel

def get_sorted_categories(responses_data):
    """Extract all unique categories from responses data and return them in order by idx number"""
    categories = set()
    
    for resp in responses_data:
        if not resp.get('annotated_thinking'):
            continue
            
        # Extract category names from annotated thinking (now idx<number> format)
        matches = CATEGORY_PATTERN.finditer(resp['annotated_thinking'])
        for match in matches:
            category = match.group(1).strip()
            categories.add(category)
    
    # Sort by numerical index extracted from idx<number> format
    def extract_idx(category_name):
        if category_name.startswith('idx'):
            try:
                return int(category_name[3:])  # Extract number after 'idx'
            except ValueError:
                return float('inf')  # Put invalid formats at the end
        return float('inf')  # Put non-idx formats at the end
    
    return sorted(list(categories), key=extract_idx)

def test_on_example(model, tokenizer, vector, layer, test_example, max_new_tokens=50, steering_token_window=None, additional_vectors=None, steering_type: str = "linear"):
    """Test the optimized vector on an example.

    Parameters
    ----------
    vector : torch.Tensor
        The *trainable* category vector we just optimised.
    additional_vectors : list[torch.Tensor] | None
        Extra *static* vectors (e.g. the bias vector) to attach alongside the main one.
    """

    if additional_vectors is None:
        additional_vectors = []

    prompt = test_example['prompt']
    target_completion = test_example['target_completion']
    
    print("\n--- Testing on example ---")
    print(f"Prompt: ...{prompt[-100:]}")
    print("\n--- Generated Completions ---")
    
    # Generate without steering
    generated_tokens = model.generate(
        **tokenizer(prompt, return_tensors='pt').to(model.device), 
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        suppress_tokens=[tokenizer.eos_token_id]
    )
    unsteered_completion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    unsteered_generated_part = unsteered_completion[len(prompt):]
    print("UNSTEERED:")
    print(unsteered_generated_part)
    print()
    
    # Figure out the correct slice for steering
    prompt_tok = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
    target_tok = tokenizer(target_completion, return_tensors='pt')['input_ids'][0]
    prompt_len = len(prompt_tok)
    target_len = len(target_tok)
    if steering_token_window is None:
        steering_start = prompt_len
    else:
        steering_start = prompt_len + max(0, target_len - steering_token_window)
    steering_token_slice = slice(steering_start, None)
    
    # Generate with steering via batch hooks (single-row slices)

    hooks = []
    slices = [steering_token_slice]  # single example generation → one row slice
    if steering_type == "linear":
        # primary vector
        hook = steering_opt.make_batch_linear_hook(vector, slices, static_vectors=[], projection_clamp=False)
        hooks.append((layer, hook))
        # additional static vectors, if any (same slice)
        if additional_vectors:
            static_vecs = [v for v in additional_vectors if isinstance(v, torch.Tensor)]
            if static_vecs:
                hooks.append((layer, steering_opt.make_batch_linear_hook(torch.zeros_like(vector), slices, static_vecs, projection_clamp=False)))
    elif steering_type == "adaptive_linear":
        assert isinstance(vector, dict) and all(k in vector for k in ("vector","W1","b1","W2","b2")), "adaptive_linear expects dict with vector,W1,b1,W2,b2"
        hook = steering_opt.make_batch_adaptive_linear_hook(vector['vector'], vector['W1'], vector['b1'], vector['W2'], vector['b2'], slices, static_vectors=[], projection_clamp=False)
        hooks.append((layer, hook))
        if additional_vectors:
            static_vecs = [v for v in additional_vectors if isinstance(v, torch.Tensor)]
            if static_vecs:
                # Attach extra static linear vectors alongside adaptive steering
                hooks.append((layer, steering_opt.make_batch_linear_hook(torch.zeros_like(vector['vector']), slices, static_vecs, projection_clamp=False)))
    elif steering_type == "resid_lora":
        assert isinstance(vector, dict) and 'A' in vector and 'B' in vector and 'alpha' in vector, "LoRA steering expects a dict with 'A', 'B', and 'alpha'"
        static_loras = [v for v in (additional_vectors or []) if isinstance(v, dict)]
        hook = steering_opt.make_batch_resid_lora_hook(vector['A'], vector['B'], vector['alpha'], slices, static_loras)
        hooks.append((layer, hook))
    else:
        raise ValueError(f"Unknown steering_type: {steering_type}")

    with steering_opt.hf_hooks_contextmanager(model, hooks): 
        generated_tokens = model.generate(
            **tokenizer(prompt, return_tensors='pt').to(model.device), 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        steered_completion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        steered_generated_part = steered_completion[len(prompt):]
        print("STEERED:")
        print(steered_generated_part)
    
    print("\nTARGET:")
    print(test_example['target_completion'])
    print("\n" + "-"*50 + "\n")
    
    return unsteered_completion, steered_completion

def save_hyperparameters(hyperparams, model_name_short, steering_vector_idx, category):
    """Save hyperparameters for a specific steering vector for reproducibility"""
    hp_dir = "results/vars/hyperparams"
    os.makedirs(hp_dir, exist_ok=True)
    
    hp_file = f"{hp_dir}/steering_vector_hyperparams_{model_name_short}_{steering_vector_idx}.json"

    result = {
        "category": category,
        "hyperparameters": hyperparams
    }
    with open(hp_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved hyperparameters to {hp_file}")

def load_synthetic_training_examples(category_name, n_training_examples, n_eval_examples):
    """Load synthetic training and evaluation examples for a specific category.
    Returns (training_examples, eval_examples) in the same format as extract_examples_for_category.
    """
    synthetic_file_path = "results/vars/synthetic_training_examples.json"
    
    if not os.path.exists(synthetic_file_path):
        raise FileNotFoundError(f"Synthetic training examples file not found at {synthetic_file_path}")
    
    with open(synthetic_file_path, 'r') as f:
        synthetic_data = json.load(f)
    
    # Find the category in the synthetic data
    category_examples = None
    for category_data in synthetic_data.get("synthetic_training_examples", []):
        if category_data.get("category_name") == category_name:
            category_examples = category_data
            break
    
    if not category_examples:
        raise ValueError(f"Category '{category_name}' not found in synthetic training examples")
    
    # Convert synthetic examples to the expected format
    training_examples = []
    eval_examples = []
    examples = category_examples.get("examples", [])
    
    # Randomly sample train and eval without overlap
    pool = examples.copy()
    random.shuffle(pool)
    selected_train = pool[:min(n_training_examples, len(pool))]
    remaining = pool[len(selected_train):]
    selected_eval = remaining[:min(n_eval_examples, len(remaining))]

    for example in selected_train:
        # Create the prompt format that matches what the model expects
        prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{example['task']}\n\nStep by step answer:\n{example['context']}"
        
        training_examples.append({
            'prompt': prompt,
            'target_completion': example['target_completion'],
            'original_question': example['task'],
            'full_thinking': example['context'],
            'activation': 1.0  # Default activation for synthetic examples
        })
    for example in selected_eval:
        prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{example['task']}\n\nStep by step answer:\n{example['context']}"
        eval_examples.append({
            'prompt': prompt,
            'target_completion': example['target_completion'],
            'original_question': example['task'],
            'full_thinking': example['context'],
            'activation': 1.0
        })
    
    print(f"Loaded {len(training_examples)} synthetic training and {len(eval_examples)} eval examples for category '{category_name}'")
    return training_examples, eval_examples

def generate_synthetic_bias_examples(n_training_examples, n_eval_examples):
    """Generate synthetic bias train/eval examples by combining examples from all categories."""
    synthetic_file_path = "results/vars/synthetic_training_examples.json"
    
    if not os.path.exists(synthetic_file_path):
        raise FileNotFoundError(f"Synthetic training examples file not found at {synthetic_file_path}")
    
    with open(synthetic_file_path, 'r') as f:
        synthetic_data = json.load(f)
    
    all_examples = []
    
    # Collect examples from all categories
    for category_data in synthetic_data.get("synthetic_training_examples", []):
        examples = category_data.get("examples", [])
        for example in examples:
            # Create the prompt format that matches what the model expects
            prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{example['task']}\n\nStep by step answer:\n{example['context']}"
            
            all_examples.append({
                'prompt': prompt,
                'target_completion': example['target_completion'],
                'original_question': example['task'],
            })
    
    if not all_examples:
        raise ValueError("No synthetic examples found in the file")
    
    print(f"Collected {len(all_examples)} candidate synthetic bias examples")
    
    # Randomly sample train and eval without overlap
    pool = all_examples.copy()
    random.shuffle(pool)
    selected_train = pool[:min(n_training_examples, len(pool))]
    remaining = pool[len(selected_train):]
    selected_eval = remaining[:min(n_eval_examples, len(remaining))]
    print(f"Selected {len(selected_train)} synthetic bias training and {len(selected_eval)} eval examples")
    return selected_train, selected_eval

def main():
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse learning rates
    try:
        learning_rates = [float(lr.strip()) for lr in args.lr.split(',')]
    except ValueError:
        raise ValueError("Learning rates must be comma-separated numbers")
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        dtype=torch.bfloat16
    )

    torch.set_default_device(model.device)

    for param in model.parameters():
        param.requires_grad = False

    # Default responses path
    model_name_short = args.model.split('/')[-1].lower()
    
    if args.thinking_model is None:
        thinking_model_name = utils.model_mapping[args.model]
        thinking_model_short = thinking_model_name.split('/')[-1].lower()
    else:
        thinking_model_name = args.thinking_model
        thinking_model_short = thinking_model_name.split('/')[-1].lower()
        # Explicit thinking model for this model
        model_name_short = f"{model_name_short}-on-{thinking_model_short}"
    
    if args.use_synthetic_examples:
        print("Using synthetic training examples - skipping response loading")
        valid_responses = []
        # For synthetic examples, we need to get categories from the synthetic data
        synthetic_file_path = "results/vars/synthetic_training_examples.json"
        if not os.path.exists(synthetic_file_path):
            raise FileNotFoundError(f"Synthetic training examples file not found at {synthetic_file_path}")
        
        with open(synthetic_file_path, 'r') as f:
            synthetic_data = json.load(f)
        
        # Extract category names from synthetic data
        all_categories = [cat_data.get("category_name") for cat_data in synthetic_data.get("synthetic_training_examples", [])]
        print(f"Found {len(all_categories)} categories in synthetic data: {all_categories}")
    else:
        responses_json_path = f"../generate-responses/results/vars/responses_{thinking_model_short}.json"
        annotated_responses_json_path = f"../generate-responses/results/vars/annotated_responses_{thinking_model_short}.json"
        
        if not os.path.exists(responses_json_path):
            raise FileNotFoundError(f"Annotated responses file not found at {responses_json_path}. Please annotate responses first.")
        
        # Load responses
        print(f"Loading annotated responses from {responses_json_path}")
        with open(responses_json_path, 'r') as f:
            responses_data = json.load(f)

        # Load annotated responses
        print(f"Loading annotated responses from {annotated_responses_json_path}")
        with open(annotated_responses_json_path, 'r') as f:
            annotated_responses_data = json.load(f)
        
        # Match responses with their annotations by index and validate
        valid_responses = []
        for i, resp in enumerate(responses_data):
            if i < len(annotated_responses_data):
                annotated_resp = annotated_responses_data[i]
                # Verify that the responses match by question_id and dataset_name
                if (resp.get('question_id') == annotated_resp.get('question_id') and 
                    resp.get('dataset_name') == annotated_resp.get('dataset_name') and
                    annotated_resp.get('annotated_thinking')):
                    # Create merged response with annotated_thinking
                    merged_resp = resp.copy()
                    merged_resp['annotated_thinking'] = annotated_resp['annotated_thinking']
                    merged_resp['thinking_process'] = extract_thinking_process(resp["full_response"])
                    valid_responses.append(merged_resp)
                else:
                    q_id = resp.get('question_id')
                    ann_q_id = annotated_resp.get('question_id')
                    if q_id != ann_q_id:
                        print(f"Warning: Mismatch at index {i} - question_id mismatch: response={q_id}, annotated={ann_q_id}")
                    elif resp.get('dataset_name') != annotated_resp.get('dataset_name'):
                        print(f"Warning: Mismatch at index {i} (q_id: {q_id}) - dataset_name mismatch: response='{resp.get('dataset_name')}', annotated='{annotated_resp.get('dataset_name')}'")
                    elif not annotated_resp.get('annotated_thinking'):
                        print(f"Warning: Mismatch at index {i} (q_id: {q_id}) - missing 'annotated_thinking' in annotated response.")
                    else:
                        # Fallback for any other unexpected mismatch
                        print(f"Warning: Mismatch at index {i} - response question_id: {q_id}, annotated question_id: {ann_q_id}")
        
        print(f"Found {len(valid_responses)} responses with annotations out of {len(responses_data)} total responses")
        
        # Get all available categories sorted alphabetically
        all_categories = get_sorted_categories(valid_responses)
    
    # Print all available categories with their indices
    print("\nAvailable categories with indices:")
    for idx, category in enumerate(all_categories):
        print(f"  [{idx}] {category}")
    
    # Check if steering_vector_idx is valid
    if args.steering_vector_idx < -1 or args.steering_vector_idx >= len(all_categories):
        raise ValueError(f"Invalid steering_vector_idx: {args.steering_vector_idx}. Must be between -1 and {len(all_categories)-1}")
    
    bias_vector = None  # may be populated later
    bias_lora = None    # may be populated later

    # Handle general bias vector case
    if args.steering_vector_idx == -1:
        target_category = "bias"
        print("\nOptimizing general bias vector (full rollouts)")
        if args.use_synthetic_examples:
            print("Using synthetic examples for bias vector")
            training_examples, eval_examples = generate_synthetic_bias_examples(
                args.n_training_examples,
                args.n_eval_examples
            )
        else:
            training_examples, eval_examples = generate_bias_examples(
                valid_responses,
                tokenizer,
                model,
                args.n_training_examples,
                args.n_eval_examples,
                model_name_short
            )
        max_activation = 1.0  # Not used for bias vector
    else:
        # Get the target category name
        target_category = all_categories[args.steering_vector_idx]
        print(f"\nOptimizing vector for category: {target_category} (index {args.steering_vector_idx})")
        
        # Extract examples only for the target category
        if args.use_synthetic_examples:
            print(f"Using synthetic training examples for category {target_category}")
            training_examples, eval_examples = load_synthetic_training_examples(target_category, args.n_training_examples, args.n_eval_examples)
            max_activation = 1.0  # Default activation for synthetic examples
        else:
            print(f"Extracting examples for category {target_category}...")
            training_examples, eval_examples, max_activation = extract_examples_for_category(
                valid_responses, 
                target_category, 
                tokenizer,
                args.n_training_examples,
                args.n_eval_examples,
                model,
                model_name_short
            )

        # --- Load bias steering (vector or lora) to attach during optimisation ---
        bias_vector = None
        bias_lora = None
        bias_path = os.path.join(args.save_path, f"{model_name_short}_bias_{args.steering_type}.pt")
        if os.path.exists(bias_path):
            try:
                bias_dict = torch.load(bias_path, map_location=model.device)
                if args.steering_type == "linear":
                    if isinstance(bias_dict, dict):
                        bias_vector = bias_dict.get("bias", None) if "bias" in bias_dict else None
                    elif isinstance(bias_dict, torch.Tensor):
                        bias_vector = bias_dict
                elif args.steering_type == "adaptive_linear":
                    # bias for adaptive acts as additional static linear vector (optional)
                    if isinstance(bias_dict, dict) and "bias" in bias_dict and isinstance(bias_dict["bias"], torch.Tensor):
                        bias_vector = bias_dict["bias"]
                    elif isinstance(bias_dict, torch.Tensor):
                        bias_vector = bias_dict
                else:
                    if isinstance(bias_dict, dict):
                        candidate = bias_dict.get("bias", bias_dict)
                        if all(k in candidate for k in ("A","B","alpha")):
                            bias_lora = candidate
            except Exception as e:
                print(f"Warning: failed to load bias steering from {bias_path}: {e}")

    # Proceed even if no separate test examples are available.
        
    print(f"Found {len(training_examples)} training examples and {len(eval_examples) if 'eval_examples' in locals() else 0} eval examples")
    
    # Extract prompts and target completions for training
    train_prompts = [ex['prompt'] for ex in training_examples]
    train_target_completions = [ex['target_completion'] for ex in training_examples]
    
    # Evaluation dataset removed – optimisation now relies solely on training loss.
    # eval_prompts = [ex['prompt'] for ex in eval_examples] if 'eval_examples' in locals() and eval_examples else None
    # eval_target_completions = [ex['target_completion'] for ex in eval_examples] if 'eval_examples' in locals() and eval_examples else None
    eval_prompts = None
    eval_target_completions = None
    
    # Store results for each learning rate
    all_results = {}
    
    # Run optimization for each learning rate
    for lr in tqdm(learning_rates, desc="Optimizing with learning rates"):
        print(f"\nOptimizing with learning rate: {lr}")
        try:
            static_vecs = None
            if args.steering_type == "linear" and bias_vector is not None:
                static_vecs = [bias_vector]
            elif args.steering_type == "resid_lora" and bias_lora is not None:
                static_vecs = [bias_lora]

            params, loss_info = steering_opt.optimize_vector_simple(
                model,
                tokenizer,
                train_prompts,
                train_target_completions,
                args.layer,
                lr=lr,
                max_iters=args.max_iters,
                optim_minibatch_size=args.optim_minibatch_size,
                base_gen_minibatch_size=args.base_gen_minibatch_size,
                warmup_steps=args.warmup_iters,
                min_lr=args.min_lr,
                starting_norm=1,
                max_norm=None,
                grad_clip=args.grad_clip,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                return_info=True,
                return_loss_history=True,
                steering_token_window=args.steering_token_window,
                include_base_objective=False,
                wandb_run=None,
                static_vectors=static_vecs,
                steering_type=args.steering_type,
                rank=args.lora_rank,
                adaptive_hidden=args.adaptive_hidden,
                eval_prompts=eval_prompts,
                eval_target_completions=eval_target_completions
            )

            if args.steering_type == "linear":
                vect_to_store = params.detach().cpu()
            elif args.steering_type == "adaptive_linear":
                vect_to_store = {
                    'vector': params['vector'].detach().cpu(),
                    'W1': params['W1'].detach().cpu(),
                    'b1': params['b1'].detach().cpu(),
                    'W2': params['W2'].detach().cpu(),
                    'b2': params['b2'].detach().cpu(),
                    'hidden': args.adaptive_hidden,
                }
            else:
                vect_to_store = {
                    'A': params['A'].detach().cpu(),
                    'B': params['B'].detach().cpu(),
                    'alpha': params['alpha'].detach().cpu(),
                    'rank': args.lora_rank,
                }
            final_loss_val = loss_info['final_loss']
            
            # Store results
            if lr not in all_results:
                all_results[lr] = []
            
            all_results[lr].append({
                'vector': vect_to_store,
                'loss_info': loss_info,
                'final_loss': final_loss_val
            })
            
        except Exception as e:
            print(f"Error optimizing vector for category {target_category} with lr {lr}: {e}")
            traceback.print_exc()
            continue
    
    if not all_results:
        print("No successful optimization runs. Exiting.")
        return
    
    # Find best learning rate and run based on final loss
    best_lr = None
    best_result = None
    best_loss = float('inf')
    
    for lr, results in all_results.items():
        for result in results:
            # Selection based solely on training loss
            final_loss = result['final_loss']
            if final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr
                best_result = result
    
    if best_result is None:
        print("No valid optimization results found. Exiting.")
        return
    
    print(f"\nBest learning rate: {best_lr} (final loss: {best_result['final_loss']:.4f})")
    # Evaluation loss removed – only training loss is reported.
    
    # Initialize wandb
    wandb_run = None
    lrs_str = "-".join([str(lr) for lr in learning_rates])
    # Use "bias" instead of -1 in run names and file paths; append steering type
    vector_id_base = "bias" if args.steering_vector_idx == -1 else f"idx{args.steering_vector_idx}"
    vector_id = f"{vector_id_base}_{args.steering_type}"
    run_name = f"{model_name_short}_{vector_id}_lr{lrs_str}_n{args.n_training_examples}"
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it with `pip install wandb`")
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )

    # Save hyperparameters for this vector
    # Norm summary depends on steering type
    if args.steering_type == "linear":
        vector_norm_value = best_result['vector'].norm().item()
    elif args.steering_type == "adaptive_linear":
        v = best_result['vector']
        vector_norm_value = float((v['vector'].norm()**2 + v['W1'].norm()**2 + v['W2'].norm()**2).sqrt().item())
    else:
        vA = best_result['vector']['A']
        vB = best_result['vector']['B']
        vAlpha = best_result['vector']['alpha']
        vector_norm_value = float(vA.norm().item() + vB.norm().item())

    hyperparams = {
        "layer": args.layer,
        "max_iters": args.max_iters,
        "learning_rates": learning_rates,
        "best_lr": best_lr,
        "context_sentences": args.context_sentences,
        "seed": args.seed,
        "n_training_examples": len(training_examples),
        "vector_norm": vector_norm_value,
        "grad_clip": args.grad_clip,
        "optim_minibatch_size": args.optim_minibatch_size,
        "base_gen_minibatch_size": args.base_gen_minibatch_size,
        "steering_token_window": args.steering_token_window,
        "steering_type": args.steering_type,
        "lora_rank": args.lora_rank if args.steering_type == "resid_lora" else None,
        "adaptive_hidden": args.adaptive_hidden if args.steering_type == "adaptive_linear" else None,
    }
    # Use "bias" instead of steering_vector_idx in hyperparams filename
    save_hyperparameters(hyperparams, model_name_short, vector_id, target_category)
    
    # Log best hyperparameters and summary to wandb
    if wandb_run:
        wandb_run.config.update(hyperparams, allow_val_change=True)
        wandb_run.summary['best_lr'] = best_lr
        wandb_run.summary['final_loss'] = best_result['final_loss']
        # Evaluation loss removed – only training loss is reported.

    # Save training losses for best learning rate in the format expected by visualization
    losses_dir = f"results/vars/losses"
    os.makedirs(losses_dir, exist_ok=True)
    # Use "bias" instead of steering_vector_idx in losses filename
    losses_path = f"{losses_dir}/losses_{model_name_short}_{vector_id}.pt"
    
    # Prepare loss data in the format expected by visualize_vector_losses.py
    best_loss_info = best_result['loss_info']
    record = {
        'train_losses': best_loss_info['loss_history'],
        'final_loss': best_loss_info['final_loss']
    }
    if best_loss_info.get('eval_loss_history') is not None:
        record['eval_losses'] = best_loss_info['eval_loss_history']
    loss_data = { best_lr: [record] }
    
    torch.save(loss_data, losses_path)
    print(f"\nSaved training losses for best learning rate {best_lr} to {losses_path}")
    
    # Show steering results on a random *training* example to visualise effect
    if training_examples:
        print(f"\nTesting {target_category} vector on a random TRAIN example (best lr: {best_lr}):")
        train_example = random.choice(training_examples)
        test_on_example(
            model, 
            tokenizer, 
            best_result['vector'], 
            args.layer, 
            train_example,
            max_new_tokens=args.test_max_tokens,
            steering_token_window=args.steering_token_window,
            additional_vectors=(
                [bias_lora] if (args.steering_type == "resid_lora" and bias_lora is not None) else (
                    [bias_vector] if bias_vector is not None else None
                )
            ),
            steering_type=args.steering_type
        )
    
    # Free memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save best vector
    # TODO: make this dynamic
    os.makedirs(args.save_path, exist_ok=True)
    # Use "bias" instead of steering_vector_idx in vectors filename
    vectors_path = f"{args.save_path}/{model_name_short}_{vector_id}.pt"
    optimized_vectors = {}
    optimized_vectors[target_category] = best_result['vector']
    torch.save(optimized_vectors, vectors_path)
    print(f"\nSaved optimized vectors to {vectors_path}")

    if wandb_run:
        artifact = wandb.Artifact(run_name, type='model')
        artifact.add_file(vectors_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

if __name__ == "__main__":
    main() 
# %%
