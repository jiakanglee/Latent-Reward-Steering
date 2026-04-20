import dotenv
dotenv.load_dotenv("../.env")

import gc
import torch
from nnsight import LanguageModel
import time
import anthropic
from openai import OpenAI
import json
import re
import numpy as np
import sys
import os
import random
import pickle
from tqdm import tqdm

from utils.responses import extract_thinking_process

def print_and_flush(message):
    """Prints a message and flushes stdout."""
    print(message)
    sys.stdout.flush()

def chat(prompt, model="gpt-4.1", max_tokens=28000):

    model_provider = ""

    if model in ["gpt-4o", "gpt-4.1"]:
        model_provider = "openai"
        client = OpenAI()
    elif model in ["claude-3-opus", "claude-3-7-sonnet", "claude-3-5-haiku"]:
        model_provider = "anthropic"
        client = anthropic.Anthropic()
    elif model in ["deepseek-v3", "gemini-2-0-think", "gemini-2-0-flash", "deepseek-r1"]:
        model_provider = "openrouter"
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # try 3 times with 3 second sleep between attempts
    for _ in range(3):
        try:
            if model_provider == "openai":
                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=1e-19,
                )
                return response.choices[0].message.content
            elif model_provider == "anthropic":
                model_mapping = {
                    "claude-3-opus": "claude-3-opus-latest",
                    "claude-3-7-sonnet": "claude-3-7-sonnet-latest",
                    "claude-3-5-haiku": "claude-3-5-haiku-latest"
                }

                if model == "claude-3-7-sonnet":
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        thinking = {
                            "type": "enabled",
                            "budget_tokens": max_tokens
                        },
                        max_tokens=max_tokens+1
                    )

                    thinking_response = response.content[0].thinking
                    answer_response = response.content[1].text

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"

                else:
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1e-19,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        max_tokens=max_tokens
                    )

                    return response.content[0].text
            elif model_provider == "openrouter":
                # Map model names to OpenRouter model IDs
                model_mapping = {
                    "deepseek-r1": "deepseek/deepseek-r1",
                    "deepseek-v3": "deepseek/deepseek-chat",
                    "gemini-2-0-think": "google/gemini-2.0-flash-thinking-exp:free",
                    "gemini-2-0-flash": "google/gemini-2.0-flash-001"
                }
                
                response = client.chat.completions.create(
                    model=model_mapping[model],
                    extra_body={},
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=1e-19,
                    max_tokens=max_tokens
                )

                if hasattr(response.choices[0].message, "reasoning"):
                    thinking_response = response.choices[0].message.reasoning
                    answer_response = response.choices[0].message.content

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"
                else:
                    return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(20)

    return None

async def chat_batch(prompts, model="gpt-4.1", max_tokens=28000, max_concurrent_requests=100, max_retries_per_item=3, json_mode=False):
    """
    Process a batch of prompts using the chat_limiter library for parallel processing.
    
    Args:
        prompts (list): List of prompts to process
        model (str): Model to use for the chat
        max_tokens (int): Maximum number of tokens per response
        max_concurrent_requests (int): Maximum number of concurrent requests
        max_retries_per_item (int): Maximum number of retries per item
        
    Returns:
        list: List of responses corresponding to the prompts
    """
    temperature = 1e-19
    if model.startswith("o3") or model.startswith("o4"):
        temperature = 1

    # Create chat completion requests
    # Newer OpenAI models (gpt-5.x, o3, o4) require max_completion_tokens instead of max_tokens
    # Handle both "gpt-5.2" and "openai/gpt-5.2" formats
    model_name_only = model.split("/")[-1] if "/" in model else model
    use_max_completion_tokens = model_name_only.startswith("gpt-5") or model_name_only.startswith("o3") or model_name_only.startswith("o4")
    if use_max_completion_tokens:
        requests = create_chat_completion_requests(
            model=model,
            prompts=prompts,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        requests = create_chat_completion_requests(
            model=model,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    # Instantiate BatchConfig, accounting for versions without a `json_mode` parameter
    try:
        config = BatchConfig(
            max_concurrent_requests=max_concurrent_requests,
            max_retries_per_item=max_retries_per_item,
            group_by_model=True,
            json_mode=json_mode,
            # print_request_initiation=True,
        )
    except TypeError:
        # Fallback for older/newer versions of chat_limiter without `json_mode`
        config = BatchConfig(
            max_concurrent_requests=max_concurrent_requests,
            max_retries_per_item=max_retries_per_item,
            group_by_model=True,
            # print_request_initiation=True,
        )
    
    # Process batch with increased timeout for reliability
    async with ChatLimiter.for_model(model, timeout=240.0) as limiter:
        results = await process_chat_completion_batch(limiter, requests, config)
    
    # Extract responses and handle errors
    responses = []
    for i, result in enumerate(results):
        # print(f"Batch request {i} result: {result}")
        if result.success:
            # Handle different response formats based on model
            response = result.result
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                # Handle thinking models that might have reasoning
                if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                    thinking_response = response.choices[0].message.reasoning
                    responses.append(f"<think>{thinking_response}\n</think>\n{content}")
                else:
                    responses.append(content)
            else:
                responses.append(str(response))
        else:
            print(f"Batch request {i} failed: {result.error_message}")
    
    return responses

def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token


def _activation_cache_paths(model_id: str, n_examples: int, layer: int) -> tuple[str, str]:
    """
    Returns (activations_pkl_path, mean_pkl_path) for the given cache key.

    NOTE: These paths intentionally mirror `process_saved_responses`'s cache layout.
    """
    activations_pkl = f"results/vars/activations_{model_id}_{n_examples}_{layer}.pkl"
    mean_pkl = f"results/vars/activations_{model_id}_{n_examples}_{layer}_mean.pkl"
    return activations_pkl, mean_pkl


def load_activation_mean(model_id: str, n_examples: int, layer: int) -> np.ndarray:
    """
    Load the centering mean vector used to build the cached activations for (model_id, n_examples, layer).

    Returns:
        mean: np.ndarray shape (d_model,) dtype float32
    """
    _acts_pkl, mean_pkl = _activation_cache_paths(model_id=model_id, n_examples=n_examples, layer=layer)
    assert os.path.exists(mean_pkl), (
        f"Missing activation mean file: {mean_pkl}\n"
        "Expected this to be produced by `utils.utils.process_saved_responses` and uploaded alongside the "
        "cached activations."
    )
    with open(mean_pkl, "rb") as f:
        payload = pickle.load(f)
    assert isinstance(payload, dict), f"Bad mean payload type: {type(payload)}"
    assert payload.get("model_id") == model_id, f"Mean model_id mismatch: {payload.get('model_id')} vs {model_id}"
    assert int(payload.get("layer")) == int(layer), f"Mean layer mismatch: {payload.get('layer')} vs {layer}"
    assert int(payload.get("n_examples")) == int(n_examples), f"Mean n_examples mismatch: {payload.get('n_examples')} vs {n_examples}"
    mean = np.asarray(payload.get("activation_mean"))
    assert mean.ndim == 1, f"Expected 1D mean, got shape {mean.shape}"
    mean = mean.astype(np.float32, copy=False)
    assert np.isfinite(mean).all(), "Non-finite values in activation mean"
    return mean


def center_and_l2_normalize_torch(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """
    Center and L2-normalize activations using a precomputed mean.

    Supported shapes:
      - x: (d_model,)
      - x: (..., d_model)
    mean:
      - (d_model,)
    Returns:
      - same shape as x, float32
    """
    assert isinstance(x, torch.Tensor), f"x must be torch.Tensor, got {type(x)}"
    assert isinstance(mean, torch.Tensor), f"mean must be torch.Tensor, got {type(mean)}"
    assert mean.ndim == 1, f"Expected mean.ndim==1, got {mean.ndim}"
    assert x.ndim >= 1, f"Expected x.ndim>=1, got {x.ndim}"
    assert x.shape[-1] == mean.shape[0], f"Bad shapes: x.shape={tuple(x.shape)} mean.shape={tuple(mean.shape)}"

    x_f32 = x.to(dtype=torch.float32)
    mean_f32 = mean.to(device=x_f32.device, dtype=torch.float32)
    centered = x_f32 - mean_f32

    denom = torch.norm(centered, dim=-1, keepdim=True)
    assert torch.isfinite(denom).all(), "Non-finite norm in center_and_l2_normalize_torch"
    assert torch.all(denom > 0), "Zero-norm encountered in center_and_l2_normalize_torch"
    return centered / denom

def center_and_normalize_activations(all_activations, overall_mean):
    """Centers and normalizes activations."""
    
    print_and_flush(f"Centering activations...")
    start_time = time.time()
    all_activations = [x - overall_mean for x in all_activations]
    all_activations = np.stack([a.reshape(-1) for a in all_activations])
    norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
    all_activations = all_activations / norms
    end_time = time.time()
    print(f"Centered and normalized activations in {end_time - start_time} seconds")

    return all_activations

def process_saved_responses(model_name, n_examples, model, tokenizer, layer_or_layers):
    """Load and process saved responses to get activations"""

    # Ensure layer_or_layers is a list
    if isinstance(layer_or_layers, (int, str)):
        layers_to_process = [int(layer_or_layers)]
    else:
        layers_to_process = [int(l) for l in layer_or_layers]

    model_id = model_name.split('/')[-1].lower()
    
    # Dictionary to store results for each layer
    results_by_layer = {}
    
    # Check for cached files for each layer
    uncached_layers = []
    for layer in layers_to_process:
        pickle_filename, _mean_filename = _activation_cache_paths(model_id=model_id, n_examples=n_examples, layer=layer)
        if os.path.exists(pickle_filename):
            print(f"Loading cached activations for layer {layer} from {pickle_filename}...")
            with open(pickle_filename, 'rb') as f:
                results_by_layer[layer] = pickle.load(f)
        else:
            uncached_layers.append(layer)

    if not uncached_layers:
        print("All requested layers were loaded from cache.")
        # If only one layer was requested, return in the old format for backward compatibility
        if len(layers_to_process) == 1:
            return results_by_layer[layers_to_process[0]]
        return results_by_layer

    print(f"Processing saved responses for layers: {uncached_layers}...")
    
    # Load responses if there are any uncached layers
    responses_json_path = f"results/vars/responses_{model_id}.json"
    print(f"Loading responses from {responses_json_path}...")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)
    
    # Limit to n_examples
    random.shuffle(responses_data)
    responses_data = responses_data[:n_examples]
    
    # Initialize data structures for uncached layers
    activations_by_layer = {layer: [] for layer in uncached_layers}
    texts_by_layer = {layer: [] for layer in uncached_layers}
    mean_by_layer = {layer: torch.zeros(1, model.config.hidden_size) for layer in uncached_layers}
    count_by_layer = {layer: 0 for layer in uncached_layers}

    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    clear_gpu_memory()

    print(f"Extracting activations for {n_examples} responses across layers {uncached_layers}...")
    for response_data in tqdm(responses_data):
        thinking_process = extract_thinking_process(response_data["full_response"])
        if not thinking_process:
            continue
            
        thinking_text = thinking_process
        full_response = response_data["full_response"]
        
        sentences = split_into_sentences(thinking_text)
        
        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        
        # Get layer activations for all uncached layers in one trace
        layer_outputs = {}
        with model.trace({
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        }) as tracer:
            for layer in uncached_layers:
                saved_output = model.model.layers[layer].output.save()
                assert torch.isfinite(saved_output).all(), f"Layer {layer}: non-finite values after save"
                layer_outputs[layer] = saved_output

        # Detach and convert to float32
        for layer in uncached_layers:
            layer_outputs[layer] = layer_outputs[layer].detach().cpu().to(torch.float32)
            assert torch.isfinite(layer_outputs[layer]).all(), f"Layer {layer}: non-finite values after detach and to float32"

        char_to_token = get_char_to_token_map(full_response, tokenizer)
        
        # Process each sentence for each layer
        for layer in uncached_layers:
            layer_output = layer_outputs[layer]
            min_token_start = float('inf')
            max_token_end = -float('inf')

            for sentence in sentences:
                text_pos = full_response.find(sentence)
                if text_pos >= 0:
                    token_start = char_to_token.get(text_pos, None)
                    token_end = char_to_token.get(text_pos + len(sentence), None)
                    
                    if token_start is not None and token_end is not None and token_start < token_end:
                        if token_start < min_token_start:
                            min_token_start = token_start
                        if token_end > max_token_end:
                            max_token_end = token_end

                        segment = layer_output[:, token_start - 1:token_end, :]
                        assert segment.shape[1] > 0, (
                            f"Empty token slice at layer {layer}: token_start={token_start}, token_end={token_end}, "
                            f"sentence='{sentence[:80]}', full_response='{full_response[:200]}'"
                        )
                        segment_activations = segment.mean(dim=1).numpy()
                        assert np.isfinite(segment_activations).all(), f"Layer {layer}: non-finite values after numpy conversion"
                        
                        activations_by_layer[layer].append(segment_activations)
                        texts_by_layer[layer].append(sentence)
            
            if min_token_start < layer_output.shape[1] and max_token_end > 0:
                vector = layer_output[:, min_token_start:max_token_end, :].mean(dim=1).cpu()
                mean_by_layer[layer] = mean_by_layer[layer] + (vector - mean_by_layer[layer]) / (count_by_layer[layer] + 1)
                count_by_layer[layer] += 1

        clear_gpu_memory()

    # Save results for each newly processed layer
    for layer in uncached_layers:
        print(f"Found {len(activations_by_layer[layer])} sentences with activations for layer {layer} across {count_by_layer[layer]} examples")
        if len(activations_by_layer[layer]) == 0:
            raise ValueError(f"No activations found for layer {layer} across {count_by_layer[layer]} examples")

        overall_running_mean = mean_by_layer[layer].cpu().numpy()
        overall_running_mean = np.asarray(overall_running_mean).reshape(-1).astype(np.float32, copy=False)
        assert overall_running_mean.shape == (model.config.hidden_size,), (
            f"Bad mean shape: {overall_running_mean.shape} vs expected {(model.config.hidden_size,)}"
        )

        # Persist the mean used for centering (required for downstream SAE usage parity).
        _pickle_filename, mean_filename = _activation_cache_paths(model_id=model_id, n_examples=n_examples, layer=layer)
        with open(mean_filename, "wb") as f:
            pickle.dump(
                {
                    "model_id": model_id,
                    "layer": int(layer),
                    "n_examples": int(n_examples),
                    "count_vectors": int(count_by_layer[layer]),
                    "activation_mean": overall_running_mean,
                },
                f,
            )
        print(f"Saved activation mean for layer {layer} to {mean_filename}")

        # Center and normalize activations
        activations_by_layer[layer] = center_and_normalize_activations(activations_by_layer[layer], overall_running_mean)
        
        result = (activations_by_layer[layer], texts_by_layer[layer])
        results_by_layer[layer] = result
        
        pickle_filename, _mean_filename = _activation_cache_paths(model_id=model_id, n_examples=n_examples, layer=layer)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved activations for layer {layer} to {pickle_filename}")

    # If only one layer was requested, return in the old format
    if len(layers_to_process) == 1:
        return results_by_layer[layers_to_process[0]]
        
    return results_by_layer


def load_model(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device="auto",
    load_in_8bit=False,
    attn_implementation=None,
):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.

    Args:
        load_in_8bit (bool): If True, load the model in 8-bit mode
        model_name (str): Name/path of the model to load
        attn_implementation: 传入 HuggingFace，如 \"flash_attention_2\"、\"sdpa\"、\"eager\"；
            None 时不设置，由 Transformers 默认决定。
    """
    lm_kw = dict(
        dispatch=True,
        load_in_8bit=load_in_8bit,
        device_map=device,
        dtype=torch.bfloat16,
    )
    if attn_implementation:
        lm_kw["attn_implementation"] = attn_implementation
    try:
        model = LanguageModel(model_name, **lm_kw)
    except Exception as e:
        if attn_implementation == "flash_attention_2":
            print_and_flush(
                f"⚠️ flash_attention_2 加载失败 ({e!r})，回退 attn_implementation=sdpa。"
            )
            print_and_flush(
                f"   当前 Python: {sys.executable} — 在同一解释器下检查: "
                f"{sys.executable} -c \"import flash_attn\"（pip 装到了别的环境时会导入失败。）"
            )
            lm_kw["attn_implementation"] = "sdpa"
            model = LanguageModel(model_name, **lm_kw)
        else:
            raise
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer

def custom_generate_with_steering(model, tokenizer, input_ids, max_new_tokens, steering_vector=None, layer=None, normalize=False, coefficient=1.0):
    """
    Generate text while steering with a specific feature vector.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        steering_vector: Vector to use for steering (should match model hidden size)
        layer: Layer index to apply steering to
        coefficient: Strength of steering (higher values = stronger effect)
        sae: Sparse Autoencoder model to use for activation-based steering
    """
    model_layers = model.model.layers

    with model.generate(
        {
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if steering_vector is not None and layer is not None:
            # Convert steering vector to correct device and dtype if needed
            steering_vector = steering_vector.to(model.device).to(model.dtype)
            avg_norm = model.model.layers[layer].output[0][:, 1:, :].norm(dim=-1).mean(dim=1)
            if normalize:
                steering_vector = steering_vector.unsqueeze(0).unsqueeze(0) * avg_norm
            model.model.layers[layer].output[0][:, 1:, :] += coefficient * steering_vector
        
        outputs = model.generator.output.save()
                    
    return outputs

def get_random_distinct_colors(labels):
    """
    Generate random distinct ANSI colors for each label.
    
    Args:
        labels: List of label names
        
    Returns:
        Dictionary mapping labels to ANSI color codes
    """
    # List of distinct ANSI colors (excluding black, white, and hard-to-see colors)
    # Format is "\033[COLORm" where COLOR is a number between 31-96
    distinct_colors = [
        "\033[31m",  # Red
        "\033[32m",  # Green
        "\033[33m",  # Yellow
        "\033[34m",  # Blue
        "\033[35m",  # Magenta
        "\033[36m",  # Cyan
        "\033[91m",  # Bright Red
        "\033[92m",  # Bright Green
        "\033[93m",  # Bright Yellow
        "\033[94m",  # Bright Blue
        "\033[95m",  # Bright Magenta
        "\033[96m",  # Bright Cyan
    ]
    
    # Shuffle the colors to randomize them
    random.shuffle(distinct_colors)
    
    # Ensure we have enough colors
    if len(labels) > len(distinct_colors):
        # If we need more colors, create additional ones with random RGB values
        additional_needed = len(labels) - len(distinct_colors)
        for _ in range(additional_needed):
            # Generate random RGB foreground color (38;2;r;g;b)
            r, g, b = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
            # Ensure colors are distinct by checking minimum distance from existing colors
            # (simplified approach)
            distinct_colors.append(f"\033[38;2;{r};{g};{b}m")
    
    # Assign colors to labels
    label_colors = {}
    for i, label in enumerate(labels):
        label_colors[label] = distinct_colors[i % len(distinct_colors)]
    
    return label_colors

# Create NumpyEncoder for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# Function to convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def process_batch_annotations(thinking_processes):
    """Annotate a batch of reasoning chains using the 7-category reasoning framework."""
    annotated_responses = []
    for thinking in thinking_processes:
        annotated_response = chat(f"""
Please annotate the following reasoning trace by marking segments with categories from the reasoning framework below. Use this format: ["<category>"] ... ["<end-section>"]. A sentence can be split into multiple segments if it exhibits different behaviors. Only use the categories provided below.

Reasoning Framework Categories:

1. problem-identification-framing – Problem Identification and Framing
   *Description:* This reflects the model's initial orientation toward the problem—an explicit commitment to focus attention on a particular question or task. It's not solving yet; it's mentally staking out the terrain and clarifying the goal.
   *Includes:* Explicit declarations of the question or topic to be addressed; clarifying scope or rephrasing the goal of the reasoning.
   *Excludes:* Any move toward analysis, solution generation, or speculation.
   *Examples:* "Okay, so I'm trying to figure out how pressure affects the boiling point of water.", "Okay, so I'm trying to figure out the ripple effects of making college education free."

2. metacognitive-setup – Metacognitive Setup and Decomposition Initiation
   *Description:* This captures the model's pre-analytic cognitive preparation—noticing uncertainty or complexity and deciding to plan, organize, or scaffold the reasoning process before diving in.
   *Includes:* Metacognitive statements about strategy or planning; moves to mentally break a problem into manageable parts.
   *Excludes:* Execution of any actual reasoning steps or guesses.
   *Examples:* "Hmm, let me think about this step by step.", "Let me try to visualize this.", "I'm not entirely sure where to start, but I think it's important to break it down step by step."

3. stepwise-calculation – Stepwise Calculation / Enumeration / Local Inference
   *Description:* This cluster captures the model's mechanistic reasoning—applying rules, performing arithmetic, listing possibilities. It's executing a mental algorithm.
   *Includes:* Arithmetic, combinatorics, enumeration of cases; explicit inferences from rules or facts.
   *Excludes:* High-level summaries or contextual reasoning.
   *Examples:* "3 times 7 is 21, and 21 times 11 is 231.", "So, the probability of drawing a red on the first draw is 4 out of 7, which is 4/7.", "Each face is a base for one pyramid, so 6 pyramids."

4. generating-alternatives – Generating Alternatives / Hypotheses
   *Description:* This cluster reflects the model's attempt to expand the hypothesis space. It's not committing to an answer—it's surfacing possible explanations, mechanisms, or paths forward.
   *Includes:* Generative thinking under uncertainty; multiple speculative branches or mechanisms.
   *Excludes:* Final answers or rule-based deductions.
   *Examples:* "Or maybe it's about controlling invasive species.", "Maybe it's just the body's way of fighting off the infection.", "I should also consider different scenarios."

5. information-seeking – Information-Seeking and Epistemic Uncertainty
   *Description:* The model confronts a knowledge gap and initiates action to resolve it. This is a pivot away from internal reasoning toward acquiring more information.
   *Includes:* Statements of uncertainty paired with information-seeking intent; declarations that external info is needed.
   *Excludes:* Internal speculation without intent to learn more; passive confusion without action.
   *Examples:* "I should probably look up some information to get a better understanding.", "Maybe I should ask someone or look it up to find out more information.", "I think I'll just have to check online or maybe ask a friend."

6. consequence-projection – Consequence Projection / Scenario Elaboration
   *Description:* This is forward simulation. The model is running a mental model of the world to ask: "What would happen if...?"
   *Includes:* Counterfactuals, conditionals, and policy simulation; exploration of second- or third-order effects.
   *Excludes:* Simple cause-effect or binary conclusions.
   *Examples:* "Also, with more free time, people might pursue further education.", "If the species affects farming, there might be compensation programs.", "Cities might save money on road repairs due to AVs."

7. conclusion-articulation – (Sub)-Conclusion Articulation
   *Description:* This is the "wrap up this step" reflex. It's when the model finishes part of the reasoning and states a result—before continuing onward.
   *Includes:* (Partial) conclusions or intermediate inferences; logic checkpoints or sanity checks.
   *Excludes:* Problem framing or speculative reasoning.
   *Examples:* "So, each face is a base for one pyramid, so 6 pyramids.", "So, the next month is December, which is D.", "So, if the surgeon is the mother, then yes, the patient is her son."

Reasoning trace to annotate:
{thinking}

Only return the annotated text using the specified format. Do not include any explanation or commentary outside the annotations.
If the last sentence is not finished, do not include it in the annotations.
""")
        annotated_responses.append(annotated_response)
    
    return annotated_responses


model_mapping = {
    "meta-llama/Llama-3.1-8B":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen2.5-0.5B":"Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B",
    "Qwen/Qwen2.5-1.5B":"Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B",
    "Qwen/Qwen2.5-Math-1.5B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-7B":"Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
    "Qwen/Qwen2.5-14B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "Qwen/Qwen2.5-32B":"Qwen/QwQ-32B",
    "meta-llama/Llama-3.3-70B-Instruct": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

def split_into_sentences(text, min_words=3):
    """
    Split text into sentences and filter based on quality criteria.
    
    Args:
        text (str): The input text to split into sentences
        
    Returns:
        list: List of cleaned sentences with at least 3 words each
    """
    # Split after sentence-ending punctuation and newlines while keeping delimiters
    # Use positive lookbehind to split after delimiters, but with edge case handling
    
    # First handle edge cases by temporarily replacing problematic patterns
    # Protect decimal numbers like "3.14" and single letter abbreviations like "E. coli"
    protected_text = text
    replacements = []
    
    # Protect decimal numbers
    for match in re.finditer(r'\d+\.\d+', text):
        placeholder = f"__DECIMAL_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)
    
    # Protect single letter abbreviations (letter followed by period and space/word)
    for match in re.finditer(r'\b[A-Za-z]\.\s+[A-Za-z]', text):
        placeholder = f"__ABBREV_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)
    
    # Protect mathematical expressions like "k!" (letter followed by exclamation)
    for match in re.finditer(r'\b[A-Za-z]!', text):
        placeholder = f"__MATH_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)
    
    # Handle consecutive punctuation by normalizing it first
    # Replace consecutive punctuation with single punctuation for splitting
    consecutive_punct_pattern = r'([.!?;])\1+'
    consecutive_matches = []
    for match in re.finditer(consecutive_punct_pattern, protected_text):
        consecutive_matches.append((match.start(), match.end(), match.group()))
    
    # Split using simple lookbehind after normalizing consecutive punctuation
    normalized_text = re.sub(consecutive_punct_pattern, r'\1', protected_text)
    sentences = re.split(r'(?<=[.!?;\n])', normalized_text)
    
    # Restore consecutive punctuation in the sentences
    if consecutive_matches:
        # Map back to original positions
        for start, end, original in consecutive_matches:
            # Find which sentence contains this punctuation and restore it
            for i, sentence in enumerate(sentences):
                if sentence and start < len(protected_text):
                    # This is a simplified restoration - may need refinement for complex cases
                    if original[0] in sentence and len(original) > 1:
                        sentences[i] = sentence.replace(original[0], original, 1)
    
    # Restore protected patterns
    for placeholder, original in replacements:
        sentences = [s.replace(placeholder, original) for s in sentences]
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s.split()) >= min_words]
    
    # Post-processing: Handle sentences that start with quotes after period splits
    # If a sentence starts with a quote, move it to the end of the previous sentence
    processed_sentences = []
    for i, sentence in enumerate(sentences):
        if i > 0 and sentence.startswith('"') and processed_sentences:
            # Move the quote to the previous sentence and remove it from current
            processed_sentences[-1] += '"'
            current_sentence = sentence[1:].strip()  # Remove quote and leading space
            if current_sentence and len(current_sentence.split()) >= min_words:
                processed_sentences.append(current_sentence)
        else:
            processed_sentences.append(sentence)
    
    return processed_sentences


def load_steering_vectors(device: str = "cpu", hyperparams_dir: str | None = None, vectors_dir: str | None = None, verbose: bool = False):
    """Load all optimized steering vectors and return a mapping of category -> vector."""
    import glob  # Local import to avoid slowing start-up when not needed

    # Resolve default directories relative to this utils.py file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train-vectors"))

    if hyperparams_dir is None:
        hyperparams_dir = os.path.join(base_dir, "results", "vars", "hyperparams")
    if vectors_dir is None:
        vectors_dir = os.path.join(base_dir, "results", "vars", "optimized_vectors")

    if verbose:
        print_and_flush(f"Loading steering vectors from:\n  Hyperparams: {hyperparams_dir}\n  Vectors:     {vectors_dir}")

    # Pattern to extract {model_name_short} and {idx} from filenames
    hp_pattern = re.compile(r"steering_vector_hyperparams_(.+?)_(idx\d+|bias)\.json")

    category_to_vector: dict[str, torch.Tensor] = {}

    for hp_path in glob.glob(os.path.join(hyperparams_dir, "steering_vector_hyperparams_*.json")):
        hp_file = os.path.basename(hp_path)
        match = hp_pattern.match(hp_file)
        if match is None:
            if verbose:
                print_and_flush(f"[load_steering_vectors] Skipping unrecognised file name: {hp_file}")
            continue

        model_name_short, idx_str = match.groups()
        # Handle both numbered indices and "bias"
        vector_path = os.path.join(vectors_dir, f"{model_name_short}_{'idx' + idx_str if idx_str.isdigit() else idx_str}.pt")

        # Load hyperparameters JSON to get the category name
        try:
            with open(hp_path, "r") as f:
                hp_data = json.load(f)
            category = hp_data.get("category")
        except Exception as e:
            if verbose:
                print_and_flush(f"[load_steering_vectors] Failed to read {hp_file}: {e}")
            continue

        if category is None:
            if verbose:
                print_and_flush(f"[load_steering_vectors] No 'category' field in {hp_file}. Skipping.")
            continue

        # Ensure the vector file exists
        if not os.path.exists(vector_path):
            if verbose:
                print_and_flush(f"[load_steering_vectors] Vector file not found for {category}: {vector_path}")
            continue

        try:
            vec_dict = torch.load(vector_path, map_location=device)
        except Exception as e:
            if verbose:
                print_and_flush(f"[load_steering_vectors] Could not load tensor from {vector_path}: {e}")
            continue

        # The saved dict is {category_name: tensor}
        if category not in vec_dict:
            # Some older runs may save just the tensor; handle that case.
            if isinstance(vec_dict, torch.Tensor):
                vector_tensor = vec_dict
            else:
                if verbose:
                    print_and_flush(f"[load_steering_vectors] Category '{category}' not in vector file {vector_path}. Keys: {list(vec_dict.keys())}")
                continue
        else:
            vector_tensor = vec_dict[category]

        # Move tensor to desired device and ensure float32/float16 is preserved
        vector_tensor = vector_tensor.to(device)
        # For bias vector, store under "bias" key regardless of what's in the JSON
        if idx_str == "bias":
            category_to_vector["bias"] = vector_tensor
        else:
            category_to_vector[category] = vector_tensor

        if verbose:
            print_and_flush(f"[load_steering_vectors] Loaded vector for '{category}' from {idx_str} (model {model_name_short})")

    if verbose:
        print_and_flush(f"[load_steering_vectors] Loaded {len(category_to_vector)} vectors.")

    return category_to_vector

# 简单版加载器 (保留作为备用)
def load_steering_vectors_simple(device="cuda", verbose=True):
    import os, glob
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train-vectors"))
    vectors_dir = os.path.join(base_dir, "results", "vars", "optimized_vectors")
    
    vectors = {}
    if verbose: print(f"Loading vectors from {vectors_dir}...")
    
    if not os.path.exists(vectors_dir):
        print(f"Warning: Vector dir not found: {vectors_dir}")
        return vectors

    for pt_file in glob.glob(os.path.join(vectors_dir, "*.pt")):
        try:
            data = torch.load(pt_file, map_location=device)
            if isinstance(data, dict):
                vectors.update(data)
            elif "bias" in pt_file:
                vectors["bias"] = data
        except:
            pass
    return vectors

# =========================================================================
# 修复后的核心函数：custom_hybrid_generate
# =========================================================================
def custom_hybrid_generate(
    thinking_model, base_model, tokenizer, input_ids, 
    max_new_tokens, baseline_config, baseline_method="probe", **kwargs
):
    device = base_model.device
    
    # 1. 向量加载 (优先从 config 拿筛选好的 15 个向量)
    if "vectors" in baseline_config and baseline_config["vectors"]:
        vectors = baseline_config["vectors"]
    else:
        if not hasattr(custom_hybrid_generate, "vectors"):
            # 这里的加载逻辑做了保底，实际运行会用你 evaluate_hybrid 传进来的
            custom_hybrid_generate.vectors = load_steering_vectors(device)
        vectors = custom_hybrid_generate.vectors
    
    probe = baseline_config["probe"]
    idx_to_label = {v: k for k, v in baseline_config["label_to_idx"].items()}
    top_k = baseline_config.get("top_k", 3)
    
    # 读取配置
    strength = baseline_config.get("strength", 0.1)
    target_layer = baseline_config.get("probe_layer", 10)

    generated_ids = input_ids.clone().to(device)
    
    print(f"\n🚀 Starting Hybrid Generation (Top-K={top_k}, Strength={strength})")

    for step in range(max_new_tokens):
        # 1. Oracle 探测 (Thinking Model 算出当前状态)
        with torch.no_grad():
            outputs = thinking_model.model(generated_ids, output_hidden_states=True)
            # 取最后一层的最后一个 token 的 hidden state
            h = outputs.hidden_states[baseline_config["probe_layer"]][:, -1, :]
            logits = probe(h)
            
        # 2. Combined Steering: 这里的逻辑就是你关心的混合过程
        k_real = min(top_k, logits.size(-1))
        vals, inds = torch.topk(logits, k=k_real, dim=-1)
        weights = torch.softmax(vals, dim=-1) # 将分数转为 0~1 的权重比例
        
        combined_vec = torch.zeros(base_model.config.hidden_size, device=device, dtype=base_model.dtype)
        do_steer = False
        
        # --- [DEBUG 记录器] ---
        steering_components = []
        # ---------------------

        for i in range(k_real):
            label = idx_to_label[inds[0, i].item()]
            w = weights[0, i].item()
            
            # 只有当 label 存在于我们加载的 vectors 字典中时才进行混合
            if label in baseline_config["forcing"] and label in vectors:
                vec = vectors[label].to(device).to(base_model.dtype)
                
                # 【核心：线性加权混合】
                combined_vec += w * vec
                do_steer = True
                
                # 记录以便打印
                steering_components.append(f"{label}({w:.2f})")
        
        # --- [DEBUG 打印输出] ---
        if do_steer:
            # 计算合成向量的模长 (Norm)，评估注入强度
            vec_norm = torch.norm(combined_vec).item()
            # 实时打印：Token 索引 | 混合成分 | 总模长
            print(f"Token {step:03d} | Steering: {' + '.join(steering_components)} | Combined Norm: {vec_norm:.4f}")
            sys.stdout.flush() 
        # ---------------------

        # 3. 注册 Hook 并注入混合向量
        hook_handle = None
        if do_steer:
            def hook(module, inp, out):
                if isinstance(out, tuple): hidden_states = out[0]
                else: hidden_states = out

                # 在指定的 target_layer 注入计算好的 combined_vec
                if hidden_states.dim() == 3:
                    hidden_states[:, -1, :] += strength * combined_vec
                elif hidden_states.dim() == 2:
                    hidden_states += strength * combined_vec
                return out

            try:
                hook_handle = base_model.model.layers[target_layer].register_forward_hook(hook)
            except:
                # 容错：如果指定的层不存在，默认注入第 10 层
                hook_handle = base_model.model.layers[10].register_forward_hook(hook)
            
        # 4. 生成下一个 token (手动执行流水线以确保 Logits 可用)
        with torch.no_grad():
            # A. 通过主干网络拿到隐层
            backbone_out = base_model.model(input_ids=generated_ids)
            # B. 通过输出头拿到概率分布
            logits_out = base_model.lm_head(backbone_out.last_hidden_state)
            # C. 贪婪采样
            next_token = torch.argmax(logits_out[:, -1, :], dim=-1).unsqueeze(-1)
            
        # 每一轮结束必须移除 Hook，否则会无限累加
        if hook_handle: hook_handle.remove()
        
        # 更新生成的序列
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # 检查是否生成结束
        if tokenizer.eos_token_id in next_token: 
            break
            
    return generated_ids, [], [], []