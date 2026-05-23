import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import os
from datasets import load_dataset
import random
import json
import gc
from messages import messages
import utils.utils as utils
from tqdm import tqdm

MAX_TOKENS_IN_INPUT = 6000

# Parse arguments
parser = argparse.ArgumentParser(description="Generate responses from models without steering vectors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to generate responses from")
parser.add_argument("--dataset", type=str, default="TIGER-Lab/MMLU-Pro",
                    help="Dataset in HuggingFace to generate responses from", choices=["legacy-messages", "TIGER-Lab/MMLU-Pro"])
parser.add_argument("--dataset_split", type=str, default="test",
                    help="Split of dataset to generate responses from")
parser.add_argument("--max_tokens", type=int, default=1000,
                    help="Maximum number of tokens to generate")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--is_base_model", action="store_true", default=False,
                    help="Whether the model is a base model")
parser.add_argument("--tensor_parallel_size", type=int, default=-1,
                    help="Number of GPUs to use for tensor parallelism")
parser.add_argument("--dtype", type=str, default="auto",
                    help="Data type for model")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="Temperature for sampling")
parser.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p for sampling")
parser.add_argument("--engine", type=str, default="nnsight", choices=["nnsight", "vllm"],
                    help="Generation engine to use")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit (nnsight engine only)")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Micro-batch size for nnsight generation only")
args, _ = parser.parse_known_args()

if args.tensor_parallel_size == -1:
    args.tensor_parallel_size = torch.cuda.device_count()

def get_prompts(tokenizer, messages_list):
    if args.is_base_model:
        prompts = [msg["content"] for msg in messages_list]
    else:
        prompts = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages_list]
    prompts_above_max_tokens = [prompt for prompt in prompts if len(prompt) > MAX_TOKENS_IN_INPUT]
    if len(prompts_above_max_tokens) > 0:
        print(f"There are {len(prompts_above_max_tokens)} prompts above MAX_TOKENS_IN_INPUT")
        for prompt in prompts_above_max_tokens:
            print(f"Length: {len(prompt)}")
        raise ValueError(f"There are {len(prompts_above_max_tokens)} prompts above MAX_TOKENS_IN_INPUT")
    return prompts

def get_messages_from_dataset(dataset_name, rows) -> dict[str, dict[str, str]]:
    if dataset_name != "TIGER-Lab/MMLU-Pro":
        raise ValueError(f"Dataset {dataset_name} not supported")

    messages_by_question_id = {}
    for row in rows:
        question_id = row["question_id"]
        messages_by_question_id[question_id] = {
            "role": "user", 
            "content": row["question"],
            "category": row["category"],
            "question": row["question"],
        }
    return messages_by_question_id

def process_model_output_batch_vllm(messages_batch, tokenizer, model):
    """Get model output for a batch of messages using vLLM"""
    # Lazy import to avoid requiring vLLM when using nnsight
    from vllm import SamplingParams
    prompts = get_prompts(tokenizer, messages_batch)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    request_outputs = model.generate(prompts, sampling_params)
    full_responses = [request_output.prompt + request_output.outputs[0].text for request_output in request_outputs]

    # Assert the questions are in the responses
    for message, response in zip(messages_batch, full_responses):
        assert message["question"] in response, f"Question {message['question']} not in response {response}"
    
    return full_responses


def process_model_output_batch_nnsight(messages_batch, tokenizer, model):
    """Get model output for a batch of messages using nnsight/HF generate."""
    prompts = get_prompts(tokenizer, messages_batch)

    # Tokenize with left padding as set in utils.load_model
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    # Shape assertions per research guidelines
    assert input_ids.dim() == 2, f"Expected 2D input_ids, got {input_ids.shape}"
    assert attention_mask.shape == input_ids.shape, f"attention_mask shape {attention_mask.shape} must match input_ids {input_ids.shape}"

    with model.generate({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, max_new_tokens=args.max_tokens, pad_token_id=tokenizer.pad_token_id, temperature=args.temperature, top_p=args.top_p) as gen:
        outputs = model.generator.output.save()

    assert outputs.shape[0] == input_ids.shape[0], "Batch size mismatch between outputs and inputs"

    # Compute per-sample prompt lengths (number of non-pad tokens)
    prompt_lengths = attention_mask.sum(dim=1)
    responses = []
    for i in range(outputs.shape[0]):
        prompt_len = int(prompt_lengths[i].item())
        assert prompt_len > 0, f"Empty prompt length for sample {i}"
        new_tokens = outputs[i][prompt_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    # Assert the questions are in the responses
    for message, response in zip(messages_batch, responses):
        assert message["question"] in response, f"Question {message['question']} not in response {response}"

    return responses


def process_messages(dataset_name, question_ids, messages_by_question_id, tokenizer, model, engine: str):
    """Process a batch of messages and return response data"""
    if engine == "vllm":
        messages_batch = [messages_by_question_id[question_id] for question_id in question_ids]
        responses = process_model_output_batch_vllm(messages_batch, tokenizer, model)

        all_data = []
        for message, response, question_id in zip(messages_batch, responses, question_ids):
            all_data.append({
                "original_message": {"role": message["role"], "content": message["content"]},
                "full_response": response,
                "question_id": question_id,
                "category": message["category"],
                "question": message["question"],
                "dataset_name": dataset_name
            })
        return all_data
    elif engine == "nnsight":
        assert args.batch_size >= 1, f"batch_size must be >= 1, got {args.batch_size}"

        all_data = []
        for start in tqdm(range(0, len(question_ids), args.batch_size)):
            sub_ids = question_ids[start:start + args.batch_size]
            messages_batch = [messages_by_question_id[qid] for qid in sub_ids]
            responses = process_model_output_batch_nnsight(messages_batch, tokenizer, model)

            for message, response, question_id in zip(messages_batch, responses, sub_ids):
                all_data.append({
                    "original_message": {"role": message["role"], "content": message["content"]},
                    "full_response": response,
                    "question_id": question_id,
                    "category": message["category"],
                    "question": message["question"],
                    "dataset_name": dataset_name
                })

            # Proactively release memory between micro-batches
            torch.cuda.empty_cache()
            gc.collect()

        return all_data
    else:
        raise ValueError(f"Engine {engine} not supported")


def save_responses(responses_data, responses_json_path):
    with open(responses_json_path, "w") as f:
        json.dump(responses_data, f, indent=2)
    print(f"Saved {len(responses_data)} responses to {responses_json_path}") 


# Main execution
if __name__ == "__main__":
    model_name = args.model

    print(f"Loading model {model_name} using engine '{args.engine}'...")
    if args.engine == "vllm":
        # Lazy import to avoid vllm requirement for nnsight
        from vllm import LLM
        from transformers import AutoTokenizer
        model = LLM(
            model=model_name,
            tensor_parallel_size=args.tensor_parallel_size if args.tensor_parallel_size != -1 else torch.cuda.device_count(),
            dtype=args.dtype,
            seed=args.seed,
            max_model_len=args.max_tokens + MAX_TOKENS_IN_INPUT, # We assume inputs are 1000 tokens long at most
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        # Use prior nnsight-based loading
        model, tokenizer = utils.load_model(model_name=model_name, load_in_8bit=args.load_in_8bit)

    # Create directories
    os.makedirs('results/vars', exist_ok=True)

    model_prefix = "base_" if args.is_base_model else ""
    responses_json_path = f"results/vars/{model_prefix}responses_{model_name.split('/')[-1].lower()}.json"

    random.seed(args.seed)

    if args.dataset == "legacy-messages":
        # Use the index in messages as question_id
        question_ids = list(range(len(messages)))
        messages_by_question_id = {}
        for i, msg in enumerate(messages):
            messages_by_question_id[i] = {
                "role": "user",
                "content": msg["content"],
                "category": "legacy-messages",
                "question": msg["content"],
            }
        print(f"Processing {len(question_ids)} questions from legacy-messages to generate responses")
    else:
        ds = load_dataset(args.dataset)
        rows = ds[args.dataset_split]
        messages_by_question_id = get_messages_from_dataset(args.dataset, rows)
        question_ids = list(messages_by_question_id.keys())
        print(f"Processing {len(question_ids)} questions in {args.dataset_split} split of {args.dataset} to generate responses")
    
    random.shuffle(question_ids)
    
    responses_data = process_messages(args.dataset, question_ids, messages_by_question_id, tokenizer, model, args.engine)
        
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

    # Save final results
    save_responses(responses_data, responses_json_path)