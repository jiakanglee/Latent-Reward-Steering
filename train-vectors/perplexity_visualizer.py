# %%
import argparse
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import json
import os
import random
from tqdm import tqdm
from utils.responses import extract_thinking_process
dotenv.load_dotenv("../.env")

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize per-token perplexity of a model")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model to analyze perplexity for")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--n_examples", type=int, default=10,
                    help="Number of examples to use for evaluation (0 = all)")
args, _ = parser.parse_known_args()

# Best prompt templates for base LMs (not instruction-tuned)
prompt_templates = [
    # 1. Raw continuation: Most natural for base LMs
    "{question}\n{answer}",
    "Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n{answer}",
    "Question:\n{question}\n\nStep by step answer:\n{answer}",
    "Solve the following question step by step.\n\nQuestion:\n{question}\n\nStep by step answer:\n{answer}",
    "I will answer the question '{question}' step by step. {answer}",
]

def calculate_token_perplexities(model, tokenizer, prompt_text):
    """Calculate perplexity for each token position in the input text"""
    input_ids = tokenizer.encode(
        prompt_text,
        return_tensors="pt"
    ).to(model.device)
    target_ids = input_ids.clone()
    with torch.no_grad():
        with model.trace({"input_ids": input_ids}) as tracer:
            logits = model.lm_head.output.save()
    logits = logits.to(torch.float32)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())
    token_perplexities = torch.exp(token_losses).cpu().numpy()[0]
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0].cpu().numpy()]
    tokens = tokens[:-1]
    return tokens, token_perplexities

def plot_template_perplexities(template_results):
    plt.figure(figsize=(10, 6))
    template_names = list(template_results.keys())
    avg_perplexities = [template_results[name]["avg_perplexity"] for name in template_names]
    std_perplexities = [template_results[name]["std_perplexity"] for name in template_names]
    bars = plt.bar([x[:15] for x in template_names], avg_perplexities, yerr=std_perplexities, 
                  color='skyblue', capsize=5, alpha=0.7)
    for bar, std in zip(bars, std_perplexities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    plt.title(f"Average Perplexity for Different Prompt Templates ({args.model})")
    plt.xlabel("Prompt Template")
    plt.ylabel("Average Perplexity (lower is better)")
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"results/figures/perplexity_templates_{args.model.split('/')[-1].lower()}.pdf")

def main():
    model_name = args.model
    print(f"Loading model {model_name}...")
    model, tokenizer = utils.load_model(model_name=model_name, load_in_8bit=args.load_in_8bit)
    thinking_model_id = utils.model_mapping[model_name].split("/")[1].lower()
    responses_json_path = f"../generate-responses/results/vars/responses_{thinking_model_id}.json"
    if not os.path.exists(responses_json_path):
        raise FileNotFoundError(f"Annotated responses file not found at {responses_json_path}. Please generate responses first.")
    print(f"Loading responses from {responses_json_path}")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)
    # Limit number of examples if requested
    if args.n_examples > 0 and args.n_examples < len(responses_data):
        responses_data = random.sample(responses_data, args.n_examples)
    template_results = {}
    for template in prompt_templates:
        all_perplexities = []
        for ex in tqdm(responses_data, desc=f"Template: {template[:30]}..."):
            question = ex["original_message"]["content"]
            answer = extract_thinking_process(ex["full_response"])
            prompt_text = template.format(question=question, answer=answer)
            _, perplexities = calculate_token_perplexities(model, tokenizer, prompt_text)
            avg_perplexity = np.mean([p for p in perplexities if p < 10000])
            all_perplexities.append(avg_perplexity)
        template_results[template] = {
            "avg_perplexity": np.mean(all_perplexities),
            "std_perplexity": np.std(all_perplexities),
            "all_runs": all_perplexities
        }
    plot_template_perplexities(template_results)
    print("\nSummary of Perplexity Results:")
    print(f"{'Template':<40} {'Avg Perplexity':<15} {'Std Dev':<10}")
    print(f"{'-'*70}")
    for template, result in template_results.items():
        print(f"{template[:35]:<40} {result['avg_perplexity']:<15.2f} {result['std_perplexity']:<10.2f}")
    print("\nIndividual Run Results (first 5 per template):")
    for template, result in template_results.items():
        print(f"\nTemplate: {template[:35]}")
        for i, perplexity in enumerate(result["all_runs"][:5]):
            print(f"  Example {i+1}: {perplexity:.2f}")
        if len(result["all_runs"]) > 5:
            print(f"  ... and {len(result['all_runs']) - 5} more examples")

if __name__ == "__main__":
    main()

# %%
