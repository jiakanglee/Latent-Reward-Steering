# %%
import argparse
import sys
import json
import re
import torch
from tqdm import tqdm
import dotenv
from utils.utils import load_model, get_char_to_token_map, center_and_l2_normalize_torch
from utils.sae import load_sae
from utils.responses import extract_thinking_process

dotenv.load_dotenv("../.env")

# Add parent directory to path for imports
sys.path.append('..')
from utils import utils

parser = argparse.ArgumentParser(description="Annotate thinking processes in generated responses")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model used to generate responses")
parser.add_argument("--layer", type=int, default=6,
                    help="Layer to analyze")
parser.add_argument("--n_clusters", type=int, default=15,
                    help="Number of clusters in the SAE")
args, _ = parser.parse_known_args()


def split_into_sentences(text):
    """Split text into sentences using regex while preserving delimiters"""
    # Split on sentence boundaries but keep the delimiters
    sentences = re.split(r'(?<=[.!?;])', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def process_responses(responses_file, model, tokenizer, sae, layer, output_file, model_name):
    """Process responses and annotate thinking processes"""
    # Load responses
    with open(responses_file, 'r') as f:
        responses_data = json.load(f)
    
    device = model.device
    
    print(f"Processing {len(responses_data)} responses...")
    
    # Create new structure for annotated responses
    annotated_responses = []
    
    for idx, response_item in tqdm(enumerate(responses_data), total=len(responses_data)):
        # Get the full response and thinking process
        full_response = response_item['full_response']
        thinking_process = extract_thinking_process(full_response)
   
        # Split into sentences
        sentences = split_into_sentences(thinking_process)
 
        # Create mapping from character positions to token positions
        char_to_token = get_char_to_token_map(full_response, tokenizer)
        
        # Tokenize the full response
        tokens = tokenizer.encode(full_response, return_tensors="pt").to(device)
        
        # Run through model to get activations
        with torch.no_grad():
            with model.trace(
                {
                    "input_ids": tokens,
                    "attention_mask": (tokens != tokenizer.pad_token_id).long()
                }
            ) as tracer:
                activations = model.model.layers[layer].output.save()
        
        # Process each sentence
        annotated_thinking = ""
        
        for sentence in sentences:
            # Find this sentence in the full response
            # Pattern to match either at start of string or after punctuation/newlines
            pattern = r'(?:^|(?:[.?!;\n]|\n\n))\s*(' + re.escape(sentence) + ')'
            match = re.search(pattern, full_response)
            sentence_pos = match.start(1) if match else -1
            if sentence_pos < 0:
                # Sentence not found in full response
                continue
                
            # Get token positions for this sentence
            token_start = char_to_token.get(sentence_pos)
            token_end = char_to_token.get(sentence_pos + len(sentence) - 1)
            
            if token_start is None or token_end is None or token_start >= token_end or token_start >= activations.shape[1] or token_end > activations.shape[1]:
                # Invalid token range
                continue
            
            # Get activations for this sentence
            sentence_activations = activations[0, token_start-1:token_end, :]
            
            # Average the activations across all tokens in the sentence first
            avg_sentence_activation = torch.mean(sentence_activations, dim=0)
            
            # Match SAE training: center by the cached mean and L2-normalize before encoding.
            x_norm = center_and_l2_normalize_torch(avg_sentence_activation, sae.activation_mean)
            latent_activation = sae.encoder((x_norm - sae.b_dec).unsqueeze(0))
        
            # Find the latent with highest activation
            top_latent_idx = torch.argmax(latent_activation.squeeze(0)).item()
            
            # Use idx<number> instead of category title
            top_activation = round(latent_activation[0, top_latent_idx].item(), 2)
            
            # Format idx tag
            idx_tag = f"idx{top_latent_idx}"
            
            # Add to annotated thinking
            annotated_thinking += f'["{top_activation}:{idx_tag}"]{sentence}["end-section"]'
        
        # Create new annotated response item with only required fields
        annotated_item = {
            'question_id': response_item['question_id'],
            'category': response_item['category'],
            'dataset_name': response_item['dataset_name'],
            'annotated_thinking': annotated_thinking.strip()
        }
        annotated_responses.append(annotated_item)
        
        # Save intermediate results every 10 items
        if (idx + 1) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(annotated_responses, f, indent=2)
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(annotated_responses, f, indent=2)
    
    return annotated_responses

# %% Get model ID from model name
model_name = args.model
model_id = model_name.split('/')[-1].lower()

# %%
responses_file = f"results/vars/responses_{model_id}.json"
output_file = f"results/vars/annotated_responses_{model_id}.json"

# Load model and tokenizer
print(f"Loading model {model_name}...")
model, tokenizer = load_model(model_name=model_name)

# %% Load SAE
print(f"Loading SAE for model {model_id}, layer {args.layer}, clusters {args.n_clusters}...")
sae, _ = load_sae(model_id, args.layer, args.n_clusters)
sae = sae.to(model.device)

# %% Process responses
processed_data = process_responses(
    responses_file, 
    model, 
    tokenizer, 
    sae, 
    args.layer,
    output_file,
    model_name
)

print(f"Annotation complete. Annotated data saved to {output_file}") 

# %%
