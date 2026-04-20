# %%
import dotenv
dotenv.load_dotenv("../.env")

import argparse
from datasets import load_dataset
from utils.utils import load_model
from utils.sae import load_sae
from utils.hybrid import custom_hybrid_generate

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--target_layer", type=int, default=15)
parser.add_argument("--n_clusters", type=int, default=19)
parser.add_argument("--load_in_8bit", type=bool, default=False)
parser.add_argument("--test_dataset", type=str, default="HuggingFaceH4/MATH-500")
parser.add_argument("--test_dataset_split", type=str, default="test")
args, _ = parser.parse_known_args()

model_id = args.model_name.split("/")[-1].lower()

# %% Load models
model, tokenizer = load_model(
    model_name=args.model_name, 
    load_in_8bit=args.load_in_8bit
)

base_model, base_tokenizer = load_model(
    model_name=args.base_model_name, 
    load_in_8bit=args.load_in_8bit
)

# %% Load SAE
sae, checkpoint = load_sae(model_id, args.target_layer, args.n_clusters)

# %% Configure baselines
sae_baseline_config = {
    "sae": sae,
    "target_layer": args.target_layer,
    "threshold": 0.5
}

random_baseline_config = {
    "forced_token_rate": 0.1,
}

norm_diff_baseline_config = {
    "threshold": 0.1,
    "target_layer": args.target_layer
}

kl_div_baseline_config = {
    "threshold": 1,
}

# %% Load test data
data_idx = 8
test_dataset = load_dataset(args.test_dataset, streaming=True).shuffle(seed=42)
test_dataset = test_dataset[args.test_dataset_split]  # type: ignore

for i, example in enumerate(test_dataset):
    if i == data_idx:
        question = example["problem"]
        break

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    add_generation_prompt=True,
    return_tensors="pt"
)

# %% Generate with SAE baseline
hybrid_result = custom_hybrid_generate(
    model, 
    base_model, 
    base_tokenizer, 
    input_ids, 
    max_new_tokens=500, 
    baseline_method="sae",
    baseline_config=sae_baseline_config,
    warmup=7, 
    show_progress=True, 
    color_output=True
)

# %%
