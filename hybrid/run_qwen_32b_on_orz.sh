#!/usr/bin/env bash
python hybrid_token.py --dataset gsm8k --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 15 --max_new_tokens 2000  --max_thinking_tokens 2000

python hybrid_token.py --dataset math500 --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 15 --max_new_tokens 2000  --max_thinking_tokens 2000

python hybrid_token.py --dataset aime24 --only-finished-thinking --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 15 --max_new_tokens 4000  --max_thinking_tokens 4000

python hybrid_token.py --dataset aime25 --only-finished-thinking --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 15 --max_new_tokens 4000  --max_thinking_tokens 4000

python hybrid_token.py --dataset mbpp --only-finished-thinking --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 15 --max_new_tokens 4000  --max_thinking_tokens 4000

python hybrid_token.py --dataset livecodebench --only-finished-thinking --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 15 --max_new_tokens 4000  --max_thinking_tokens 4000

python hybrid_token.py --dataset medqa --only-finished-thinking --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 15 --max_new_tokens 4000  --max_thinking_tokens 4000

python hybrid_token.py --dataset legalbench --only-finished-thinking --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 15 --max_new_tokens 4000  --max_thinking_tokens 4000
