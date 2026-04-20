#!/usr/bin/env bash
python hybrid_token.py --dataset math500 --thinking_model qwen/QwQ-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 10 --max_new_tokens 2000  --max_thinking_tokens 2000 --random-firing

python hybrid_token.py --dataset aime24 --thinking_model qwen/QwQ-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 10 --max_new_tokens 4000  --max_thinking_tokens 4000 --random-firing

python hybrid_token.py --dataset aime25 --thinking_model qwen/QwQ-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 10 --max_new_tokens 4000  --max_thinking_tokens 4000 --random-firing

python hybrid_token.py --dataset mbpp --thinking_model qwen/QwQ-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 10 --max_new_tokens 4000  --max_thinking_tokens 4000 --random-firing

python hybrid_token.py --dataset livecodebench --thinking_model qwen/QwQ-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 10 --max_new_tokens 4000  --max_thinking_tokens 4000 --random-firing

python hybrid_token.py --dataset medqa --thinking_model qwen/QwQ-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 10 --max_new_tokens 4000  --max_thinking_tokens 4000 --random-firing

python hybrid_token.py --dataset legalbench --thinking_model qwen/QwQ-32B --base_model Qwen/Qwen2.5-32B --steering_layer 24  --sae_layer 27 --n_clusters 10 --max_new_tokens 4000  --max_thinking_tokens 4000 --random-firing


