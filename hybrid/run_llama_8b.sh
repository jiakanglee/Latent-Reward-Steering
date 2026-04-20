#!/usr/bin/env bash
python hybrid_token.py --dataset gsm8k --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 2000 --max_thinking_tokens 2000 --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --token_windows 0 -1 -15 -50 -100

python hybrid_token.py --dataset math500 --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 2000 --max_thinking_tokens 2000 --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --token_windows 0 -1 -15 -50 -100

python hybrid_token.py --dataset aime24 --only-finished-thinking --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 4000 --max_thinking_tokens 4000 --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --token_windows 0 -1 -15 -50 -100

python hybrid_token.py --dataset aime25 --only-finished-thinking --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 4000 --max_thinking_tokens 4000 --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --token_windows 0 -1 -15 -50 -100

python hybrid_token.py --dataset mbpp --only-finished-thinking --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 4000 --max_thinking_tokens 4000 --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --token_windows 0 -1 -15 -50 -100

python hybrid_token.py --dataset livecodebench --only-finished-thinking --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 4000 --max_thinking_tokens 4000 --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --token_windows 0 -1 -15 -50 -100

python hybrid_token.py --dataset medqa --only-finished-thinking --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 4000 --max_thinking_tokens 4000 --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --token_windows 0 -1 -15 -50 -100

python hybrid_token.py --dataset legalbench --only-finished-thinking --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 4000 --max_thinking_tokens 4000 --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --token_windows 0 -1 -15 -50 -100
