#!/bin/bash

# Llama 8B
python visualize_vector_losses.py --model meta-llama/Llama-3.1-8B --smoothing_sigma 100 --steering_strategy linear
python visualize_vector_losses.py --model meta-llama/Llama-3.1-8B --smoothing_sigma 100 --steering_strategy adaptive_linear
python visualize_vector_losses.py --model meta-llama/Llama-3.1-8B --smoothing_sigma 100 --steering_strategy resid_lora

# Llama 70B
# python visualize_vector_losses.py --model meta-llama/Llama-3.3-70B-Instruct --smoothing_sigma 100 --steering_strategy linear

# Qwen 32B (trained on QwQ)
python visualize_vector_losses.py --model Qwen/Qwen2.5-32B --smoothing_sigma 100 --steering_strategy linear
python visualize_vector_losses.py --model Qwen/Qwen2.5-32B --smoothing_sigma 100 --steering_strategy adaptive_linear

# Qwen 32B (trained on Deepseek)
python visualize_vector_losses.py --model Qwen/Qwen2.5-32B --thinking_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --smoothing_sigma 100 --steering_strategy linear

# Qwen 14B
python visualize_vector_losses.py --model Qwen/Qwen2.5-14B --smoothing_sigma 100 --steering_strategy linear

# Qwen 1.5B
python visualize_vector_losses.py --model Qwen/Qwen2.5-Math-1.5B --smoothing_sigma 100 --steering_strategy linear
