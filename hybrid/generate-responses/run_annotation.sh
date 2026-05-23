#!/usr/bin/env bash
python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer 6 --n_clusters 15

python annotate_thinking.py --model qwen/QwQ-32B --layer 27 --n_clusters 10

python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 4 --n_clusters 15

python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 38 --n_clusters 5

python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --layer 27 --n_clusters 15

python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --layer 11 --n_clusters 10

python annotate_thinking.py --model Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B --layer 8 --n_clusters 10

python annotate_thinking.py --model Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B --layer 8 --n_clusters 5

python annotate_thinking.py --model Open-Reasoner-Zero/Open-Reasoner-Zero-7B --layer 20 --n_clusters 10

python annotate_thinking.py --model Open-Reasoner-Zero/Open-Reasoner-Zero-32B --layer 27 --n_clusters 15