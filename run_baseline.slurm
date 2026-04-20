#!/bin/bash
#SBATCH --job-name=SAE_Steer_Eval
#SBATCH --output=log2/steer_%j.out
#SBATCH --error=log2/steer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a4000:1

# --- 配置变量 ---
MODEL_NAME="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
SAE_FILE="/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/train-saes/results/vars/saes/sae_open-reasoner-zero-7b_layer20_clusters10.pt"
LAYER_INDEX=20
MAX_TOKENS=1000
# 跑前 200 个
RUN_CONFIG="--example_idx 2 8 10 12 15 34 37 61 65 81 93 97 101 104"
# ---------------

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p logs

python -u run_baseline.py \
  --model "$MODEL_NAME" \
  --sae_path "$SAE_FILE" \
  --sae_layer $LAYER_INDEX \
  $RUN_CONFIG \
  --max_new_tokens $MAX_TOKENS \
  --output_file "log2/batch_steer_layer${LAYER_INDEX}.jsonl" \
  --print_response \
  --print_activations\
  --multiplier 1.0