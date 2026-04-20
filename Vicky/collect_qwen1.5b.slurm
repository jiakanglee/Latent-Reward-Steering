#!/bin/bash
#SBATCH --job-name=Collect_Qwen1.5B
#SBATCH --output=logs/collect_qwen1.5b_%j.out
#SBATCH --error=logs/collect_qwen1.5b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a4000:1

# --- 配置变量 ---
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LAYER_INDEX=20
N_CLUSTERS=10
MAX_TOKENS=2000
DATASET="math500"
OUTPUT_FILE="Vicky/collected_sae_latents_qwen1.5b_10dim_${MAX_TOKENS}_${DATASET}.pt"
# ---------------

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p logs
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p logs

python -u Vicky/generate_data_qwen1.5b.py \
  --model "$MODEL_NAME" \
  --sae_layer $LAYER_INDEX \
  --n_clusters $N_CLUSTERS \
  --max_token $MAX_TOKENS \
  --dataset $DATASET \
  --output_file "$OUTPUT_FILE"
