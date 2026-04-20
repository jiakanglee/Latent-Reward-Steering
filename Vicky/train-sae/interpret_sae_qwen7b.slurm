#!/bin/bash
#SBATCH --job-name=InterpSAE_Qwen7B
#SBATCH --output=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae/logs/interp_sae_qwen7b_%A_%a.out
#SBATCH --error=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae/logs/interp_sae_qwen7b_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab2
#SBATCH --array=0-5

LAYERS=(4 8 12 16 20 24)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae
mkdir -p logs

echo "Interpreting SAE for layer ${LAYER}..."

python -u interpret_sae_qwen7b.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --layer ${LAYER} \
    --n_examples 12032 \
    --n_clusters 5 10 15 20 25 30 35 40 45 50 \
    --top_k 20
