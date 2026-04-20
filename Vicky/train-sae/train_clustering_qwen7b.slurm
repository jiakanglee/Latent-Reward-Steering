#!/bin/bash
#SBATCH --job-name=TrainSAE_Qwen7B
#SBATCH --output=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae/logs/train_clustering_qwen7b_%A_%a.out
#SBATCH --error=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae/logs/train_clustering_qwen7b_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:4
#SBATCH --array=0-5

LAYERS=(4 8 12 16 20 24)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae
mkdir -p logs

echo "Training SAE for layer ${LAYER}..."

python -u train_clustering.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --layer ${LAYER} \
    --n_examples 12032 \
    --clusters 5 10 15 20 25 30 35 40 45 50 \
    --clustering_methods sae_topk
