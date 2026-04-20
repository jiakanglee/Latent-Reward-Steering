#!/bin/bash
#SBATCH --job-name=GenAct_Qwen7B
#SBATCH --output=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae/logs/generate_activations_qwen7b_%j.out
#SBATCH --error=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae/logs/generate_activations_qwen7b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:4

source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/train-sae
mkdir -p logs

python -u generate_activations.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --layers 4 8 12 16 20 24 \
    --n_examples 12032
