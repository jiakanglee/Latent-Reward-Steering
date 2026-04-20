#!/bin/bash
#SBATCH --job-name=GenResp_Qwen7B
#SBATCH --output=logs/generate_qwen7b_%j.out
#SBATCH --error=logs/generate_qwen7b_%j.err
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
export PYTHONPATH=/common/home/jl3614/Desktop/Python_project/thinking-llms-interp:$PYTHONPATH

python -u generate_responses.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --engine nnsight \
    --batch_size 32
