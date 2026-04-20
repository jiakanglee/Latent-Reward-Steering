#!/bin/bash
#SBATCH --job-name=Patch_Llama8B_SAE_Mean
#SBATCH --output=logs/patch_llama8b_sae_mean_%j.out
#SBATCH --error=logs/patch_llama8b_sae_mean_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:4

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p logs
mkdir -p Vicky/logs
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p Vicky/logs

python -u Vicky/patch_llama8b_sae_mean.py

echo "Patch done."
