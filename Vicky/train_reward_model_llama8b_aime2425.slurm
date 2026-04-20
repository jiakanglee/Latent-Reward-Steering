#!/bin/bash
#SBATCH --job-name=TrainRM_Llama8B_AIME2425
#SBATCH --output=logs/train_rm_llama8b_aime2425_%j.out
#SBATCH --error=logs/train_rm_llama8b_aime2425_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab6
#SBATCH --gres=gpu:a5000:4

# --- 配置变量 ---
DATA_FILE="/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/collected_sae_latents_8b_15dim_4000_aime24_25.pt"
SAVE_PATH="/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/transformer_reward_model_llama8b_aime2425.pt"
EPOCHS=30
LR=5e-4
HIDDEN_DIM=128
# ---------------

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p Vicky/logs
export PYTHONPATH=$PWD:$PYTHONPATH

echo "======================================================"
echo "🚀 Training Reward Model for Llama-8B (AIME24+25)"
echo "Date      : $(date)"
echo "Node      : $(hostname)"
echo "Data      : $DATA_FILE"
echo "Save      : $SAVE_PATH"
echo "======================================================"

python -u Vicky/train_latent_classifier_llama8b.py \
    --data_file "$DATA_FILE" \
    --save_path "$SAVE_PATH" \
    --epochs $EPOCHS \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM

echo "======================================================"
echo "✅ Training finished. Date: $(date)"
echo "======================================================"
