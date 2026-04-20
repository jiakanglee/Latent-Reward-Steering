#!/bin/bash
#SBATCH --job-name=TrainRM_Qwen1.5B
#SBATCH --output=logs/train_rm_qwen1.5b_%j.out
#SBATCH --error=logs/train_rm_qwen1.5b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a4000:1

# --- 配置变量 ---
DATA_FILE="/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/collected_sae_latents_qwen1.5b_10dim_2000_math500.pt"
SAVE_PATH="/common/home/jl3614/Desktop/Python_project/thinking-llms-interp/Vicky/transformer_reward_model_qwen1.5b.pt"
EPOCHS=30
LR=5e-4
HIDDEN_DIM=128
# ---------------

export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p logs
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p logs

echo "======================================================"
echo "🚀 Training Reward Model for Qwen-1.5B"
echo "Date      : $(date)"
echo "Node      : $(hostname)"
echo "Data      : $DATA_FILE"
echo "Save      : $SAVE_PATH"
echo "======================================================"

python -u train_reward_model_prefix.py \
    --data_file "$DATA_FILE" \
    --save_path "$SAVE_PATH" \
    --epochs $EPOCHS \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM

echo "======================================================"
echo "✅ Training finished. Date: $(date)"
echo "======================================================"
