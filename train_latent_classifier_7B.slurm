#!/bin/bash
#SBATCH --job-name=Train_Transformer
#SBATCH --output=log2/train_tf_%j.out
#SBATCH --error=log2/train_tf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab2
#SBATCH --gres=gpu:a4000:1

# === 1. 环境配置 ===
export PYTHONNOUSERSITE=1
source /common/home/jl3614/miniconda/etc/profile.d/conda.sh
conda activate stllms_env

cd /common/home/jl3614/Desktop/Python_project/thinking-llms-interp
mkdir -p log2
export PYTHONPATH=$PWD:$PYTHONPATH

# 检查 Python 解释器
if [ -x "$PWD/.venv/bin/python3" ]; then PYCMD="$PWD/.venv/bin/python3"; else PYCMD="python"; fi

echo "🚀 Starting Transformer Reward Model Training (Balanced)..."
echo "Date: $(date)"
echo "Node: $(hostname)"

# === 2. 参数设置 ===
# ⚠️ 确保这个文件名和你硬盘上的一致
DATA_FILE="collected_sae_latents_10dim_2000.pt"

# 模型保存路径
SAVE_PATH="transformer_reward_model.pt"

# Transformer 超参数 (与 LSTM 不同)
# d_model=128 对 10维特征足够了，太大容易过拟合
HIDDEN_DIM=128   
# Transformer 需要多跑几轮来拟合
EPOCHS=50
# Transformer 对学习率敏感，建议比 LSTM 小一点 (0.0005 vs 0.003)

# === 3. 执行训练 ===
echo "📋 Configuration:"
echo "   Input Data: $DATA_FILE"
echo "   Output Model: $SAVE_PATH"
echo "   Epochs: $EPOCHS"
echo "   Script: train_latent_classifier_7B.py"

# $PYCMD -u train_latent_classifier_7B.py \
#     --data_file "$DATA_FILE" \
#     --save_path "$SAVE_PATH" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --hidden_dim $HIDDEN_DIM

# echo "✅ Training Finished for learning rate $LR at $(date)"

# $PYCMD -u train_latent_classifier_7B.py \
#     --data_file "$DATA_FILE" \
#     --save_path "$SAVE_PATH" \
#     --epochs $EPOCHS \
#     --lr 0.001 \
#     --hidden_dim $HIDDEN_DIM
# echo "✅ Training Finished for learning rate 0.001 at $(date)"

# $PYCMD -u train_latent_classifier_7B.py \
#     --data_file "$DATA_FILE" \
#     --save_path "$SAVE_PATH" \
#     --epochs $EPOCHS \
#     --lr 0.0006 \
#     --hidden_dim $HIDDEN_DIM
# echo "✅ Training Finished for learning rate 0.0006 at $(date)"

$PYCMD -u train_latent_classifier_7B.py \
    --data_file "$DATA_FILE" \
    --save_path "$SAVE_PATH" \
    --epochs $EPOCHS \
    --lr 0.0003 \
    --hidden_dim $HIDDEN_DIM
echo "✅ Training Finished for learning rate 0.0003 at $(date)"

# $PYCMD -u train_latent_classifier_7B.py \
#     --data_file "$DATA_FILE" \
#     --save_path "$SAVE_PATH" \
#     --epochs $EPOCHS \
#     --lr 0.0001 \
#     --hidden_dim $HIDDEN_DIM
# echo "✅ Training Finished for learning rate 0.0001 at $(date)"